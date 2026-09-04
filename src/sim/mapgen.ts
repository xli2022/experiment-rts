/**
 * Map generation.
 *
 * Two layouts share one carver. `Lanes` is the 1v1 map described below;
 * `Quarters` is the four-start co-op map, documented at `quartersLayout`.
 * Everything after the layout is common to both, which is the point of keeping
 * the shape as data: a new map is a new polyline set, not a new generator.
 *
 * ## Lanes: a three-lane layout inspired by the MOBA standard
 *
 * Two bases sit in opposite corners, joined by three routes:
 *
 *   - a **middle** lane running between them through the centre,
 *   - two **outer** lanes that hug the map edge via the other two corners.
 *
 * Short **connectors** cut between the lanes, so an army can rotate between
 * routes without walking all the way home, and small open **clearings** sit off
 * the lanes as room to fight in and to build forward.
 *
 * ## Why carve rather than scatter
 *
 * The previous generator scattered random rock blobs and then repaired
 * connectivity if it happened to seal a base in. That produces terrain but not
 * *structure*: no chokepoint worth holding, no flanking route, no reason to look
 * at the minimap. Carving from a solid block inverts the problem — the layout is
 * designed, and the open space is the deliberate part.
 *
 * ## Symmetry is a property of the brush, not of the layout
 *
 * Every write goes through `open`, which opens the tile **and** its 180-degree
 * opposite. Nothing downstream can break the mirror: lanes may wobble, be carved
 * in any order, or overlap, and the two halves still come out bit-identical. The
 * alternative — writing each shape twice and trusting both copies to stay in
 * step — failed the moment the corridor brush gained a random wobble, because
 * each copy drew its own wobble from the same stream.
 *
 * A consequence worth knowing: a lane spanning both halves is carved twice, once
 * directly and once as the mirror of its opposite. That is harmless — carving is
 * a union — and it is why the middle lane, which is its own mirror, ends up a
 * little wider than the outers. Mid being the widest and most contested route is
 * exactly what we want anyway.
 */

import { Rng } from './rng.js';
import { MapLayout, Tile } from './types.js';

/** A point in tile space. */
export interface Pt {
  x: number;
  y: number;
}

export interface LaneSpec {
  /** Waypoints the lane passes through, in order. */
  points: Pt[];
  /** Half-width in tiles. The carved corridor is roughly twice this across. */
  radius: number;
}

export interface Layout {
  /**
   * Start locations, in *mirrored halves*: entry `i` and entry `i + n/2` are
   * exact 180-degree rotations of one another.
   *
   * The convention is what makes team fairness structural rather than something
   * to keep re-checking. Team 0 takes the first half, team 1 the second, so the
   * two sides are rotations of each other by construction — and player `p`'s
   * opposite number is always `p + n/2`, whatever the layout.
   *
   * Allies within a half are *not* mirrors of each other, and cannot be: only
   * one symmetry is guaranteed, and it is the one between the teams.
   */
  bases: Pt[];
  /**
   * Second base sites, with room for a Command Post and a mineral line.
   *
   * Same mirrored-halves convention as `bases` — the mineral line in the second
   * half is laid out reflected, so a pair is an exact 180-degree rotation of
   * each other rather than merely the same shape in the same orientation.
   */
  expansions: Pt[];
  lanes: LaneSpec[];
  clearings: { at: Pt; radius: number }[];
}

/**
 * Index of the canonical (first-half) entry that `i` is a rotation of, and
 * whether `i` is itself the rotated copy.
 *
 * Everything laid out at a start or an expansion — Command Post, mineral line,
 * starting workers — is placed from the canonical site and rotated for the
 * other half. Applying the same offsets to a mirrored site instead is *nearly*
 * the same thing, and is the bug that put one player's whole base a tile nearer
 * the middle of the map.
 */
export function mirroredHalf(index: number, count: number): { canonical: number; flip: boolean } {
  const half = count >> 1;
  return { canonical: half === 0 ? index : index % half, flip: index >= half };
}

/** Where the bases sit, as a fraction in from the corner. */
const BASE_INSET = 0.16;
/** How far the outer lanes' corner waypoint sits from the map edge. */
const EDGE_INSET = 0.11;

/** Main lanes are wide enough for an army and some building. */
const LANE_RADIUS = 4;
/** The corner-to-corner route across the middle. */
const RIVER_RADIUS = 3;
/** Connectors are tighter — they are flanking routes, not highways. */
const CONNECTOR_RADIUS = 3;
/** The open ground each player builds in. */
const BASE_RADIUS = 11;
/** Pockets off the lanes. */
const CLEARING_RADIUS = 4;
/**
 * The pocket an expansion sits in.
 *
 * Wide enough for a Command Post, its mineral line, and room to walk around
 * both — a base site you cannot actually finish building on is worse than no
 * base site at all.
 */
const EXPANSION_RADIUS = 10;

/** Cliffs stop getting taller past this many tiles from open ground. */
const MAX_ELEVATION = 6;

/** Rotate a point 180 degrees about the centre of a `size` x `size` map. */
export function mirror(p: Pt, size: number): Pt {
  return { x: size - 1 - p.x, y: size - 1 - p.y };
}

/** Steps needed to walk a segment with a king-move brush. */
function segSteps(p0: Pt, p1: Pt): number {
  return Math.max(Math.abs(p1.x - p0.x), Math.abs(p1.y - p0.y));
}

/**
 * Sample a polyline at fraction `t` of its length.
 *
 * Connectors are anchored to points *on* the lanes rather than to hand-written
 * coordinates, so moving a lane drags its connectors with it instead of leaving
 * a corridor that runs into a cliff.
 */
export function pointAt(points: Pt[], t: number): Pt {
  let total = 0;
  for (let i = 0; i + 1 < points.length; i++) total += segSteps(points[i]!, points[i + 1]!);

  let want = Math.round(total * t);
  for (let i = 0; i + 1 < points.length; i++) {
    const p0 = points[i]!;
    const p1 = points[i + 1]!;
    const steps = segSteps(p0, p1);
    if (want <= steps || i === points.length - 2) {
      let f = steps === 0 ? 0 : want / steps;
      if (f < 0) f = 0;
      if (f > 1) f = 1;
      return {
        x: Math.round(p0.x + (p1.x - p0.x) * f),
        y: Math.round(p0.y + (p1.y - p0.y) * f),
      };
    }
    want -= steps;
  }
  return points[points.length - 1]!;
}

/**
 * The point on a polyline closest to `target`.
 *
 * Connectors are cut from a lane to whichever part of the neighbouring lane is
 * actually nearest, so they come out short and roughly perpendicular. Matching
 * the two lanes by equal fraction instead produced a 50-tile diagonal gash,
 * because the outer lanes are half again as long as mid.
 */
export function nearestOn(points: Pt[], target: Pt): Pt {
  let best = points[0]!;
  let bestD = Infinity;
  for (let i = 0; i + 1 < points.length; i++) {
    const p0 = points[i]!;
    const p1 = points[i + 1]!;
    const steps = segSteps(p0, p1);
    for (let s = 0; s <= steps; s++) {
      const f = steps === 0 ? 0 : s / steps;
      const x = Math.round(p0.x + (p1.x - p0.x) * f);
      const y = Math.round(p0.y + (p1.y - p0.y) * f);
      const d = (x - target.x) * (x - target.x) + (y - target.y) * (y - target.y);
      if (d < bestD) {
        bestD = d;
        best = { x, y };
      }
    }
  }
  return best;
}

/**
 * How many start locations a layout has, and therefore how many players fit.
 *
 * Always even: the whole fairness argument rests on the roster splitting into
 * two mirrored halves.
 */
export function layoutStarts(layout: MapLayout): number {
  return layout === MapLayout.Quarters ? 4 : 2;
}

/**
 * The side length a layout is designed for.
 *
 * Four economies want more ground than two. The `Lanes` map keeps its size so
 * every existing replay, checksum and screenshot still means what it did.
 */
export function layoutSize(layout: MapLayout): number {
  return layout === MapLayout.Quarters ? QUARTERS_SIZE : LANES_SIZE;
}

/** Side length of the 1v1 map. Mirrors `MAP_SIZE` in `map.ts`. */
const LANES_SIZE = 128;
/**
 * Side length of the 2v2 map.
 *
 * Bigger than the duel map because it holds twice the economy, and no bigger:
 * every extra tile is flow-field and fog work every tick, and four bases at
 * this size still put the front line within a comfortable march of home.
 */
const QUARTERS_SIZE = 152;

/**
 * The layout for a map of this size, as data rather than tiles.
 *
 * Kept separate from carving so it can be inspected and tested: asserting things
 * about six polylines is far easier than asserting them about 16,384 tiles.
 */
export function buildLayout(size: number, layout: MapLayout = MapLayout.Lanes): Layout {
  return layout === MapLayout.Quarters ? quartersLayout(size) : lanesLayout(size);
}

/**
 * `Quarters`: four starts, two to a side, with a back lane of their own.
 *
 * Bases sit in the four corners and the teams take a side each — the first half
 * of `bases` along the top edge, their 180-degree opposites along the bottom.
 * That is the only arrangement of four corners that keeps allies adjacent *and*
 * keeps the two teams exact rotations of each other; splitting them diagonally
 * would put each player's partner across the map from them, which is a 1v1
 * played twice rather than a co-op match.
 *
 * Four kinds of route, and each exists for a reason:
 *
 *   - a **back lane** joining a team's two bases, so reinforcing a partner does
 *     not mean walking through the middle and past the enemy;
 *   - two **flanks** down the left and right edges, where each player faces the
 *     opponent directly opposite them;
 *   - two **diagonals** crossing at the centre, which is what lets a team
 *     concentrate on one opponent instead of fighting two separate 1v1s;
 *   - short **approaches** from each base to its own natural expansion.
 *
 * One spec often carves two routes: the mirrored brush means authoring the left
 * flank also cuts the right one, and the top back lane also cuts the bottom.
 */
function quartersLayout(size: number): Layout {
  const inset = Math.floor(size * BASE_INSET);
  // The two canonical bases: this team's corners. Their rotations are the
  // other team's, and the brush carves both from either.
  const a0: Pt = { x: inset, y: inset };
  const a1: Pt = { x: size - 1 - inset, y: inset };
  const b0 = mirror(a0, size);
  const b1 = mirror(a1, size);
  const mid: Pt = { x: (size >> 1) - 1, y: (size >> 1) - 1 };

  // Back lane: along this team's edge, bowed toward the corner so it hugs the
  // edge rather than cutting the corner off the map. Its mirror is the other
  // team's back lane, so one spec gives both sides the same road home.
  const backBow: Pt = { x: size >> 1, y: Math.floor(size * 0.1) };
  const back: LaneSpec = { points: [a0, backBow, a1], radius: CONNECTOR_RADIUS };

  // Flanks: straight down an edge, base to base. `a0 -> b1` runs down the left,
  // and its rotation is `b0 -> a1` down the right, so the two players on a team
  // each get an edge and each faces one opponent across it.
  const flank: LaneSpec = { points: [a0, b1], radius: LANE_RADIUS };

  // Diagonals: each base to the one diagonally opposite, through the centre.
  // Both are their own mirrors, so both are carved twice and come out the
  // widest routes on the map — which is right, because the centre is what a
  // team has to hold to fight together rather than separately.
  const diagonalA: LaneSpec = { points: [a0, mid, b0], radius: RIVER_RADIUS };
  const diagonalB: LaneSpec = { points: [a1, mid, b1], radius: RIVER_RADIUS };

  // Naturals sit off the diagonal a third of the way toward the middle: close
  // enough to defend from home, far enough that taking one is a commitment.
  const natural0 = pointAt([a0, mid], 0.42);
  const natural1 = pointAt([a1, mid], 0.42);
  const approach0: LaneSpec = { points: [a0, natural0], radius: CONNECTOR_RADIUS };
  const approach1: LaneSpec = { points: [a1, natural1], radius: CONNECTOR_RADIUS };

  // The contested pair: halfway down each flank, equidistant from both teams.
  // Its own rotation is the far flank's site, so the two are a mirrored pair
  // like every other expansion.
  const contested = nearestOn(flank.points, { x: inset, y: size >> 1 });

  const lanes: LaneSpec[] = [back, flank, diagonalA, diagonalB, approach0, approach1];

  const expansions: Pt[] = [
    natural0,
    natural1,
    contested,
    mirror(natural0, size),
    mirror(natural1, size),
    mirror(contested, size),
  ];

  const clearings = [
    { at: natural0, radius: EXPANSION_RADIUS },
    { at: natural1, radius: EXPANSION_RADIUS },
    { at: contested, radius: EXPANSION_RADIUS - 2 },
    // The centre crossroads, where both diagonals and both teams meet.
    { at: mid, radius: CLEARING_RADIUS + 4 },
    // Where the back lane meets each flank: the ground a team stages on before
    // committing, and the last open space before its own front door.
    { at: pointAt(back.points, 0.5), radius: CLEARING_RADIUS },
  ];

  return { bases: [a0, a1, b0, b1], expansions, lanes, clearings };
}

/** `Lanes`: the two-start duel map. See the module comment. */
function lanesLayout(size: number): Layout {
  const inset = Math.floor(size * BASE_INSET);
  const a: Pt = { x: inset, y: inset };
  const b = mirror(a, size);
  // Two adjacent tiles straddling the exact centre, so a lane through the middle
  // is symmetric about it rather than pivoting on one side.
  const mid: Pt = { x: (size >> 1) - 1, y: (size >> 1) - 1 };

  // Middle lane: an S through the centre. Bowed off the true diagonal so it is a
  // route with corners to fight over rather than a ruler-straight line.
  //
  // It leaves the base at 45 degrees and only bends once clear of it. Bending
  // immediately pointed mid the same way as an outer lane, and the two carved
  // into each other into a single open flank.
  const leave: Pt = { x: Math.floor(size * 0.25), y: Math.floor(size * 0.27) };
  const bow: Pt = { x: Math.floor(size * 0.36), y: Math.floor(size * 0.45) };
  const middle: LaneSpec = {
    points: [a, leave, bow, mid, mirror(mid, size), mirror(bow, size), mirror(leave, size), b],
    radius: LANE_RADIUS,
  };

  // Outer lanes run out to a corner and then along to the far base. One spec
  // produces both: the mirrored brush carves the opposite corner's lane too.
  const edge = Math.floor(size * EDGE_INSET);
  const cornerLow: Pt = { x: edge, y: size - 1 - edge };
  const outer: LaneSpec = {
    points: [a, cornerLow, b],
    radius: LANE_RADIUS,
  };

  // The river: corner to corner across the middle, crossing mid dead centre.
  // It is the one route that touches all three lanes, which makes the centre a
  // four-way crossroads and the natural thing to fight over. Its endpoints are
  // each other's mirror, so it is its own reflection.
  const river: LaneSpec = {
    points: [cornerLow, mirror(cornerLow, size)],
    radius: RIVER_RADIUS,
  };

  // Connectors tie mid to whichever outer lane runs nearest. Because mid bows
  // toward one side in its first half, that is the low outer lane near home and
  // — after mirroring — the high one out near the enemy. Flanking therefore
  // costs a commitment rather than being a free rotation.
  const connectorFromMid = (t: number): LaneSpec => {
    const from = pointAt(middle.points, t);
    return { points: [from, nearestOn(outer.points, from)], radius: CONNECTOR_RADIUS };
  };
  const connectors = [connectorFromMid(0.16), connectorFromMid(0.32)];

  const lanes: LaneSpec[] = [middle, outer, river, ...connectors];

  // The expansion sits in the pocket the outer connector cuts, roughly a third
  // of the way out from home. Off both lanes, so taking it is a decision rather
  // than something that happens on the way past — but with an entrance at each
  // end, so holding it costs something too.
  const expansion = pointAt(connectors[1]!.points, 0.5);
  const expansions: Pt[] = [expansion, mirror(expansion, size)];

  // Clearings: pockets of open ground where corridors meet. Somewhere to stage
  // an army, or put a forward building, that is not a chokepoint.
  const clearings = [
    { at: pointAt(connectors[0]!.points, 0.5), radius: CLEARING_RADIUS },
    { at: expansion, radius: EXPANSION_RADIUS },
    // Where the river meets the outer lane — the map's two far corners.
    { at: cornerLow, radius: CLEARING_RADIUS + 2 },
    // The centre crossroads.
    { at: mid, radius: CLEARING_RADIUS + 2 },
  ];

  return { bases: [a, b], expansions, lanes, clearings };
}

/**
 * Carve the layout into a tile grid that starts out solid.
 *
 * `tiles` and `elevation` are modified in place. Everything not carved stays
 * cliff, which is what gives the map its shape.
 */
export function carveLayout(
  tiles: Uint8Array,
  elevation: Uint8Array,
  size: number,
  seed: number,
  kind: MapLayout = MapLayout.Lanes,
): { bases: Pt[]; expansions: Pt[]; layout: Layout } {
  tiles.fill(Tile.Cliff);

  const layout = buildLayout(size, kind);
  const rng = new Rng(seed ^ 0x2f6b3c11);

  /**
   * Open a tile and its 180-degree opposite.
   *
   * This is the whole symmetry guarantee. Keep it that way: an `open` that
   * writes one tile turns fairness back into something to keep re-checking.
   */
  const open = (x: number, y: number): void => {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    tiles[y * size + x] = Tile.Ground;
    tiles[(size - 1 - y) * size + (size - 1 - x)] = Tile.Ground;
  };

  const disc = (cx: number, cy: number, r: number): void => {
    const rr = r * r;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > rr) continue;
        open(cx + dx, cy + dy);
      }
    }
  };

  /**
   * Carve a corridor along a polyline.
   *
   * The radius wobbles by at most a tile as it goes, so lanes do not read as
   * drawn with a ruler, and never narrows enough to pinch a corridor shut.
   */
  const corridor = (lane: LaneSpec): void => {
    for (let s = 0; s + 1 < lane.points.length; s++) {
      const p0 = lane.points[s]!;
      const p1 = lane.points[s + 1]!;
      const steps = segSteps(p0, p1);
      for (let i = 0; i <= steps; i++) {
        const t = steps === 0 ? 0 : i / steps;
        const x = Math.round(p0.x + (p1.x - p0.x) * t);
        const y = Math.round(p0.y + (p1.y - p0.y) * t);
        const jitter = rng.nextInt(3) - 1;
        disc(x, y, Math.max(2, lane.radius + jitter));
      }
    }
  };

  // Bases first, so lane carving can only ever widen them.
  for (const base of layout.bases) disc(base.x, base.y, BASE_RADIUS);
  for (const lane of layout.lanes) corridor(lane);
  for (const c of layout.clearings) disc(c.at.x, c.at.y, c.radius);

  // A solid border, so units cannot walk along the very edge of the world and so
  // the map reads as an arena rather than a cropped plane.
  for (let i = 0; i < size; i++) {
    for (let b = 0; b < 2; b++) {
      tiles[b * size + i] = Tile.Cliff;
      tiles[(size - 1 - b) * size + i] = Tile.Cliff;
      tiles[i * size + b] = Tile.Cliff;
      tiles[i * size + (size - 1 - b)] = Tile.Cliff;
    }
  }

  raiseCliffs(tiles, elevation, size);
  return { bases: layout.bases, expansions: layout.expansions, layout };
}

/**
 * Give every cliff tile a height from its distance to the nearest open ground.
 *
 * Purely cosmetic — the simulation decides movement from walkability alone — but
 * it is what turns a uniform grey slab into ridges that rise away from the
 * lanes, and it costs one breadth-first sweep at load.
 *
 * The queue is a plain array with a moving head and is seeded in index order, so
 * the result is identical on every machine even though nothing checksums it.
 */
function raiseCliffs(tiles: Uint8Array, elevation: Uint8Array, size: number): void {
  const n = size * size;
  elevation.fill(0);

  const queue: number[] = [];
  const seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (tiles[i] !== Tile.Cliff) {
      seen[i] = 1;
      queue.push(i);
    }
  }

  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    const cx = cur % size;
    const cy = (cur / size) | 0;
    // Saturate rather than stop: the deep interior of a massif is uniformly
    // tall, but it must still be *tall*, not left at ground height.
    const next = Math.min(elevation[cur]! + 1, MAX_ELEVATION);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
        const ni = ny * size + nx;
        if (seen[ni] === 1) continue;
        seen[ni] = 1;
        elevation[ni] = next;
        queue.push(ni);
      }
    }
  }
}
