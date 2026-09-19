/**
 * Straight-line visibility over the tile grid, and the path smoothing built on
 * it.
 *
 * ## Why this exists
 *
 * A* and the flow field both answer in *tiles*, and a unit that walks the
 * answer literally can only ever travel in the eight directions a tile grid
 * offers. Any real bearing that is not a multiple of 45 degrees is then
 * approximated by alternating two of them, which is a staircase — the zig-zag
 * that made units look drunk on open ground. Worse, the flow-field follower
 * re-aims at the next tile centre every time it crosses a tile boundary, and
 * it crosses at a *corner* rather than at the centre it was walking to, so each
 * new leg starts from a different offset and the heading swings 20-some degrees
 * every tile rather than cleanly alternating.
 *
 * The fix in both cases is the same: stop steering at the adjacent tile and
 * steer at the furthest tile you can still reach in a straight line. That needs
 * one primitive — "is the segment from here to there clear of obstacles" — and
 * that is `lineOfSightClear`.
 *
 * ## Determinism
 *
 * The traversal is an exact integer DDA. Deciding whether the segment crosses
 * an x boundary or a y boundary next is a comparison of two products rather
 * than of two quotients, so nothing is ever divided and nothing is ever
 * rounded: the tile sequence is a property of the segment, not of the
 * arithmetic. Both products are bounded by `(mapSize + 1) * FIX_ONE` times
 * `mapSize * FIX_ONE`, which for a 128 tile map is about 7.1e13 — far below the
 * 2^53 where float64 stops representing integers exactly.
 *
 * ## Rotation equivariance
 *
 * The walk happens in the requesting player's canonical frame, like every other
 * tile decision in the simulation. A segment is turned into tiles by flooring,
 * and a floor is not symmetric under the map's 180-degree rotation: a point
 * exactly on a tile boundary lands in the higher tile, and so does its
 * rotation, which is the *lower* tile once rotated back. Unit positions sit on
 * exact boundaries more often than one would guess — building centres, the
 * approach points beside them and formation slots all do — so a test that
 * floored in absolute coordinates would let one half of the map smooth a
 * corner its opposite number did not, and a differently shaped path is walked
 * at a different real speed. See `GameMap.tileOfPosFor`, which exists for the
 * same reason.
 *
 * Everything else here is direction-agnostic by construction: the exact-corner
 * test is a property of the line, so the same segment gives the same answer
 * walked from either end.
 */

import { FIX_HALF, FIX_ONE, FIX_SHIFT, fromInt, type Fix } from '../fixed.js';
import type { GameMap } from '../map.js';

/** Walkability of a tile named in the canonical frame. */
function walkableAt(map: GameMap, tx: number, ty: number, flip: boolean): boolean {
  if (!flip) return map.isWalkable(tx, ty);
  // Out of range stays out of range under the rotation, so `isWalkable`'s own
  // bounds check still catches it.
  return map.isWalkable(map.width - 1 - tx, map.height - 1 - ty);
}

/**
 * True when a unit can walk the straight segment from (x0, y0) to (x1, y1)
 * without crossing anything unwalkable.
 *
 * Every tile the segment passes through must be walkable. Where it passes
 * exactly through a lattice corner — which is every diagonal between two tile
 * centres, so this is the common case and not an edge case — both tiles
 * flanking that corner must be walkable too. That is the same rule A* and the
 * flow field apply to a diagonal step, and applying it here is what stops a
 * smoothed path slipping through a gap the unsmoothed one was forbidden.
 *
 * `flip` is `World.flipOf` for the unit the question is asked on behalf of.
 */
export function lineOfSightClear(
  map: GameMap,
  x0: Fix,
  y0: Fix,
  x1: Fix,
  y1: Fix,
  flip = false,
): boolean {
  // Into the canonical frame, where flooring is mirror-consistent.
  let ax = x0;
  let ay = y0;
  let bx = x1;
  let by = y1;
  if (flip) {
    const w = fromInt(map.width);
    const h = fromInt(map.height);
    ax = w - x0;
    ay = h - y0;
    bx = w - x1;
    by = h - y1;
  }

  let tx = ax >> FIX_SHIFT;
  let ty = ay >> FIX_SHIFT;
  const ex = bx >> FIX_SHIFT;
  const ey = by >> FIX_SHIFT;

  if (!walkableAt(map, tx, ty, flip)) return false;
  if (tx === ex && ty === ey) return true;

  const dx = bx - ax;
  const dy = by - ay;
  const stepX = dx > 0 ? 1 : -1;
  const stepY = dy > 0 ? 1 : -1;
  const adx = dx < 0 ? -dx : dx;
  const ady = dy < 0 ? -dy : dy;

  // How far along each axis the segment travels before it leaves the tile it
  // starts in. Every boundary after that is one whole tile further on, which is
  // why stepping only ever adds FIX_ONE below.
  let toX = dx > 0 ? ((tx + 1) << FIX_SHIFT) - ax : ax - (tx << FIX_SHIFT);
  let toY = dy > 0 ? ((ty + 1) << FIX_SHIFT) - ay : ay - (ty << FIX_SHIFT);

  // Boundaries still to cross on each axis. Counting them rather than
  // comparing against the end tile keeps the loop finite no matter what the
  // arithmetic does at the very last boundary.
  let nx = ex > tx ? ex - tx : tx - ex;
  let ny = ey > ty ? ey - ty : ty - ey;

  while (nx > 0 || ny > 0) {
    // Which boundary does the segment meet first? Comparing `toX / adx` with
    // `toY / ady` without doing either division.
    if (nx > 0 && ny > 0 && toX * ady === toY * adx) {
      // Exactly through a corner. Both flanking tiles or no passage.
      if (!walkableAt(map, tx + stepX, ty, flip)) return false;
      if (!walkableAt(map, tx, ty + stepY, flip)) return false;
      tx += stepX;
      ty += stepY;
      toX += FIX_ONE;
      toY += FIX_ONE;
      nx--;
      ny--;
    } else if (ny === 0 || (nx > 0 && toX * ady < toY * adx)) {
      tx += stepX;
      toX += FIX_ONE;
      nx--;
    } else {
      ty += stepY;
      toY += FIX_ONE;
      ny--;
    }
    if (!walkableAt(map, tx, ty, flip)) return false;
  }

  return true;
}

/** Centre of a tile, in world coordinates. */
export function tileCentreX(map: GameMap, tile: number): Fix {
  return fromInt(map.tileXOf(tile)) + FIX_HALF;
}

/** Centre of a tile, in world coordinates. */
export function tileCentreY(map: GameMap, tile: number): Fix {
  return fromInt(map.tileYOf(tile)) + FIX_HALF;
}

/**
 * String-pull a tile path down to the corners that actually matter.
 *
 * A* returns one node per tile, and walking that literally is what draws the
 * staircase. Here each node is dropped when the unit can see past it to a later
 * one: a route across open ground collapses to a single leg, and a route around
 * a cliff keeps exactly the corners it has to turn at.
 *
 * The scan is greedy and forward — hold the last node still visible from the
 * anchor, and commit it the moment sight breaks — which costs one visibility
 * test per node rather than the quadratic sweep that picking the furthest
 * visible node would. Losing sight of a node and regaining it further along is
 * possible around a concave obstacle, and there the result keeps a corner it
 * could in principle have skipped. That is a slightly longer path, never an
 * invalid one, and it is not worth 48 times the work to recover.
 *
 * The first leg is measured from the unit's own position rather than from the
 * tile it stands in, because that is where it will actually walk from.
 */
export function smoothPath(
  map: GameMap,
  path: readonly number[],
  fromX: Fix,
  fromY: Fix,
  flip = false,
  out: number[] = [],
): number[] {
  out.length = 0;
  if (path.length === 0) return out;

  let anchorX = fromX;
  let anchorY = fromY;
  // Furthest node known reachable in a straight line from the anchor.
  let last = 0;

  for (let j = 1; j < path.length; j++) {
    const cx = tileCentreX(map, path[j]!);
    const cy = tileCentreY(map, path[j]!);
    if (lineOfSightClear(map, anchorX, anchorY, cx, cy, flip)) {
      last = j;
      continue;
    }
    // Sight broke here, so the unit has to turn at the last node it could see.
    out.push(path[last]!);
    anchorX = tileCentreX(map, path[last]!);
    anchorY = tileCentreY(map, path[last]!);
    // `last` and `j` are adjacent nodes of the original route, and A* has
    // already vetted that step including its corner, so `j` is reachable from
    // the node just committed without testing it again.
    last = j;
  }

  out.push(path[path.length - 1]!);
  return out;
}
