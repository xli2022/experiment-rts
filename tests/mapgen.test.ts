/**
 * The generated map.
 *
 * Two properties matter enough to pin mechanically.
 *
 * **Symmetry** decides whether a mirror match is a game or a coin flip. It is
 * guaranteed structurally — every write opens a tile and its opposite — but that
 * guarantee lives in one small function that a later edit could quietly bypass,
 * and the failure mode is a map that looks fine and is unfair. An earlier version
 * carved each shape twice and trusted the copies to stay in step; they diverged
 * on 926 tiles the moment the corridor brush gained a random wobble, because
 * each copy drew its own wobble from the same stream.
 *
 * **Route redundancy** is the thing the layout exists for. Three trunk routes
 * joined by connectors is a claim about topology, so it is tested as one: sever
 * any single route and the bases must still reach each other; sever all three
 * and they must not.
 */

import { describe, expect, it } from 'vitest';
import { PATCHES_PER_BASE, defOf } from '../src/config/rules.js';
import { GameMap, MAP_SIZE, generateMap } from '../src/sim/map.js';
import { buildLayout, mirror, nearestOn, pointAt } from '../src/sim/mapgen.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Tile } from '../src/sim/types.js';

const SEEDS = [0x51ce7a11, 0, 1, 7, 99, 0x7fffffff, 0xdecafbad | 0];

/** Reachability between the two starts, with some regions optionally sealed. */
function startsConnected(map: GameMap, blocks: { x: number; y: number; r: number }[]): boolean {
  const size = map.width;
  const blocked = Uint8Array.from(map.tiles);
  for (const b of blocks) {
    for (let y = b.y - b.r; y <= b.y + b.r; y++) {
      for (let x = b.x - b.r; x <= b.x + b.r; x++) {
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        const dx = x - b.x;
        const dy = y - b.y;
        if (dx * dx + dy * dy > b.r * b.r) continue;
        blocked[y * size + x] = Tile.Cliff;
      }
    }
  }

  const a = map.starts[0]!;
  const goal = map.index(map.starts[1]!.tileX, map.starts[1]!.tileY);
  const seen = new Uint8Array(size * size);
  const queue = [map.index(a.tileX, a.tileY)];
  seen[queue[0]!] = 1;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    if (cur === goal) return true;
    const cx = cur % size;
    const cy = (cur / size) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
      const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const ni = ny * size + nx;
      if (seen[ni] === 1 || blocked[ni] === Tile.Cliff) continue;
      seen[ni] = 1;
      queue.push(ni);
    }
  }
  return false;
}

/** The three trunk routes, identified by a point only that route passes through. */
function trunkChokes(size: number): { centre: { x: number; y: number; r: number }; low: { x: number; y: number; r: number }; high: { x: number; y: number; r: number } } {
  const edge = Math.floor(size * 0.11);
  // Wider than any corridor at these points, so a block really does sever the
  // route rather than leaving a lip of open ground to squeeze past.
  const r = 18;
  const low = { x: edge, y: size - 1 - edge, r };
  return {
    centre: { x: (size >> 1) - 1, y: (size >> 1) - 1, r },
    low,
    high: { x: size - 1 - low.x, y: size - 1 - low.y, r },
  };
}

describe('map generation', () => {
  it('is exactly symmetric under a 180 degree rotation', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed);
      let mismatched = 0;
      for (let y = 0; y < map.height; y++) {
        for (let x = 0; x < map.width; x++) {
          const here = map.index(x, y);
          const there = map.index(map.width - 1 - x, map.height - 1 - y);
          if (map.tiles[here] !== map.tiles[there]) mismatched++;
          if (map.elevation[here] !== map.elevation[there]) mismatched++;
        }
      }
      expect(`seed ${seed}: ${mismatched} mismatched`).toBe(`seed ${seed}: 0 mismatched`);
    }
  });

  it('puts the starts opposite each other', () => {
    const map = generateMap(1);
    const a = map.starts[0]!;
    const b = map.starts[1]!;
    expect(mirror({ x: a.tileX, y: a.tileY }, map.width)).toEqual({ x: b.tileX, y: b.tileY });
  });

  it('gives both players room to open', () => {
    for (const seed of SEEDS) {
      const sim = new Simulation(seed);
      const { pool, map } = sim.world;

      const hqDef = defOf(EntityType.CommandPost);
      for (const start of map.starts) {
        expect(map.isGroundWalkable(start.tileX, start.tileY)).toBe(true);
        // The base disc must hold the Command Post plus space to walk around it.
        let clear = 0;
        for (let dy = -8; dy <= 8; dy++) {
          for (let dx = -8; dx <= 8; dx++) {
            if (map.isGroundWalkable(start.tileX + dx, start.tileY + dy)) clear++;
          }
        }
        expect(clear).toBeGreaterThan(220);
        void hqDef;
      }

      // Every mineral patch must have fitted, for both players equally — a base
      // one patch short is a losing opening that no amount of skill recovers.
      const patches = [0, 0];
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1 || pool.type[i] !== EntityType.MineralPatch) continue;
        const nearer =
          Math.abs(pool.tileX[i]! - map.starts[0]!.tileX) +
          Math.abs(pool.tileY[i]! - map.starts[0]!.tileY) <
          Math.abs(pool.tileX[i]! - map.starts[1]!.tileX) +
            Math.abs(pool.tileY[i]! - map.starts[1]!.tileY)
            ? 0
            : 1;
        patches[nearer]!++;
      }
      expect(patches).toEqual([PATCHES_PER_BASE, PATCHES_PER_BASE]);
    }
  });

  it('leaves the bases connected', () => {
    for (const seed of SEEDS) {
      expect(startsConnected(generateMap(seed), [])).toBe(true);
    }
  });

  it('offers three independent routes between the bases', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed);
      const { centre, low, high } = trunkChokes(map.width);

      // Any one route severed: the other two still carry an army.
      expect(startsConnected(map, [centre])).toBe(true);
      expect(startsConnected(map, [low])).toBe(true);
      expect(startsConnected(map, [high])).toBe(true);
      // Any two severed: the survivor still connects.
      expect(startsConnected(map, [low, high])).toBe(true);
      expect(startsConnected(map, [centre, low])).toBe(true);
      expect(startsConnected(map, [centre, high])).toBe(true);
      // All three: nothing gets through. Without this the map would be an open
      // field with decorative rocks rather than a set of lanes.
      expect(startsConnected(map, [centre, low, high])).toBe(false);
    }
  });

  it('carves lanes, not a field and not a maze', () => {
    for (const seed of SEEDS) {
      const map = generateMap(seed);
      let open = 0;
      for (let i = 0; i < map.tiles.length; i++) {
        if (map.tiles[i] !== Tile.Cliff) open++;
      }
      const fraction = open / map.tiles.length;
      expect(fraction).toBeGreaterThan(0.28);
      expect(fraction).toBeLessThan(0.5);
    }
  });

  it('seals the border', () => {
    const map = generateMap(3);
    for (let i = 0; i < map.width; i++) {
      expect(map.isGroundWalkable(i, 0)).toBe(false);
      expect(map.isGroundWalkable(i, map.height - 1)).toBe(false);
      expect(map.isGroundWalkable(0, i)).toBe(false);
      expect(map.isGroundWalkable(map.width - 1, i)).toBe(false);
    }
  });

  it('raises cliffs away from open ground and leaves ground flat', () => {
    const map = generateMap(11);
    let raised = 0;
    for (let i = 0; i < map.tiles.length; i++) {
      if (map.tiles[i] === Tile.Cliff) {
        expect(map.elevation[i]).toBeGreaterThan(0);
        if (map.elevation[i]! > 1) raised++;
      } else {
        expect(map.elevation[i]).toBe(0);
      }
    }
    // Not just a one-tile lip everywhere: there are real massifs.
    expect(raised).toBeGreaterThan(1000);
  });

  it('is a pure function of the seed', () => {
    const a = generateMap(0x1234);
    const b = generateMap(0x1234);
    const c = generateMap(0x1235);
    expect(Array.from(a.tiles)).toEqual(Array.from(b.tiles));
    expect(Array.from(a.tiles)).not.toEqual(Array.from(c.tiles));
  });
});

describe('layout geometry', () => {
  it('samples a polyline by distance along it', () => {
    const line = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
    ];
    expect(pointAt(line, 0)).toEqual({ x: 0, y: 0 });
    expect(pointAt(line, 0.25)).toEqual({ x: 5, y: 0 });
    expect(pointAt(line, 0.5)).toEqual({ x: 10, y: 0 });
    expect(pointAt(line, 1)).toEqual({ x: 10, y: 10 });
    // Out of range clamps rather than extrapolating off the map.
    expect(pointAt(line, 1.4)).toEqual({ x: 10, y: 10 });
  });

  it('finds the nearest point on a polyline', () => {
    const line = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
    ];
    expect(nearestOn(line, { x: 3, y: 7 })).toEqual({ x: 3, y: 0 });
    expect(nearestOn(line, { x: -5, y: 1 })).toEqual({ x: 0, y: 0 });
  });

  it('keeps every lane inside the map', () => {
    const layout = buildLayout(MAP_SIZE);
    for (const lane of layout.lanes) {
      for (const p of lane.points) {
        expect(p.x).toBeGreaterThanOrEqual(0);
        expect(p.y).toBeGreaterThanOrEqual(0);
        expect(p.x).toBeLessThan(MAP_SIZE);
        expect(p.y).toBeLessThan(MAP_SIZE);
      }
    }
    expect(layout.lanes.length).toBeGreaterThanOrEqual(4);
  });
});
