import { describe, expect, it } from 'vitest';
import { defOf, PATCHES_PER_EXPANSION } from '../src/config/rules.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { GameMap, mirrorTile } from '../src/sim/map.js';
import { mirroredHalf } from '../src/sim/mapgen.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, MapLayout } from '../src/sim/types.js';

const SEEDS = [0x51ce7a11, 0, 1, 7, 99, 0x7fffffff, 0xdecafbad | 0];
type Site = { tileX: number; tileY: number };

/** Four-neighbour walking distance: checks terrain, independent of steering. */
function walkingDistance(map: GameMap, start: Site, goal: Site): number {
  const distance = new Int32Array(map.width * map.height).fill(-1);
  const queue = [map.index(start.tileX, start.tileY)];
  distance[queue[0]!] = 0;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    const x = map.tileXOf(cur);
    const y = map.tileYOf(cur);
    if (x === goal.tileX && y === goal.tileY) return distance[cur]!;
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx!;
      const ny = y + dy!;
      if (!map.isGroundWalkable(nx, ny)) continue;
      const next = map.index(nx, ny);
      if (distance[next] !== -1) continue;
      distance[next] = distance[cur]! + 1;
      queue.push(next);
    }
  }
  return -1;
}

describe('map tactical routes', () => {
  it('puts duel flank minerals equally within reach of both players', () => {
    for (const seed of SEEDS) {
      const { map } = new Simulation(duelMatch(seed)).world;
      expect(map.expansions).toHaveLength(4);
      // Each half holds a natural followed by a contested river crossroads.
      for (const index of [1, 3]) {
        const site = map.expansions[index]!;
        const fromHome = walkingDistance(map, map.starts[0]!, site);
        const fromAway = walkingDistance(map, map.starts[1]!, site);
        expect(fromHome).toBeGreaterThan(0);
        expect(fromAway).toBe(fromHome);
        // The old empty flank corners took 99 steps to reach. Keep this route
        // short enough to contest while the direct centre route still exists.
        expect(fromHome).toBeLessThanOrEqual(90);
      }
      expect(walkingDistance(map, map.expansions[0]!, map.expansions[1]!)).toBeLessThanOrEqual(60);
      expect(walkingDistance(map, map.starts[0]!, map.expansions[0]!)).toBeLessThan(
        walkingDistance(map, map.starts[0]!, map.expansions[1]!),
      );
    }
  });

  it('lets every co-op natural rotate directly onto its own flank', () => {
    for (const seed of SEEDS) {
      const { map } = new Simulation(coopMatch(seed)).world;
      // Skip the contested site at the end of each mirrored half.
      for (const [player, index] of [
        [0, 0],
        [1, 1],
        [2, 3],
        [3, 4],
      ]) {
        const natural = map.expansions[index!]!;
        const flank = { tileX: map.starts[player!]!.tileX, tileY: natural.tileY };
        // Previously this was a 41-45 tile detour back toward home. A lateral
        // connection reaches the edge without crossing the contested centre.
        expect(walkingDistance(map, natural, flank)).toBe(Math.abs(natural.tileX - flank.tileX));
        expect(walkingDistance(map, natural, flank)).toBeLessThanOrEqual(24);
      }
    }
  });

  it.each([MapLayout.Lanes, MapLayout.Quarters])(
    'keeps every expansion buildable, stocked and reachable on layout %i',
    (layout) => {
      const hqSize = defOf(EntityType.CommandPost).footprint;
      const half = hqSize >> 1;
      for (const seed of SEEDS) {
        const config = layout === MapLayout.Lanes ? duelMatch(seed) : coopMatch(seed);
        const { map, pool } = new Simulation(config).world;
        for (let e = 0; e < map.expansions.length; e++) {
          const site = map.expansions[e]!;
          const pair = mirroredHalf(e, map.expansions.length);
          const canonical = map.expansions[pair.canonical]!;
          const tx = pair.flip
            ? mirrorTile(map.width, canonical.tileX - half, hqSize)
            : canonical.tileX - half;
          const ty = pair.flip
            ? mirrorTile(map.height, canonical.tileY - half, hqSize)
            : canonical.tileY - half;
          expect(map.canPlace(tx, ty, hqSize)).toBe(true);
          let patches = 0;
          for (let i = 0; i < pool.count; i++) {
            if (pool.alive[i] !== 1 || pool.type[i] !== EntityType.MineralPatch) continue;
            const dx = pool.tileX[i]! - site.tileX;
            const dy = pool.tileY[i]! - site.tileY;
            if (dx * dx + dy * dy <= 12 * 12) patches++;
          }
          expect(patches).toBe(PATCHES_PER_EXPANSION);
          for (const start of map.starts) {
            expect(walkingDistance(map, start, site)).toBeGreaterThan(0);
          }
        }
      }
    },
  );
});
