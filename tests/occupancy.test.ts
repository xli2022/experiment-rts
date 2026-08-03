/**
 * What standing on a tile means.
 *
 * There are two different questions a tile can answer — "can something walk
 * here?" and "can something be built here?" — and they are not the same
 * question. A structure blocks both. A mineral patch blocks only building: it is
 * scenery you mine, not a wall.
 *
 * The distinction has teeth because a mineral line is eight patches in a tight
 * arc right beside the Command Post. Treated as solid, it is a fence across the
 * busiest few tiles on the map, and every worker trip and every unit crossing
 * its own base pays for it.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { reapDead } from '../src/sim/systems/combat.js';
import { AStar } from '../src/sim/pathing/astar.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType } from '../src/sim/types.js';

/** Every tile covered by a mineral patch, and the patch entity indices. */
function mineralTiles(sim: Simulation): { tiles: number[]; patches: number[] } {
  const { pool, map } = sim.world;
  const footprint = defOf(EntityType.MineralPatch).footprint;
  const tiles: number[] = [];
  const patches: number[] = [];
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.type[i] !== EntityType.MineralPatch) continue;
    patches.push(i);
    for (let dy = 0; dy < footprint; dy++) {
      for (let dx = 0; dx < footprint; dx++) {
        tiles.push(map.index(pool.tileX[i]! + dx, pool.tileY[i]! + dy));
      }
    }
  }
  return { tiles, patches };
}

describe('mineral patches', () => {
  it('can be walked over', () => {
    const sim = new Simulation(0x51ce7a11);
    const { map } = sim.world;
    const { tiles, patches } = mineralTiles(sim);
    expect(patches.length).toBeGreaterThan(0);

    for (const t of tiles) {
      expect(map.isWalkable(map.tileXOf(t), map.tileYOf(t))).toBe(true);
    }
  });

  it('cannot be built on', () => {
    const sim = new Simulation(0x51ce7a11);
    const { map } = sim.world;
    const { patches } = mineralTiles(sim);

    for (const i of patches) {
      const tx = sim.world.pool.tileX[i]!;
      const ty = sim.world.pool.tileY[i]!;
      expect(map.canPlace(tx, ty, 1)).toBe(false);
      // Nor overlapping one with something larger.
      expect(map.canPlace(tx - 1, ty - 1, 3)).toBe(false);
    }
  });

  it('does not make a unit path around the mineral line', () => {
    const sim = new Simulation(0x51ce7a11);
    const { map, pool } = sim.world;
    const astar = new AStar(map);

    // A patch with clear ground on both sides along one axis: crossing it is
    // the shortest route only if the patch itself is passable.
    const { patches } = mineralTiles(sim);
    let crossed = 0;
    for (const i of patches) {
      const tx = pool.tileX[i]!;
      const ty = pool.tileY[i]!;
      const before = map.index(tx, ty - 2);
      const after = map.index(tx, ty + 3);
      if (!map.isWalkable(tx, ty - 2) || !map.isWalkable(tx, ty + 3)) continue;

      const path = astar.find(map, before, after, []);
      expect(path.length).toBeGreaterThan(0);
      // Straight through is five tiles of travel; anything much longer means it
      // went round the patch.
      expect(path.length).toBeLessThanOrEqual(6);
      crossed++;
    }
    expect(crossed).toBeGreaterThan(0);
  });

  it('still lets workers mine', () => {
    // The whole point of a patch is the economy, so prove the change did not
    // quietly break gathering by letting workers walk straight through the
    // thing they are supposed to stop at.
    const sim = new Simulation(0x51ce7a11);
    const { pool } = sim.world;

    const workers: number[] = [];
    let patch = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (pool.owner[i] === 0 && pool.type[i] === EntityType.Worker) workers.push(pool.idAt(i));
      if (patch < 0 && pool.type[i] === EntityType.MineralPatch) patch = i;
    }
    expect(workers.length).toBeGreaterThan(0);
    expect(patch).toBeGreaterThanOrEqual(0);

    const before = sim.world.player(0).minerals;
    sim.step([
      { type: CommandType.Harvest, player: 0, units: workers, target: pool.idAt(patch) },
    ]);
    for (let t = 0; t < 600; t++) sim.step([]);
    expect(sim.world.player(0).minerals).toBeGreaterThan(before);
  });
});

describe('structures', () => {
  it('block both walking and building, and free their ground when destroyed', () => {
    const sim = new Simulation(0x51ce7a11);
    const { map, pool } = sim.world;

    let hq = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.CommandPost) hq = i;
    }
    expect(hq).toBeGreaterThanOrEqual(0);

    const tx = pool.tileX[hq]!;
    const ty = pool.tileY[hq]!;
    expect(map.isWalkable(tx, ty)).toBe(false);
    expect(map.canPlace(tx, ty, 1)).toBe(false);

    // Kill it. `reapDead` is driven by the tick's death list, so it is invoked
    // directly rather than by staging a whole battle.
    pool.hp[hq] = 0;
    sim.world.events.deaths.push(hq);
    reapDead(sim.world);

    expect(map.isWalkable(tx, ty)).toBe(true);
    expect(map.canPlace(tx, ty, 1)).toBe(true);
  });
});
