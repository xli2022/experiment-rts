/**
 * Reaching a building from whichever side you arrive on.
 *
 * Buildings were measured as the circle inscribed in their own footprint, which
 * is wrong in both directions at once: it reaches half a tile past the middle
 * of each face and falls short of every corner. On a Command Post, whose
 * footprint is four tiles, the diagonals were out of range entirely.
 *
 * Worse, units pathed at the building's *centre*. That tile is not walkable, so
 * A* substituted the nearest walkable tile to it — the same tile for everyone,
 * whichever direction they came from. A worker arriving from the far side
 * walked around the building to reach it, which is what this was reported as.
 */

import { describe, expect, it } from 'vitest';
import { CommandType } from '../src/sim/commands.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

function findOwn(sim: Simulation, type: EntityType): number {
  const pool = sim.world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === type) return i;
  }
  return -1;
}

/**
 * Drop a loaded worker `dx,dy` from the Command Post and let it deliver.
 *
 * Returns how far it actually walked and how long it took, so a detour shows up
 * as distance rather than as a pass/fail nobody can size.
 */
function deliverFrom(dx: number, dy: number): { ticks: number; walked: number; direct: number } {
  const sim = new Simulation(0x51ce7a11);
  const { pool, map } = sim.world;
  const cp = findOwn(sim, EntityType.CommandPost);
  const patch = (() => {
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.MineralPatch) return i;
    }
    return -1;
  })();

  const cx = toFloat(pool.posX[cp]!);
  const cy = toFloat(pool.posY[cp]!);
  const sx = Math.round(cx + dx);
  const sy = Math.round(cy + dy);
  if (!map.isWalkable(sx, sy)) throw new Error(`(${dx},${dy}) is not open ground`);

  const id = pool.spawn(
    EntityType.Worker,
    0 as PlayerId,
    Math.round((sx + 0.5) * FIX),
    Math.round((sy + 0.5) * FIX),
  );
  const w = id & 0xffff;
  pool.carrying[w] = 8;
  pool.harvestPatch[w] = pool.idAt(patch);
  pool.order[w] = Order.Harvest;

  const before = sim.world.players[0]!.minerals;
  let walked = 0;
  let px = toFloat(pool.posX[w]!);
  let py = toFloat(pool.posY[w]!);
  for (let t = 0; t < 400; t++) {
    sim.step([]);
    const nx = toFloat(pool.posX[w]!);
    const ny = toFloat(pool.posY[w]!);
    walked += Math.hypot(nx - px, ny - py);
    px = nx;
    py = ny;
    if (sim.world.players[0]!.minerals > before) {
      // Straight-line distance to the footprint's near face, which is the best
      // any route could do.
      const half = 2; // Command Post footprint 4
      const gx = Math.max(0, Math.abs(dx) - half);
      const gy = Math.max(0, Math.abs(dy) - half);
      return { ticks: t, walked, direct: Math.hypot(gx, gy) };
    }
  }
  throw new Error(`worker at (${dx},${dy}) never delivered`);
}

describe('delivering to a Command Post', () => {
  it('works from every side without walking around', () => {
    // The regression: before this, a worker on the far side took up to 2.6x the
    // direct distance to get in.
    for (const [dx, dy] of [
      [0, -5],
      [0, 5],
      [-5, 0],
      [5, 0],
    ]) {
      const r = deliverFrom(dx!, dy!);
      const ratio = r.walked / Math.max(0.5, r.direct);
      expect(`from (${dx},${dy}) walked ${ratio.toFixed(1)}x direct: ${ratio < 1.6}`).toBe(
        `from (${dx},${dy}) walked ${ratio.toFixed(1)}x direct: true`,
      );
    }
  });

  it('reaches the corners, which the inscribed circle never could', () => {
    // A 4-tile footprint puts its corners 2.83 from centre; the old reach was
    // 2.67, so a diagonal approach could not deliver at all without sliding
    // round to a face first.
    for (const [dx, dy] of [
      [4, 4],
      [-4, -4],
      [4, -4],
      [-4, 4],
    ]) {
      const r = deliverFrom(dx!, dy!);
      const ratio = r.walked / Math.max(0.5, r.direct);
      expect(`corner (${dx},${dy}): ${ratio < 1.6}`).toBe(`corner (${dx},${dy}): true`);
    }
  });

  it('moves more minerals over a real harvest cycle', () => {
    // The end-to-end consequence, and the only number a player would notice.
    const sim = new Simulation(0x51ce7a11);
    const pool = sim.world.pool;
    const cp = findOwn(sim, EntityType.CommandPost);
    const cx = toFloat(pool.posX[cp]!);
    const cy = toFloat(pool.posY[cp]!);

    const workers: number[] = [];
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Worker) {
        workers.push(pool.idAt(i));
      }
    }
    let patch = -1;
    let best = Infinity;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || pool.type[i] !== EntityType.MineralPatch) continue;
      const d = Math.hypot(toFloat(pool.posX[i]!) - cx, toFloat(pool.posY[i]!) - cy);
      if (d < best) {
        best = d;
        patch = i;
      }
    }

    const start = sim.world.players[0]!.minerals;
    sim.step([
      { type: CommandType.Harvest, player: 0, units: workers, target: pool.idAt(patch) },
    ]);
    for (let t = 0; t < 2000; t++) sim.step([]);
    const mined = sim.world.players[0]!.minerals - start;

    // Six workers over 100 seconds. The centre-pathing version managed well
    // under this; the figure is a floor, not a target to tune against.
    expect(`mined ${mined} in 2000 ticks, over 900: ${mined > 900}`).toBe(
      `mined ${mined} in 2000 ticks, over 900: true`,
    );
  });
});
