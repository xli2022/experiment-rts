/**
 * Arrival spread, and the cost it must not have.
 *
 * A group move used to send every unit to the same point, so an army arrived in
 * a heap and spent the next second shoving itself apart. Spreading the
 * destinations fixes that — and the obvious way to do it is a trap.
 *
 * Grouped moves navigate by flow field, and the field is cached per *goal tile*:
 * one Dijkstra sweep serves the whole army, which is the entire reason
 * GROUP_PATH_THRESHOLD exists. Spreading the goal gives every unit its own tile
 * and therefore its own sweep. It is not a crash or a deadlock, just a silent
 * collapse — measured, the test match went from 5 seconds to over 150.
 *
 * So the shared destination steers the group and the spread point only decides
 * where each unit finally stands.
 */

import { describe, expect, it } from 'vitest';
import { GROUP_PATH_THRESHOLD } from '../src/config/rules.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt, toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

/** An army of `n` units on open ground near player 0's start. */
function army(n: number): { sim: Simulation; ids: number[] } {
  const sim = new Simulation(0x51ce7a11);
  const { pool, map } = sim.world;
  const start = map.starts[0]!;
  const ids: number[] = [];
  for (let r = 3; r < 40 && ids.length < n; r++) {
    for (let dy = -r; dy <= r && ids.length < n; dy++) {
      for (let dx = -r; dx <= r && ids.length < n; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        if (!map.isWalkable(x, y)) continue;
        ids.push(
          pool.spawn(
            EntityType.Rifleman,
            0 as PlayerId,
            Math.round((x + 0.5) * FIX),
            Math.round((y + 0.5) * FIX),
          ),
        );
      }
    }
  }
  return { sim, ids };
}

/** A walkable tile a good distance from the start, to march toward. */
function target(sim: Simulation): { x: number; y: number } {
  const { map } = sim.world;
  const start = map.starts[0]!;
  for (let r = 14; r < 40; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        if (map.isWalkable(x, y)) return { x, y };
      }
    }
  }
  throw new Error('nowhere to march to');
}

function march(sim: Simulation, ids: number[], to: { x: number; y: number }): void {
  const cmd: Command = {
    type: CommandType.AttackMove,
    player: 0,
    units: ids,
    x: fromInt(to.x) + 32768,
    y: fromInt(to.y) + 32768,
  };
  sim.step([cmd]);
}

describe('arrival spread', () => {
  it('gives the members of a group distinct destinations', () => {
    const { sim, ids } = army(9);
    const to = target(sim);
    march(sim, ids, to);

    const pool = sim.world.pool;
    const seen = new Set<string>();
    for (const id of ids) {
      const i = id & 0xffff;
      seen.add(`${pool.orderX[i]},${pool.orderY[i]}`);
    }
    expect(`${seen.size} distinct destinations for ${ids.length} units`).toBe(
      `${ids.length} distinct destinations for ${ids.length} units`,
    );
  });

  it('still points the whole group at one flow-field goal', () => {
    // This is the regression. One sweep serves the army; one sweep per unit is
    // a silent 30x on the heaviest system in the simulation.
    const { sim, ids } = army(GROUP_PATH_THRESHOLD + 6);
    const to = target(sim);
    march(sim, ids, to);

    const pool = sim.world.pool;
    const goals = new Set<number>();
    for (const id of ids) {
      const g = pool.flowGoal[id & 0xffff]!;
      if (g >= 0) goals.add(g);
    }
    expect(`${goals.size} flow goal(s) for ${ids.length} units`).toBe(
      `1 flow goal(s) for ${ids.length} units`,
    );
  });

  it('leaves a lone unit exactly where it was told to go', () => {
    const { sim, ids } = army(1);
    const to = target(sim);
    march(sim, ids, [to][0]!);
    const i = ids[0]! & 0xffff;
    expect(`${toFloat(sim.world.pool.orderX[i]!)},${toFloat(sim.world.pool.orderY[i]!)}`).toBe(
      `${to.x + 0.5},${to.y + 0.5}`,
    );
  });

  it('lands the army spread out rather than stacked', () => {
    const { sim, ids } = army(9);
    const to = target(sim);
    march(sim, ids, to);
    for (let t = 0; t < 400; t++) sim.step([]);

    const pool = sim.world.pool;
    let arrived = 0;
    const cells = new Set<string>();
    for (const id of ids) {
      const i = id & 0xffff;
      if (pool.alive[i] !== 1) continue;
      if (pool.order[i] !== Order.None) continue;
      arrived++;
      cells.add(`${Math.round(toFloat(pool.posX[i]!))},${Math.round(toFloat(pool.posY[i]!))}`);
    }
    expect(arrived).toBeGreaterThan(4);
    // Each on its own tile, not piled onto one.
    expect(cells.size).toBeGreaterThan(arrived / 2);
  });
});
