/**
 * Units ramp up to speed instead of snapping to it.
 *
 * Constant velocity is most of what made movement look mechanical: a unit went
 * from nothing to full pace in one tick and stopped dead in another.
 *
 * The trap this pins is that there is more than one mover. Direct steering, flow
 * fields and A* path-following are three separate code paths, and A* — the one
 * that carries most individual orders — walks its waypoints in a loop of its
 * own rather than going through `stepToward`. Adding acceleration to the shared
 * helper alone changed nothing at all for it, and nothing failed to say so.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

/** A unit at rest on open ground, plus somewhere to send it. */
function ready(type: EntityType): {
  sim: Simulation;
  unit: number;
  id: number;
  dest: { x: number; y: number };
} {
  const sim = new Simulation(0x51ce7a11);
  const { pool, map } = sim.world;
  const start = map.starts[0]!;

  let spot: { x: number; y: number } | null = null;
  for (let r = 4; r < 24 && !spot; r++) {
    for (let dx = -r; dx <= r && !spot; dx++) {
      const x = start.tileX + dx;
      const y = start.tileY - r;
      if (map.isWalkable(x, y)) spot = { x, y };
    }
  }
  const id = pool.spawn(
    type,
    0 as PlayerId,
    Math.round((spot!.x + 0.5) * FIX),
    Math.round((spot!.y + 0.5) * FIX),
  );

  let dest: { x: number; y: number } | null = null;
  for (let r = 10; r < 30 && !dest; r++) {
    for (let dx = -r; dx <= r && !dest; dx++) {
      const x = spot!.x + dx;
      const y = spot!.y + r;
      if (map.isWalkable(x, y)) dest = { x, y };
    }
  }
  return { sim, unit: id & 0xffff, id, dest: dest! };
}

/** Distance covered on each of the first `ticks` ticks after being ordered off. */
function profile(type: EntityType, ticks: number): number[] {
  const { sim, unit, id, dest } = ready(type);
  const pool = sim.world.pool;
  const cmd: Command = {
    type: CommandType.Move,
    player: 0,
    units: [id],
    x: Math.round((dest.x + 0.5) * FIX),
    y: Math.round((dest.y + 0.5) * FIX),
  };
  sim.step([cmd]);

  const steps: number[] = [];
  let px = pool.posX[unit]!;
  let py = pool.posY[unit]!;
  for (let t = 0; t < ticks; t++) {
    sim.step([]);
    steps.push(Math.hypot(pool.posX[unit]! - px, pool.posY[unit]! - py) / FIX);
    px = pool.posX[unit]!;
    py = pool.posY[unit]!;
  }
  return steps;
}

describe('units get up to speed', () => {
  it('starts below top speed and climbs to it', () => {
    for (const type of [EntityType.Rifleman, EntityType.Brawler, EntityType.Worker]) {
      const def = defOf(type);
      const top = toFloat(def.speedPerTick);
      const steps = profile(type, 12);

      // The regression: an unaccelerated mover is at full pace on tick one.
      expect(`${def.name} first tick below top: ${steps[0]! < top * 0.9}`).toBe(
        `${def.name} first tick below top: true`,
      );
      expect(`${def.name} reaches top: ${steps[11]! > top * 0.95}`).toBe(
        `${def.name} reaches top: true`,
      );
      // Monotonic while ramping — no stutter on the way up.
      for (let i = 1; i < 4; i++) {
        expect(steps[i]!).toBeGreaterThanOrEqual(steps[i - 1]! - 1e-6);
      }
    }
  });

  it('never exceeds the unit’s top speed', () => {
    for (const type of [EntityType.Rifleman, EntityType.Gunship]) {
      const top = toFloat(defOf(type).speedPerTick);
      for (const step of profile(type, 40)) {
        // A percent of slack: separation nudges a unit a fraction of a unit
        // sideways in the same tick, and fixed-point rounding adds its own
        // ULP. The point is that nothing travels at *double* speed.
        expect(step).toBeLessThanOrEqual(top * 1.01);
      }
    }
  });

  it('takes a fraction of a second, not a crawl', () => {
    // Slow enough to read as weight, fast enough that a click still feels
    // answered — at 20 ticks a second this is about a fifth of one.
    const top = toFloat(defOf(EntityType.Rifleman).speedPerTick);
    const steps = profile(EntityType.Rifleman, 20);
    const reached = steps.findIndex((s) => s >= top * 0.99);
    expect(reached).toBeGreaterThan(0);
    expect(reached).toBeLessThan(8);
  });

  it('still gets there', () => {
    // A ramp that never quite reaches the destination would strand every order.
    const { sim, unit, id, dest } = ready(EntityType.Rifleman);
    const pool = sim.world.pool;
    sim.step([
      {
        type: CommandType.Move,
        player: 0,
        units: [id],
        x: Math.round((dest.x + 0.5) * FIX),
        y: Math.round((dest.y + 0.5) * FIX),
      },
    ]);
    let arrived = false;
    for (let t = 0; t < 400 && !arrived; t++) {
      sim.step([]);
      const d = Math.hypot(
        pool.posX[unit]! / FIX - (dest.x + 0.5),
        pool.posY[unit]! / FIX - (dest.y + 0.5),
      );
      if (d < 1) arrived = true;
    }
    expect(arrived).toBe(true);
  });
});
