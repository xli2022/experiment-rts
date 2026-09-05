/**
 * What can shoot at what, and what that means for chasing.
 *
 * Reported as melee units trailing after Beamdrones. There were two halves to it:
 * nothing in combat distinguished air from ground at all, so a Slicebot could
 * swing a sword at something twenty feet up — and because acquisition is what
 * `engageNearby` walks toward, acquiring an aircraft was also an instruction to
 * follow it around.
 *
 * A worker's 0.6-tile reach is a melee swing by the same argument, so it is
 * ground-only too. Everything with a real weapon still covers both layers.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, NO_ENTITY, Order, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

/**
 * Put an attacker `gap` tiles from a pinned enemy and run.
 *
 * Deliberately far from either base: workers have a weapon of their own and
 * will shoot anything parked beside them, which silently contaminated the first
 * version of this measurement.
 */
function duel(
  attacker: EntityType,
  target: EntityType,
  gap: number,
  order?: (a: number, b: number) => CommandType,
): { hit: boolean; walked: number } {
  const sim = new Simulation(0x51ce7a11);
  const { pool, map } = sim.world;
  const start = map.starts[0]!;

  let spot: { x: number; y: number } | null = null;
  for (let r = 14; r < 34 && !spot; r++) {
    for (let dx = -r; dx <= r && !spot; dx++) {
      const x = start.tileX + dx;
      const y = start.tileY + r;
      if (map.isWalkable(x, y) && map.isWalkable(x + Math.ceil(gap), y)) spot = { x, y };
    }
  }
  if (!spot) throw new Error('nowhere clear of the base to stage a fight');

  const ax = spot.x + 0.5;
  const ay = spot.y + 0.5;
  const a =
    pool.spawn(attacker, 0 as PlayerId, Math.round(ax * FIX), Math.round(ay * FIX)) & 0xffff;
  const b =
    pool.spawn(target, 1 as PlayerId, Math.round((ax + gap) * FIX), Math.round(ay * FIX)) & 0xffff;

  if (order) {
    sim.step([
      { type: order(a, b), player: 0, units: [pool.idAt(a)], target: pool.idAt(b) } as never,
    ]);
  }

  const hp0 = pool.hp[b]!;
  const x0 = toFloat(pool.posX[a]!);
  let hit = false;
  for (let t = 0; t < 150; t++) {
    pool.posX[b] = Math.round((ax + gap) * FIX);
    pool.posY[b] = Math.round(ay * FIX);
    pool.order[b] = Order.Hold;
    const before = pool.hp[b]!;
    sim.step([]);
    // Only shots from our attacker count; nothing else should be in range, but
    // asserting on hp alone is how the base's workers got mistaken for a melee
    // unit landing impossible blows.
    const shots = sim.world.events.shots;
    for (let k = 0; k + 1 < shots.length; k += 2) {
      if (shots[k] === a && shots[k + 1] === b) hit = true;
    }
    void before;
    pool.hp[b] = hp0;
  }
  return { hit, walked: Math.abs(toFloat(pool.posX[a]!) - x0) };
}

describe('melee cannot touch air', () => {
  it('does not chase a Beamdrone it could never hit', () => {
    // The report. Acquisition drives `engageNearby`, so refusing the target is
    // what stops the chase — the two are the same fix.
    const r = duel(EntityType.Slicebot, EntityType.Beamdrone, 3);
    expect(`hit ${r.hit}, walked ${r.walked.toFixed(2)}`).toBe('hit false, walked 0.00');
  });

  it('applies to workers too, whose reach is shorter still', () => {
    const r = duel(EntityType.Worker, EntityType.Beamdrone, 2);
    expect(`hit ${r.hit}, walked ${r.walked.toFixed(2)}`).toBe('hit false, walked 0.00');
  });

  it('still fights ground units exactly as before', () => {
    // Non-vacuity: the same staging with a ground target must still engage, or
    // this suite would pass with combat switched off entirely.
    const r = duel(EntityType.Slicebot, EntityType.Burstbot, 3);
    expect(r.hit).toBe(true);
    expect(r.walked).toBeGreaterThan(0.5);
  });

  it('refuses an explicit attack order onto a flyer', () => {
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const start = map.starts[0]!;
    const x = Math.round((start.tileX + 0.5) * FIX);
    const y = Math.round((start.tileY + 8.5) * FIX);
    const slicebot = pool.spawn(EntityType.Slicebot, 0 as PlayerId, x, y) & 0xffff;
    const burstbot = pool.spawn(EntityType.Burstbot, 0 as PlayerId, x, y) & 0xffff;
    const beamdrone = pool.spawn(EntityType.Beamdrone, 1 as PlayerId, x + 4 * FIX, y) & 0xffff;

    sim.step([
      {
        type: CommandType.Attack,
        player: 0,
        units: [pool.idAt(slicebot), pool.idAt(burstbot)],
        target: pool.idAt(beamdrone),
      },
    ]);

    // Dropped per unit, not for the whole order: the Burstbot still goes.
    expect(`slicebot order ${pool.order[slicebot]}, burstbot order ${pool.order[burstbot]}`).toBe(
      `slicebot order ${Order.None}, burstbot order ${Order.Attack}`,
    );
    expect(pool.orderTarget[slicebot]).toBe(NO_ENTITY);
  });
});

describe('everything else still covers both layers', () => {
  it('lets ranged units and turrets shoot air', () => {
    for (const type of [EntityType.Burstbot, EntityType.Beamdrone, EntityType.Turret]) {
      expect(`${defOf(type).name} hits air: ${defOf(type).canHitAir}`).toBe(
        `${defOf(type).name} hits air: true`,
      );
    }
    expect(duel(EntityType.Burstbot, EntityType.Beamdrone, 3).hit).toBe(true);
  });

  it('lets air shoot ground', () => {
    expect(duel(EntityType.Beamdrone, EntityType.Slicebot, 3).hit).toBe(true);
  });
});
