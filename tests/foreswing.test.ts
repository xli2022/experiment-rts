/** Authoritative attack wind-up and its presentation event. */

import { describe, expect, it } from 'vitest';
import { DEFS, defOf, isValidAttackTiming } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { applyDamage, reapDead } from '../src/sim/systems/combat.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, NO_ENTITY, Order, type EntityId, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

function hasPair(values: readonly number[], attacker: number, target: number): boolean {
  for (let k = 0; k + 1 < values.length; k += 2) {
    if (values[k] === attacker && values[k + 1] === target) return true;
  }
  return false;
}

/** Put a stationary Slicebot duel on open ground away from either base. */
function stageDuel(): {
  sim: Simulation;
  attacker: number;
  attackerId: EntityId;
  target: number;
  targetId: EntityId;
} {
  const sim = new Simulation(0x51ce7a11);
  const { map, pool } = sim.world;
  const start = map.starts[0]!;

  let spot: { x: number; y: number } | undefined;
  for (let r = 12; r < 40 && !spot; r++) {
    for (let dx = -r; dx <= r && !spot; dx++) {
      const x = start.tileX + dx;
      const y = start.tileY + r;
      if (map.isWalkable(x, y) && map.isWalkable(x + 1, y)) spot = { x, y };
    }
  }
  if (!spot) throw new Error('no open ground for foreswing duel');

  const at = (n: number) => Math.round(n * FIX);
  const attackerId = pool.spawn(
    EntityType.Slicebot,
    0 as PlayerId,
    at(spot.x + 0.5),
    at(spot.y + 0.5),
  );
  const targetId = pool.spawn(
    EntityType.Burstbot,
    1 as PlayerId,
    at(spot.x + 1.5),
    at(spot.y + 0.5),
  );
  const attacker = attackerId & 0xffff;
  const target = targetId & 0xffff;
  pool.order[attacker] = Order.None;
  pool.order[target] = Order.Hold;
  return { sim, attacker, attackerId, target, targetId };
}

describe('attack foreswing rules', () => {
  it('gives Slicebot a readable wind-up within its unchanged attack period', () => {
    const slicebot = defOf(EntityType.Slicebot);
    expect(slicebot.attackForeswing).toBeGreaterThan(0);
    expect(slicebot.attackForeswing).toBeLessThan(slicebot.attackCooldown);
  });

  it('keeps every configured foreswing inside its weapon cycle', () => {
    for (const def of DEFS) {
      expect(isValidAttackTiming(def.attackForeswing, def.attackCooldown)).toBe(true);
    }
  });

  it('allows 0/0 but rejects foreswing equal to a positive cooldown', () => {
    expect(isValidAttackTiming(0, 0)).toBe(true);
    expect(isValidAttackTiming(0, 1)).toBe(true);
    expect(isValidAttackTiming(8, 9)).toBe(true);
    expect(isValidAttackTiming(9, 9)).toBe(false);
  });
});

describe('authoritative wind-up timing', () => {
  it('starts first, preserves HP through foreswing, and hits on the exact impact tick', () => {
    const { sim, attacker, target, targetId } = stageDuel();
    const { pool } = sim.world;
    const def = defOf(EntityType.Slicebot);
    const hpBefore = pool.hp[target]!;
    let startTick = -1;
    let impactTick = -1;

    for (let step = 0; step < 80 && impactTick < 0; step++) {
      sim.step([]);
      if (hasPair(sim.world.events.attackStarts, attacker, target)) {
        startTick = sim.world.tick;
        expect(pool.attackWindup[attacker]).toBe(def.attackForeswing);
        expect(pool.attackTarget[attacker]).toBe(targetId);
        expect(pool.hp[target]).toBe(hpBefore);
        expect(hasPair(sim.world.events.shots, attacker, target)).toBe(false);
      }
      if (hasPair(sim.world.events.shots, attacker, target)) {
        impactTick = sim.world.tick;
        expect(sim.world.events.attackImpacts).toContain(attacker);
      } else if (startTick >= 0) {
        expect(pool.hp[target]).toBe(hpBefore);
      }
    }

    expect(startTick).toBeGreaterThan(0);
    expect(impactTick - startTick).toBe(def.attackForeswing);
    expect(pool.hp[target]).toBe(hpBefore - def.damage);
    expect(pool.attackWindup[attacker]).toBe(0);
    expect(pool.attackTarget[attacker]).toBe(NO_ENTITY);
  });

  it('counts cooldown from attack start, preserving time between impacts', () => {
    const { sim, attacker, target } = stageDuel();
    const impacts: number[] = [];
    for (let step = 0; step < 100 && impacts.length < 2; step++) {
      sim.step([]);
      if (hasPair(sim.world.events.shots, attacker, target)) impacts.push(sim.world.tick);
    }
    expect(impacts).toHaveLength(2);
    expect(impacts[1]! - impacts[0]!).toBe(defOf(EntityType.Slicebot).attackCooldown);
  });

  it('lets Stop cancel a pending blow without refunding cooldown', () => {
    const { sim, attacker, attackerId, target } = stageDuel();
    const { pool } = sim.world;
    const def = defOf(EntityType.Slicebot);

    while (pool.attackWindup[attacker] === 0) sim.step([]);
    while (pool.attackWindup[attacker]! > 1) sim.step([]);
    const cooldownBefore = pool.attackCooldown[attacker]!;
    const hpBefore = pool.hp[target]!;
    sim.step([{ type: CommandType.Stop, player: 0, units: [attackerId] }]);

    expect(pool.attackWindup[attacker]).toBe(0);
    expect(pool.attackTarget[attacker]).toBe(NO_ENTITY);
    expect(pool.attackCooldown[attacker]).toBe(cooldownBefore - 1);
    expect(sim.world.events.attackImpacts).not.toContain(attacker);
    for (let t = 0; t <= def.attackForeswing; t++) {
      expect(hasPair(sim.world.events.shots, attacker, target)).toBe(false);
      sim.step([]);
    }
    expect(pool.hp[target]).toBe(hpBefore);
  });

  it('reports the resolve tick for a whiff without reporting a shot', () => {
    const { sim, attacker, target } = stageDuel();
    const { pool } = sim.world;
    while (pool.attackWindup[attacker] === 0) sim.step([]);
    const hpBefore = pool.hp[target]!;

    // Leave reach after attack start. The locked swing still resolves on time,
    // but impact validation correctly refuses damage at this distance.
    pool.posX[target] = pool.posX[attacker]! + 10 * FIX;
    while (!sim.world.events.attackImpacts.includes(attacker)) sim.step([]);

    expect(hasPair(sim.world.events.shots, attacker, target)).toBe(false);
    expect(pool.hp[target]).toBe(hpBefore);
    sim.step([]);
    expect(sim.world.events.attackImpacts).not.toContain(attacker);
  });

  it('cancels a wind-up whose generation-tagged target dies', () => {
    const { sim, attacker, target, targetId } = stageDuel();
    const { pool } = sim.world;
    while (pool.attackWindup[attacker] === 0) sim.step([]);

    applyDamage(sim.world, target, pool.hp[target]!);
    reapDead(sim.world);

    expect(pool.isAlive(targetId)).toBe(false);
    expect(pool.attackWindup[attacker]).toBe(0);
    expect(pool.attackTarget[attacker]).toBe(NO_ENTITY);
  });
});

describe('wind-up state lifecycle', () => {
  it('checksums both pending fields and resets a recycled slot', () => {
    const { sim, attacker, attackerId, targetId } = stageDuel();
    const { pool } = sim.world;
    while (pool.attackWindup[attacker] === 0) sim.step([]);

    const baseline = sim.checksum();
    const windup = pool.attackWindup[attacker]!;
    pool.attackWindup[attacker] = windup - 1;
    expect(sim.checksum()).not.toBe(baseline);
    pool.attackWindup[attacker] = windup;
    pool.attackTarget[attacker] = NO_ENTITY;
    expect(sim.checksum()).not.toBe(baseline);
    pool.attackTarget[attacker] = targetId;
    expect(sim.checksum()).toBe(baseline);

    const x = pool.posX[attacker]!;
    const y = pool.posY[attacker]!;
    pool.destroy(attackerId);
    const replacement = pool.spawn(EntityType.Slicebot, 0 as PlayerId, x, y);
    expect(replacement & 0xffff).toBe(attacker);
    expect(pool.attackWindup[attacker]).toBe(0);
    expect(pool.attackTarget[attacker]).toBe(NO_ENTITY);
  });
});
