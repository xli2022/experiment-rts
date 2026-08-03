/**
 * Unit roster and the counter triangle.
 *
 * The triangle is the only place in the game where unit choice matters, so it
 * gets pinned explicitly: a subtle sign error here would quietly make one unit
 * strictly best and collapse the whole mechanic without breaking anything.
 */

import { describe, expect, it } from 'vitest';
import { COUNTER_PCT, counterBonusPct, DEFS, defOf } from '../src/config/rules.js';
import { ENTITY_TYPE_COUNT, EntityType } from '../src/sim/types.js';
import { Simulation } from '../src/sim/tick.js';

describe('entity definitions', () => {
  it('has one def per type, in enum order', () => {
    // The simulation looks defs up by numeric type, so a misaligned table would
    // silently give every unit the wrong stats.
    expect(DEFS.length).toBe(ENTITY_TYPE_COUNT);
    for (let i = 0; i < DEFS.length; i++) {
      expect(DEFS[i]!.type).toBe(i);
    }
  });

  it('describes the gunship as a flying, non-colliding unit', () => {
    const g = defOf(EntityType.Gunship);
    expect(g.flying).toBe(true);
    expect(g.collides).toBe(false);
    expect(g.isBuilding).toBe(false);
    expect(g.attackRange).toBeGreaterThan(0);
  });

  it('keeps every ground unit non-flying', () => {
    for (const type of [EntityType.Worker, EntityType.Rifleman, EntityType.Brawler]) {
      expect(defOf(type).flying).toBe(false);
    }
  });

  it('lets the barracks train all three combat units', () => {
    const produces = defOf(EntityType.Barracks).produces;
    expect(produces).toContain(EntityType.Rifleman);
    expect(produces).toContain(EntityType.Brawler);
    expect(produces).toContain(EntityType.Gunship);
  });
});

describe('counter triangle', () => {
  const RANGED = EntityType.Rifleman;
  const MELEE = EntityType.Brawler;
  const AIR = EntityType.Gunship;

  it('applies ranged > air > melee > ranged', () => {
    expect(counterBonusPct(RANGED, AIR)).toBe(COUNTER_PCT);
    expect(counterBonusPct(AIR, MELEE)).toBe(COUNTER_PCT);
    expect(counterBonusPct(MELEE, RANGED)).toBe(COUNTER_PCT);
  });

  it('gives no bonus the other way round', () => {
    // If both directions were bonused the triangle would cancel out entirely.
    expect(counterBonusPct(AIR, RANGED)).toBe(100);
    expect(counterBonusPct(MELEE, AIR)).toBe(100);
    expect(counterBonusPct(RANGED, MELEE)).toBe(100);
  });

  it('is a strict cycle with no dominant unit', () => {
    // Each of the three counters exactly one other and is countered by exactly
    // one other. Anything else means some unit is simply the best pick.
    const trio = [RANGED, MELEE, AIR];
    for (const attacker of trio) {
      const beats = trio.filter((t) => t !== attacker && counterBonusPct(attacker, t) > 100);
      const beatenBy = trio.filter((t) => t !== attacker && counterBonusPct(t, attacker) > 100);
      expect(beats.length).toBe(1);
      expect(beatenBy.length).toBe(1);
      expect(beats[0]).not.toBe(beatenBy[0]);
    }
  });

  it('never bonuses a unit against itself', () => {
    for (const t of [RANGED, MELEE, AIR, EntityType.Worker]) {
      expect(counterBonusPct(t, t)).toBe(100);
    }
  });

  it('leaves workers and buildings out of the triangle', () => {
    for (const t of [RANGED, MELEE, AIR]) {
      expect(counterBonusPct(EntityType.Worker, t)).toBe(100);
      expect(counterBonusPct(t, EntityType.CommandPost)).toBe(100);
    }
  });
});

describe('flying units in a real match', () => {
  it('get built and fly over terrain the ground army cannot cross', () => {
    const sim = new Simulation(0x51ce7a11);
    sim.botPlayers.add(0);
    sim.botPlayers.add(1);

    let sawGunship = false;
    let gunshipOverSolidGround = false;

    for (let t = 0; t < 20 * 60 * 12 && !gunshipOverSolidGround; t++) {
      sim.step([]);
      const pool = sim.world.pool;
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1) continue;
        if (pool.type[i] !== EntityType.Gunship) continue;
        sawGunship = true;
        // A flyer standing over an unwalkable tile proves it is genuinely
        // ignoring terrain rather than just being a fast ground unit.
        const tile = sim.world.map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
        if (tile < 0) continue;
        const tx = sim.world.map.tileXOf(tile);
        const ty = sim.world.map.tileYOf(tile);
        if (!sim.world.map.isWalkable(tx, ty)) gunshipOverSolidGround = true;
      }
    }

    expect(sawGunship).toBe(true);
    expect(gunshipOverSolidGround).toBe(true);
  });
});
