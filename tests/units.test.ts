/**
 * Unit roster, and the rule that a unit's damage is a single number.
 *
 * There used to be a counter triangle applying a hidden 2x to three matchups.
 * It is gone so that the figure on the info panel is the figure a player gets,
 * and that is worth a test: a multiplier reintroduced anywhere between the def
 * and `applyDamage` breaks nothing visibly and makes the HUD quietly wrong.
 */

import { describe, expect, it } from 'vitest';
import { DEFS, defOf } from '../src/config/rules.js';
import { ENTITY_TYPE_COUNT, EntityType, type PlayerId } from '../src/sim/types.js';
import { HeadlessMatch } from '../src/ai/headless.js';
import { duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { scriptedAgents } from './helpers/agents.js';

describe('entity definitions', () => {
  it('has one def per type, in enum order', () => {
    // The simulation looks defs up by numeric type, so a misaligned table would
    // silently give every unit the wrong stats.
    expect(DEFS.length).toBe(ENTITY_TYPE_COUNT);
    for (let i = 0; i < DEFS.length; i++) {
      expect(DEFS[i]!.type).toBe(i);
    }
  });

  it('describes the Beamdrone as a flying, non-colliding unit', () => {
    const g = defOf(EntityType.Beamdrone);
    expect(g.flying).toBe(true);
    expect(g.collides).toBe(false);
    expect(g.isBuilding).toBe(false);
    expect(g.attackRange).toBeGreaterThan(0);
  });

  it('keeps every ground unit non-flying', () => {
    for (const type of [EntityType.Worker, EntityType.Burstbot, EntityType.Slicebot]) {
      expect(defOf(type).flying).toBe(false);
    }
  });

  it('lets the barracks train all three combat units', () => {
    const produces = defOf(EntityType.Barracks).produces;
    expect(produces).toContain(EntityType.Burstbot);
    expect(produces).toContain(EntityType.Slicebot);
    expect(produces).toContain(EntityType.Beamdrone);
  });
});

describe('damage is a single number', () => {
  const FIX = 65536;

  /**
   * Clear ground a long way from either start.
   *
   * Workers carry a 0.6-reach weapon and will shoot anything that comes near
   * their base, so a fight staged near a start location measures the workers as
   * much as the units under test.
   */
  function clearSpot(sim: Simulation): { x: number; y: number } {
    const map = sim.world.map;
    const start = map.starts[0]!;
    for (let r = 12; r < 40; r++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = start.tileX + dx;
        const y = start.tileY + r;
        if (map.isWalkable(x, y) && map.isWalkable(x + 1, y) && map.isWalkable(x + 2, y)) {
          return { x, y };
        }
      }
    }
    throw new Error('no clear ground');
  }

  /**
   * Stage one real fight and report what one blow actually took off.
   *
   * Both units are healed every tick, so neither can die mid-measurement — a
   * corpse reads as a full-health unit that stopped taking damage — and the
   * reading is taken on a tick carrying exactly one shot from this attacker at
   * this target, so nothing else can be folded into the number.
   */
  function damageDealt(attacker: EntityType, target: EntityType): number {
    const sim = new Simulation(0x51ce7a11);
    const spot = clearSpot(sim);
    const pool = sim.world.pool;
    const at = (n: number) => Math.round(n * FIX);
    const a = pool.spawn(attacker, 0 as PlayerId, at(spot.x + 0.5), at(spot.y + 0.5)) & 0xffff;
    const b = pool.spawn(target, 1 as PlayerId, at(spot.x + 1.7), at(spot.y + 0.5)) & 0xffff;
    const attackerHp = defOf(attacker).maxHp;
    const targetHp = defOf(target).maxHp;

    for (let t = 0; t < 400; t++) {
      pool.hp[a] = attackerHp;
      pool.hp[b] = targetHp;
      sim.step([]);
      const shots = sim.world.events.shots;
      let fired = 0;
      for (let k = 0; k < shots.length; k += 2) {
        if (shots[k] === a && shots[k + 1] === b) fired++;
      }
      if (fired === 1) return targetHp - pool.hp[b]!;
    }
    return -1;
  }

  const ARMED = [EntityType.Burstbot, EntityType.Slicebot, EntityType.Beamdrone, EntityType.Worker];

  for (const attacker of ARMED) {
    for (const target of ARMED) {
      // Melee cannot reach a flyer at all, which is the one matchup rule left
      // and is expressed as `canHitAir`, not as a number.
      if (defOf(target).flying && !defOf(attacker).canHitAir) continue;

      it(`has a ${defOf(attacker).name} hit a ${defOf(target).name} for its listed damage`, () => {
        expect(damageDealt(attacker, target)).toBe(defOf(attacker).damage);
      });
    }
  }
});

describe('flying units in a real match', () => {
  it('get built and fly over terrain the ground army cannot cross', () => {
    const config = duelMatch(0x51ce7a11, { botPlayers: [0, 1] });
    const sim = new HeadlessMatch(config, scriptedAgents(config));

    let sawBeamdrone = false;
    let beamdroneOverSolidGround = false;

    for (let t = 0; t < 20 * 60 * 12 && !beamdroneOverSolidGround; t++) {
      sim.step();
      const pool = sim.world.pool;
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1) continue;
        if (pool.type[i] !== EntityType.Beamdrone) continue;
        sawBeamdrone = true;
        // A flyer standing over an unwalkable tile proves it is genuinely
        // ignoring terrain rather than just being a fast ground unit.
        const tile = sim.world.map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
        if (tile < 0) continue;
        const tx = sim.world.map.tileXOf(tile);
        const ty = sim.world.map.tileYOf(tile);
        if (!sim.world.map.isWalkable(tx, ty)) beamdroneOverSolidGround = true;
      }
    }

    expect(sawBeamdrone).toBe(true);
    expect(beamdroneOverSolidGround).toBe(true);
  });
});
