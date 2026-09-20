/**
 * The abilities the robot line was built around, each measured in a real fight.
 *
 * Every one of them is a claim the info panel makes out loud — "splash 2.2",
 * "armour 4", "hits 3 at once" — so every one of them gets a test that a player
 * reading the panel would recognise as the thing they were promised. The point
 * is not the arithmetic; it is that none of these can quietly stop happening
 * while the panel goes on advertising it.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { abilityText, CHILL_SPEED, defOf, DEFS, MIN_DAMAGE } from '../src/config/rules.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { EntityType, seconds, type PlayerId } from '../src/sim/types.js';
import { activityOf } from '../src/ui/status.js';
import { ProceduralModelProvider } from '../src/render/models/procedural.js';

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
  for (let r = 14; r < 48; r++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = start.tileX + dx;
      const y = start.tileY + r;
      let open = true;
      for (let k = -2; k <= 14; k++) if (!map.isWalkable(x + k, y)) open = false;
      if (open) return { x, y };
    }
  }
  throw new Error('no clear ground');
}

describe('the robot line', () => {
  let sim: Simulation;
  let spot: { x: number; y: number };

  beforeEach(() => {
    sim = new Simulation(0x51ce7a11);
    spot = clearSpot(sim);
  });

  /** Spawn at an offset from the clear spot, in tiles, and return the slot. */
  function put(type: EntityType, player: number, dx: number, dy = 0): number {
    const x = Math.round((spot.x + 0.5 + dx) * FIX);
    const y = Math.round((spot.y + 0.5 + dy) * FIX);
    const id = sim.world.pool.spawn(type, player as PlayerId, x, y);
    return id & 0xffff;
  }

  /**
   * Run until the attacker lands a blow, holding everyone at full health so
   * nothing dies mid-measurement, and report what each slot lost on that tick.
   *
   * Units are held on Hold so nothing walks out of the arrangement being
   * measured — these tests are about what one attack reaches, not about who
   * chases whom.
   */
  function damageOnFirstHit(attacker: number, watch: number[], ticks = 300): Map<number, number> {
    const pool = sim.world.pool;
    const full = (i: number) => defOf(pool.type[i]! as EntityType).maxHp;
    for (let t = 0; t < ticks; t++) {
      for (const i of watch) pool.hp[i] = full(i);
      pool.hp[attacker] = full(attacker);
      sim.step([]);
      const shots = sim.world.events.shots;
      let fired = false;
      for (let k = 0; k < shots.length; k += 2) if (shots[k] === attacker) fired = true;
      if (!fired) continue;
      const out = new Map<number, number>();
      for (const i of watch) out.set(i, full(i) - pool.hp[i]!);
      return out;
    }
    throw new Error('the attacker never fired');
  }

  it('has a Sentry shell take everything standing inside its blast', () => {
    const gun = defOf(EntityType.Sentry);
    const sentry = put(EntityType.Sentry, 0, 0);
    // One aimed at, one beside it well inside the 2.2 blast, one clear of it.
    const aimed = put(EntityType.Burstbot, 1, 6);
    const beside = put(EntityType.Burstbot, 1, 7.2);
    const clear = put(EntityType.Burstbot, 1, 6, 5);

    const hurt = damageOnFirstHit(sentry, [aimed, beside, clear]);
    expect(hurt.get(aimed)).toBe(gun.damage);
    expect(hurt.get(beside)).toBe(gun.damage);
    expect(hurt.get(clear)).toBe(0);
  });

  it('keeps a blast off a flyer when the weapon that threw it cannot reach one', () => {
    // The Sentry's barrel points at the sky and it still cannot hit anything in
    // it: splash is the same weapon, under the same rule.
    const sentry = put(EntityType.Sentry, 0, 0);
    const ground = put(EntityType.Burstbot, 1, 6);
    const air = put(EntityType.Beamdrone, 1, 6.6);

    const hurt = damageOnFirstHit(sentry, [ground, air]);
    expect(hurt.get(ground)).toBe(defOf(EntityType.Sentry).damage);
    expect(hurt.get(air)).toBe(0);
  });

  it('will not let a Sentry fire at something inside its minimum range', () => {
    const sentry = put(EntityType.Sentry, 0, 0);
    put(EntityType.Burstbot, 1, 1.4);
    const pool = sim.world.pool;
    for (let t = 0; t < 120; t++) {
      pool.hp[sentry] = defOf(EntityType.Sentry).maxHp;
      sim.step([]);
      const shots = sim.world.events.shots;
      for (let k = 0; k < shots.length; k += 2) {
        expect(shots[k], 'the Sentry fired at point-blank range').not.toBe(sentry);
      }
    }
  });

  it('has an Arclight strike three enemies at once and no more', () => {
    const coils = defOf(EntityType.Arclight);
    const arc = put(EntityType.Arclight, 0, 0);
    // Four in reach, spread so no two are the same distance away.
    const near = [
      put(EntityType.Burstbot, 1, 2),
      put(EntityType.Burstbot, 1, 2.6),
      put(EntityType.Burstbot, 1, 3.2),
      put(EntityType.Burstbot, 1, 3.8),
    ];

    const hurt = damageOnFirstHit(arc, near);
    const struck = near.filter((i) => hurt.get(i)! > 0);
    expect(struck.length).toBe(coils.maxTargets);
    for (const i of struck) expect(hurt.get(i)).toBe(coils.damage);
    // The three it took are the three nearest it, not whichever three the
    // spatial grid happened to reach first.
    expect(struck).toEqual(near.slice(0, 3));
  });

  it('has a Piercebot bolt take everything on the line, and nothing off it', () => {
    const rail = defOf(EntityType.Piercebot);
    const bot = put(EntityType.Piercebot, 0, 0);
    const between = put(EntityType.Burstbot, 1, 3);
    const aimed = put(EntityType.Burstbot, 1, 6);
    const aside = put(EntityType.Burstbot, 1, 3, 2.5);
    // Well clear of `between`'s distance, so which one the rail aims at is not
    // a tie for the comparator to break.
    const behind = put(EntityType.Burstbot, 1, -4.5);

    const hurt = damageOnFirstHit(bot, [between, aimed, aside, behind]);
    // It aims at the nearest, so the one further down the line is beyond the
    // shot and the one behind it was never on it.
    expect(hurt.get(between)).toBe(rail.damage);
    expect(hurt.get(aside)).toBe(0);
    expect(hurt.get(behind)).toBe(0);
    expect(hurt.get(aimed)).toBe(0);
  });

  it('takes a Dark Golem armour off every hit, whoever throws it', () => {
    const golem = defOf(EntityType.DarkGolem);
    const target = put(EntityType.DarkGolem, 1, 0);
    const burst = put(EntityType.Burstbot, 0, 3);

    const hurt = damageOnFirstHit(burst, [target]);
    expect(hurt.get(target)).toBe(defOf(EntityType.Burstbot).damage - golem.armor);
  });

  it('never lets armour reduce a hit to nothing', () => {
    // A Worker's 5 is below the Dark Golem's 4 plus the floor, so this is the
    // matchup where armour would otherwise start rounding to immunity.
    const target = put(EntityType.DarkGolem, 1, 0);
    const worker = put(EntityType.Worker, 0, 1.2);

    const hurt = damageOnFirstHit(worker, [target]);
    expect(hurt.get(target)).toBe(MIN_DAMAGE);
  });

  it('chills what an Ice Golem hits, for exactly as long as the panel says', () => {
    const pool = sim.world.pool;
    put(EntityType.IceGolem, 0, 0);
    const victim = put(EntityType.Burstbot, 1, 3.5);

    let chilled = 0;
    for (let t = 0; t < 200 && chilled === 0; t++) {
      pool.hp[victim] = defOf(EntityType.Burstbot).maxHp;
      sim.step([]);
      chilled = pool.chill[victim]!;
    }
    expect(chilled).toBe(defOf(EntityType.IceGolem).chillTicks);
    expect(toFloat(CHILL_SPEED)).toBe(0.5);
  });

  it('moves a chilled unit at half its speed, and at its own speed after', () => {
    // Two identical walks, one of them chilled, so the only difference between
    // the two distances is the thing under test. Both start from rest, so both
    // pay the same ramp.
    function walked(chill: number, ticks: number): number {
      const run = new Simulation(0x51ce7a11);
      const at = clearSpot(run);
      const pool = run.world.pool;
      const id = pool.spawn(
        EntityType.Burstbot,
        0 as PlayerId,
        Math.round((at.x + 0.5) * FIX),
        Math.round((at.y + 0.5) * FIX),
      );
      const i = id & 0xffff;
      const order: Command = {
        type: CommandType.Move,
        player: 0 as PlayerId,
        units: [id],
        x: Math.round((at.x + 0.5 + 12) * FIX),
        y: pool.posY[i]!,
      };
      run.step([order]);
      const from = pool.posX[i]!;
      for (let t = 0; t < ticks; t++) {
        if (chill > 0) pool.chill[i] = chill;
        run.step([]);
      }
      return pool.posX[i]! - from;
    }

    const WINDOW = 20;
    const free = walked(0, WINDOW);
    const slowed = walked(seconds(10), WINDOW);
    expect(free).toBeGreaterThan(0);
    expect(slowed).toBeGreaterThan(0);
    // Half speed, within the ramp's slack: both accelerate from a standstill,
    // and a chilled unit reaches its lower ceiling sooner.
    expect(slowed / free).toBeGreaterThan(0.4);
    expect(slowed / free).toBeLessThan(0.7);
  });

  it('has a Boomwalker die with its payload', () => {
    const pool = sim.world.pool;
    const bomb = put(EntityType.Boomwalker, 0, 0);
    const a = put(EntityType.Burstbot, 1, 1.0);
    const b = put(EntityType.Burstbot, 1, 1.9);

    const hurt = damageOnFirstHit(bomb, [a, b]);
    expect(hurt.get(a)).toBe(defOf(EntityType.Boomwalker).damage);
    expect(hurt.get(b)).toBe(defOf(EntityType.Boomwalker).damage);
    // The detonation is queued as a death, so the reap at the end of the tick
    // takes the Boomwalker with it.
    expect(pool.alive[bomb]).toBe(0);
  });

  it('has a Fixomatic mend a damaged unit beside it, but never itself', () => {
    const pool = sim.world.pool;
    const fixer = put(EntityType.Fixomatic, 0, 0);
    const hurtUnit = put(EntityType.Slicebot, 0, 2);
    pool.hp[hurtUnit] = 20;
    pool.hp[fixer] = 10;

    for (let t = 0; t < seconds(2); t++) sim.step([]);
    expect(pool.hp[hurtUnit]).toBeGreaterThan(20);
    expect(pool.hp[fixer]).toBe(10);
  });

  it('does not let a Fixomatic mend past full, or mend a structure at all', () => {
    const pool = sim.world.pool;
    put(EntityType.Fixomatic, 0, 0);
    const unit = put(EntityType.Slicebot, 0, 2);
    const depot = put(EntityType.Depot, 0, -3);
    // A unit a few points down, and a structure most of the way down. Long
    // enough for the repairer to top the unit up several times over.
    pool.hp[unit] = defOf(EntityType.Slicebot).maxHp - 5;
    pool.hp[depot] = 100;

    for (let t = 0; t < seconds(6); t++) sim.step([]);
    expect(pool.hp[unit]).toBe(defOf(EntityType.Slicebot).maxHp);
    expect(pool.hp[depot]).toBe(100);
  });

  it('says a working Fixomatic is repairing, not engaging', () => {
    const pool = sim.world.pool;
    const fixer = put(EntityType.Fixomatic, 0, 0);
    const unit = put(EntityType.Slicebot, 0, 2);
    pool.hp[unit] = 30;

    // Idle until something needs it, then repairing — never "engaging", which
    // is what the panel said while `combatTarget` was assumed to be an enemy.
    expect(activityOf(sim.world, fixer)).toBe('idle');
    for (let t = 0; t < seconds(1); t++) sim.step([]);
    expect(activityOf(sim.world, fixer)).toBe('repairing');
  });

  it('never has a Fixomatic hurt anything', () => {
    const fixer = defOf(EntityType.Fixomatic);
    expect(fixer.damage).toBe(0);
    expect(fixer.repairAmount).toBeGreaterThan(0);
  });
});

describe('the roster', () => {
  it('trains all twelve robots across the three production buildings', () => {
    const line = [
      ...defOf(EntityType.Barracks).produces,
      ...defOf(EntityType.Factory).produces,
      ...defOf(EntityType.Airport).produces,
    ];
    expect(line.length).toBe(12);
    expect(new Set(line).size).toBe(12);
    // The Command Post's Worker is the only unit none of them makes.
    for (const def of DEFS) {
      if (def.isBuilding || def.type === EntityType.Worker) continue;
      expect(line, `${def.name} is trained nowhere`).toContain(def.type);
    }
  });

  it('gives every entity type a procedural model of its own', () => {
    // The Factory shipped invisible: the renderer built its instanced pools
    // from a list of types written out by hand, and a structure has no authored
    // model to fall back on — the procedural mesh is the only thing that ever
    // draws it. The pool list comes from `DEFS` now; this is the other half,
    // that the provider has a real shape for each of them rather than the
    // single grey box it returns for anything it does not recognise.
    const provider = new ProceduralModelProvider();
    try {
      for (const def of DEFS) {
        const spec = provider.get(def.type);
        expect(spec.parts.length, `${def.name} has only the fallback box`).toBeGreaterThan(1);
        expect(spec.radius, `${def.name} ring`).toBeCloseTo(toFloat(def.radius), 5);
      }
    } finally {
      provider.dispose();
    }
  });

  it('says every ability out loud on the info panel', () => {
    for (const def of DEFS) {
      const said = abilityText(def).join(' · ');
      if (def.splashRadius > 0) expect(said).toContain('splash');
      if (def.maxTargets > 1) expect(said).toContain(`hits ${def.maxTargets}`);
      if (def.pierce) expect(said).toContain('pierces');
      if (def.armor > 0) expect(said).toContain(`armour ${def.armor}`);
      if (def.chillTicks > 0) expect(said).toContain('chills');
      if (def.repairAmount > 0) expect(said).toContain('repairs');
      if (def.detonates) expect(said).toContain('detonates');
      if (def.minRange > 0) expect(said).toContain('min range');
    }
  });

  it('splits infantry, vehicles and aircraft into their own buildings', () => {
    expect(defOf(EntityType.Barracks).produces.length).toBe(5);
    expect(defOf(EntityType.Factory).produces.length).toBe(5);
    expect(defOf(EntityType.Airport).produces.length).toBe(2);
  });

  it('keeps every Barracks unit within reach of a Barracks budget', () => {
    // The Barracks is the building you have in the first two minutes. Nothing
    // it makes may cost more than the Factory that unlocks the rest, or the
    // tech step would be the cheaper way to a bigger unit.
    const factory = defOf(EntityType.Factory).mineralCost;
    for (const light of defOf(EntityType.Barracks).produces) {
      expect(defOf(light).mineralCost).toBeLessThan(factory);
    }
  });
});
