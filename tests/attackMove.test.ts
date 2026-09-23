/**
 * Attack-move, against what a StarCraft player expects it to do.
 *
 * The order a player uses more than any other, and the one whose failure is
 * least visible: an army that stops halfway looks like a pathing hiccup rather
 * than a missing state transition.
 *
 * ## A trap in measuring this
 *
 * Every scenario keeps the unit under test at full health. Staging two evenly
 * matched Burstbots instead means both die, and a corpse keeps its last `order`
 * value forever — so the dead unit reads as "still attack-moving, never
 * resumed". That produced a confident and completely wrong diagnosis once
 * already. Where survival matters, these assert on it explicitly.
 */

import { describe, expect, it } from 'vitest';
import { defOf, GROUP_PATH_THRESHOLD } from '../src/config/rules.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromFloat, fromInt, toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order, Tile, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;
const RUN = 20;

interface Field {
  sim: Simulation;
  ids: number[];
  unit: number;
  origin: { x: number; y: number };
  dest: { x: number; y: number };
}

/** A clear corridor well away from either base, and `n` Burstbots at one end. */
function field(n = 1, type = EntityType.Burstbot): Field {
  const sim = new Simulation(0x51ce7a11);
  const { pool, map } = sim.world;
  const start = map.starts[0]!;

  let c: { x: number; y: number } | null = null;
  for (let r = 12; r < 40 && !c; r++) {
    for (let dx = -r; dx <= r && !c; dx++) {
      const x = start.tileX + dx;
      const y = start.tileY + r;
      let ok = true;
      for (let k = 0; k <= RUN && ok; k++) ok = map.isWalkable(x + k, y);
      if (ok) c = { x, y };
    }
  }
  if (!c) throw new Error('no corridor clear of the bases');

  const ids: number[] = [];
  for (let k = 0; k < n; k++) {
    ids.push(
      pool.spawn(
        type,
        0 as PlayerId,
        Math.round((c.x + 0.5) * FIX),
        Math.round((c.y + 0.5 + k * 0.6) * FIX),
      ),
    );
  }
  return { sim, ids, unit: ids[0]! & 0xffff, origin: c, dest: { x: c.x + RUN, y: c.y } };
}

function command(f: Field, type: CommandType.Move | CommandType.AttackMove): Command {
  return {
    type,
    player: 0,
    units: f.ids,
    x: fromInt(f.dest.x) + 32768,
    y: fromInt(f.dest.y) + 32768,
  } as Command;
}

/** Step the sim, holding our own units at full health throughout. */
function play(f: Field, ticks: number): void {
  const maxHp = defOf(EntityType.Burstbot).maxHp;
  for (let t = 0; t < ticks; t++) {
    for (const id of f.ids) f.sim.world.pool.hp[id & 0xffff] = maxHp;
    f.sim.step([]);
  }
}

function xOf(f: Field): number {
  return toFloat(f.sim.world.pool.posX[f.unit]!);
}

/** Something weak to fight, so the scenario is about movement not attrition. */
function enemyAt(f: Field, dx: number, dy = 0): number {
  return (
    f.sim.world.pool.spawn(
      EntityType.Worker,
      1 as PlayerId,
      Math.round((f.origin.x + 0.5 + dx) * FIX),
      Math.round((f.origin.y + 0.5 + dy) * FIX),
    ) & 0xffff
  );
}

function firedThisTick(f: Field): boolean {
  const shots = f.sim.world.events.shots;
  for (let k = 0; k + 1 < shots.length; k += 2) if (shots[k] === f.unit) return true;
  return false;
}

/** An isolated retreat opposite to the attack-mover's intended advance. */
function pursuitField(mode: 'private' | 'grouped' | 'flying', player: PlayerId = 0) {
  const sim = new Simulation(1);
  const { pool, map } = sim.world;
  map.tiles.fill(Tile.Ground);
  map.occupied.fill(0);
  map.sealTerrain();
  const type = mode === 'flying' ? EntityType.Beamdrone : EntityType.Burstbot;
  const point = (x: number, y: number) => ({
    x: fromFloat(player === 0 ? x : map.width - x),
    y: fromFloat(player === 0 ? y : map.height - y),
  });
  const start = point(40.5, 40.5);
  const goal = point(40.5, 80.5);
  const unit = pool.spawn(type, player, start.x, start.y);
  const foeStart = point(40.5, 34.5);
  const enemy = pool.spawn(EntityType.Worker, 1 - player, foeStart.x, foeStart.y);
  pool.order[enemy & 0xffff] = Order.Hold;
  // A real group command chooses the shared flow field. Its other members are
  // far away and stopped next tick, so they cannot push the pursuer off its trap.
  const others: number[] = [];
  if (mode === 'grouped') {
    for (let k = 1; k < GROUP_PATH_THRESHOLD; k++) {
      const at = point(90.5 + k, 40.5);
      others.push(pool.spawn(type, player, at.x, at.y));
    }
  }
  sim.step([
    { type: CommandType.AttackMove, player, units: [unit, ...others], x: goal.x, y: goal.y },
  ]);
  const index = unit & 0xffff;
  const foe = enemy & 0xffff;
  const y = () =>
    player === 0 ? toFloat(pool.posY[index]!) : map.height - toFloat(pool.posY[index]!);
  const step = (tick: number) => {
    // Stage the foe's retreat independently of worker economy and attrition.
    // The attack-mover receives no further order and is never repositioned.
    const at = point(40.5, 34.5 - Math.min(6, tick * 0.1));
    pool.posX[foe] = at.x;
    pool.posY[foe] = at.y;
    pool.order[foe] = Order.Hold;
    pool.hp[index] = defOf(type).maxHp;
    pool.hp[foe] = defOf(EntityType.Worker).maxHp;
    sim.step(
      tick === 1 && others.length ? [{ type: CommandType.Stop, player, units: others }] : [],
    );
    expect(pool.isAlive(unit)).toBe(true);
    expect(pool.isAlive(enemy)).toBe(true);
  };
  return { sim, unit, enemy, index, foe, point, goal, y, step };
}

describe('attack-move', () => {
  it('has aircraft stop in weapon range and resume only after the fight ends', () => {
    const f = field(1, EntityType.Beamdrone);
    const pool = f.sim.world.pool;
    const foe = enemyAt(f, 5);
    pool.order[foe] = Order.Hold;
    f.sim.step([command(f, CommandType.AttackMove)]);
    for (let t = 0; t < 50 && !firedThisTick(f); t++) f.sim.step([]);
    expect(firedThisTick(f)).toBe(true);
    const firingX = pool.posX[f.unit]!;
    const firingY = pool.posY[f.unit]!;

    for (let t = 0; t < 15; t++) {
      pool.hp[foe] = defOf(EntityType.Worker).maxHp;
      f.sim.step([]);
      expect(pool.posX[f.unit]).toBe(firingX);
      expect(pool.posY[f.unit]).toBe(firingY);
    }

    pool.destroy(pool.idAt(foe));
    play(f, 200);
    expect(pool.alive[f.unit]).toBe(1);
    expect(pool.order[f.unit]).toBe(Order.None);
    expect(Math.abs(xOf(f) - (f.dest.x + 0.5))).toBeLessThan(0.5);
  });

  it('walks to the destination when nothing is in the way', () => {
    const f = field();
    f.sim.step([command(f, CommandType.AttackMove)]);
    play(f, 500);
    expect(Math.abs(xOf(f) - (f.dest.x + 0.5))).toBeLessThan(2);
    expect(f.sim.world.pool.order[f.unit]).toBe(Order.None);
  });

  it('stops to kill something on the way, then carries on', () => {
    // The headline bug. Combat clears the route when a unit stands its ground,
    // and nothing put it back — so an army given one attack-move across the map
    // stopped at the first thing it killed and held an order it would never
    // finish for the rest of the match.
    const f = field();
    const foe = enemyAt(f, 10);
    f.sim.step([command(f, CommandType.AttackMove)]);

    let fought = false;
    for (let t = 0; t < 700; t++) {
      f.sim.world.pool.hp[f.unit] = defOf(EntityType.Burstbot).maxHp;
      f.sim.step([]);
      if (firedThisTick(f)) fought = true;
    }

    expect(`survived: ${f.sim.world.pool.alive[f.unit] === 1}`).toBe('survived: true');
    expect(`fought: ${fought}`).toBe('fought: true');
    expect(`killed it: ${f.sim.world.pool.alive[foe] !== 1}`).toBe('killed it: true');
    expect(`arrived at ${xOf(f).toFixed(1)} of ${(f.dest.x + 0.5).toFixed(1)}`).toBe(
      `arrived at ${xOf(f).toFixed(1)} of ${(f.dest.x + 0.5).toFixed(1)}`,
    );
    expect(Math.abs(xOf(f) - (f.dest.x + 0.5))).toBeLessThan(3);
  });

  it('resumes a grouped advance on its shared field, not a new one each', () => {
    // Grouped orders navigate by flow field, and `clearPath` wipes the goal
    // along with the route. Resuming from each unit's own destination would
    // turn one Dijkstra sweep into one per unit.
    const f = field(GROUP_PATH_THRESHOLD + 2);
    enemyAt(f, 10);
    f.sim.step([command(f, CommandType.AttackMove)]);
    play(f, 700);

    expect(Math.abs(xOf(f) - (f.dest.x + 0.5))).toBeLessThan(4);

    const goals = new Set<number>();
    for (const id of f.ids) {
      const g = f.sim.world.pool.navGoal[id & 0xffff]!;
      if (g >= 0) goals.add(g);
    }
    expect(`distinct flow goals: ${goals.size}`).toBe('distinct flow goals: 1');
  });

  it('walks at an enemy it can see but not yet shoot', () => {
    // Acquisition without approach means a unit strolls past something it has
    // already picked a fight with.
    const f = field();
    const reach = toFloat(defOf(EntityType.Burstbot).attackRange);
    enemyAt(f, 10, reach + 1.5);
    f.sim.step([command(f, CommandType.AttackMove)]);

    let fought = false;
    for (let t = 0; t < 500; t++) {
      f.sim.world.pool.hp[f.unit] = defOf(EntityType.Burstbot).maxHp;
      f.sim.step([]);
      if (firedThisTick(f)) fought = true;
    }
    expect(`engaged the flanker: ${fought}`).toBe('engaged the flanker: true');
  });

  it('does not let a fleeing enemy tow it off its objective', () => {
    // The leash is measured from where the chase began. Measured from the
    // unit's current position it slides along with the target and the chase
    // ratchets — a unit dragged sideways followed one 14.6 tiles off route.
    const f = field();
    const foe = enemyAt(f, 8);
    const pool = f.sim.world.pool;
    f.sim.step([command(f, CommandType.AttackMove)]);

    for (let t = 0; t < 500; t++) {
      pool.hp[f.unit] = defOf(EntityType.Burstbot).maxHp;
      pool.posY[foe] = Math.round((f.origin.y + 0.5 - t * 0.05) * FIX);
      pool.hp[foe] = defOf(EntityType.Worker).maxHp;
      f.sim.step([]);
    }
    const strayed = Math.abs(toFloat(pool.posY[f.unit]!) - (f.origin.y + 0.5));
    expect(`strayed ${strayed.toFixed(1)} tiles, under 7: ${strayed < 7}`).toBe(
      `strayed ${strayed.toFixed(1)} tiles, under 7: true`,
    );
  });

  it.each(['private', 'grouped', 'flying'] as const)(
    'resumes its %s advance after a fleeing enemy stops beyond the pursuit leash',
    (mode) => {
      for (const player of [0, 1]) {
        const f = pursuitField(mode, player);
        let closestToFoe = f.y();
        for (let t = 1; t <= 600; t++) {
          f.step(t);
          closestToFoe = Math.min(closestToFoe, f.y());
        }
        // The old radius-only check moved back inside the leash toward the
        // objective, then immediately chased outward again in the same tick.
        // It could spend the rest of the match alive, 46 tiles from its goal.
        expect(closestToFoe).toBeLessThan(35);
        expect(f.y()).toBeGreaterThan(80);
        expect(f.sim.world.pool.order[f.index]).toBe(Order.None);
      }
    },
  );

  it('can pursue a new enemy after leaving the abandoned fight behind', () => {
    const f = pursuitField('private');
    let tick = 0;
    while (tick < 300 && (tick < 60 || f.y() < 45)) f.step(++tick);
    expect(f.y()).toBeGreaterThanOrEqual(45);
    const pool = f.sim.world.pool;
    const encounterX = pool.posX[f.index]!;
    const next = pool.spawn(EntityType.Worker, 1, encounterX + fromInt(7), pool.posY[f.index]!);
    const ni = next & 0xffff;
    pool.order[ni] = Order.Hold;
    let fought = false;
    for (let t = 0; t < 100; t++) {
      pool.hp[ni] = defOf(EntityType.Worker).maxHp;
      f.step(++tick);
      const shots = f.sim.world.events.shots;
      for (let k = 0; k < shots.length; k += 2) {
        if (shots[k] === f.index && shots[k + 1] === ni) fought = true;
      }
    }
    expect(pool.isAlive(next)).toBe(true);
    expect(pool.posX[f.index]).toBeGreaterThan(encounterX + FIX / 2);
    expect(fought).toBe(true);
  });

  it('still stops to fire if a disengaged target comes into weapon range', () => {
    const f = pursuitField('private');
    let previousY = f.y();
    let resumed = false;
    for (let t = 1; t <= 150; t++) {
      f.step(t);
      if (previousY < 35 && f.y() > previousY) {
        resumed = true;
        break;
      }
      previousY = f.y();
    }
    expect(resumed).toBe(true);
    const pool = f.sim.world.pool;
    pool.posY[f.foe] = pool.posY[f.index]! - fromInt(4);
    f.sim.step([]);
    const firingY = pool.posY[f.index]!;
    let fought = false;
    for (let t = 0; t < 30; t++) {
      pool.hp[f.foe] = defOf(EntityType.Worker).maxHp;
      f.sim.step([]);
      expect(pool.posY[f.index]).toBe(firingY);
      const shots = f.sim.world.events.shots;
      for (let k = 0; k < shots.length; k += 2) {
        if (shots[k] === f.index && shots[k + 1] === f.foe) fought = true;
      }
    }
    expect(pool.isAlive(f.unit)).toBe(true);
    expect(pool.isAlive(f.enemy)).toBe(true);
    expect(fought).toBe(true);
  });

  it('holds where it arrives and still shoots what comes near', () => {
    const f = field();
    f.sim.step([command(f, CommandType.AttackMove)]);
    play(f, 500);
    expect(f.sim.world.pool.order[f.unit]).toBe(Order.None);

    f.sim.world.pool.spawn(
      EntityType.Worker,
      1 as PlayerId,
      Math.round((xOf(f) + 2) * FIX),
      f.sim.world.pool.posY[f.unit]!,
    );
    let fought = false;
    for (let t = 0; t < 200; t++) {
      f.sim.step([]);
      if (firedThisTick(f)) fought = true;
    }
    expect(`defends its ground: ${fought}`).toBe('defends its ground: true');
  });
});

describe('a plain move is not an attack-move', () => {
  it('walks past an enemy without stopping to fight it', () => {
    // The distinction is the entire reason both orders exist.
    const f = field();
    enemyAt(f, 10);
    f.sim.step([command(f, CommandType.Move)]);

    // Only while the order is actually held. Once the unit arrives it goes
    // idle, and an idle unit returning fire is correct — counting that as a
    // failure would be testing the wrong thing.
    let foughtUnderOrder = false;
    let tickedUnderOrder = 0;
    for (let t = 0; t < 500; t++) {
      // Held across the whole tick: movement finishes the order mid-tick and
      // combat runs after it, so the arrival tick is legitimately idle by the
      // time anything shoots.
      const before = f.sim.world.pool.order[f.unit] === Order.Move;
      f.sim.world.pool.hp[f.unit] = defOf(EntityType.Burstbot).maxHp;
      f.sim.step([]);
      const onOrder = before && f.sim.world.pool.order[f.unit] === Order.Move;
      if (onOrder) {
        tickedUnderOrder++;
        if (firedThisTick(f)) foughtUnderOrder = true;
      }
    }
    expect(tickedUnderOrder).toBeGreaterThan(50);
    expect(`fired while on a move order: ${foughtUnderOrder}`).toBe(
      'fired while on a move order: false',
    );
  });
});
