/**
 * Idle units defending themselves.
 *
 * Combat acquires a target within sight and then only fires if it is already
 * within weapon range. Nothing ever closed the gap, so a unit whose weapon is
 * shorter than its eyes would stand still while an enemy it had already picked
 * out shot at it. For a rifleman that is nearly invisible — its range covers
 * most of what it can see — but a brawler reaches 1.3 tiles and sees 7, so it
 * only ever fought things already touching it. It was reported as melee units
 * simply not attacking.
 *
 * The leash matters as much as the engagement: an idle line that chases every
 * straggler unravels, so a unit walks the last few tiles and no further.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order } from '../src/sim/types.js';

const FIX = 65536;
const COMBAT_TYPES = [EntityType.Rifleman, EntityType.Brawler, EntityType.Gunship];

/**
 * Open ground near the base, wide enough to place both units and walk between.
 *
 * Found rather than hard-coded: the map is more than half cliff, and a unit
 * standing in rock is ejected by `clampToMap` every tick, which quietly ruins
 * any measurement of how far it walked.
 */
function openSpot(sim: Simulation, span: number): { x: number; y: number } {
  const map = sim.world.map;
  const start = map.starts[0]!;
  for (let r = 4; r < 30; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        let clear = true;
        for (let k = -1; k <= span + 1 && clear; k++) {
          clear =
            map.isWalkable(x + k, y) &&
            map.isWalkable(x + k, y + 1) &&
            map.isWalkable(x + k, y - 1);
        }
        if (clear) return { x, y };
      }
    }
  }
  throw new Error('no open ground near the start');
}

/**
 * Put an attacker `dist` tiles from a stationary enemy and run.
 *
 * The enemy is pinned in place and healed each tick, so this measures only
 * whether the attacker closes and engages.
 */
function fight(
  attacker: EntityType,
  dist: number,
  order: Order = Order.None,
  ticks = 200,
): { hit: boolean; walked: number } {
  const sim = new Simulation(0x51ce7a11);
  const pool = sim.world.pool;
  const spot = openSpot(sim, Math.ceil(dist));
  const ax = spot.x + 0.5;
  const ay = spot.y + 0.5;

  const a = pool.spawn(attacker, 0, Math.round(ax * FIX), Math.round(ay * FIX)) & 0xffff;
  const b =
    pool.spawn(EntityType.Rifleman, 1, Math.round((ax + dist) * FIX), Math.round(ay * FIX)) &
    0xffff;
  const hp0 = pool.hp[b]!;
  const x0 = pool.posX[a]!;

  for (let t = 0; t < ticks; t++) {
    pool.posX[b] = Math.round((ax + dist) * FIX);
    pool.posY[b] = Math.round(ay * FIX);
    pool.hp[b] = hp0;
    pool.order[b] = Order.Hold;
    pool.order[a] = order;
    sim.step([]);
    if (pool.hp[b]! < hp0) return { hit: true, walked: (pool.posX[a]! - x0) / FIX };
  }
  return { hit: false, walked: (pool.posX[a]! - x0) / FIX };
}

describe('engaging on sight', () => {
  it('walks up to an enemy it can see but cannot yet reach', () => {
    for (const type of COMBAT_TYPES) {
      const def = defOf(type);
      // Comfortably beyond weapon range for every type, well inside sight.
      const dist = toFloat(def.attackRange) + 1.5;
      const result = fight(type, dist);
      expect(`${def.name} at ${dist.toFixed(1)}: ${result.hit}`).toBe(
        `${def.name} at ${dist.toFixed(1)}: true`,
      );
      expect(result.walked).toBeGreaterThan(0);
    }
  });

  it('fights what is already in range without moving', () => {
    for (const type of COMBAT_TYPES) {
      const result = fight(type, 1);
      expect(`${defOf(type).name}: ${result.hit}`).toBe(`${defOf(type).name}: true`);
      expect(Math.abs(result.walked)).toBeLessThan(0.4);
    }
  });

  it('does not chase past its leash', () => {
    // Beyond the leash the unit holds its ground, so an idle line does not
    // unravel one straggler at a time.
    const result = fight(EntityType.Brawler, 9);
    expect(result.hit).toBe(false);
    expect(Math.abs(result.walked)).toBeLessThan(0.4);
  });

  it('still ignores enemies while on a plain move order', () => {
    // That distinction is the whole reason attack-move exists.
    const result = fight(EntityType.Brawler, 3, Order.Move);
    expect(result.hit).toBe(false);
  });

  it('holds position on Hold, but still shoots what comes to it', () => {
    const away = fight(EntityType.Brawler, 3, Order.Hold);
    expect(away.hit).toBe(false);
    expect(Math.abs(away.walked)).toBeLessThan(0.4);

    const adjacent = fight(EntityType.Brawler, 1, Order.Hold);
    expect(adjacent.hit).toBe(true);
  });
});

describe('weapon reach', () => {
  it('measures to the target’s edge, so buildings are hittable', () => {
    // A Command Post's centre is two tiles from its edge, and its footprint is
    // not walkable — so a melee unit that measured centre-to-centre against its
    // own 0.9 reach could never touch one.
    const sim = new Simulation(0x51ce7a11);
    const pool = sim.world.pool;
    let post = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 1 && pool.type[i] === EntityType.CommandPost) {
        post = i;
      }
    }
    expect(post).toBeGreaterThanOrEqual(0);

    const hqDef = defOf(EntityType.CommandPost);
    const brawler = defOf(EntityType.Brawler);
    const reach = toFloat(brawler.attackRange) + toFloat(hqDef.radius);
    // Closest a brawler can physically stand to the centre: outside the
    // footprint, by its own radius.
    const closest = hqDef.footprint / 2 + toFloat(brawler.radius);
    expect(reach).toBeGreaterThan(closest);
  });

  it('lets an attack order reach a building', () => {
    const sim = new Simulation(0x51ce7a11);
    const pool = sim.world.pool;
    let post = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 1 && pool.type[i] === EntityType.CommandPost) {
        post = i;
      }
    }
    const spawnX = pool.posX[post]! + Math.round(3.2 * FIX);
    const brawler =
      pool.spawn(EntityType.Brawler, 0, spawnX, pool.posY[post]!) & 0xffff;
    const hp0 = pool.hp[post]!;

    sim.step([
      {
        type: CommandType.Attack,
        player: 0,
        units: [pool.idAt(brawler)],
        target: pool.idAt(post),
      },
    ]);
    for (let t = 0; t < 120; t++) {
      sim.step([]);
      if (pool.hp[post]! < hp0) break;
    }
    expect(pool.hp[post]!).toBeLessThan(hp0);
  });
});
