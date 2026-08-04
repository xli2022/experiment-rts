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

/**
 * Somewhere a good distance from the start to march to, with room around it.
 *
 * `clear` is the radius of open ground the caller needs. Taking the first
 * walkable tile instead lands on the edge of a rock formation, where the
 * formation slots legitimately fall back to the raw target — fine behaviour,
 * but it makes "every unit gets its own point" untestable.
 */
function target(sim: Simulation, clear = 0): { x: number; y: number } {
  const { map } = sim.world;
  const start = map.starts[0]!;
  const open = (x: number, y: number): boolean => {
    for (let dy = -clear; dy <= clear; dy++) {
      for (let dx = -clear; dx <= clear; dx++) {
        if (!map.isWalkable(x + dx, y + dy)) return false;
      }
    }
    return true;
  };
  for (let r = 14; r < 40; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        if (open(x, y)) return { x, y };
      }
    }
  }
  throw new Error('nowhere to march to');
}

/**
 * A walkable tile boxed in by rock on `walls` of its four sides.
 *
 * The hard case for arrival, and the one that separates the two halves of the
 * fix: with room to spread out, steering the last stretch at each unit's own
 * slot settles a group on its own. A pocket has no room, so the units that
 * cannot get in have to give up instead.
 */
function pocketTarget(sim: Simulation, walls = 3): { x: number; y: number } {
  const { map } = sim.world;
  const start = map.starts[0]!;
  for (let r = 10; r < 40; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        if (!map.isWalkable(x, y)) continue;
        let solid = 0;
        if (!map.isWalkable(x + 1, y)) solid++;
        if (!map.isWalkable(x - 1, y)) solid++;
        if (!map.isWalkable(x, y + 1)) solid++;
        if (!map.isWalkable(x, y - 1)) solid++;
        if (solid >= walls) return { x, y };
      }
    }
  }
  throw new Error(`no tile walled on ${walls} sides`);
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
    const to = target(sim, 3);
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

  it('never puts a formation slot inside rock', () => {
    // The check used to be `tileOfPos(...) < 0`, which asks whether a point is
    // on the map. Solid rock is on the map. A unit handed a slot inside a cliff
    // walked at it, was stopped by the terrain, and shoved at it from then on,
    // because it could never get near enough to count as having arrived.
    const { sim, ids } = army(9);
    const to = pocketTarget(sim);
    march(sim, ids, to);

    const { map, pool } = sim.world;
    for (const id of ids) {
      const i = id & 0xffff;
      const tile = map.tileOfPos(pool.orderX[i]!, pool.orderY[i]!);
      expect(tile).toBeGreaterThanOrEqual(0);
      expect(map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))).toBe(true);
    }
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

/**
 * A unit that cannot reach the exact point it was given has to stop anyway.
 *
 * Arrival is "within half a tile of your point", and in a crowd most units can
 * never satisfy it: their spot is taken, and separation pushes them out of it
 * faster than they can walk back in. Nothing else ended the order, so they
 * shoved at the destination for the rest of the match — measured on open
 * ground, 20 of 24 units in a single group move never came to rest.
 */
describe('coming to rest', () => {
  function marchAndSettle(n: number, clear: number): {
    sim: Simulation;
    ids: number[];
    to: { x: number; y: number };
    restless: number;
  } {
    const { sim, ids } = army(n);
    const to = target(sim, clear);
    march(sim, ids, to);
    for (let t = 0; t < 1200; t++) sim.step([]);

    const pool = sim.world.pool;
    const restless = ids.filter(
      (id) => pool.alive[id & 0xffff] === 1 && pool.order[id & 0xffff] !== Order.None,
    ).length;
    return { sim, ids, to, restless };
  }

  it('brings a large group entirely to rest', () => {
    expect(marchAndSettle(24, 4).restless).toBe(0);
  });

  it('brings a group to rest even ordered into a corner', () => {
    // No room for the formation at all: the slots are mostly rock, so most of
    // the group is aimed at the one tile the player clicked.
    const { sim, ids } = army(12);
    const to = pocketTarget(sim);
    march(sim, ids, to);
    for (let t = 0; t < 1200; t++) sim.step([]);

    const pool = sim.world.pool;
    const restless = ids.filter(
      (id) => pool.alive[id & 0xffff] === 1 && pool.order[id & 0xffff] !== Order.None,
    ).length;
    expect(restless).toBe(0);
  });

  it('stops them at the destination, not short of it', () => {
    // The failure mode of a give-up rule is giving up too early, which looks
    // exactly like the pathfinding being broken. Every unit must finish within
    // formation distance of where the player clicked.
    const { sim, ids, to } = marchAndSettle(24, 4);
    const pool = sim.world.pool;

    let furthest = 0;
    for (const id of ids) {
      const i = id & 0xffff;
      if (pool.alive[i] !== 1) continue;
      furthest = Math.max(
        furthest,
        Math.hypot(toFloat(pool.posX[i]!) - (to.x + 0.5), toFloat(pool.posY[i]!) - (to.y + 0.5)),
      );
    }
    // A 24-strong block at one-tile spacing reaches about 2.8 to its corner.
    expect(furthest).toBeLessThan(4.5);
  });

  it('settles promptly rather than waiting out the give-up timer', () => {
    // Both halves of the fix bring a group to rest, but only one does it well.
    // Steering the last stretch at each unit's own slot fans the group into its
    // formation; without it they funnel into the goal tile, scrum, and stop
    // where they stand once the stall timer fires — tick 261 against 183 for
    // this march, every one of those extra 78 spent shoving.
    const { sim, ids } = army(24);
    const to = target(sim, 4);
    march(sim, ids, to);

    const pool = sim.world.pool;
    let settledAt = -1;
    for (let t = 0; t < 800 && settledAt < 0; t++) {
      sim.step([]);
      const moving = ids.filter(
        (id) => pool.alive[id & 0xffff] === 1 && pool.order[id & 0xffff] !== Order.None,
      ).length;
      if (moving === 0) settledAt = t;
    }
    expect(`settled at tick ${settledAt}`).toBe(
      settledAt >= 0 && settledAt < 220 ? `settled at tick ${settledAt}` : 'settled before tick 220',
    );
  });

  it('does not stop a unit that is still making progress', () => {
    // The stall rule must not fire on a lone unit walking an ordinary route,
    // which is the way a give-up rule usually goes wrong.
    const { sim, ids } = army(1);
    const to = target(sim, 2);
    march(sim, ids, to);

    const i = ids[0]! & 0xffff;
    const pool = sim.world.pool;
    let stoppedEarly = false;
    for (let t = 0; t < 600; t++) {
      sim.step([]);
      if (pool.order[i] === Order.None) {
        const gap = Math.hypot(
          toFloat(pool.posX[i]!) - (to.x + 0.5),
          toFloat(pool.posY[i]!) - (to.y + 0.5),
        );
        stoppedEarly = gap > 1.5;
        break;
      }
    }
    expect(stoppedEarly).toBe(false);
  });
});

/**
 * A destination a unit cannot stand on.
 *
 * Nothing stopped a right-click on a cliff being issued as an order, and the
 * unit then had a destination it could never reach: A* aims at the nearest
 * walkable tile and stops there, but arrival is measured against the ordered
 * point. For a group it was worse — the flow field cannot route to a solid
 * tile, so the whole formation pushed at the rock for the rest of the match,
 * six tiles short, and the give-up rule never applied because it only runs
 * near the destination.
 */
describe('ordered onto solid rock', () => {
  /** Rock with walkable ground somewhere near it — a cliff, not the map edge. */
  function rock(sim: Simulation): { x: number; y: number } {
    const { map } = sim.world;
    const start = map.starts[0]!;
    for (let r = 12; r < 44; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          const x = start.tileX + dx;
          const y = start.tileY + dy;
          if (map.isWalkable(x, y)) continue;
          // Solid for a tile in every direction, so this is not a lone pebble.
          let solid = 0;
          for (let a = -1; a <= 1; a++) {
            for (let b = -1; b <= 1; b++) if (!map.isWalkable(x + a, y + b)) solid++;
          }
          if (solid === 9) return { x, y };
        }
      }
    }
    throw new Error('no rock');
  }

  function marchIntoRock(n: number): { sim: Simulation; ids: number[] } {
    const { sim, ids } = army(n);
    march(sim, ids, rock(sim));
    for (let t = 0; t < 1500; t++) sim.step([]);
    return { sim, ids };
  }

  it('gives every unit a destination it can stand on', () => {
    const { sim, ids } = army(12);
    march(sim, ids, rock(sim));
    const { map, pool } = sim.world;
    for (const id of ids) {
      const i = id & 0xffff;
      const tile = map.tileOfPos(pool.orderX[i]!, pool.orderY[i]!);
      expect(tile).toBeGreaterThanOrEqual(0);
      expect(map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))).toBe(true);
    }
  });

  it('brings a lone unit to rest', () => {
    const { sim, ids } = marchIntoRock(1);
    expect(sim.world.pool.order[ids[0]! & 0xffff]).toBe(Order.None);
  });

  it('brings a whole group to rest', () => {
    const { sim, ids } = marchIntoRock(12);
    const pool = sim.world.pool;
    const restless = ids.filter(
      (id) => pool.alive[id & 0xffff] === 1 && pool.order[id & 0xffff] !== Order.None,
    ).length;
    expect(restless).toBe(0);
  });

  it('walks them to the substituted point rather than abandoning the order', () => {
    // Stopping is not enough on its own — giving up where they stood would also
    // "come to rest". They have to actually get to the ground they were sent to.
    const { sim, ids } = marchIntoRock(12);
    const pool = sim.world.pool;
    let furthest = 0;
    for (const id of ids) {
      const i = id & 0xffff;
      if (pool.alive[i] !== 1) continue;
      furthest = Math.max(
        furthest,
        Math.hypot(
          toFloat(pool.posX[i]!) - toFloat(pool.orderX[i]!),
          toFloat(pool.posY[i]!) - toFloat(pool.orderY[i]!),
        ),
      );
    }
    expect(furthest).toBeLessThan(4.5);
  });
});
