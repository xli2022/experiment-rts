/**
 * Worker repair.
 *
 * Build and repair are deliberately the same order: a worker sent to a structure
 * finishes it if it is unfinished and patches it if it is hurt. These tests pin
 * both halves, plus the release condition — a worker that never lets go of a
 * fully-repaired building would silently drain the economy.
 */

import { describe, expect, it } from 'vitest';
import { CommandType } from '../src/sim/commands.js';
import { defOf, REPAIR_HP_PER_TICK } from '../src/config/rules.js';
import { idIndex } from '../src/sim/entities.js';
import { BuildState, EntityType, NO_ENTITY, Order } from '../src/sim/types.js';
import { Simulation } from '../src/sim/tick.js';

/** A match with one worker parked next to a finished building of its own. */
function setup() {
  const sim = new Simulation(0x51ce7a11);
  const world = sim.world;
  const pool = world.pool;

  let worker = -1;
  let post = -1;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.owner[i] !== 0) continue;
    if (worker < 0 && pool.type[i] === EntityType.Worker) worker = i;
    if (post < 0 && pool.type[i] === EntityType.CommandPost) post = i;
  }
  return { sim, world, pool, worker, post };
}

describe('repair', () => {
  it('restores health to a damaged building', () => {
    const { sim, world, pool, worker, post } = setup();
    expect(worker).toBeGreaterThanOrEqual(0);
    expect(post).toBeGreaterThanOrEqual(0);

    const def = defOf(EntityType.CommandPost);
    pool.hp[post] = 200;
    // Put the worker on the doorstep so it is in reach immediately.
    pool.posX[worker] = pool.posX[post]!;
    pool.posY[worker] = pool.posY[post]!;

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.CommandPost,
        tileX: pool.tileX[post]!,
        tileY: pool.tileY[post]!,
      },
    ]);

    expect(pool.order[worker]).toBe(Order.Build);
    const before = pool.hp[post]!;
    for (let t = 0; t < 10; t++) sim.step([]);
    const after = pool.hp[post]!;

    expect(after).toBeGreaterThan(before);
    expect(after).toBeLessThanOrEqual(def.maxHp);
    // Roughly the advertised rate, allowing for the tick the order landed on.
    expect(after - before).toBeGreaterThanOrEqual(REPAIR_HP_PER_TICK * 8);
    void world;
  });

  it('stops and frees the worker once the building is whole', () => {
    const { sim, pool, worker, post } = setup();
    const def = defOf(EntityType.CommandPost);
    pool.hp[post] = def.maxHp - REPAIR_HP_PER_TICK * 2;
    pool.posX[worker] = pool.posX[post]!;
    pool.posY[worker] = pool.posY[post]!;

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.CommandPost,
        tileX: pool.tileX[post]!,
        tileY: pool.tileY[post]!,
      },
    ]);
    for (let t = 0; t < 20; t++) sim.step([]);

    expect(pool.hp[post]).toBe(def.maxHp);
    // Released, not left standing there forever.
    expect(pool.order[worker]).toBe(Order.None);
    expect(pool.orderTarget[worker]).toBe(NO_ENTITY);
  });

  it('never overheals', () => {
    const { sim, pool, worker, post } = setup();
    const def = defOf(EntityType.CommandPost);
    pool.hp[post] = def.maxHp - 1;
    pool.posX[worker] = pool.posX[post]!;
    pool.posY[worker] = pool.posY[post]!;

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.CommandPost,
        tileX: pool.tileX[post]!,
        tileY: pool.tileY[post]!,
      },
    ]);
    for (let t = 0; t < 30; t++) sim.step([]);
    expect(pool.hp[post]).toBe(def.maxHp);
  });

  it('ignores an order onto an undamaged building', () => {
    const { sim, pool, worker, post } = setup();
    pool.posX[worker] = pool.posX[post]!;
    pool.posY[worker] = pool.posY[post]!;
    const originalOrder = pool.order[worker];

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.CommandPost,
        tileX: pool.tileX[post]!,
        tileY: pool.tileY[post]!,
      },
    ]);

    // Nothing to do there: the worker should not be captured by a no-op job. It
    // also must not have started a second Command Post on top of the first.
    let posts = 0;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.CommandPost) posts++;
    }
    expect(posts).toBe(2); // one per player
    expect(pool.order[worker]).toBe(originalOrder);
  });

  it('still finishes an unfinished building', () => {
    const { sim, pool, worker } = setup();
    const barracksDef = defOf(EntityType.Barracks);
    sim.world.players[0]!.minerals = 1000;

    // Place a site somewhere clear near the worker.
    const tileX = sim.world.map.starts[0]!.tileX + 6;
    const tileY = sim.world.map.starts[0]!.tileY + 6;
    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.Barracks,
        tileX,
        tileY,
      },
    ]);

    let site = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.Barracks) site = i;
    }
    expect(site).toBeGreaterThanOrEqual(0);

    for (let t = 0; t < barracksDef.buildTicks + 400; t++) {
      sim.step([]);
      if (pool.buildState[site] === BuildState.Complete) break;
    }
    expect(pool.buildState[site]).toBe(BuildState.Complete);
    expect(pool.hp[site]).toBe(barracksDef.maxHp);
    void idIndex;
  });
});
