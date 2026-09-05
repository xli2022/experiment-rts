/**
 * What a player remembers, and — just as much — what they cannot know.
 *
 * A building seen once stays known until its ground is seen without it; a
 * unit seen once is forgotten when its ground is seen empty, or after a while
 * either way. And the memory never reads the state of anything it cannot
 * currently see, which is the contract that keeps it from being a fog leak.
 */

import { describe, expect, it } from 'vitest';
import { EntityMemory } from '../src/ai/neural/memory.js';
import { UNIT_MEMORY_TICKS } from '../src/ai/neural/spec.js';
import { fromInt } from '../src/sim/fixed.js';
import { duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, NO_ENTITY } from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { Visibility } from '../src/vision/visibility.js';

/** A duel world with the starting workers moved far out of the way. */
function quietWorld(): { sim: Simulation; world: World } {
  const sim = new Simulation(duelMatch(0x51ce7a11, { botPlayers: [] }));
  const world = sim.world;
  for (let i = 0; i < world.pool.count; i++) {
    if (world.pool.alive[i] === 1 && world.pool.type[i] === EntityType.Worker) {
      world.pool.posX[i] = fromInt(world.pool.owner[i] === 0 ? 4 : 124);
      world.pool.posY[i] = fromInt(world.pool.owner[i] === 0 ? 4 : 124);
    }
  }
  return { sim, world };
}

function spawnUnit(world: World, type: EntityType, owner: number, x: number, y: number): number {
  const id = world.pool.spawn(type, owner, fromInt(x) + 32768, fromInt(y) + 32768);
  expect(id).not.toBe(NO_ENTITY);
  return id;
}

function moveTo(world: World, id: number, x: number, y: number): void {
  const i = id & 0xffff;
  world.pool.posX[i] = fromInt(x) + 32768;
  world.pool.posY[i] = fromInt(y) + 32768;
}

/** Advance the clock without running the simulation, then look again. */
function look(world: World, vis: Visibility, mem: EntityMemory, ticks = 1): void {
  for (let t = 0; t < ticks; t++) {
    world.tick++;
    vis.update(world, 0);
    mem.update(world, vis);
  }
}

describe('entity memory', () => {
  it('remembers a unit it saw and forgets it after a while', () => {
    const { world } = quietWorld();
    const vis = new Visibility(world.map);
    const mem = new EntityMemory(0);
    const scout = spawnUnit(world, EntityType.Burstbot, 0, 55, 60);
    const enemy = spawnUnit(world, EntityType.Burstbot, 1, 60, 60);

    look(world, vis, mem);
    expect(mem.get(enemy)?.lastSeen).toBe(world.tick);
    expect(mem.get(enemy)?.tileX).toBe(60);

    moveTo(world, scout, 20, 20);
    look(world, vis, mem, UNIT_MEMORY_TICKS);
    expect(mem.get(enemy)).toBeDefined();
    look(world, vis, mem, 1);
    expect(mem.get(enemy)).toBeUndefined();
    // The mineral patches the scout passed stay known; only the unit is gone.
    expect(mem.entries.filter((e) => e.type !== EntityType.MineralPatch)).toEqual([]);
  });

  it('forgets a unit the moment its ground is seen without it', () => {
    const { world } = quietWorld();
    const vis = new Visibility(world.map);
    const mem = new EntityMemory(0);
    const scout = spawnUnit(world, EntityType.Burstbot, 0, 55, 60);
    const enemy = spawnUnit(world, EntityType.Burstbot, 1, 60, 60);
    look(world, vis, mem);
    expect(mem.get(enemy)).toBeDefined();

    moveTo(world, scout, 20, 20);
    look(world, vis, mem, 10);
    moveTo(world, enemy, 100, 100); // gone, unseen
    look(world, vis, mem, 10);
    expect(mem.get(enemy)).toBeDefined(); // still believed to be there

    moveTo(world, scout, 55, 60);
    look(world, vis, mem);
    expect(mem.get(enemy)).toBeUndefined();
  });

  it('keeps a building until its ground is seen without it', () => {
    const { sim, world } = quietWorld();
    const vis = new Visibility(world.map);
    const mem = new EntityMemory(0);
    const scout = spawnUnit(world, EntityType.Burstbot, 0, 55, 60);
    const depot = world.placeBuilding(EntityType.Depot, 1, 60, 60);
    expect(depot).not.toBe(NO_ENTITY);
    look(world, vis, mem);
    expect(mem.get(depot)?.tileX).toBe(60);
    expect(mem.get(depot)?.tileY).toBe(60);

    moveTo(world, scout, 20, 20);
    look(world, vis, mem, UNIT_MEMORY_TICKS * 4);
    expect(mem.get(depot)).toBeDefined();

    // Destroyed while nobody was looking: still remembered...
    world.pool.destroy(depot);
    expect(world.pool.isAlive(depot)).toBe(false);
    void sim;
    look(world, vis, mem, 10);
    expect(mem.get(depot)).toBeDefined();

    // ...until the ground is seen again.
    moveTo(world, scout, 55, 60);
    look(world, vis, mem);
    expect(mem.get(depot)).toBeUndefined();
  });

  it('keeps a mineral patch once discovered', () => {
    const { world } = quietWorld();
    const vis = new Visibility(world.map);
    const mem = new EntityMemory(0);
    let patch = -1;
    for (let i = 0; i < world.pool.count; i++) {
      if (world.pool.type[i] === EntityType.MineralPatch) {
        patch = i;
        break;
      }
    }
    expect(patch).toBeGreaterThanOrEqual(0);
    const scout = spawnUnit(
      world,
      EntityType.Burstbot,
      0,
      world.pool.tileX[patch]! + 3,
      world.pool.tileY[patch]! + 3,
    );
    look(world, vis, mem);
    const id = world.pool.idAt(patch);
    expect(mem.get(id)?.resourceAmount).toBeGreaterThan(0);
    moveTo(world, scout, 20, 20);
    look(world, vis, mem, UNIT_MEMORY_TICKS * 4);
    expect(mem.get(id)).toBeDefined();
  });

  it('never reads the state of an entity it cannot see', () => {
    const { world } = quietWorld();
    const vis = new Visibility(world.map);
    const mem = new EntityMemory(0);
    const scout = spawnUnit(world, EntityType.Burstbot, 0, 55, 60);
    spawnUnit(world, EntityType.Burstbot, 1, 60, 60);
    look(world, vis, mem);
    moveTo(world, scout, 20, 20);

    // Every liveness query the memory makes goes through `isAlive`; a memory
    // that peeked at what it remembers would call it with an unseen handle.
    let peeks = 0;
    const pool = new Proxy(world.pool, {
      get(target, key, receiver) {
        if (key === 'isAlive') {
          return (id: number) => {
            peeks++;
            return target.isAlive(id);
          };
        }
        return Reflect.get(target, key, receiver);
      },
    });
    const spied = Object.create(world, { pool: { value: pool } }) as World;
    for (let t = 0; t < 50; t++) {
      spied.tick++;
      vis.update(spied, 0);
      mem.update(spied, vis);
    }
    expect(mem.entries.filter((e) => e.type !== EntityType.MineralPatch).length).toBe(1);
    expect(peeks).toBe(0);
  });
});
