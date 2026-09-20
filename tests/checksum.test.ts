import { describe, expect, it } from 'vitest';
import { fromInt } from '../src/sim/fixed.js';
import { EntityType } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

describe('simulation checksum coverage', () => {
  it('detects an entity with the wrong owner before either player issues an order', () => {
    const world = new World(1);
    const unit = world.pool.spawn(EntityType.Worker, 0, fromInt(20), fromInt(20)) & 0xffff;
    const before = world.checksum();
    world.pool.owner[unit] = 1;
    expect(world.checksum()).not.toBe(before);
  });

  it('detects a queued path request that one peer will skip', () => {
    const world = new World(1);
    const unit = world.pool.spawn(EntityType.Worker, 0, fromInt(20), fromInt(20)) & 0xffff;
    world.pathQueue.push(unit);
    world.pool.pathPending[unit] = 1;
    const before = world.checksum();
    world.pool.pathPending[unit] = 0;
    expect(world.checksum()).not.toBe(before);
  });

  it('detects a different remaining route while ignoring unused path storage', () => {
    const world = new World(1);
    const unit = world.pool.spawn(EntityType.Worker, 0, fromInt(20), fromInt(20)) & 0xffff;
    world.pool.setPath(unit, [world.map.index(21, 20)]);
    const before = world.checksum();
    world.pool.pathNodes[0] = world.map.index(20, 21);
    expect(world.checksum()).not.toBe(before);

    const changed = world.checksum();
    world.pool.pathNodes[1] = world.map.index(22, 20);
    expect(world.checksum()).toBe(changed);
  });
});
