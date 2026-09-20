import { describe, expect, it } from 'vitest';
import { CommandType } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { ENTITY_TYPE_COUNT, EntityType } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

describe('build command validation', () => {
  it.each([
    -1,
    ENTITY_TYPE_COUNT,
    999,
    3.5,
    Number.NaN,
    EntityType.Worker,
    EntityType.MineralPatch,
  ])('ignores an invalid building type %s without changing the simulation', (building) => {
    const world = new World(1);
    const worker = world.pool.spawn(EntityType.Worker, 0, fromInt(20), fromInt(20));
    world.player(0).minerals = 5000;
    const before = world.checksum();

    expect(() =>
      executeCommand(world, {
        type: CommandType.Build,
        player: 0,
        worker,
        building: building as EntityType,
        tileX: 24,
        tileY: 24,
      }),
    ).not.toThrow();
    expect(world.checksum()).toBe(before);
  });
});
