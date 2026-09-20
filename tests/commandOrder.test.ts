import { describe, expect, it } from 'vitest';
import { CommandType, sortCommands, type Command } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { FIX_HALF, fromInt } from '../src/sim/fixed.js';
import { coopMatch } from '../src/sim/match.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { EntityType, Tile } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

describe('commands in the same tick', () => {
  it.each([CommandType.Move, CommandType.AttackMove] as const)(
    'keeps mirrored formation destinations when other players also place buildings (%i)',
    (type) => {
      const world = new World(coopMatch(1, { botPlayers: [] }));
      const { pool, map } = world;
      map.tiles.fill(Tile.Ground);
      map.occupied.fill(0);
      map.sealTerrain();
      const width = fromInt(map.width);
      const height = fromInt(map.height);
      const units = [0, 2].map((player) =>
        Array.from({ length: 10 }, () =>
          pool.spawn(EntityType.Burstbot, player, fromInt(30), fromInt(30)),
        ),
      );
      const worker1 = pool.spawn(EntityType.Worker, 1, fromInt(120), fromInt(60));
      const worker3 = pool.spawn(EntityType.Worker, 3, width - fromInt(120), height - fromInt(60));
      world.player(1).minerals = 1000;
      world.player(3).minerals = 1000;
      const x = fromInt(40) + FIX_HALF;
      const y = fromInt(100) + FIX_HALF;
      // Slot eight is (+1,-1) from this target. Each allied builder will put a
      // Depot on the opposite army's corresponding slot during the same tick.
      // Player-first execution used to place player 1's Depot between the two
      // moves, making only player 2 fall back to the formation's centre.
      const commands: Command[] = [
        { type, player: 0, units: units[0]!, x, y },
        {
          type: CommandType.Build,
          player: 1,
          worker: worker1,
          building: EntityType.Depot,
          tileX: map.width - 43,
          tileY: map.height - 100,
        },
        { type, player: 2, units: units[1]!, x: width - x, y: height - y },
        {
          type: CommandType.Build,
          player: 3,
          worker: worker3,
          building: EntityType.Depot,
          tileX: 41,
          tileY: 98,
        },
      ];
      for (const command of sortCommands(commands.reverse())) executeCommand(world, command);
      expect(world.player(1).minerals).toBe(900);
      expect(world.player(3).minerals).toBe(900);
      for (let k = 0; k < units[0]!.length; k++) {
        const a = idIndex(units[0]![k]!);
        const b = idIndex(units[1]![k]!);
        expect(pool.orderX[a]! + pool.orderX[b]!, `slot ${k} x`).toBe(width);
        expect(pool.orderY[a]! + pool.orderY[b]!, `slot ${k} y`).toBe(height);
      }
    },
  );
});
