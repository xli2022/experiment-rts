import { describe, expect, it } from 'vitest';
import { chunkCommands } from '../src/ai/agent.js';
import { CommandType, type AttackMoveCommand, type MoveCommand } from '../src/sim/commands.js';
import { ENTITY_CAPACITY, idIndex } from '../src/sim/entities.js';
import { fromFloat, fromInt, toFloat } from '../src/sim/fixed.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order, Tile } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

type MovementCommand = MoveCommand | AttackMoveCommand;

function openWorld(world = new World(1)): World {
  world.map.tiles.fill(Tile.Ground);
  world.map.occupied.fill(0);
  world.map.sealTerrain();
  return world;
}

describe.each([CommandType.Move, CommandType.AttackMove] as const)(
  'chunked movement %i',
  (type) => {
    it.each([48, 130])('keeps %i distinct mirrored destinations for a whole army', (count) => {
      for (const entity of [EntityType.Burstbot, EntityType.Beamdrone]) {
        const world = openWorld();
        const pool = world.pool;
        const armies = [0, 1].map((player) =>
          Array.from({ length: count }, () =>
            pool.spawn(entity, player, fromInt(player === 0 ? 30 : 98), fromInt(40)),
          ),
        );
        for (const player of [0, 1]) {
          const command: MovementCommand = {
            type,
            player,
            units: armies[player]!,
            x: fromInt(64),
            y: fromInt(64),
          };
          for (const part of chunkCommands([command])) executeCommand(world, part);
          const destinations = command.units.map((id) => {
            const i = idIndex(id);
            expect(pool.order[i]).toBe(type === CommandType.Move ? Order.Move : Order.AttackMove);
            return `${pool.orderX[i]},${pool.orderY[i]}`;
          });
          expect(new Set(destinations).size).toBe(count);
        }
        for (let unit = 0; unit < count; unit++) {
          const a = idIndex(armies[0]![unit]!);
          const b = idIndex(armies[1]![unit]!);
          expect(pool.orderX[a]! + pool.orderX[b]!).toBe(fromInt(world.map.width));
          expect(pool.orderY[a]! + pool.orderY[b]!).toBe(fromInt(world.map.height));
        }
      }
    });

    it('keeps even a one-unit final chunk on the shared flow field', () => {
      const world = openWorld();
      const pool = world.pool;
      const units = Array.from({ length: 49 }, () =>
        pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(40)),
      );
      for (const command of chunkCommands([
        { type, player: 0, units, x: fromInt(64), y: fromInt(64) },
      ])) {
        executeCommand(world, command);
      }
      const goals = units.map((id) => pool.flowGoal[idIndex(id)]!);
      expect(new Set(goals)).toEqual(new Set([world.map.index(64, 64)]));
      expect(units.every((id) => pool.navGoal[idIndex(id)] === goals[0])).toBe(true);
      expect(world.pathQueue).toHaveLength(0);
    });

    it('reaches a far-side slot beyond the old three-tile formation approach', () => {
      const sim = new Simulation(1);
      const world = openWorld(sim.world);
      const unit = world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(64));
      const i = idIndex(unit);
      executeCommand(world, {
        type,
        player: 0,
        units: [unit],
        x: fromInt(64),
        y: fromInt(64),
        formationOffset: 120,
      });
      // Slot120 is five tiles right and five up: a shared field aimed at64,64
      // used to pull this unit back to the centre indefinitely.
      expect(world.pool.orderX[i]).toBe(fromInt(69));
      expect(world.pool.orderY[i]).toBe(fromInt(59));
      for (let tick = 0; tick < 500; tick++) sim.step([]);
      expect(world.pool.order[i]).toBe(Order.None);
      expect(
        Math.hypot(world.pool.posX[i]! - fromInt(69), world.pool.posY[i]! - fromInt(59)),
      ).toBeLessThanOrEqual(fromInt(1) / 2);
    });

    it('settles a 130-unit ground army near its full formation', () => {
      const sim = new Simulation(1);
      const world = openWorld(sim.world);
      const pool = world.pool;
      const units = Array.from({ length: 130 }, (_, j) =>
        pool.spawn(
          EntityType.Burstbot,
          0,
          fromFloat(35 + ((j % 13) - 6) * 1.1),
          fromFloat(64 + (Math.floor(j / 13) - 5) * 1.1),
        ),
      );
      for (const command of chunkCommands([
        { type, player: 0, units, x: fromInt(80), y: fromInt(64) },
      ])) {
        executeCommand(world, command);
      }
      for (let tick = 0; tick < 800; tick++) sim.step([]);
      const errors = units.map((unit) => {
        const i = idIndex(unit);
        expect(pool.order[i]).toBe(Order.None);
        expect(Math.hypot(toFloat(pool.posX[i]!) - 80, toFloat(pool.posY[i]!) - 64)).toBeLessThan(
          9,
        );
        return Math.hypot(
          toFloat(pool.posX[i]! - pool.orderX[i]!),
          toFloat(pool.posY[i]! - pool.orderY[i]!),
        );
      });
      // Separation can displace crowded inner slots. The army must settle in
      // the destination area, while outer slots actually reach the perimeter
      // instead of being repeatedly pulled back into the shared goal's scrum.
      expect(errors.reduce((sum, error) => sum + error, 0) / errors.length).toBeLessThan(1.5);
      const outer = errors.slice(96);
      expect(outer.reduce((sum, error) => sum + error, 0) / outer.length).toBeLessThan(1);
      expect(Math.max(...outer)).toBeLessThan(3);
    });

    it('keeps a wide ground formation on the reachable side of a cliff', () => {
      const world = openWorld();
      for (let y = 58; y <= 70; y++) world.map.tiles[world.map.index(66, y)] = Tile.Cliff;
      const unit = world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(64));
      const i = idIndex(unit);
      // The slot itself is ground, but the shared goal cannot directly reach it.
      expect(world.map.isWalkable(69, 59)).toBe(true);
      executeCommand(world, {
        type,
        player: 0,
        units: [unit],
        x: fromInt(64),
        y: fromInt(64),
        formationOffset: 120,
      });
      expect(world.pool.orderX[i]).toBe(fromInt(64));
      expect(world.pool.orderY[i]).toBe(fromInt(64));
      expect(world.pool.flowGoal[i]).toBe(world.map.index(64, 64));
      expect(world.pathQueue).toHaveLength(0);
    });

    it('preserves inherited formation offsets and is stable when chunked again', () => {
      const units = Array.from({ length: 50 }, (_, i) => i + 1);
      const source: MovementCommand = {
        type,
        player: 0,
        units,
        x: fromInt(64),
        y: fromInt(64),
        formationOffset: 17,
      };
      const chunks = chunkCommands([source]) as MovementCommand[];
      expect(chunks.map((command) => command.formationOffset)).toEqual([17, 41, 65]);
      expect(chunks.flatMap((command) => command.units)).toEqual(units);
      expect(chunkCommands(chunks)).toEqual(chunks);
      expect(source.formationOffset).toBe(17);
      expect(source.units).toHaveLength(50);
    });

    it.each([-1, 0.5, NaN, Infinity, ENTITY_CAPACITY, ENTITY_CAPACITY - 1, null, '24'])(
      'rejects invalid formation offset %s before changing unit orders',
      (offset) => {
        const world = openWorld();
        const units = Array.from({ length: 2 }, () =>
          world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(40)),
        );
        const before = world.checksum();
        executeCommand(world, {
          type,
          player: 0,
          units,
          x: fromInt(64),
          y: fromInt(64),
          formationOffset: offset as number,
        });
        expect(world.checksum()).toBe(before);
      },
    );

    it('leaves ordinary unsplit movement unchanged when the offset is omitted', () => {
      const a = openWorld();
      const b = openWorld();
      const units = Array.from({ length: 3 }, () => {
        const id = a.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(40));
        expect(b.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(40))).toBe(id);
        return id;
      });
      const command: MovementCommand = { type, player: 0, units, x: fromInt(64), y: fromInt(64) };
      executeCommand(a, command);
      executeCommand(b, { ...command, formationOffset: 0 });
      expect(a.checksum()).toBe(b.checksum());
    });
  },
);
