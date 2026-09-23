import { describe, expect, it } from 'vitest';
import { botThink } from '../src/ai/bot.js';
import { HeadlessMatch } from '../src/ai/headless.js';
import { ScriptedAgent } from '../src/ai/scripted.js';
import { CommandType } from '../src/sim/commands.js';
import { fromFloat, toFloat } from '../src/sim/fixed.js';
import { matchConfig } from '../src/sim/match.js';
import { executeCommand, standableTarget } from '../src/sim/systems/orders.js';
import { BuildState, EntityType, MapLayout, Order, Tile } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

const SEED = 0x51ce7a11;

function add(world: World, type: EntityType, player: number, x: number, y: number): number {
  const i = world.pool.spawn(type, player, fromFloat(x), fromFloat(y)) & 0xffff;
  world.pool.buildState[i] = BuildState.Complete;
  world.pool.tileX[i] = Math.floor(x);
  world.pool.tileY[i] = Math.floor(y);
  return i;
}

function army(world: World, type: EntityType, x: number, y: number): number[] {
  return Array.from({ length: 6 }, () => add(world, type, 0, x, y));
}

function publicSites(world: World): { x: number; y: number }[] {
  return [
    world.map.starts[1]!,
    ...world.map.expansions
      .slice()
      .sort((a, b) => world.map.index(a.tileX, a.tileY) - world.map.index(b.tileX, b.tileY)),
  ].map((site) => ({ x: fromFloat(site.tileX + 0.5), y: fromFloat(site.tileY + 0.5) }));
}

function attackMoves(world: World) {
  return botThink(world, 0).filter((command) => command.type === CommandType.AttackMove);
}

describe('scripted bot destination progress', () => {
  it.each([EntityType.Burstbot, EntityType.Beamdrone])(
    'scouts three distinct sites before revisiting one with unit type %i',
    (type) => {
      const script = new ScriptedAgent();
      const destinations: string[] = [];
      const match = new HeadlessMatch(SEED, [
        [
          0,
          {
            act(world, player) {
              const commands = script.act(world, player);
              for (const command of commands) {
                if (command.type !== CommandType.AttackMove) continue;
                const target = `${command.x},${command.y}`;
                if (destinations.at(-1) !== target) destinations.push(target);
              }
              return commands;
            },
          },
        ],
      ]);
      try {
        const world = match.world;
        // An enemy whose original base was cleared now owns a remote depot.
        // Keep the normal simulation, command delay, and six-unit push threshold.
        for (let i = 0; i < world.pool.count; i++) {
          if (world.pool.alive[i] === 1) world.pool.destroy(world.pool.idAt(i));
        }
        world.map.occupied.fill(0);
        for (const player of world.players) {
          player.minerals = 100;
          player.supplyUsed = 0;
          player.supplyMax = 200;
        }
        add(world, EntityType.Depot, 0, 20.5, 20.5);
        add(world, EntityType.Depot, 1, 23.5, 104.5);
        add(world, EntityType.Worker, 1, 23.5, 106.5);
        const start = world.map.starts[1]!;
        for (let k = 0; k < 6; k++) {
          add(world, type, 0, start.tileX + 0.5 + (k % 3), start.tileY + 0.5 + Math.floor(k / 3));
        }
        while (world.tick < 4500 && destinations.length < 3 && !world.matchOver) match.step();
        const publicTargets = new Set(publicSites(world).map(({ x, y }) => `${x},${y}`));
        expect(destinations).toHaveLength(3);
        expect(destinations.every((destination) => publicTargets.has(destination))).toBe(true);
        expect(new Set(destinations).size).toBe(3);
        expect(match.driver.statsFor(0)?.rejected).toBe(0);
      } finally {
        match.dispose();
      }
    },
  );

  it('continues after a completed scout order instead of returning to the hidden start', () => {
    const world = new World(SEED);
    world.player(0).minerals = 100;
    const sites = publicSites(world);
    const completed = sites[1]!;
    const units = army(world, EntityType.Burstbot, toFloat(completed.x), toFloat(completed.y));
    for (const i of units) {
      world.pool.order[i] = Order.None;
      world.pool.orderX[i] = completed.x;
      world.pool.orderY[i] = completed.y;
    }
    world.tick = 60;
    expect(attackMoves(world)).toMatchObject([{ ...sites[2] }]);
  });

  it('does not let one old order retained during wind-up reverse the main army', () => {
    const world = new World(SEED);
    world.player(0).minerals = 100;
    const sites = publicSites(world);
    const current = sites[2]!;
    const units = army(world, EntityType.Burstbot, toFloat(current.x), toFloat(current.y));
    for (const i of units) {
      world.pool.order[i] = Order.AttackMove;
      world.pool.orderX[i] = current.x;
      world.pool.orderY[i] = current.y;
    }
    const oldest = units[0]!;
    world.pool.orderX[oldest] = sites[1]!.x;
    world.pool.orderY[oldest] = sites[1]!.y;
    world.pool.attackWindup[oldest] = 2;
    world.tick = 60;
    expect(attackMoves(world)).toMatchObject([
      {
        ...sites[3],
        units: units.slice(1).map((i) => world.pool.idAt(i)),
      },
    ]);
  });

  it('advances a completed majority even when one unit still has an old active order', () => {
    const world = new World(SEED);
    world.player(0).minerals = 100;
    const sites = publicSites(world);
    const completed = sites[2]!;
    const units = army(world, EntityType.Burstbot, toFloat(completed.x), toFloat(completed.y));
    for (const i of units) {
      world.pool.order[i] = Order.None;
      world.pool.orderX[i] = completed.x;
      world.pool.orderY[i] = completed.y;
    }
    const laggard = units[0]!;
    world.pool.order[laggard] = Order.AttackMove;
    world.pool.orderX[laggard] = sites[1]!.x;
    world.pool.orderY[laggard] = sites[1]!.y;
    world.tick = 60;
    expect(attackMoves(world)).toMatchObject([
      {
        ...sites[3],
        units: units.map((i) => world.pool.idAt(i)),
      },
    ]);
  });

  it('does not recall a melee army for aircraft or unarmed support units', () => {
    for (const threat of [EntityType.Beamdrone, EntityType.Fixomatic]) {
      const world = new World(SEED);
      world.player(0).minerals = 100;
      add(world, EntityType.Depot, 0, 20.5, 20.5);
      army(world, EntityType.Slicebot, 40.5, 20.5);
      const before = attackMoves(world);
      expect(before).toHaveLength(1);
      add(world, threat, 1, 38.5, 20.5);
      expect(attackMoves(world)).toEqual(before);
    }
  });

  it('approaches an observed threat and keeps its last position when visibility is lost', () => {
    const world = new World(SEED);
    world.map.tiles.fill(Tile.Ground);
    world.player(0).minerals = 100;
    add(world, EntityType.Depot, 0, 20.5, 20.5);
    const units = army(world, EntityType.Burstbot, 31.5, 20.5);
    const hostile = add(world, EntityType.Slicebot, 1, 38.5, 20.5);
    const observed = { x: fromFloat(38.5), y: fromFloat(20.5) };
    const first = attackMoves(world);
    expect(first).toMatchObject([observed]);
    executeCommand(world, first[0]!);

    // The spotting part of the army pulls back and a fresh unit joins it.
    // A different visible building must not cancel inspection of this threat.
    for (const i of units) world.pool.posX[i] = fromFloat(28.5);
    world.pool.posX[hostile] = fromFloat(80.5);
    world.pool.posY[hostile] = fromFloat(80.5);
    const reinforcement = add(world, EntityType.Burstbot, 0, 28.5, 20.5);
    add(world, EntityType.Depot, 1, 30.5, 26.5);
    world.tick = 60;
    const continuing = attackMoves(world);
    expect(continuing).toMatchObject([{ ...observed, units: [world.pool.idAt(reinforcement)] }]);

    // Hidden movement cannot alter the last observed destination.
    world.pool.posX[hostile] = fromFloat(95.5);
    expect(attackMoves(world)).toEqual(continuing);
  });

  it('keeps ground units progressing when a visible flyer is over unreachable cliffs', () => {
    const world = new World(matchConfig(MapLayout.Quarters, SEED));
    world.map.tiles.fill(Tile.Ground);
    world.player(0).minerals = 100;
    for (let y = 12; y <= 28; y++) {
      for (let x = 43; x <= 59; x++) world.map.tiles[world.map.index(x, y)] = Tile.Cliff;
    }
    add(world, EntityType.Depot, 0, 40.5, 20.5);
    const units = army(world, EntityType.Burstbot, 30.5, 20.5);
    // An ally reveals the flyer without giving this player an aerial defender.
    add(world, EntityType.Beamdrone, 1, 51.5, 21.5);
    add(world, EntityType.Beamdrone, 2, 51.5, 20.5);
    const target = { x: fromFloat(51.5), y: fromFloat(20.5) };
    expect(standableTarget(world, 0, target.x, target.y, { x: 0, y: 0 })).toBe(false);
    const orders = attackMoves(world);
    expect(orders.length).toBeGreaterThan(0);
    expect(orders[0]).not.toMatchObject(target);
    for (const command of orders) executeCommand(world, command);
    expect(units.every((i) => world.pool.order[i] === Order.AttackMove)).toBe(true);

    // Once an aircraft is available, it can answer the same visible threat.
    const aircraft = add(world, EntityType.Beamdrone, 0, 40.5, 20.5);
    world.tick = 20;
    const response = attackMoves(world);
    expect(response).toMatchObject([target]);
    for (const command of response) executeCommand(world, command);
    expect(world.pool.order[aircraft]).toBe(Order.AttackMove);
    expect(
      Math.hypot(
        toFloat(world.pool.orderX[aircraft]! - target.x),
        toFloat(world.pool.orderY[aircraft]! - target.y),
      ),
    ).toBeLessThanOrEqual(2);
  });
});
