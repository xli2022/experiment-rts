import { afterEach, describe, expect, it, vi } from 'vitest';
import { CommandType } from '../src/sim/commands.js';
import { fromFloat, fromInt, vecDist } from '../src/sim/fixed.js';
import { AStar } from '../src/sim/pathing/astar.js';
import { ConstructionPaths } from '../src/sim/pathing/construction.js';
import { FlowFieldCache } from '../src/sim/pathing/flowfield.js';
import { lineOfSightClear } from '../src/sim/pathing/los.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { movementSystem } from '../src/sim/systems/movement.js';
import { BuildState, EntityType, Order, Tile } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

afterEach(() => vi.restoreAllMocks());

function setup(owner = 0) {
  const world = new World(1);
  world.map.tiles.fill(Tile.Ground);
  world.map.occupied.fill(0);
  world.map.sealTerrain();
  const point = (value: number) => fromFloat(owner === 0 ? value : world.map.width - value);
  const worker = world.pool.spawn(EntityType.Worker, owner, point(10.5), point(4.5));
  return {
    world,
    worker,
    index: worker & 0xffff,
    owner,
    point,
    astar: new AStar(world.map),
    fields: new FlowFieldCache(world.map.width * world.map.height),
    construction: new ConstructionPaths(),
  };
}
type Context = ReturnType<typeof setup>;

function move(c: Context, type: CommandType.Move | CommandType.AttackMove = CommandType.Move) {
  executeCommand(c.world, {
    type,
    player: c.owner,
    units: [c.worker],
    x: c.point(10.5),
    y: c.point(22.5),
  });
}

function foundation(c: Context, tx = 9, ty = 9) {
  const x = c.owner === 0 ? tx : c.world.map.width - tx - 4;
  const y = c.owner === 0 ? ty : c.world.map.height - ty - 4;
  const id = c.world.placeBuilding(EntityType.CommandPost, c.owner, x, y);
  c.world.pool.buildState[id & 0xffff] = BuildState.Site;
  return id;
}

function step(c: Context) {
  const { world, index } = c;
  const x = world.pool.posX[index]!,
    y = world.pool.posY[index]!;
  world.grid.rebuild(world.pool);
  movementSystem(world, c.astar, c.fields, c.construction);
  world.tick++;
  // Repair must route around the footprint, not rely on terrain ejection.
  expect(
    lineOfSightClear(
      world.map,
      x,
      y,
      world.pool.posX[index]!,
      world.pool.posY[index]!,
      world.flipOf(c.owner),
    ),
  ).toBe(true);
}

function advance(c: Context, ticks: number) {
  for (let i = 0; i < ticks; i++) step(c);
}

function expectArrival(c: Context) {
  const pool = c.world.pool;
  expect(pool.order[c.index]).toBe(Order.None);
  expect(
    vecDist(pool.posX[c.index]!, pool.posY[c.index]!, c.point(10.5), c.point(22.5)),
  ).toBeLessThanOrEqual(fromFloat(0.5));
}

describe('private path repair after new occupancy', () => {
  it.each([CommandType.Move, CommandType.AttackMove] as const)(
    'repairs a long smoothed route for order %i',
    (type) => {
      const c = setup();
      const searches = vi.spyOn(c.astar, 'find');
      move(c, type);
      advance(c, 10);
      expect(c.world.pool.pathLen[c.index]).toBe(1);
      foundation(c);
      // The destination remains walkable; it is the middle of the long leg that changed.
      expect(c.world.map.isWalkable(10, 22)).toBe(true);
      advance(c, 590);
      expectArrival(c);
      expect(searches.mock.calls.length).toBeGreaterThan(1);
      expect(searches.mock.calls.length).toBeLessThanOrEqual(4);
    },
  );

  it('does not reroute or change movement for an off-route occupancy change', () => {
    const control = setup(),
      changed = setup();
    const searches = vi.spyOn(changed.astar, 'find');
    move(control);
    move(changed);
    for (let tick = 0; tick < 300; tick++) {
      if (tick === 10) foundation(changed, 30, 30);
      step(control);
      step(changed);
      for (const field of [
        'posX',
        'posY',
        'order',
        'pathLen',
        'pathCursor',
        'pathPending',
      ] as const)
        expect(changed.world.pool[field][changed.index]).toBe(
          control.world.pool[field][control.index],
        );
    }
    expectArrival(changed);
    expect(searches).toHaveBeenCalledTimes(1);
  });

  it('retains one request when a chase refresh and blocked step happen together', () => {
    const c = setup();
    const pool = c.world.pool;
    const patch = c.world.placeBuilding(EntityType.MineralPatch, 0, 9, 22);
    executeCommand(c.world, {
      type: CommandType.Harvest,
      player: 0,
      units: [c.worker],
      target: patch,
    });
    advance(c, 2);
    expect(pool.pathLen[c.index]).toBeGreaterThan(0);
    expect(pool.pathPending[c.index]).toBe(0);
    // Put the next private-path step at the new footprint's edge, on the
    // chase's refresh tick. The refresh queues before step validation runs.
    pool.posY[c.index] = fromFloat(8.99);
    c.world.tick = 10 - (pool.serial[c.index]! % 10);
    foundation(c);
    const searches = vi.spyOn(c.astar, 'find');
    step(c);
    expect(pool.pathLen[c.index]).toBe(0);
    expect(pool.pathPending[c.index]).toBe(1);
    expect(c.world.pathQueue).toEqual([c.index]);
    expect(searches).not.toHaveBeenCalled();
    step(c);
    expect(pool.pathPending[c.index]).toBe(0);
    expect(pool.pathLen[c.index]).toBeGreaterThan(0);
    expect(c.world.pathQueue).toHaveLength(0);
    expect(searches).toHaveBeenCalledTimes(1);
  });

  it('repairs both mirrored players on identical ticks', () => {
    const a = setup(0),
      b = setup(1);
    const sa = vi.spyOn(a.astar, 'find'),
      sb = vi.spyOn(b.astar, 'find');
    move(a);
    move(b);
    for (let tick = 0; tick < 600; tick++) {
      if (tick === 10) {
        foundation(a);
        foundation(b);
      }
      step(a);
      step(b);
      const pa = a.world.pool,
        pb = b.world.pool;
      expect(pa.posX[a.index]! + pb.posX[b.index]!).toBe(fromInt(a.world.map.width));
      expect(pa.posY[a.index]! + pb.posY[b.index]!).toBe(fromInt(a.world.map.height));
      for (const field of [
        'order',
        'pathLen',
        'pathCursor',
        'pathPending',
        'pathCooldown',
      ] as const)
        expect(pa[field][a.index]).toBe(pb[field][b.index]);
      expect(sa.mock.calls.length).toBe(sb.mock.calls.length);
    }
    expectArrival(a);
    expectArrival(b);
  });

  it('stops an unreachable move after one failed repair instead of retrying every tick', () => {
    const c = setup();
    c.world.map.tiles.fill(Tile.Cliff);
    for (let y = 0; y < c.world.map.height; y++)
      for (let x = 9; x < 13; x++) c.world.map.tiles[c.world.map.index(x, y)] = Tile.Ground;
    const searches = vi.spyOn(c.astar, 'find');
    move(c);
    advance(c, 10);
    foundation(c);
    advance(c, 590);
    expect(c.world.pool.order[c.index]).toBe(Order.None);
    expect(c.world.pool.pathLen[c.index]).toBe(0);
    expect(c.world.pool.pathPending[c.index]).toBe(0);
    expect(c.world.pathQueue).toHaveLength(0);
    expect(searches).toHaveBeenCalledTimes(2);
    expect(c.world.pool.pathCooldown[c.index]).toBeGreaterThan(0);
  });

  it('keeps the existing failed-search cooldown for an unreachable entity target', () => {
    const c = setup();
    c.world.map.tiles.fill(Tile.Cliff);
    for (let y = 0; y < c.world.map.height; y++)
      for (let x = 9; x < 13; x++) c.world.map.tiles[c.world.map.index(x, y)] = Tile.Ground;
    const patch = c.world.placeBuilding(EntityType.MineralPatch, 0, 9, 22);
    const failed: number[] = [];
    const find = c.astar.find.bind(c.astar);
    vi.spyOn(c.astar, 'find').mockImplementation((...args) => {
      const result = find(...args);
      if (!result.length) failed.push(c.world.tick);
      return result;
    });
    executeCommand(c.world, {
      type: CommandType.Harvest,
      player: 0,
      units: [c.worker],
      target: patch,
    });
    advance(c, 10);
    foundation(c);
    advance(c, 590);
    expect(c.world.pool.order[c.index]).toBe(Order.Harvest);
    expect(failed.length).toBeGreaterThan(1);
    expect(failed.length).toBeLessThanOrEqual(16);
    for (let i = 1; i < failed.length; i++)
      expect(failed[i]! - failed[i - 1]!).toBeGreaterThanOrEqual(40);
  });

  it('can stop beside a destination that becomes a building footprint', () => {
    const c = setup();
    const searches = vi.spyOn(c.astar, 'find');
    move(c);
    advance(c, 10);
    foundation(c, 9, 21);
    advance(c, 590);
    expect(c.world.pool.order[c.index]).toBe(Order.None);
    const tile = c.world.map.tileOfPos(c.world.pool.posX[c.index]!, c.world.pool.posY[c.index]!);
    expect(c.world.map.isWalkable(c.world.map.tileXOf(tile), c.world.map.tileYOf(tile))).toBe(true);
    expect(searches.mock.calls.length).toBeLessThanOrEqual(4);
  });
});
