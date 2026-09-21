import { describe, expect, it } from 'vitest';
import {
  allocAction,
  allocMasks,
  computeMasks,
  decode,
  encode,
  legalise,
} from '../src/ai/neural/actions.js';
import { allocFrame, cellOf, RowKind } from '../src/ai/neural/frame.js';
import { EntityMemory } from '../src/ai/neural/memory.js';
import { allocObservation, NO_RECENT, ObservationEncoder } from '../src/ai/neural/observation.js';
import {
  ActionType,
  ENTITY_FEATURE_COUNT,
  GRID,
  GRID_CHANNELS,
  N_ENT,
  SCALARS,
} from '../src/ai/neural/spec.js';
import { defOf, PATCH_AMOUNT } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { fromFloat } from '../src/sim/fixed.js';
import { UNOCCUPIED } from '../src/sim/map.js';
import { coopMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import {
  BuildState,
  EntityType,
  NEUTRAL,
  NO_ENTITY,
  Order,
  Tile,
  type PlayerId,
} from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { EXPLORED, VISIBLE, Visibility } from '../src/vision/visibility.js';
import { mirrorX, mirrorY, twinMap } from './helpers/mirror.js';

function emptyWorld(): World {
  const world = new Simulation(coopMatch(0x51ce7a11, { botPlayers: [] })).world;
  for (let i = 0; i < world.pool.count; i++) {
    if (world.pool.alive[i] === 1) world.pool.destroy(world.pool.idAt(i));
  }
  world.map.tiles.fill(Tile.Ground);
  world.map.occupied.fill(UNOCCUPIED);
  world.tick = 100;
  for (const player of world.players) {
    player.minerals = 1000;
    player.supplyUsed = 0;
    player.supplyMax = 200;
  }
  return world;
}

function spawn(
  world: World,
  type: EntityType,
  owner: PlayerId,
  x = 20,
  y = 20,
  complete = true,
): number {
  const def = defOf(type);
  const half = def.isBuilding ? def.footprint / 2 : 0.5;
  const id = world.pool.spawn(type, owner, fromFloat(x + half), fromFloat(y + half));
  const i = idIndex(id);
  world.pool.tileX[i] = x;
  world.pool.tileY[i] = y;
  world.pool.buildState[i] = complete ? BuildState.Complete : BuildState.Site;
  if (type === EntityType.MineralPatch) world.pool.resourceAmount[i] = 500;
  return id;
}

function eyes(world: World, viewer: PlayerId = 0) {
  const vis = new Visibility(world.map);
  vis.state.fill(VISIBLE);
  const mem = new EntityMemory(viewer);
  mem.update(world, vis);
  const encoder = new ObservationEncoder(world, viewer);
  const obs = allocObservation(),
    frame = allocFrame(),
    masks = allocMasks();
  const update = () => {
    encoder.encode(vis, mem, NO_RECENT, obs, frame);
    computeMasks(world, frame, vis, mem, masks);
  };
  update();
  return { vis, mem, obs, frame, masks, update };
}

function grid(
  eye: ReturnType<typeof eyes>,
  channel: (typeof GRID_CHANNELS)[number],
  x: number,
  y: number,
): number {
  return eye.obs.grid[GRID_CHANNELS.indexOf(channel) * GRID * GRID + cellOf(x, y)]!;
}

describe('observation pointer capacity', () => {
  it('keeps discovered harvest and visible attack targets above 160 friendly entities', () => {
    const world = emptyWorld();
    const worker = spawn(world, EntityType.Worker, 0);
    for (let i = 0; i < 19; i++) spawn(world, EntityType.Burstbot, 0);
    for (let i = 0; i < 150; i++) spawn(world, EntityType.Burstbot, 1);
    const enemy = spawn(world, EntityType.Burstbot, 2, 30, 30);
    const patch = spawn(world, EntityType.MineralPatch, NEUTRAL, 25, 25);
    const e = eyes(world);
    expect(e.obs.entityMask.reduce((a, b) => a + b, 0)).toBe(N_ENT);
    expect(e.frame.rowOf.has(worker)).toBe(true);
    expect(e.frame.rowKind[e.frame.rowOf.get(enemy)!]).toBe(RowKind.EnemyVisible);
    expect(e.frame.rowKind[e.frame.rowOf.get(patch)!]).toBe(RowKind.Patch);
    expect(e.masks.type[ActionType.Harvest]).toBe(1);
    expect(e.masks.type[ActionType.Attack]).toBe(1);
    const command = {
      type: CommandType.Harvest,
      player: 0,
      units: [worker],
      target: patch,
    } as const;
    const action = allocAction();
    expect(encode({ ...command, units: [...command.units] }, e.frame, action)).toBe(true);
    expect(legalise(action, e.masks)).toBe(true);
    expect(decode(action, world, e.frame)).toEqual(command);
  });

  it('aggregates omitted own/allied/enemy/resource state independently of pointers', () => {
    const world = emptyWorld();
    const depot = spawn(world, EntityType.Depot, 0, 100, 120);
    for (let i = 0; i < 200; i++) spawn(world, EntityType.Worker, 0);
    const ally = spawn(world, EntityType.Burstbot, 1, 120, 40);
    for (let i = 0; i < 16; i++) spawn(world, EntityType.Burstbot, 2, 25, 25);
    const enemy = spawn(world, EntityType.Airport, 2, 130, 130);
    for (let i = 0; i < 8; i++) spawn(world, EntityType.MineralPatch, NEUTRAL, 22 + i, 22);
    const patch = spawn(world, EntityType.MineralPatch, NEUTRAL, 140, 140);
    const e = eyes(world);
    for (const id of [depot, ally, enemy, patch]) expect(e.frame.rowOf.has(id)).toBe(false);
    expect([...e.frame.rowKind].filter((kind) => kind === RowKind.Patch)).toHaveLength(8);
    expect([...e.frame.rowKind].filter((kind) => kind === RowKind.EnemyVisible)).toHaveLength(16);
    expect([...e.frame.rowKind].filter((kind) => kind === RowKind.OwnUnit)).toHaveLength(136);
    expect(grid(e, 'ownBuildings', 101, 121)).toBe(0.25);
    expect(grid(e, 'allyEntities', 120, 40)).toBe(0.125);
    expect(grid(e, 'enemyBuildings', 131, 131)).toBe(0.25);
    expect(grid(e, 'minerals', 140, 140)).toBeCloseTo(500 / (4 * PATCH_AMOUNT));
    expect(e.obs.scalars[SCALARS.indexOf('own:Depot')]).toBe(1 / 16);
    expect(e.obs.scalars[SCALARS.indexOf('enemy:Airport')]).toBe(1 / 16);
  });

  it('balances oversized own roles and producer types, reclaiming unused target reserves', () => {
    const world = emptyWorld();
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Burstbot, 0);
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Worker, 0);
    for (let i = 0; i < 70; i++) spawn(world, EntityType.CommandPost, 0);
    const producers = [EntityType.Barracks, EntityType.Factory, EntityType.Airport].map((type) =>
      spawn(world, type, 0),
    );
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Depot, 0, 40, 40, false);
    const e = eyes(world);
    expect(e.obs.entityMask.reduce((a, b) => a + b, 0)).toBe(N_ENT);
    expect(
      [...e.frame.rowKind].every(
        (kind) => kind === RowKind.OwnUnit || kind === RowKind.OwnBuilding,
      ),
    ).toBe(true);
    expect(e.masks.type[ActionType.Move]).toBe(1);
    expect(e.masks.type[ActionType.Build]).toBe(1);
    for (const id of producers)
      expect(e.masks.selection[ActionType.Train * N_ENT + e.frame.rowOf.get(id)!]).toBe(1);
    const types = [...e.frame.rows].map((id) => world.pool.type[idIndex(id)]);
    expect(types).toContain(EntityType.Worker);
    expect(types).toContain(EntityType.Burstbot);
    expect(types).toContain(EntityType.Depot);
  });

  it('keeps an omitted owned site resumable at zero minerals without charging twice', () => {
    const world = emptyWorld();
    const worker = spawn(world, EntityType.Worker, 0, 101, 100);
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Barracks, 0, 30, 30, false);
    const site = spawn(world, EntityType.Barracks, 0, 104, 100, false);
    world.player(0).minerals = 0;
    const e = eyes(world);
    expect(e.frame.rowOf.has(site)).toBe(false);
    expect(e.frame.constructionSites.get(100 * world.map.width + 104)?.id).toBe(site);
    const command = {
      type: CommandType.Build,
      player: 0,
      worker,
      building: EntityType.Barracks,
      tileX: 104,
      tileY: 100,
    } as const;
    const action = allocAction();
    expect(encode(command, e.frame, action)).toBe(true);
    expect(legalise(action, e.masks)).toBe(true);
    expect(decode(action, world, e.frame)).toEqual(command);
    const count = world.pool.count;
    executeCommand(world, decode(action, world, e.frame)!);
    expect(world.pool.order[idIndex(worker)]).toBe(Order.Build);
    expect(world.pool.orderTarget[idIndex(worker)]).toBe(site);
    expect(world.player(0).minerals).toBe(0);
    expect(world.pool.count).toBe(count);
  });

  it('does not consult hidden live enemy or resource state when choosing or aggregating memory', () => {
    const world = emptyWorld();
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Worker, 0);
    const enemy = spawn(world, EntityType.Airport, 2, 100, 100);
    const patch = spawn(world, EntityType.MineralPatch, NEUTRAL, 25, 25);
    const e = eyes(world);
    world.tick++;
    e.vis.state.fill(EXPLORED);
    e.mem.update(world, e.vis);
    e.update();
    expect(e.frame.rowKind[e.frame.rowOf.get(enemy)!]).toBe(RowKind.EnemyRemembered);
    const before = structuredClone(e.obs),
      rows = e.frame.rows.slice(),
      masks = structuredClone(e.masks);
    world.pool.posX[idIndex(enemy)] = fromFloat(130);
    world.pool.hp[idIndex(enemy)] = 1;
    world.pool.resourceAmount[idIndex(patch)] = 0;
    spawn(world, EntityType.DarkGolem, 2, 120, 120);
    e.mem.update(world, e.vis);
    e.update();
    expect(e.obs).toEqual(before);
    expect(e.frame.rows).toEqual(rows);
    expect(e.masks).toEqual(masks);
    expect(e.masks.type[ActionType.Harvest]).toBe(1);
    expect(e.masks.type[ActionType.Attack]).toBe(0);
  });

  it('uses freshest remembered threats only when none are visible, without making them attack targets', () => {
    const world = emptyWorld();
    for (let i = 0; i < 180; i++) spawn(world, EntityType.Worker, 0);
    const enemies = Array.from({ length: 18 }, (_, i) =>
      spawn(world, EntityType.Burstbot, 2, 70 + i, 70),
    );
    const e = eyes(world);
    world.tick++;
    e.vis.state.fill(EXPLORED);
    const newest = idIndex(enemies[17]!);
    e.vis.state[world.map.tileOfPosFor(world.pool.posX[newest]!, world.pool.posY[newest]!, false)] =
      VISIBLE;
    e.mem.update(world, e.vis);
    world.tick++;
    e.vis.state.fill(EXPLORED);
    e.mem.update(world, e.vis);
    e.update();
    expect([...e.frame.rowKind].filter((kind) => kind === RowKind.EnemyRemembered)).toHaveLength(
      16,
    );
    expect(e.frame.rowOf.has(enemies[17]!)).toBe(true);
    expect(e.frame.rowOf.has(enemies[16]!)).toBe(false);
    expect(e.masks.type[ActionType.Attack]).toBe(0);
  });

  it('has byte-identical overflow observations and masks in mirrored seats', () => {
    const world = emptyWorld(),
      pool = world.pool;
    const pair = (type: EntityType, owner: PlayerId, x: number, y: number, complete = true) => {
      const a = spawn(world, type, owner, x, y, complete);
      const b = spawn(world, type, owner === NEUTRAL ? NEUTRAL : owner + 2, x, y, complete);
      const ai = idIndex(a),
        bi = idIndex(b),
        f = defOf(type).footprint;
      pool.posX[bi] = mirrorX(world, pool.posX[ai]!);
      pool.posY[bi] = mirrorY(world, pool.posY[ai]!);
      pool.tileX[bi] = world.map.width - x - f;
      pool.tileY[bi] = world.map.height - y - f;
    };
    for (let i = 0; i < 190; i++)
      pair(
        i % 2 ? EntityType.Worker : EntityType.Burstbot,
        0,
        15 + (i % 12),
        15 + Math.floor(i / 12),
      );
    for (const type of [
      EntityType.CommandPost,
      EntityType.Barracks,
      EntityType.Factory,
      EntityType.Airport,
    ])
      pair(type, 0, 40 + type, 40);
    for (let i = 0; i < 30; i++) pair(EntityType.Depot, 0, 45 + (i % 10), 50, false);
    for (let i = 0; i < 170; i++) pair(EntityType.Burstbot, 1, 45 + (i % 10), 60);
    for (let i = 0; i < 12; i++) pair(EntityType.MineralPatch, NEUTRAL, 10 + i * 3, 8);
    const a = eyes(world, 0),
      b = eyes(world, 2);
    expect(a.obs).toEqual(b.obs);
    expect(a.masks).toEqual(b.masks);
    const twins = twinMap(world, world);
    for (let row = 0; row < N_ENT; row++)
      expect(pool.idAt(twins[idIndex(a.frame.rows[row]!)]!)).toBe(b.frame.rows[row]);
  });

  it('preserves under-capacity canonical row order and clears previously full padding', () => {
    const world = emptyWorld();
    const own = Array.from({ length: 180 }, () => spawn(world, EntityType.Worker, 0));
    const ally = spawn(world, EntityType.Burstbot, 1, 30, 30);
    const enemy = spawn(world, EntityType.Burstbot, 2, 40, 40);
    const patch = spawn(world, EntityType.MineralPatch, NEUTRAL, 25, 25);
    const e = eyes(world);
    for (const id of own.slice(2)) world.pool.destroy(id);
    e.update();
    expect([...e.frame.rows.slice(0, 5)]).toEqual([own[0], own[1], ally, enemy, patch]);
    expect([...e.frame.rows.slice(5)].every((id) => id === NO_ENTITY)).toBe(true);
    expect([...e.obs.entityMask.slice(5)].every((x) => x === 0)).toBe(true);
    expect([...e.obs.entities.slice(5 * ENTITY_FEATURE_COUNT)].every((x) => x === 0)).toBe(true);
  });
});
