/**
 * The neural bot's observation is fog-limited and seat-blind.
 *
 * Seat-blind: the map is symmetric under a 180-degree rotation and so is the
 * simulation, so in a mirrored match the second seat's observation must be
 * byte-for-byte the first seat's, row for row — that is what lets one policy
 * play every seat. Fog-limited: no row is an enemy the side cannot see or
 * does not remember, and no cell in unexplored ground carries an entity.
 */

import { describe, expect, it } from 'vitest';
import { RowKind, allocFrame, cellOf } from '../src/ai/neural/frame.js';
import { EntityMemory } from '../src/ai/neural/memory.js';
import {
  allocObservation,
  encodeCritic,
  NO_RECENT,
  ObservationEncoder,
} from '../src/ai/neural/observation.js';
import {
  CRITIC_LEN,
  ENTITY_FEATURE_COUNT,
  ENTITY_FEATURES,
  GRID,
  GRID_CHANNELS,
  N_ENT,
} from '../src/ai/neural/spec.js';
import { HeadlessMatch } from '../src/ai/headless.js';
import { buildingUpgrade, defOf, MAX_PRODUCTION_QUEUE } from '../src/config/rules.js';
import { fromFloat, fromInt } from '../src/sim/fixed.js';
import { idIndex } from '../src/sim/entities.js';
import { OCCUPIED_SOLID, UNOCCUPIED } from '../src/sim/map.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, NO_ENTITY, Order, type PlayerId } from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { EXPLORED, VISIBLE, Visibility } from '../src/vision/visibility.js';
import { scriptedAgents } from './helpers/agents.js';
import { fullScript, mirrorCommand, twinMap } from './helpers/mirror.js';

const SEED = 0x51ce7a11;
const F = ENTITY_FEATURE_COUNT;
const CELLS = GRID * GRID;

/** One seat's eyes: visibility, memory and encoder kept in step every tick. */
class Eyes {
  readonly vis: Visibility;
  readonly mem: EntityMemory;
  readonly encoder: ObservationEncoder;
  readonly obs = allocObservation();
  readonly frame = allocFrame();
  constructor(
    world: World,
    readonly viewer: PlayerId,
  ) {
    this.vis = new Visibility(world.map);
    this.mem = new EntityMemory(viewer);
    this.encoder = new ObservationEncoder(world, viewer);
  }
  look(world: World): void {
    this.vis.update(world, this.viewer);
    this.mem.update(world, this.vis);
  }
  encode(): void {
    this.encoder.encode(this.vis, this.mem, NO_RECENT, this.obs, this.frame);
  }
}

function firstDifference(a: Float32Array, b: Float32Array): string | null {
  for (let k = 0; k < a.length; k++) {
    if (a[k] !== b[k]) return `index ${k}: ${a[k]} vs ${b[k]}`;
  }
  return null;
}

describe('the observation', () => {
  it('exposes precise owned movement goals in the canonical frame and clears unused goals', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const pool = world.pool;
    const x = fromFloat(31.25),
      y = fromFloat(45.5);
    const dx = fromFloat(16.25),
      dy = fromFloat(-8.5);
    const ids = [
      pool.spawn(EntityType.Burstbot, 0, x, y),
      pool.spawn(
        EntityType.Burstbot,
        1,
        fromInt(world.map.width) - x,
        fromInt(world.map.height) - y,
      ),
    ];
    const eyes = [new Eyes(world, 0), new Eyes(world, 1)];
    const columns = [ENTITY_FEATURES.indexOf('orderDx'), ENTITY_FEATURES.indexOf('orderDy')];
    for (const order of [
      Order.Move,
      Order.AttackMove,
      Order.Hold,
      Order.Attack,
      Order.Build,
      Order.Harvest,
      Order.None,
    ]) {
      const values = eyes.map((eye, player) => {
        const i = idIndex(ids[player]!);
        const sign = player === 0 ? 1 : -1;
        pool.order[i] = order;
        pool.orderX[i] = pool.posX[i]! + sign * dx;
        pool.orderY[i] = pool.posY[i]! + sign * dy;
        eye.look(world);
        eye.encode();
        const row = eye.frame.rowOf.get(ids[player]!)!;
        return columns.map((column) => eye.obs.entities[row * F + column]);
      });
      expect(values[0]).toEqual(values[1]);
      if (order === Order.Move || order === Order.AttackMove) {
        expect(values[0]![0]).toBeCloseTo(16.25 / Math.max(world.map.width, world.map.height), 8);
        expect(values[0]![1]).toBeCloseTo(-8.5 / Math.max(world.map.width, world.map.height), 8);
      } else expect(values[0]).toEqual([0, 0]);
    }
  });

  it('distinguishes owned production tails while preserving the legacy observation columns', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const id = world.pool.spawn(EntityType.Barracks, 0, fromInt(30), fromInt(30));
    const i = idIndex(id);
    world.pool.prodPush(i, EntityType.Burstbot);
    world.pool.prodPush(i, EntityType.Burstbot);
    world.pool.prodProgress[i] = 60;
    const eyes = new Eyes(world, 0);
    eyes.look(world);
    eyes.encode();
    const row = eyes.frame.rowOf.get(id)!;
    const firstNew = ENTITY_FEATURES.indexOf('orderDx');
    const before = eyes.obs.entities.slice(row * F, (row + 1) * F);
    world.pool.prodQueue[i * MAX_PRODUCTION_QUEUE + 1] = EntityType.Slicebot;
    eyes.encode();
    const after = eyes.obs.entities.slice(row * F, (row + 1) * F);
    expect(after.slice(0, firstNew)).toEqual(before.slice(0, firstNew));
    expect(before[ENTITY_FEATURES.indexOf('queued:Burstbot')]).toBe(
      Math.fround(2 / MAX_PRODUCTION_QUEUE),
    );
    expect(after[ENTITY_FEATURES.indexOf('queued:Burstbot')]).toBe(
      Math.fround(1 / MAX_PRODUCTION_QUEUE),
    );
    expect(after[ENTITY_FEATURES.indexOf('queued:Slicebot')]).toBe(
      Math.fround(1 / MAX_PRODUCTION_QUEUE),
    );
    world.pool.prodCount[i] = 0;
    eyes.encode();
    expect([...eyes.obs.entities.slice(row * F + firstNew, (row + 1) * F)]).toEqual(
      new Array(F - firstNew).fill(0),
    );
  });

  it('keeps allied, visible and remembered enemy, neutral, and unused private columns zero', () => {
    const world = new Simulation(coopMatch(SEED, { botPlayers: [] })).world;
    const pool = world.pool;
    for (const owner of [0, 1, 2]) {
      for (const type of [EntityType.Burstbot, EntityType.Barracks, EntityType.Depot]) {
        const id = pool.spawn(type, owner, fromInt(30), fromInt(30));
        const i = idIndex(id);
        pool.order[i] = Order.AttackMove;
        pool.orderX[i] = fromInt(60);
        pool.orderY[i] = fromInt(45);
        pool.prodPush(i, EntityType.Burstbot);
      }
    }
    const eyes = new Eyes(world, 0);
    const firstNew = ENTITY_FEATURES.indexOf('orderDx');
    for (const visible of [true, false]) {
      eyes.vis.state.fill(visible ? VISIBLE : EXPLORED);
      eyes.mem.update(world, eyes.vis);
      eyes.encode();
      let nonOwnRows = 0;
      for (let row = 0; row < N_ENT; row++) {
        const kind = eyes.frame.rowKind[row]!;
        const values = [...eyes.obs.entities.slice(row * F + firstNew, (row + 1) * F)];
        if (kind !== RowKind.OwnUnit && kind !== RowKind.OwnBuilding) {
          expect(values).toEqual(new Array(F - firstNew).fill(0));
          if (kind !== RowKind.Empty) nonOwnRows++;
        } else {
          const def = defOf(pool.type[idIndex(eyes.frame.rows[row]!)]!);
          if (def.isBuilding) expect(values.slice(0, 2)).toEqual([0, 0]);
          if (def.produces.length === 0)
            expect(values.slice(2)).toEqual(new Array(F - firstNew - 2).fill(0));
        }
      }
      expect(nonOwnRows).toBeGreaterThan(0);
      expect([...eyes.frame.rowKind]).toContain(
        visible ? RowKind.EnemyVisible : RowKind.EnemyRemembered,
      );
      world.tick++;
    }
  });

  it('exposes own building technology and progress without leaking enemy research', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    const buildings = [0, 1].map((owner) =>
      world.pool.spawn(EntityType.Barracks, owner, fromInt(30), fromInt(30)),
    );
    for (const id of buildings) {
      const i = idIndex(id);
      world.pool.buildingLevel[i] = 1;
      world.pool.upgrading[i] = 1;
      world.pool.upgradeProgress[i] = buildingUpgrade(EntityType.Barracks)!.buildTicks / 2;
    }
    eyes.vis.state.fill(VISIBLE);
    eyes.mem.update(world, eyes.vis);
    eyes.encode();
    const technology = (id: number) =>
      ['buildingLevel', 'upgrading', 'upgradeProgress'].map(
        (feature) =>
          eyes.obs.entities[
            eyes.frame.rowOf.get(id)! * F +
              ENTITY_FEATURES.indexOf(feature as (typeof ENTITY_FEATURES)[number])
          ],
      );
    expect(technology(buildings[0]!)).toEqual([0.5, 1, 0.5]);
    expect(technology(buildings[1]!)).toEqual([0, 0, 0]);
    const i = idIndex(buildings[0]!);
    world.pool.buildingLevel[i] = 2;
    world.pool.upgrading[i] = 0;
    world.pool.upgradeProgress[i] = 0;
    eyes.encode();
    expect(technology(buildings[0]!)).toEqual([1, 0, 0]);
  });

  it('keeps a remembered mineral patch from becoming a solid obstacle', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    const patch = world.pool.type.findIndex((type) => type === EntityType.MineralPatch);
    const footprint = defOf(EntityType.MineralPatch).footprint;
    for (let y = world.pool.tileY[patch]!; y < world.pool.tileY[patch]! + footprint; y++) {
      for (let x = world.pool.tileX[patch]!; x < world.pool.tileX[patch]! + footprint; x++) {
        eyes.vis.state[world.map.index(x, y)] = VISIBLE;
      }
    }
    eyes.mem.update(world, eyes.vis);
    expect(eyes.mem.get(world.pool.idAt(patch))).toBeDefined();
    eyes.encode();
    const offset = GRID_CHANNELS.indexOf('buildable') * CELLS;
    const before = eyes.obs.grid.slice(offset, offset + CELLS);
    world.tick++;
    for (let t = 0; t < eyes.vis.state.length; t++) {
      if (eyes.vis.state[t] === VISIBLE) eyes.vis.state[t] = EXPLORED;
    }
    eyes.mem.update(world, eyes.vis);
    eyes.encode();
    expect(firstDifference(eyes.obs.grid.slice(offset, offset + CELLS), before)).toBeNull();
  });

  it('does not reveal new occupancy on explored ground that is now hidden', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    eyes.vis.state.fill(EXPLORED);
    world.map.occupied.fill(UNOCCUPIED);
    eyes.encode();
    const before = eyes.obs.grid.slice();
    world.map.occupied.fill(OCCUPIED_SOLID);
    eyes.encode();
    expect(firstDifference(eyes.obs.grid, before)).toBeNull();
  });

  it('retains remembered building occupancy and marks hidden patches as remembered', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    eyes.vis.state.fill(VISIBLE);
    eyes.mem.update(world, eyes.vis);
    eyes.encode();
    const buildable = GRID_CHANNELS.indexOf('buildable') * CELLS;
    const before = eyes.obs.grid.slice(buildable, buildable + CELLS);
    const enemyPost = world.pool.type.findIndex(
      (type, i) => type === EntityType.CommandPost && world.pool.owner[i] === 1,
    );
    const cell = cellOf(world.pool.tileX[enemyPost]!, world.pool.tileY[enemyPost]!);
    world.tick++;
    eyes.vis.state.fill(EXPLORED);
    eyes.mem.update(world, eyes.vis);
    eyes.encode();
    const remembered = eyes.obs.grid.slice(buildable, buildable + CELLS);
    expect(remembered[cell]).toBe(before[cell]);
    // The enemy base remains blocked even if it is destroyed while hidden.
    for (let i = 0; i < world.pool.count; i++) {
      if (world.pool.owner[i] === 1 && world.pool.type[i] === EntityType.CommandPost) {
        const footprint = defOf(EntityType.CommandPost).footprint;
        for (let y = world.pool.tileY[i]!; y < world.pool.tileY[i]! + footprint; y++) {
          for (let x = world.pool.tileX[i]!; x < world.pool.tileX[i]! + footprint; x++) {
            world.map.occupied[world.map.index(x, y)] = UNOCCUPIED;
          }
        }
        world.pool.destroy(world.pool.idAt(i));
      }
    }
    eyes.mem.update(world, eyes.vis);
    eyes.encode();
    expect(
      firstDifference(eyes.obs.grid.slice(buildable, buildable + CELLS), remembered),
    ).toBeNull();
    const visibleCol = ENTITY_FEATURES.indexOf('visibleNow');
    const patch = eyes.frame.rowKind.findIndex((kind) => kind === RowKind.Patch);
    expect(patch).toBeGreaterThanOrEqual(0);
    expect(eyes.obs.entities[patch * F + visibleCol]).toBe(0);
  });

  it('is the same from both seats of a mirrored duel, row for row', () => {
    const sim = new Simulation(duelMatch(SEED, { botPlayers: [] }));
    const world = sim.world;
    const eyes = [new Eyes(world, 0), new Eyes(world, 1)];
    let compared = 0;
    for (let t = 0; t < 4000 && !world.matchOver; t++) {
      const own = fullScript(world, t);
      const twins = twinMap(world, world);
      const all = own.slice();
      for (const c of own) all.push(mirrorCommand(world, c, twins));
      sim.step(all);
      for (const e of eyes) e.look(world);
      if (t % 100 !== 99) continue;

      for (const e of eyes) e.encode();
      const [a, b] = eyes as [Eyes, Eyes];
      const diff =
        firstDifference(a.obs.entities, b.obs.entities) ??
        firstDifference(a.obs.grid, b.obs.grid) ??
        firstDifference(a.obs.scalars, b.obs.scalars);
      if (diff !== null) {
        const row = Math.floor(Number(diff.match(/index (\d+)/)?.[1] ?? 0) / F);
        throw new Error(
          `tick ${world.tick}: ${diff} (entity row ${row}, feature ${ENTITY_FEATURES[Number(diff.match(/index (\d+)/)?.[1] ?? 0) % F]})`,
        );
      }
      expect([...a.obs.entityMask]).toEqual([...b.obs.entityMask]);
      // Every row stands for the twin of the row it mirrors.
      const twinsNow = twinMap(world, world);
      for (let r = 0; r < N_ENT; r++) {
        const idA = a.frame.rows[r]!;
        const idB = b.frame.rows[r]!;
        expect(a.frame.rowKind[r]).toBe(b.frame.rowKind[r]);
        if (idA === NO_ENTITY) {
          expect(idB).toBe(NO_ENTITY);
          continue;
        }
        // A remembered entity may be dead by now; its twin then is too, and
        // neither can be looked up. The living ones must be twins.
        if (!world.pool.isAlive(idA)) {
          expect(world.pool.isAlive(idB)).toBe(false);
          continue;
        }
        const twin = twinsNow[idIndex(idA)]!;
        expect(twin).toBeGreaterThanOrEqual(0);
        expect(world.pool.idAt(twin)).toBe(idB);
      }
      compared++;
    }
    expect(compared).toBeGreaterThan(20);
    // And the match was worth comparing: rows of every kind appeared.
    const kinds = new Set([...eyes[0]!.frame.rowKind]);
    expect(kinds.has(RowKind.OwnUnit)).toBe(true);
    expect(kinds.has(RowKind.OwnBuilding)).toBe(true);
    expect(kinds.has(RowKind.Patch)).toBe(true);
  });

  it('is the same from mirrored seats of a four-bot match on the four-corner map', () => {
    const config = coopMatch(SEED, { botPlayers: [0, 1, 2, 3] });
    const match = new HeadlessMatch(config, scriptedAgents(config));
    const world = match.world;
    const eyes = [new Eyes(world, 0), new Eyes(world, 2), new Eyes(world, 1), new Eyes(world, 3)];
    let compared = 0;
    for (let t = 0; t < 3000 && !world.matchOver; t++) {
      match.step();
      for (const e of eyes) e.look(world);
      if (t % 250 !== 249) continue;
      for (const e of eyes) e.encode();
      for (const [a, b] of [
        [eyes[0]!, eyes[1]!],
        [eyes[2]!, eyes[3]!],
      ]) {
        expect(firstDifference(a.obs.entities, b.obs.entities)).toBeNull();
        expect(firstDifference(a.obs.grid, b.obs.grid)).toBeNull();
        expect(firstDifference(a.obs.scalars, b.obs.scalars)).toBeNull();
      }
      // Allies are not mirrors of each other, and their observations differ.
      expect(firstDifference(eyes[0]!.obs.scalars, eyes[2]!.obs.scalars)).not.toBeNull();
      compared++;
    }
    expect(compared).toBeGreaterThan(5);
    const kinds = new Set([...eyes[0]!.frame.rowKind]);
    expect(kinds.has(RowKind.Ally)).toBe(true);
  });

  it('shows enemies only while seen or remembered, and nothing in unexplored ground', () => {
    const config = duelMatch(SEED, { botPlayers: [0, 1] });
    const match = new HeadlessMatch(config, scriptedAgents(config));
    const world = match.world;
    const eyes = new Eyes(world, 0);
    const visibleCol = ENTITY_FEATURES.indexOf('visibleNow');
    const exploredChannel = GRID_CHANNELS.indexOf('explored');
    let sawEnemy = false;
    let remembered = false;
    for (let t = 0; t < 5000 && !world.matchOver; t++) {
      match.step();
      eyes.look(world);
      if (t % 50 !== 49) continue;
      eyes.encode();
      for (let r = 0; r < N_ENT; r++) {
        const kind = eyes.frame.rowKind[r]!;
        const id = eyes.frame.rows[r]!;
        if (kind === RowKind.EnemyVisible) {
          sawEnemy = true;
          expect(world.pool.isAlive(id)).toBe(true);
          expect(eyes.vis.canSee(world, idIndex(id), 0)).toBe(true);
          expect(eyes.obs.entities[r * F + visibleCol]).toBe(1);
        } else if (kind === RowKind.EnemyRemembered) {
          remembered = true;
          expect(eyes.mem.get(id)).toBeDefined();
          expect(eyes.obs.entities[r * F + visibleCol]).toBe(0);
        }
      }
      // Every living enemy the side can see is in the table as visible.
      for (let i = 0; i < world.pool.count; i++) {
        if (world.pool.alive[i] !== 1 || !world.isHostile(i, 0)) continue;
        const id = world.pool.idAt(i);
        const row = eyes.frame.rowOf.get(id);
        if (eyes.vis.canSee(world, i, 0))
          expect(eyes.frame.rowKind[row!]).toBe(RowKind.EnemyVisible);
        else if (row !== undefined) expect(eyes.frame.rowKind[row]).toBe(RowKind.EnemyRemembered);
      }
      for (let cell = 0; cell < CELLS; cell++) {
        if (eyes.obs.grid[exploredChannel * CELLS + cell] !== 0) continue;
        for (const channel of [
          'ownBuildings',
          'ownUnits',
          'enemyBuildings',
          'enemyUnitsVisible',
          'minerals',
        ]) {
          expect(eyes.obs.grid[GRID_CHANNELS.indexOf(channel as never) * CELLS + cell]).toBe(0);
        }
      }
    }
    expect(sawEnemy).toBe(true);
    expect(remembered).toBe(true);
  });

  it('orders own rows by serial and fills the critic view with the whole truth', () => {
    const config = duelMatch(SEED, { botPlayers: [0, 1] });
    const match = new HeadlessMatch(config, scriptedAgents(config));
    const world = match.world;
    const eyes = new Eyes(world, 1);
    for (let t = 0; t < 1500; t++) {
      match.step();
      eyes.look(world);
    }
    eyes.encode();
    let last = -1;
    let own = 0;
    for (let r = 0; r < N_ENT; r++) {
      const kind = eyes.frame.rowKind[r]!;
      if (kind !== RowKind.OwnUnit && kind !== RowKind.OwnBuilding) continue;
      const serial = world.pool.serial[idIndex(eyes.frame.rows[r]!)]!;
      expect(serial).toBeGreaterThan(last);
      last = serial;
      own++;
    }
    expect(own).toBeGreaterThan(8);

    const critic = new Float32Array(CRITIC_LEN);
    encodeCritic(world, 1, critic);
    // Player 1 comes first in its own critic view; the enemy's true economy follows.
    expect(critic[0]).toBeCloseTo(world.player(1).minerals / 1000, 6);
    expect(critic[11]).toBeCloseTo(world.player(0).minerals / 1000, 6);
    expect(critic[3]).toBeGreaterThan(0);
  });
});
