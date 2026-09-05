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
import { RowKind, allocFrame } from '../src/ai/neural/frame.js';
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
import { idIndex } from '../src/sim/entities.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { NO_ENTITY, type PlayerId } from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { Visibility } from '../src/vision/visibility.js';
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
