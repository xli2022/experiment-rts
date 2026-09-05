/**
 * The human vocabulary, round-tripped and masked.
 *
 * A command the scripted bot or a person issues must encode into a decision
 * and decode back into the same command, up to what the vocabulary cannot say
 * (more than 24 units, a point finer than a tile). Every decision the masks
 * allow must be one the simulation accepts. And the whole thing must be
 * seat-blind: a command and its rotation encode into the same decision.
 */

import { describe, expect, it } from 'vitest';
import { botThink, THINK_INTERVAL } from '../src/ai/bot.js';
import { HeadlessMatch } from '../src/ai/headless.js';
import {
  actionFromInts,
  actionToInts,
  allocAction,
  allocMasks,
  computeMasks,
  decode,
  encode,
  selectsMany,
  usesLocation,
  usesTarget,
} from '../src/ai/neural/actions.js';
import { allocFrame, cellOf, subOf, tileOfCell } from '../src/ai/neural/frame.js';
import { EntityMemory } from '../src/ai/neural/memory.js';
import { allocObservation, NO_RECENT, ObservationEncoder } from '../src/ai/neural/observation.js';
import { sampleUniform } from '../src/ai/neural/random.js';
import {
  ACTION_INTS,
  ACTION_TYPE_COUNT,
  ActionType,
  N_ENT,
  SELECTION_MAX,
} from '../src/ai/neural/spec.js';
import { defOf } from '../src/config/rules.js';
import { CommandType, MAX_COMMAND_UNITS, type Command } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { FIX_HALF, fromInt, toInt } from '../src/sim/fixed.js';
import { duelMatch } from '../src/sim/match.js';
import { Rng } from '../src/sim/rng.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, NO_ENTITY, Order, type PlayerId } from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { Visibility } from '../src/vision/visibility.js';
import { scriptedAgents } from './helpers/agents.js';
import { fullScript, mirrorCommand, twinMap } from './helpers/mirror.js';

const SEED = 0x51ce7a11;

class Eyes {
  readonly vis: Visibility;
  readonly mem: EntityMemory;
  readonly encoder: ObservationEncoder;
  readonly obs = allocObservation();
  readonly frame = allocFrame();
  readonly masks = allocMasks();
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
  see(world: World): void {
    this.encoder.encode(this.vis, this.mem, NO_RECENT, this.obs, this.frame);
    computeMasks(world, this.frame, this.vis, this.mem, this.masks);
  }
}

/** The tile centre a point quantises to. */
function centreOf(x: number): number {
  return fromInt(toInt(x)) + FIX_HALF;
}

describe('encode and decode', () => {
  it('round-trip every command the scripted bot issues, up to the vocabulary', () => {
    const config = duelMatch(SEED, { botPlayers: [0, 1] });
    const match = new HeadlessMatch(config, scriptedAgents(config));
    const world = match.world;
    const eyes = new Eyes(world, 0);
    const action = allocAction();
    const seen = new Set<CommandType>();
    let checked = 0;
    for (let t = 0; t < 4500 && !world.matchOver; t++) {
      match.step();
      eyes.look(world);
      if (world.tick % THINK_INTERVAL !== 0) continue;
      eyes.see(world);
      for (const command of botThink(world, 0)) {
        if (!encode(command, eyes.frame, action)) continue;
        const back = decode(action, world, eyes.frame);
        expect(back).not.toBeNull();
        expect(back!.type).toBe(command.type);
        seen.add(command.type);
        checked++;
        if ('units' in command && 'units' in back!) {
          const inFrame = command.units
            .filter((id) => eyes.frame.rowOf.has(id))
            .slice(0, SELECTION_MAX);
          expect(back.units).toEqual(inFrame);
          expect(back.units.length).toBeLessThanOrEqual(MAX_COMMAND_UNITS);
        }
        if ('target' in command && 'target' in back!) expect(back.target).toBe(command.target);
        if (command.type === CommandType.Build && back!.type === CommandType.Build) {
          expect(back.worker).toBe(command.worker);
          expect(back.building).toBe(command.building);
          expect(back.tileX).toBe(command.tileX);
          expect(back.tileY).toBe(command.tileY);
        }
        if (command.type === CommandType.Train && back!.type === CommandType.Train) {
          expect(back.building).toBe(command.building);
          expect(back.unit).toBe(command.unit);
        }
        if (
          (command.type === CommandType.Move || command.type === CommandType.AttackMove) &&
          'x' in back!
        ) {
          expect(back.x).toBe(centreOf(command.x));
          expect(back.y).toBe(centreOf(command.y));
        }
      }
    }
    expect(checked).toBeGreaterThan(60);
    for (const type of [
      CommandType.Harvest,
      CommandType.Train,
      CommandType.Build,
      CommandType.AttackMove,
    ]) {
      expect(seen.has(type), `no ${CommandType[type]} command seen`).toBe(true);
    }
  });

  it('give the same decision for a command and its rotation, from the two seats', () => {
    const sim = new Simulation(duelMatch(SEED, { botPlayers: [] }));
    const world = sim.world;
    const eyes = [new Eyes(world, 0), new Eyes(world, 1)];
    const a = allocAction();
    const b = allocAction();
    const ia = new Int32Array(ACTION_INTS);
    const ib = new Int32Array(ACTION_INTS);
    let compared = 0;
    for (let t = 0; t < 4000 && !world.matchOver; t++) {
      const own = fullScript(world, t);
      const twins = twinMap(world, world);
      const all = own.slice();
      for (const c of own) all.push(mirrorCommand(world, c, twins));
      if (own.length > 0) {
        for (const e of eyes) e.see(world);
        for (let k = 0; k < own.length; k++) {
          if (!encode(own[k]!, eyes[0]!.frame, a)) continue;
          expect(encode(all[own.length + k]!, eyes[1]!.frame, b)).toBe(true);
          actionToInts(a, ia);
          actionToInts(b, ib);
          expect([...ia]).toEqual([...ib]);
          compared++;
        }
      }
      sim.step(all);
      for (const e of eyes) e.look(world);
    }
    // The script is sparing with commands — a handful of harvests, trains,
    // builds, a rally and the attack-moves — and every one of them is checked.
    expect(compared).toBeGreaterThan(20);
  });

  it('survive the flat integer form', () => {
    const action = allocAction();
    action.type = ActionType.Build;
    action.entityType = EntityType.Barracks;
    action.cell = 123;
    action.sub = 7;
    action.selection[0] = 9;
    const ints = new Int32Array(ACTION_INTS);
    actionToInts(action, ints);
    const back = allocAction();
    actionFromInts(ints, back);
    expect(back).toEqual(action);
    expect(() => actionToInts(action, new Int32Array(3))).toThrow();
  });

  it('map cells and sub-cells to tiles and back', () => {
    for (const [tx, ty] of [
      [0, 0],
      [5, 9],
      [127, 127],
      [151, 3],
    ]) {
      const cell = cellOf(tx!, ty!);
      const sub = subOf(tx!, ty!);
      expect(tileOfCell(cell, sub)).toEqual({ tx, ty });
    }
    expect(cellOf(160, 0)).toBe(-1);
  });
});

describe('the masks', () => {
  /** Apply a command and say whether the simulation took it. */
  function accepted(world: World, command: Command): boolean {
    const pool = world.pool;
    const minerals = world.player(command.player).minerals;
    const orderOf = (id: number): number => pool.order[idIndex(id)]!;
    switch (command.type) {
      case CommandType.Move:
      case CommandType.AttackMove: {
        executeCommand(world, command);
        const want = command.type === CommandType.Move ? Order.Move : Order.AttackMove;
        return command.units.some((id) => pool.isAlive(id) && orderOf(id) === want);
      }
      case CommandType.Attack:
      case CommandType.Harvest: {
        executeCommand(world, command);
        const want = command.type === CommandType.Attack ? Order.Attack : Order.Harvest;
        return command.units.some((id) => pool.isAlive(id) && orderOf(id) === want);
      }
      case CommandType.Stop:
        executeCommand(world, command);
        return command.units.some((id) => pool.isAlive(id) && orderOf(id) === Order.None);
      case CommandType.Hold:
        executeCommand(world, command);
        return command.units.some((id) => pool.isAlive(id) && orderOf(id) === Order.Hold);
      case CommandType.Build: {
        executeCommand(world, command);
        return world.player(command.player).minerals < minerals;
      }
      case CommandType.Train: {
        const before = pool.prodCount[idIndex(command.building)]!;
        executeCommand(world, command);
        return pool.prodCount[idIndex(command.building)]! > before;
      }
      case CommandType.CancelTrain: {
        const before = pool.prodCount[idIndex(command.building)]!;
        executeCommand(world, command);
        return pool.prodCount[idIndex(command.building)]! < before;
      }
      case CommandType.SetRally: {
        // The point may be snapped to the nearest standable tile, as a human's
        // click would be; what matters is that a rally was set.
        pool.hasRally[idIndex(command.building)] = 0;
        executeCommand(world, command);
        return pool.hasRally[idIndex(command.building)] === 1;
      }
      default:
        return false;
    }
  }

  it('only ever allow decisions the simulation accepts', () => {
    // A random legal decision at every step of a bot match, applied to the
    // world: what the masks call legal, the simulation must take. The two
    // exceptions are the simulation's per-unit rules the vocabulary does not
    // see — melee units told to attack a flyer, a patch mined out since it was
    // last seen — and they are counted, not excused.
    const config = duelMatch(SEED, { botPlayers: [1] });
    const match = new HeadlessMatch(config, scriptedAgents(config));
    const world = match.world;
    const eyes = new Eyes(world, 0);
    const rng = new Rng(0xabc);
    const action = allocAction();
    const byType = new Map<number, { tried: number; refused: number }>();
    let tried = 0;
    let refused = 0;
    let nulls = 0;
    for (let t = 0; t < 5000 && !world.matchOver; t++) {
      match.step();
      eyes.look(world);
      // Nobody plays slot 0 but this test, and its random orders keep the
      // workers from mining; a periodic grant keeps building and training
      // affordable so those heads get sampled too.
      if (t % 200 === 0) world.players[0]!.minerals = 600;
      if (t % 5 !== 0) continue;
      for (let k = 0; k < 2; k++) {
        // Masks describe the world as it stands; a command just applied may
        // have spent the minerals or emptied the queue the next one relied on.
        eyes.see(world);
        sampleUniform(eyes.masks, rng, action);
        if (action.type === ActionType.Noop) continue;
        expect(eyes.masks.type[action.type]).toBe(1);
        const command = decode(action, world, eyes.frame);
        if (command === null) {
          nulls++;
          continue;
        }
        expect(command.type).not.toBe(CommandType.Surrender);
        if ('units' in command) expect(command.units.length).toBeLessThanOrEqual(MAX_COMMAND_UNITS);
        const stat = byType.get(action.type) ?? { tried: 0, refused: 0 };
        stat.tried++;
        tried++;
        if (!accepted(world, command)) {
          stat.refused++;
          refused++;
        }
        byType.set(action.type, stat);
      }
    }
    expect(tried).toBeGreaterThan(1500);
    expect(nulls).toBe(0);
    const report = [...byType.entries()]
      .map(([type, s]) => `${ActionType[type]} ${s.refused}/${s.tried}`)
      .join(', ');
    expect(refused / tried, report).toBeLessThan(0.03);
    for (let type = 1; type < ACTION_TYPE_COUNT; type++) {
      expect(byType.get(type)?.tried ?? 0, `${ActionType[type]} never sampled`).toBeGreaterThan(0);
    }
  });

  it('follow the simulation rules for training, building, cancelling and rallying', () => {
    const config = duelMatch(SEED, { botPlayers: [] });
    const sim = new Simulation(config);
    const world = sim.world;
    const eyes = new Eyes(world, 0);
    const start = world.map.starts[0]!;
    const barracks = world.placeBuilding(EntityType.Barracks, 0, start.tileX + 5, start.tileY + 6);
    world.pool.buildState[idIndex(barracks)] = BuildState.Complete;
    world.recomputeSupply();
    const post = world.pool.idAt(
      [...Array(world.pool.count).keys()].find(
        (i) =>
          world.pool.alive[i] === 1 &&
          world.pool.owner[i] === 0 &&
          world.pool.type[i] === EntityType.CommandPost,
      )!,
    );

    world.players[0]!.minerals = 0;
    eyes.look(world);
    eyes.see(world);
    const rowB = eyes.frame.rowOf.get(barracks)!;
    const rowP = eyes.frame.rowOf.get(post)!;
    // Broke: nothing to train, nothing to build, but rallies are free.
    expect(eyes.masks.type[ActionType.Train]).toBe(0);
    expect(eyes.masks.type[ActionType.Build]).toBe(0);
    expect(eyes.masks.selection[ActionType.SetRally * N_ENT + rowB]).toBe(1);
    expect(eyes.masks.selection[ActionType.SetRally * N_ENT + rowP]).toBe(1);
    expect(eyes.masks.type[ActionType.CancelTrain]).toBe(0);

    world.players[0]!.minerals = 5000;
    eyes.see(world);
    expect(eyes.masks.type[ActionType.Train]).toBe(1);
    expect(eyes.masks.rowEntityType[rowB * 9 + EntityType.Burstbot]).toBe(1);
    expect(eyes.masks.rowEntityType[rowB * 9 + EntityType.Worker]).toBe(0);
    expect(eyes.masks.rowEntityType[rowP * 9 + EntityType.Worker]).toBe(1);
    expect(eyes.masks.type[ActionType.Build]).toBe(1);
    for (const building of [
      EntityType.CommandPost,
      EntityType.Depot,
      EntityType.Barracks,
      EntityType.Turret,
    ]) {
      expect(eyes.masks.buildType[building]).toBe(1);
    }
    expect(eyes.masks.buildType[EntityType.Burstbot]).toBe(0);
    // A building may only go where the side can see.
    const CELLS = 1600;
    const visible = eyes.obs.grid.subarray(3 * CELLS, 4 * CELLS);
    for (let cell = 0; cell < CELLS; cell++) {
      if (visible[cell] === 0)
        expect(eyes.masks.buildCell[EntityType.Depot * CELLS + cell]).toBe(0);
    }
    expect(
      [
        ...eyes.masks.buildCell.subarray(EntityType.Depot * CELLS, (EntityType.Depot + 1) * CELLS),
      ].some((v) => v === 1),
    ).toBe(true);

    // Queue something, and cancelling becomes possible.
    executeCommand(world, {
      type: CommandType.Train,
      player: 0,
      building: barracks,
      unit: EntityType.Burstbot,
    });
    eyes.see(world);
    expect(eyes.masks.type[ActionType.CancelTrain]).toBe(1);
    expect(eyes.masks.selection[ActionType.CancelTrain * N_ENT + rowB]).toBe(1);
    expect(eyes.masks.selection[ActionType.CancelTrain * N_ENT + rowP]).toBe(0);

    // Every type's heads agree with its type bit.
    for (let type = 1; type < ACTION_TYPE_COUNT; type++) {
      const anySelection = [
        ...eyes.masks.selection.subarray(type * N_ENT, (type + 1) * N_ENT),
      ].some((v) => v === 1);
      if (eyes.masks.type[type] === 1) expect(anySelection).toBe(true);
      if (usesTarget(type) && eyes.masks.type[type] === 1) {
        expect(
          [...eyes.masks.target.subarray(type * N_ENT, (type + 1) * N_ENT)].some((v) => v === 1),
        ).toBe(true);
      }
      if (usesLocation(type) && type !== ActionType.Build && eyes.masks.type[type] === 1) {
        expect(
          [...eyes.masks.cell.subarray(type * CELLS, (type + 1) * CELLS)].some((v) => v === 1),
        ).toBe(true);
      }
      void selectsMany;
    }
    void defOf;
    void NO_ENTITY;
  });
});
