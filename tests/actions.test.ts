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
  BUILDINGS,
  computeMasks,
  decode,
  encode,
  legalise,
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
import { buildingUpgrade, defOf, productionOptions } from '../src/config/rules.js';
import { CommandType, MAX_COMMAND_UNITS, type Command } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { FIX_HALF, fromInt, toInt } from '../src/sim/fixed.js';
import { UNOCCUPIED } from '../src/sim/map.js';
import { duelMatch, matchConfig } from '../src/sim/match.js';
import { Rng } from '../src/sim/rng.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { Simulation } from '../src/sim/tick.js';
import {
  BuildState,
  EntityType,
  ENTITY_TYPE_COUNT,
  MapLayout,
  NO_ENTITY,
  Order,
  Tile,
  type PlayerId,
} from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { VISIBLE, Visibility } from '../src/vision/visibility.js';
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
  it.each([0, 1])(
    'resumes the exact owned construction site without paying again from seat %i',
    (player) => {
      const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
      world.map.tiles.fill(Tile.Ground);
      world.map.occupied.fill(UNOCCUPIED);
      const worker = world.pool.spawn(EntityType.Worker, player, fromInt(39), fromInt(39));
      const site = world.placeBuilding(EntityType.Airport, player, 40, 40);
      world.pool.buildState[idIndex(site)] = BuildState.UnderConstruction;
      world.pool.buildProgress[idIndex(site)] = 10;
      const eyes = new Eyes(world, player);
      const command: Command = {
        type: CommandType.Build,
        player,
        worker,
        building: EntityType.Airport,
        tileX: 40,
        tileY: 40,
      };
      const action = allocAction();
      for (const minerals of [0, 1000]) {
        world.player(player).minerals = minerals;
        eyes.look(world);
        eyes.see(world);
        expect(encode(command, eyes.frame, action)).toBe(true);
        expect(legalise(action, eyes.masks)).toBe(true);
        expect(decode(action, world, eyes.frame)).toEqual(command);
        const count = world.pool.count;
        executeCommand(world, decode(action, world, eyes.frame)!);
        expect(world.pool.order[idIndex(worker)]).toBe(Order.Build);
        expect(world.pool.orderTarget[idIndex(worker)]).toBe(site);
        expect(world.player(player).minerals).toBe(minerals);
        expect(world.pool.count).toBe(count);
        expect(world.pool.buildProgress[idIndex(site)]).toBe(10);
        if (minerals === 0) {
          expect([...eyes.masks.buildCell].reduce((sum, bit) => sum + bit, 0)).toBe(1);
          // Sub-cells are unmasked: with no funds every sub-cell choice must
          // resolve to the same existing site, not a nearby new foundation.
          for (let sub = 0; sub < 16; sub++) {
            action.sub = sub;
            expect(decode(action, world, eyes.frame)).toEqual(command);
          }
          action.entityType = EntityType.Factory;
          expect(legalise(action, eyes.masks)).toBe(false);
          expect(decode(action, world, eyes.frame)).toBeNull();
        }
      }
      world.pool.buildState[idIndex(site)] = BuildState.Complete;
      expect(decode(action, world, eyes.frame)).toBeNull();
      world.pool.buildState[idIndex(site)] = BuildState.UnderConstruction;
      world.pool.destroy(site);
      world.map.occupied.fill(UNOCCUPIED);
      expect(decode(action, world, eyes.frame)).toBeNull();
    },
  );

  it.each([
    { owner: 0, state: BuildState.Complete, reason: 'completed own' },
    { owner: 1, state: BuildState.UnderConstruction, reason: 'allied' },
    { owner: 2, state: BuildState.UnderConstruction, reason: 'enemy' },
  ])('does not offer free resume of a $reason building', ({ owner, state }) => {
    const world = new Simulation(matchConfig(MapLayout.Quarters, SEED, { botPlayers: [] })).world;
    world.map.tiles.fill(Tile.Ground);
    world.map.occupied.fill(UNOCCUPIED);
    const worker = world.pool.spawn(EntityType.Worker, 0, fromInt(39), fromInt(39));
    const site = world.placeBuilding(EntityType.Airport, owner, 40, 40);
    world.pool.buildState[idIndex(site)] = state;
    world.player(0).minerals = 0;
    const eyes = new Eyes(world, 0);
    eyes.look(world);
    eyes.see(world);
    const action = allocAction();
    expect(
      encode(
        {
          type: CommandType.Build,
          player: 0,
          worker,
          building: EntityType.Airport,
          tileX: 40,
          tileY: 40,
        },
        eyes.frame,
        action,
      ),
    ).toBe(true);
    expect(eyes.masks.type[ActionType.Build]).toBe(0);
    expect(legalise(action, eyes.masks)).toBe(false);
    expect(decode(action, world, eyes.frame)).toBeNull();
  });

  it('round-trips upgrade commands and matches each unlocked production tier', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    const action = allocAction();
    for (const producer of [EntityType.Barracks, EntityType.Factory, EntityType.Airport]) {
      const id = world.pool.spawn(producer, 0, fromInt(30), fromInt(30));
      const i = idIndex(id);
      world.pool.buildState[i] = BuildState.Complete;
      world.player(0).minerals = 1000;
      eyes.look(world);
      eyes.see(world);
      const row = eyes.frame.rowOf.get(id)!;
      const allowed = (type: ActionType) => eyes.masks.selection[type * N_ENT + row];
      const training = () =>
        Array.from({ length: ENTITY_TYPE_COUNT }, (_, type) => type).filter(
          (type) => eyes.masks.rowEntityType[row * ENTITY_TYPE_COUNT + type] === 1,
        );
      expect(training()).toEqual([...productionOptions(producer)].sort((a, b) => a - b));
      const upgrade = buildingUpgrade(producer);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(upgrade ? 1 : 0);
      if (!upgrade) continue;
      world.player(0).minerals = upgrade.mineralCost - 1;
      eyes.see(world);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(0);
      world.player(0).minerals = 1000;
      world.pool.prodPush(i, productionOptions(producer)[0]!);
      eyes.see(world);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(0);
      world.pool.prodCount[i] = 0;
      for (const type of [CommandType.UpgradeBuilding, CommandType.CancelUpgrade] as const) {
        const command = { type, player: 0, building: id };
        expect(encode(command, eyes.frame, action)).toBe(true);
        expect(decode(action, world, eyes.frame)).toEqual(command);
      }
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building: id });
      eyes.see(world);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(0);
      expect(allowed(ActionType.CancelUpgrade)).toBe(1);
      expect(allowed(ActionType.Train)).toBe(0);
      expect(training()).toEqual([]);
      executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building: id });
      eyes.see(world);
      expect(allowed(ActionType.CancelUpgrade)).toBe(0);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(1);
      world.pool.buildingLevel[i] = 2;
      eyes.see(world);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(0);
      expect(training()).toEqual([...productionOptions(producer, 2)].sort((a, b) => a - b));
      world.pool.buildState[i] = BuildState.UnderConstruction;
      eyes.see(world);
      expect(allowed(ActionType.Train)).toBe(0);
      expect(allowed(ActionType.UpgradeBuilding)).toBe(0);
    }
  });

  it('rejects teacher subformations the action vocabulary cannot express', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    const eyes = new Eyes(world, 0);
    eyes.look(world);
    eyes.see(world);
    const worker = [...eyes.frame.rowOf.keys()].find(
      (id) =>
        world.pool.owner[idIndex(id)] === 0 && world.pool.type[idIndex(id)] === EntityType.Worker,
    )!;
    const action = allocAction();
    for (const type of [CommandType.Move, CommandType.AttackMove] as const) {
      const command = { type, player: 0, units: [worker], x: fromInt(20), y: fromInt(20) };
      expect(encode(command, eyes.frame, action)).toBe(true);
      expect(encode({ ...command, formationOffset: 0 }, eyes.frame, action)).toBe(true);
      expect(encode({ ...command, formationOffset: MAX_COMMAND_UNITS }, eyes.frame, action)).toBe(
        false,
      );
    }
  });

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
  it('does not reveal hidden occupancy at the edge of a visible build cell', () => {
    const world = new Simulation(duelMatch(SEED, { botPlayers: [] })).world;
    world.map.tiles.fill(Tile.Ground);
    world.map.occupied.fill(UNOCCUPIED);
    const eyes = new Eyes(world, 0);
    // Only one tile is seen. A four-tile-wide footprint extends into fog.
    eyes.vis.state[world.map.index(60, 60)] = VISIBLE;
    eyes.see(world);
    const before = eyes.masks.buildCell.slice();
    for (let t = 0; t < world.map.occupied.length; t++) {
      if (eyes.vis.state[t] !== VISIBLE) world.map.occupied[t] = 123;
    }
    eyes.see(world);
    expect(eyes.masks.buildCell).toEqual(before);
  });

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
        const worker = idIndex(command.worker);
        const site = pool.orderTarget[worker]!;
        return (
          world.player(command.player).minerals < minerals ||
          (pool.order[worker] === Order.Build &&
            pool.isAlive(site) &&
            pool.owner[idIndex(site)] === command.player &&
            pool.tileX[idIndex(site)] === command.tileX &&
            pool.tileY[idIndex(site)] === command.tileY)
        );
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
      case CommandType.UpgradeBuilding:
        executeCommand(world, command);
        return pool.upgrading[idIndex(command.building)] === 1;
      case CommandType.CancelUpgrade:
        executeCommand(world, command);
        return (
          pool.upgrading[idIndex(command.building)] === 0 &&
          world.player(command.player).minerals > minerals
        );
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
    const byType = new Map<number, { tried: number; refused: number }>();
    let tried = 0;
    let refused = 0;
    let nulls = 0;
    // Use two matches so an earlier victory does not erase the sample budget.
    for (const seed of [SEED, SEED + 1]) {
      const config = duelMatch(seed, { botPlayers: [1] });
      const match = new HeadlessMatch(config, scriptedAgents(config));
      const world = match.world;
      // Random workers rarely finish a production building. Start with one
      // operational Barracks so research and its cancellation are exercised
      // by the same simulation-acceptance fuzz as all other action heads.
      const start = world.map.starts[0]!;
      const producer = world.placeBuilding(
        EntityType.Barracks,
        0,
        start.tileX + 5,
        start.tileY + 6,
      );
      world.pool.buildState[idIndex(producer)] = BuildState.Complete;
      const eyes = new Eyes(world, 0);
      const rng = new Rng(0xabc);
      const action = allocAction();
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
          if ('units' in command)
            expect(command.units.length).toBeLessThanOrEqual(MAX_COMMAND_UNITS);
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
      match.dispose();
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
    expect(eyes.masks.rowEntityType[rowB * ENTITY_TYPE_COUNT + EntityType.Burstbot]).toBe(1);
    expect(eyes.masks.rowEntityType[rowB * ENTITY_TYPE_COUNT + EntityType.Worker]).toBe(0);
    expect(eyes.masks.rowEntityType[rowP * ENTITY_TYPE_COUNT + EntityType.Worker]).toBe(1);
    expect(eyes.masks.type[ActionType.Build]).toBe(1);
    // Every structure a worker can raise, including Factory and Airport.
    for (const building of BUILDINGS) {
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
