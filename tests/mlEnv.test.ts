/**
 * The training environment is the game as the neural bot will meet it.
 *
 * Same seed and same decisions give the same match, so a run is reproducible;
 * a decision executes exactly when the browser would execute it; every
 * decision the masks allow is taken; and the teacher's labels are decisions
 * the student could have made from the same observation.
 *
 * "From the same observation" is load-bearing in that last one: a label is
 * checked against the masks it was encoded with, never against a later world.
 */

import { describe, expect, it, vi } from 'vitest';
import { DECISION_TICKS } from '../src/ai/cadence.js';
import { executeTickFor } from '../src/ai/headless.js';
import { actionToInts, allocAction, encode, legalise } from '../src/ai/neural/actions.js';
import { sampleUniform } from '../src/ai/neural/random.js';
import { ScriptedAgent } from '../src/ai/scripted.js';
import {
  ACTION_INTS,
  ActionType,
  ENTITY_FEATURE_COUNT,
  ENTITY_FEATURES,
  SCALARS,
} from '../src/ai/neural/spec.js';
import { STARTING_WORKERS } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { fromInt } from '../src/sim/fixed.js';
import { UNOCCUPIED } from '../src/sim/map.js';
import { Rng } from '../src/sim/rng.js';
import { BuildState, EntityType, MapLayout, Order, Tile } from '../src/sim/types.js';
import { MatchEnv, parseSlots } from '../tools/ml/env.js';

const SEED = 0x51ce7a11;

function noop(): Int32Array {
  const ints = new Int32Array(ACTION_INTS).fill(-1);
  ints[0] = ActionType.Noop;
  return ints;
}

describe('the training environment', () => {
  it('drops a stale Build intent instead of labelling a nearby new foundation', () => {
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('teacher,idle'),
    });
    const world = env.world;
    world.map.tiles.fill(Tile.Ground);
    world.map.occupied.fill(UNOCCUPIED);
    world.player(0).minerals = 1000;
    const site = world.placeBuilding(EntityType.Airport, 0, 40, 40);
    world.pool.buildState[idIndex(site)] = BuildState.Complete;
    const worker = world.pool.spawn(EntityType.Worker, 0, fromInt(39), fromInt(39));
    const command = {
      type: CommandType.Build as const,
      player: 0,
      worker,
      building: EntityType.Airport,
      tileX: 40,
      tileY: 40,
    };
    const think = vi.spyOn(ScriptedAgent.prototype, 'act').mockImplementation(() => [command]);
    try {
      env.step(new Map());
      const slot = env.observe(0);
      const action = allocAction();
      expect(encode(command, slot.frame, action)).toBe(true);
      expect(legalise(action, slot.masks)).toBe(true); // Coarse cell has free nearby tiles.
      expect(slot.label[0]).toBe(-1);
      expect(env.teacherCoverage(0)).toMatchObject({
        nonNoop: 0,
        dropped: 1,
        droppedCommands: { Build: 1 },
      });
    } finally {
      think.mockRestore();
      env.dispose();
    }
  });

  it('pairs a teacher label with the actual decision-source tick and neural issue delay', () => {
    let sourceTick = -1;
    const think = vi.spyOn(ScriptedAgent.prototype, 'act').mockImplementation((world, player) => {
      if (world.tick !== DECISION_TICKS) return [];
      sourceTick = world.tick;
      const index = world.pool.type.findIndex(
        (type, i) => type === EntityType.Worker && world.pool.owner[i] === player,
      );
      return [{ type: CommandType.Hold, player, units: [world.pool.idAt(index)] }];
    });
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('teacher,idle'),
    });
    try {
      expect(env.observe(0).label[0]).toBe(-1);
      expect(env.step(new Map()).issued[0]).toBe(0);
      const labelled = env.observe(0);
      expect(sourceTick).toBe(DECISION_TICKS);
      expect(labelled.frame.tick).toBe(sourceTick);
      expect(labelled.observation.scalars[SCALARS.indexOf('tick')]).toBeCloseTo(
        sourceTick / 24000,
        8,
      );
      expect(labelled.label[0]).toBe(ActionType.Hold);
      const worker = idIndex(labelled.frame.rows[labelled.label[5]!]!);
      const executesAt = executeTickFor(sourceTick + 1);
      expect(env.step(new Map()).issued[0]).toBe(1);
      expect(env.tick).toBeLessThan(executesAt);
      expect(env.world.pool.order[worker]).not.toBe(Order.Hold);
      env.step(new Map());
      expect(env.tick).toBeGreaterThan(executesAt);
      expect(env.world.pool.order[worker]).toBe(Order.Hold);
      const coverage = env.teacherCoverage(0);
      expect(coverage).toMatchObject({
        decisions: 3,
        valid: 3,
        nonNoop: 1,
        dropped: 0,
        actions: { Hold: 1, Noop: 2 },
      });
      coverage.actions.Hold = 99;
      expect(env.teacherCoverage(0).actions.Hold).toBe(1);
    } finally {
      env.dispose();
      think.mockRestore();
    }
  });

  it('executes upgrade and cancellation decisions and remembers their building selection', () => {
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,idle'),
    });
    const world = env.world;
    const id = world.pool.spawn(EntityType.Barracks, 0, fromInt(30), fromInt(30));
    const index = idIndex(id);
    world.pool.buildState[index] = BuildState.Complete;
    world.player(0).minerals = 1000;
    for (const type of [ActionType.UpgradeBuilding, ActionType.CancelUpgrade]) {
      const slot = env.observe(0);
      expect(slot.masks.type[type]).toBe(1);
      const action = allocAction();
      action.type = type;
      action.selection[0] = slot.frame.rowOf.get(id)!;
      const ints = new Int32Array(ACTION_INTS);
      actionToInts(action, ints);
      expect(env.step(new Map([[0, ints]])).issued[0]).toBe(1);
      const recent = env.observe(0);
      expect(
        recent.observation.scalars[
          SCALARS.indexOf(
            type === ActionType.UpgradeBuilding ? 'prev:UpgradeBuilding' : 'prev:CancelUpgrade',
          )
        ],
      ).toBe(1);
      expect(
        recent.observation.entities[
          recent.frame.rowOf.get(id)! * ENTITY_FEATURE_COUNT +
            ENTITY_FEATURES.indexOf('inLastCommand')
        ],
      ).toBe(1);
      env.step(new Map([[0, noop()]]));
      expect(world.pool.upgrading[index]).toBe(type === ActionType.UpgradeBuilding ? 1 : 0);
    }
    expect(world.player(0).minerals).toBe(1000);
    env.dispose();
  });

  it('is a pure function of the seed and the decisions', () => {
    const make = () =>
      new MatchEnv({ seed: SEED, layout: MapLayout.Lanes, slots: parseSlots('policy,scripted') });
    const a = make();
    const b = make();
    const rng = new Rng(3);
    const action = allocAction();
    const ints = new Int32Array(ACTION_INTS);
    for (let step = 0; step < 300; step++) {
      const slot = a.observe(0);
      b.observe(0);
      sampleUniform(slot.masks, rng, action);
      actionToInts(action, ints);
      const ra = a.step(new Map([[0, ints]]));
      const rb = b.step(new Map([[0, ints]]));
      expect(rb.rewards[0]).toBe(ra.rewards[0]);
      if (a.world.checksum() !== b.world.checksum())
        throw new Error(`runs diverged at step ${step}`);
    }
    expect(a.world.player(1).supplyUsed).toBeGreaterThan(0);
  });

  it('executes a decision when the browser would', () => {
    // A decision made at the boundary is issued on the first tick after it,
    // exactly as the neural agent's reply is, so it lands on the runner's
    // schedule for that tick.
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,idle'),
    });
    for (let k = 0; k < 3; k++) env.step(new Map([[0, noop()]]));
    const boundary = env.tick;
    expect(boundary).toBe(3 * DECISION_TICKS);
    const slot = env.observe(0);
    const world = env.world;
    let workerRow = -1;
    for (let r = 0; r < slot.frame.rows.length; r++) {
      const id = slot.frame.rows[r]!;
      if (
        id >= 0 &&
        world.pool.type[idIndex(id)] === EntityType.Worker &&
        world.pool.owner[idIndex(id)] === 0
      ) {
        workerRow = r;
        break;
      }
    }
    expect(workerRow).toBeGreaterThanOrEqual(0);
    const action = allocAction();
    action.type = ActionType.Move;
    action.selection[0] = workerRow;
    action.cell = 25 * 40 + 25;
    action.sub = 5;
    const ints = new Int32Array(ACTION_INTS);
    actionToInts(action, ints);
    const worker = idIndex(slot.frame.rows[workerRow]!);
    // Issued on tick boundary + 1, so it runs on the runner's schedule for
    // that tick — inside the second step from here, not the first.
    const executesAt = executeTickFor(boundary + 1);
    expect(executesAt).toBe(boundary + 6);
    env.step(new Map([[0, ints]]));
    expect(env.tick).toBe(boundary + DECISION_TICKS);
    expect(world.pool.order[worker]).toBe(Order.None);
    env.step(new Map([[0, noop()]]));
    expect(env.tick).toBeGreaterThanOrEqual(executesAt);
    expect(world.pool.order[worker]).toBe(Order.Move);
  });

  it('takes every decision the masks allow', () => {
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,scripted'),
    });
    const rng = new Rng(11);
    const action = allocAction();
    const ints = new Int32Array(ACTION_INTS);
    let decided = 0;
    let issued = 0;
    for (let step = 0; step < 1500 && !env.done; step++) {
      const slot = env.observe(0);
      sampleUniform(slot.masks, rng, action);
      actionToInts(action, ints);
      const result = env.step(new Map([[0, ints]]));
      if (action.type !== ActionType.Noop) {
        decided++;
        // A legal decision always decodes into a command the driver issues.
        expect(result.issued[0]).toBe(1);
        issued++;
      }
    }
    // Random play loses to the scripted bot well inside the step budget.
    expect(decided).toBeGreaterThan(500);
    expect(issued).toBe(decided);
  });

  it('labels every decision of a teacher with what the student could have said', () => {
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('teacher,scripted@20'),
    });
    const world = env.world;
    const action = allocAction();
    const types = new Set<number>();
    let labelled = 0;
    let dropped = 0;
    let peakSupply = world.player(0).supplyUsed;
    for (let step = 0; step < 2500 && !env.done; step++) {
      const slot = env.observe(0);
      if (slot.label[0] === -1) {
        // The teacher said something the student could not have; not a lesson.
        dropped++;
      } else if (slot.label[0] !== ActionType.Noop) {
        labelled++;
        types.add(slot.label[0]!);
        // The label decodes against the very frame it was made in.
        action.type = slot.label[0]!;
        action.entityType = slot.label[1]!;
        action.target = slot.label[2]!;
        action.cell = slot.label[3]!;
        action.sub = slot.label[4]!;
        for (let k = 0; k < action.selection.length; k++) action.selection[k] = slot.label[5 + k]!;
        // Masks and row indices belong to the same decision boundary as the
        // command. The teacher issues it on the following tick, like a policy.
        expect(legalise(action, slot.masks)).toBe(true);
        expect(slot.masks.type[action.type]).toBe(1);
      }
      env.step(new Map());
      peakSupply = Math.max(peakSupply, world.player(0).supplyUsed);
    }
    expect(labelled).toBeGreaterThan(50);
    expect(types.size).toBeGreaterThanOrEqual(4);
    expect(types.has(ActionType.UpgradeBuilding)).toBe(true);
    expect(dropped).toBeLessThan(labelled / 2);
    // The teacher can lose this match; it must have developed beyond its
    // opening units while producing useful labels, not survive the final tick.
    expect(peakSupply).toBeGreaterThan(STARTING_WORKERS);
  });

  it('rewards the winner, charges for time, and resets on the next seed', () => {
    const env = new MatchEnv({
      seed: SEED,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,scripted'),
      maxTicks: 400,
      shaping: 0,
    });
    let total = 0;
    let steps = 0;
    let last = env.step(new Map([[0, noop()]]));
    total += last.rewards[0]!;
    steps++;
    while (!last.done) {
      last = env.step(new Map([[0, noop()]]));
      total += last.rewards[0]!;
      steps++;
    }
    expect(last.truncated).toBe(true);
    expect(steps).toBe(100);
    expect(total).toBeCloseTo(-1e-4 * steps, 6);
    const before = env.world;
    env.reset();
    expect(env.world).not.toBe(before);
    expect(env.tick).toBe(0);
    expect(env.world.config.seed).toBe((SEED + 1) >>> 0);
  });

  it('parses slot lists', () => {
    expect(parseSlots('policy,scripted@20,teacher,idle')).toEqual([
      { kind: 'policy' },
      { kind: 'scripted', thinkInterval: 20 },
      { kind: 'teacher' },
      { kind: 'idle' },
    ]);
    expect(() => parseSlots('human')).toThrow();
  });
});
