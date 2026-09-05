/**
 * The training environment is the game as the neural bot will meet it.
 *
 * Same seed and same decisions give the same match, so a run is reproducible;
 * a decision executes exactly when the browser would execute it; every
 * decision the masks allow is taken; and the teacher's labels are decisions
 * the student could have made from the same observation.
 */

import { describe, expect, it } from 'vitest';
import { DECISION_TICKS } from '../src/ai/cadence.js';
import { executeTickFor } from '../src/ai/headless.js';
import { actionToInts, allocAction, decode } from '../src/ai/neural/actions.js';
import { sampleUniform } from '../src/ai/neural/random.js';
import { ACTION_INTS, ActionType } from '../src/ai/neural/spec.js';
import { idIndex } from '../src/sim/entities.js';
import { Rng } from '../src/sim/rng.js';
import { EntityType, MapLayout, Order } from '../src/sim/types.js';
import { MatchEnv, parseSlots } from '../tools/ml/env.js';

const SEED = 0x51ce7a11;

function noop(): Int32Array {
  const ints = new Int32Array(ACTION_INTS).fill(-1);
  ints[0] = ActionType.Noop;
  return ints;
}

describe('the training environment', () => {
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
        expect(decode(action, world, slot.frame)).not.toBeNull();
        expect(slot.masks.type[action.type]).toBe(1);
      }
      env.step(new Map());
    }
    expect(labelled).toBeGreaterThan(50);
    expect(types.size).toBeGreaterThanOrEqual(4);
    expect(dropped).toBeLessThan(labelled / 2);
    expect(world.player(0).supplyUsed).toBeGreaterThan(0);
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
