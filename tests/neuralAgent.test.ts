/**
 * The neural agent around a runtime it does not wait for: one request per
 * decision boundary, none while one is outstanding, the answer decoded
 * against the frame it was asked in, and a runtime that fails counted and
 * moved past.
 */

import { describe, expect, it } from 'vitest';
import type { Agent } from '../src/ai/agent.js';
import { AgentDriver } from '../src/ai/driver.js';
import { createHostedAgents } from '../src/ai/factory.js';
import { actionToInts, allocAction, type Action, type Masks } from '../src/ai/neural/actions.js';
import { NeuralAgent, type ActRequest, type NeuralRuntime } from '../src/ai/neural/agent.js';
import { sampleUniform } from '../src/ai/neural/random.js';
import { ACTION_INTS, ActionType, DECISION_TICKS, GRID, N_ENT } from '../src/ai/neural/spec.js';
import { LockstepRunner } from '../src/net/lockstep.js';
import { SoloTransport } from '../src/net/localTransport.js';
import { duelMatch } from '../src/sim/match.js';
import { Rng } from '../src/sim/rng.js';
import { Simulation } from '../src/sim/tick.js';
import { BotKind } from '../src/sim/types.js';

const SEED = 0x51ce7a11;
const MS_PER_TICK = 50;
const CELLS = GRID * GRID;

interface Pending {
  readonly request: ActRequest;
  readonly resolve: (ints: Int32Array) => void;
  readonly reject: (error: Error) => void;
}

class FakeRuntime implements NeuralRuntime {
  readonly requests: ActRequest[] = [];
  private pending: Pending[] = [];
  disposed = false;

  act(request: ActRequest): Promise<Int32Array> {
    this.requests.push(request);
    return new Promise((resolve, reject) => this.pending.push({ request, resolve, reject }));
  }

  get inFlight(): number {
    return this.pending.length;
  }

  /** Answer everything outstanding with a uniformly random legal decision. */
  answer(rng: Rng): void {
    for (const p of this.pending.splice(0)) {
      const action = allocAction();
      sampleUniform(p.request.masks, rng, action);
      p.resolve(intsOf(action));
    }
  }

  answerWith(action: Action): void {
    for (const p of this.pending.splice(0)) p.resolve(intsOf(action));
  }

  fail(): void {
    for (const p of this.pending.splice(0)) p.reject(new Error('no model'));
  }

  dispose(): void {
    this.disposed = true;
  }
}

function intsOf(action: Action): Int32Array {
  const ints = new Int32Array(ACTION_INTS);
  actionToInts(action, ints);
  return ints;
}

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

function firstLegal(mask: Uint8Array, offset: number, length: number): number {
  for (let k = 0; k < length; k++) if (mask[offset + k] === 1) return k;
  return -1;
}

function stage(agent: Agent) {
  const config = duelMatch(SEED, { botPlayers: [1], kind: BotKind.Neural });
  const sim = new Simulation(config);
  let driver: AgentDriver;
  const runner = new LockstepRunner(sim, new SoloTransport(), {
    onStep: () => driver.tick(sim.world),
  });
  driver = new AgentDriver(createHostedAgents(config, 0, { neural: () => agent }), (c, p) =>
    runner.issue(c, p),
  );
  return { sim, runner, driver, tick: () => runner.update(MS_PER_TICK) };
}

describe('the neural agent', () => {
  it('asks once per decision boundary and skips boundaries while an answer is outstanding', async () => {
    const runtime = new FakeRuntime();
    const agent = new NeuralAgent(runtime);
    const { tick, sim } = stage(agent);
    for (let t = 0; t < 40; t++) tick();
    expect(sim.world.tick).toBe(40);
    expect(runtime.requests).toHaveLength(1);
    expect(agent.stats.decisions).toBe(1);
    expect(agent.stats.skipped).toBe(40 / DECISION_TICKS - 1);
    expect(runtime.requests[0]!.player).toBe(1);

    runtime.answer(new Rng(3));
    await flush();
    for (let t = 0; t < DECISION_TICKS; t++) tick();
    expect(runtime.requests).toHaveLength(2);
    expect(agent.stats.skipped).toBe(40 / DECISION_TICKS - 1);
  });

  it('decodes an answer against the frame it was asked in, and the driver issues it', async () => {
    const runtime = new FakeRuntime();
    const agent = new NeuralAgent(runtime);
    const { tick, driver } = stage(agent);
    for (let t = 0; t < DECISION_TICKS; t++) tick();
    const masks: Masks = runtime.requests[0]!.masks;
    const action = allocAction();
    action.type = ActionType.Move;
    action.selection[0] = firstLegal(masks.selection, ActionType.Move * N_ENT, N_ENT);
    action.cell = firstLegal(masks.cell, ActionType.Move * CELLS, CELLS);
    action.sub = 5;
    expect(action.selection[0]).toBeGreaterThanOrEqual(0);
    expect(action.cell).toBeGreaterThanOrEqual(0);
    // The answer arrives much later than it was asked for; the frame it was
    // made in is what it is read against.
    for (let t = 0; t < 30; t++) tick();
    runtime.answerWith(action);
    await flush();
    tick();
    expect(agent.stats.issued).toBe(1);
    expect(driver.statsFor(1)!.issued).toBe(1);
    expect(driver.statsFor(1)!.rejected).toBe(0);
  });

  it('plays a match through the hosted path when every answer is a legal decision', async () => {
    const runtime = new FakeRuntime();
    const agent = new NeuralAgent(runtime);
    const { tick, driver, runner } = stage(agent);
    const rng = new Rng(11);
    for (let t = 0; t < 1600; t++) {
      tick();
      if (runtime.inFlight > 0) {
        runtime.answer(rng);
        await flush();
      }
    }
    expect(agent.stats.decisions).toBeGreaterThan(300);
    expect(agent.stats.skipped).toBe(0);
    expect(agent.stats.failed).toBe(0);
    expect(agent.stats.issued).toBeGreaterThan(30);
    // The last answer may still be waiting for the next tick to be read.
    expect(agent.stats.issued + agent.stats.noops).toBeGreaterThanOrEqual(
      agent.stats.decisions - 1,
    );
    expect(driver.statsFor(1)!.issued).toBe(agent.stats.issued);
    expect(driver.statsFor(1)!.rejected).toBe(0);
    expect(runner.droppedByBudget).toBe(0);
    // Every request carries arrays of its own: they are the runtime's to keep.
    expect(new Set(runtime.requests.map((r) => r.observation.entities)).size).toBe(
      runtime.requests.length,
    );
  });

  it('counts a runtime that fails and keeps asking', async () => {
    const runtime = new FakeRuntime();
    const agent = new NeuralAgent(runtime);
    const { tick } = stage(agent);
    for (let t = 0; t < 40; t++) {
      tick();
      if (runtime.inFlight > 0) {
        runtime.fail();
        await flush();
      }
    }
    expect(agent.stats.failed).toBe(40 / DECISION_TICKS);
    expect(agent.stats.skipped).toBe(0);
    expect(agent.stats.issued).toBe(0);
  });

  it('asks nothing once disposed, and disposes the runtime', () => {
    const runtime = new FakeRuntime();
    const agent = new NeuralAgent(runtime);
    const { tick } = stage(agent);
    for (let t = 0; t < DECISION_TICKS; t++) tick();
    expect(runtime.requests).toHaveLength(1);
    agent.dispose();
    expect(runtime.disposed).toBe(true);
    for (let t = 0; t < 40; t++) tick();
    expect(runtime.requests).toHaveLength(1);
  });
});
