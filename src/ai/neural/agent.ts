/**
 * The neural bot as an `Agent`.
 *
 * It sees what a human sees (`observation.ts`), asks a model — behind a
 * `NeuralRuntime`, which in the browser is a Web Worker running ONNX Runtime
 * — and says what the model chose (`actions.ts`), one command per decision at
 * most. Inference is asynchronous and the world does not wait for it: a
 * decision is posted every `DECISION_TICKS`, its answer is picked up by the
 * next `act`, and a decision that comes due while one is still in flight is
 * skipped rather than queued, so a slow machine only slows the bot's reactions
 * and never the match.
 *
 * The frame a decision was made in is kept until its answer arrives. Handles
 * are generation-checked, so a unit that died in the meantime is dropped and
 * one that moved is still itself.
 */

import { CommandType, type Command } from '../../sim/commands.js';
import type { EntityId, PlayerId } from '../../sim/types.js';
import type { World } from '../../sim/world.js';
import { Visibility } from '../../vision/visibility.js';
import type { Agent } from '../agent.js';
import {
  actionFromInts,
  allocAction,
  allocMasks,
  computeMasks,
  decode,
  type Action,
  type Masks,
} from './actions.js';
import { allocFrame, type Frame } from './frame.js';
import { EntityMemory } from './memory.js';
import {
  allocObservation,
  ObservationEncoder,
  type Observation,
  type RecentActions,
} from './observation.js';
import { ACTION_INTS, ActionType, DECISION_TICKS } from './spec.js';

/** One decision's worth of inputs, handed to the runtime. The arrays are the runtime's to keep. */
export interface ActRequest {
  readonly seq: number;
  readonly player: PlayerId;
  readonly observation: Observation;
  readonly masks: Masks;
  readonly temperature: number;
}

/** What a runtime can say about itself, for the debug overlay and the smoke test. */
export interface NeuralRuntimeStats {
  acts: number;
  failures: number;
  timeouts: number;
  /** Round trip of the last answered decision, main thread to main thread. */
  lastLatencyMs: number;
  /** The model alone, inside the worker, for the last answered decision. */
  lastInferenceMs: number;
  warmupMs: number;
}

/** Whatever runs the model. Resolves to `ACTION_INTS` integers; see `actionToInts`. */
export interface NeuralRuntime {
  act(request: ActRequest): Promise<Int32Array>;
  dispose?(): void;
  readonly stats?: NeuralRuntimeStats;
}

export interface NeuralOptions {
  /** Sampling temperature. One is the policy as trained; lower is greedier. */
  readonly temperature?: number;
}

export interface NeuralStats {
  decisions: number;
  noops: number;
  skipped: number;
  failed: number;
  issued: number;
  lastLatencyMs: number;
}

interface Pending {
  readonly frame: Frame;
  readonly startedMs: number;
}

export class NeuralAgent implements Agent {
  readonly stats: NeuralStats = {
    decisions: 0,
    noops: 0,
    skipped: 0,
    failed: 0,
    issued: 0,
    lastLatencyMs: 0,
  };
  private readonly temperature: number;
  private vis: Visibility | null = null;
  private mem: EntityMemory | null = null;
  private encoder: ObservationEncoder | null = null;
  private seq = 0;
  private inFlight: Pending | null = null;
  private reply: { frame: Frame; ints: Int32Array } | null = null;
  private readonly action: Action = allocAction();
  private readonly recent: RecentActions & { lastUnits: Set<EntityId> } = {
    prevType: ActionType.Noop,
    sinceNonNoop: 0,
    recentCommands: 0,
    lastUnits: new Set(),
  };
  private readonly commandTicks: number[] = [];
  private disposed = false;

  constructor(
    private readonly runtime: NeuralRuntime,
    options: NeuralOptions = {},
    private readonly now: () => number = () => Date.now(),
  ) {
    this.temperature = options.temperature ?? 1;
  }

  act(world: World, player: PlayerId): Command[] {
    if (this.disposed) return [];
    if (!this.vis || !this.mem || !this.encoder) {
      this.vis = new Visibility(world.map);
      this.mem = new EntityMemory(player);
      this.encoder = new ObservationEncoder(world, player);
    }
    this.vis.update(world, player);
    this.mem.update(world, this.vis);

    const out: Command[] = [];
    const reply = this.reply;
    if (reply) {
      this.reply = null;
      actionFromInts(reply.ints, this.action);
      const command = decode(this.action, world, reply.frame);
      this.recent.prevType = this.action.type;
      this.recent.lastUnits.clear();
      if (command === null) {
        this.stats.noops++;
        this.recent.sinceNonNoop++;
      } else {
        this.recent.sinceNonNoop = 0;
        if ('units' in command) for (const id of command.units) this.recent.lastUnits.add(id);
        if ('worker' in command) this.recent.lastUnits.add(command.worker);
        // A Train, CancelTrain or SetRally names a building handle; a Build's
        // `building` is a type, not a handle.
        if (
          command.type === CommandType.Train ||
          command.type === CommandType.CancelTrain ||
          command.type === CommandType.SetRally
        ) {
          this.recent.lastUnits.add(command.building);
        }
        this.commandTicks.push(world.tick);
        this.stats.issued++;
        out.push(command);
      }
    }
    while (this.commandTicks.length > 0 && world.tick - this.commandTicks[0]! > 200)
      this.commandTicks.shift();
    this.recent.recentCommands = this.commandTicks.length;

    if (world.tick % DECISION_TICKS === 0) {
      if (this.inFlight) {
        this.stats.skipped++;
      } else {
        this.request(world, player);
      }
    }
    return out;
  }

  private request(world: World, player: PlayerId): void {
    const observation = allocObservation();
    const masks = allocMasks();
    const frame = allocFrame();
    this.encoder!.encode(this.vis!, this.mem!, this.recent, observation, frame);
    computeMasks(world, frame, this.vis!, this.mem!, masks);
    const seq = ++this.seq;
    const pending: Pending = { frame, startedMs: this.now() };
    this.inFlight = pending;
    this.stats.decisions++;
    this.runtime
      .act({ seq, player, observation, masks, temperature: this.temperature })
      .then((ints) => {
        if (this.inFlight !== pending) return;
        this.inFlight = null;
        this.stats.lastLatencyMs = this.now() - pending.startedMs;
        if (ints.length < ACTION_INTS)
          throw new Error(`runtime returned ${ints.length} ints, expected ${ACTION_INTS}`);
        this.reply = { frame, ints };
      })
      .catch(() => {
        if (this.inFlight === pending) this.inFlight = null;
        this.stats.failed++;
      });
  }

  /** The runtime's own numbers, when it keeps any. */
  runtimeStats(): NeuralRuntimeStats | null {
    return this.runtime.stats ? { ...this.runtime.stats } : null;
  }

  dispose(): void {
    this.disposed = true;
    this.runtime.dispose?.();
  }
}
