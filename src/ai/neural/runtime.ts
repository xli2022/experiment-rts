/**
 * The model behind `NeuralAgent`, running in a Web Worker.
 *
 * The main thread encodes; the worker infers; nothing waits. A request is a
 * message with the observation's arrays transferred, and its answer comes back
 * by id or not at all: a reply later than `timeoutMs` is a failure the agent
 * counts and moves past, because a hosted bot that stalls the renderer, or
 * worse the lockstep, would be a bug in the wrong place. Model latency feeds
 * nothing but the bot's own reaction time — never `adaptDelay`.
 *
 * `load` refuses a model whose `specVersion` is not this build's: the codec
 * decides what every input means, and a model trained against another one
 * would run happily and play nonsense.
 */

import { SPEC } from './spec.js';
import type { ActRequest, NeuralRuntime, NeuralRuntimeStats } from './agent.js';
import type { ActMessage, FromWorker, InitMessage, ToWorker } from './messages.js';

/** `public/models/policy.json`, written by `rtsml-export`. */
export interface PolicyManifest {
  readonly specVersion: number;
  /** File name beside the manifest. */
  readonly model: string;
  readonly sha256?: string;
  readonly bytes?: number;
  readonly quantized?: boolean;
  readonly evaluation?: unknown;
}

export function isPolicyManifest(value: unknown): value is PolicyManifest {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.specVersion === 'number' && typeof v.model === 'string';
}

/** The part of `Worker` the runtime uses, so a test can stand one in. */
export interface WorkerLike {
  postMessage(message: ToWorker, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((event: MessageEvent<FromWorker>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
}

export interface WorkerRuntimeOptions {
  /** A decision unanswered after this long is failed. */
  readonly timeoutMs?: number;
  /** Loading the model and warming up may take this long. */
  readonly loadTimeoutMs?: number;
  readonly now?: () => number;
}

interface Pending {
  readonly resolve: (ints: Int32Array) => void;
  readonly reject: (error: Error) => void;
  readonly startedMs: number;
  readonly timer: ReturnType<typeof setTimeout>;
}

export class WorkerRuntime implements NeuralRuntime {
  readonly stats: NeuralRuntimeStats = {
    acts: 0,
    failures: 0,
    timeouts: 0,
    lastLatencyMs: 0,
    lastInferenceMs: 0,
    warmupMs: 0,
  };
  /** Told about every failure, for a banner; the agent only counts them. */
  onFailure: ((message: string) => void) | null = null;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private ready: { resolve: () => void; reject: (error: Error) => void } | null = null;
  private terminated = false;
  private readonly timeoutMs: number;
  private readonly now: () => number;

  private constructor(
    private readonly worker: WorkerLike,
    readonly manifest: PolicyManifest,
    options: WorkerRuntimeOptions,
  ) {
    this.timeoutMs = options.timeoutMs ?? 2000;
    this.now = options.now ?? (() => Date.now());
    worker.onmessage = (event) => this.receive(event.data);
    worker.onerror = (event) => this.fail(event.message || 'the inference worker crashed');
  }

  /** Start the worker on a model and wait until it has inferred once. */
  static load(
    worker: WorkerLike,
    init: Omit<InitMessage, 'type'>,
    manifest: PolicyManifest,
    options: WorkerRuntimeOptions = {},
  ): Promise<WorkerRuntime> {
    if (manifest.specVersion !== SPEC.version) {
      worker.terminate();
      return Promise.reject(
        new Error(
          `The neural model was trained for codec version ${manifest.specVersion}; this build is version ${SPEC.version}.`,
        ),
      );
    }
    const runtime = new WorkerRuntime(worker, manifest, options);
    const loadTimeoutMs = options.loadTimeoutMs ?? 30000;
    return new Promise<WorkerRuntime>((resolve, reject) => {
      const timer = setTimeout(() => {
        runtime.fail(`the neural model did not load within ${loadTimeoutMs} ms`);
      }, loadTimeoutMs);
      runtime.ready = {
        resolve: () => {
          clearTimeout(timer);
          resolve(runtime);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        },
      };
      worker.postMessage({ type: 'init', ...init });
    });
  }

  get disposed(): boolean {
    return this.terminated;
  }

  act(request: ActRequest): Promise<Int32Array> {
    if (this.terminated) return Promise.reject(new Error('the neural runtime is disposed'));
    const id = this.nextId++;
    const o = request.observation;
    const m = request.masks;
    const message: ActMessage = {
      type: 'act',
      id,
      entities: o.entities,
      entityMask: o.entityMask,
      grid: o.grid,
      scalars: o.scalars,
      maskType: m.type,
      maskSelection: m.selection,
      maskTarget: m.target,
      maskCell: m.cell,
      maskBuildCell: m.buildCell,
      maskRowEntityType: m.rowEntityType,
      maskBuildType: m.buildType,
      temperature: request.temperature,
    };
    const transfer = [
      o.entities.buffer,
      o.entityMask.buffer,
      o.grid.buffer,
      o.scalars.buffer,
      m.type.buffer,
      m.selection.buffer,
      m.target.buffer,
      m.cell.buffer,
      m.buildCell.buffer,
      m.rowEntityType.buffer,
      m.buildType.buffer,
    ];
    this.stats.acts++;
    return new Promise<Int32Array>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return;
        this.stats.timeouts++;
        this.stats.failures++;
        this.onFailure?.(`decision ${id} was not answered within ${this.timeoutMs} ms`);
        reject(new Error('the neural model timed out'));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, startedMs: this.now(), timer });
      this.worker.postMessage(message, transfer);
    });
  }

  private receive(message: FromWorker): void {
    switch (message.type) {
      case 'ready': {
        this.stats.warmupMs = message.warmupMs;
        const ready = this.ready;
        this.ready = null;
        ready?.resolve();
        return;
      }
      case 'result': {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        clearTimeout(pending.timer);
        this.stats.lastLatencyMs = this.now() - pending.startedMs;
        this.stats.lastInferenceMs = message.ms;
        pending.resolve(message.action);
        return;
      }
      case 'error': {
        if (message.id === undefined) {
          this.fail(message.message);
          return;
        }
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        clearTimeout(pending.timer);
        this.stats.failures++;
        this.onFailure?.(message.message);
        pending.reject(new Error(message.message));
        return;
      }
    }
  }

  /** The worker is gone or never came: fail everything, loading included. */
  private fail(message: string): void {
    const ready = this.ready;
    this.ready = null;
    ready?.reject(new Error(message));
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      this.stats.failures++;
      pending.reject(new Error(message));
    }
    this.pending.clear();
    if (ready === null) this.onFailure?.(message);
    this.dispose();
  }

  dispose(): void {
    if (this.terminated) return;
    this.terminated = true;
    this.worker.terminate();
    const ready = this.ready;
    this.ready = null;
    ready?.reject(new Error('the neural runtime was disposed while loading'));
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error('the neural runtime is disposed'));
    }
    this.pending.clear();
  }
}
