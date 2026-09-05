/**
 * The worker runtime's side of the protocol: decisions go out by id with their
 * arrays transferred, answers come back by id or time out, and a worker that
 * dies takes every outstanding decision with it — never the match.
 */

import { describe, expect, it } from 'vitest';
import { allocMasks } from '../src/ai/neural/actions.js';
import type { ActRequest } from '../src/ai/neural/agent.js';
import type { ActMessage, FromWorker, ToWorker } from '../src/ai/neural/messages.js';
import { allocObservation } from '../src/ai/neural/observation.js';
import {
  isPolicyManifest,
  WorkerRuntime,
  type WorkerLike,
  type WorkerRuntimeOptions,
} from '../src/ai/neural/runtime.js';
import { ACTION_INTS, SPEC } from '../src/ai/neural/spec.js';

class FakeWorker implements WorkerLike {
  readonly posted: ToWorker[] = [];
  readonly transfers: (Transferable[] | undefined)[] = [];
  terminated = false;
  onmessage: ((event: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  postMessage(message: ToWorker, transfer?: Transferable[]): void {
    this.posted.push(message);
    this.transfers.push(transfer);
  }

  terminate(): void {
    this.terminated = true;
  }

  reply(message: FromWorker): void {
    this.onmessage?.({ data: message } as MessageEvent<FromWorker>);
  }

  crash(message: string): void {
    this.onerror?.({ message } as ErrorEvent);
  }

  lastAct(): ActMessage {
    const last = this.posted[this.posted.length - 1]!;
    if (last.type !== 'act') throw new Error('last message was not an act');
    return last;
  }
}

const manifest = { specVersion: SPEC.version, model: 'policy.onnx' };
const init = { modelUrl: 'models/policy.onnx', wasmUrl: 'ort.wasm', numThreads: 1 };

async function loaded(
  options: WorkerRuntimeOptions = {},
): Promise<{ worker: FakeWorker; runtime: WorkerRuntime }> {
  const worker = new FakeWorker();
  const loading = WorkerRuntime.load(worker, init, manifest, options);
  expect(worker.posted[0]).toEqual({ type: 'init', ...init });
  worker.reply({ type: 'ready', warmupMs: 7 });
  const runtime = await loading;
  expect(runtime.stats.warmupMs).toBe(7);
  return { worker, runtime };
}

function request(seq = 1): ActRequest {
  return { seq, player: 1, observation: allocObservation(), masks: allocMasks(), temperature: 0.8 };
}

describe('loading', () => {
  it('refuses a model for another codec version without starting it', async () => {
    const worker = new FakeWorker();
    await expect(
      WorkerRuntime.load(worker, init, { specVersion: SPEC.version + 1, model: 'x' }),
    ).rejects.toThrow(/codec version/);
    expect(worker.posted).toHaveLength(0);
    expect(worker.terminated).toBe(true);
  });

  it('fails when the worker reports an error before it is ready', async () => {
    const worker = new FakeWorker();
    const loading = WorkerRuntime.load(worker, init, manifest);
    worker.reply({ type: 'error', message: 'no such model' });
    await expect(loading).rejects.toThrow('no such model');
    expect(worker.terminated).toBe(true);
  });

  it('gives up after the load timeout', async () => {
    const worker = new FakeWorker();
    await expect(WorkerRuntime.load(worker, init, manifest, { loadTimeoutMs: 10 })).rejects.toThrow(
      /did not load/,
    );
    expect(worker.terminated).toBe(true);
  });

  it('recognises a manifest', () => {
    expect(isPolicyManifest(manifest)).toBe(true);
    expect(isPolicyManifest({ model: 'x' })).toBe(false);
    expect(isPolicyManifest(null)).toBe(false);
  });
});

describe('acting', () => {
  it('sends the observation with its buffers transferred and resolves by id', async () => {
    let clock = 100;
    const { worker, runtime } = await loaded({ now: () => clock });
    const req = request();
    const answer = runtime.act(req);
    const sent = worker.lastAct();
    expect(sent.entities).toBe(req.observation.entities);
    expect(sent.maskBuildCell).toBe(req.masks.buildCell);
    expect(sent.temperature).toBe(0.8);
    expect(worker.transfers[worker.transfers.length - 1]).toHaveLength(11);
    clock = 112;
    const action = new Int32Array(ACTION_INTS).fill(-1);
    action[0] = 3;
    worker.reply({ type: 'result', id: sent.id, action, ms: 9 });
    await expect(answer).resolves.toBe(action);
    expect(runtime.stats.acts).toBe(1);
    expect(runtime.stats.lastLatencyMs).toBe(12);
  });

  it('keeps two decisions apart by id', async () => {
    const { worker, runtime } = await loaded();
    const first = runtime.act(request(1));
    const firstId = worker.lastAct().id;
    const second = runtime.act(request(2));
    const secondId = worker.lastAct().id;
    expect(secondId).not.toBe(firstId);
    const a = new Int32Array(ACTION_INTS).fill(2);
    const b = new Int32Array(ACTION_INTS).fill(5);
    worker.reply({ type: 'result', id: secondId, action: b, ms: 1 });
    worker.reply({ type: 'result', id: firstId, action: a, ms: 1 });
    await expect(first).resolves.toBe(a);
    await expect(second).resolves.toBe(b);
  });

  it('times out an unanswered decision and ignores its late answer', async () => {
    const { worker, runtime } = await loaded({ timeoutMs: 10 });
    const failures: string[] = [];
    runtime.onFailure = (message) => failures.push(message);
    const answer = runtime.act(request());
    const id = worker.lastAct().id;
    await expect(answer).rejects.toThrow(/timed out/);
    expect(runtime.stats.timeouts).toBe(1);
    expect(runtime.stats.failures).toBe(1);
    expect(failures).toHaveLength(1);
    worker.reply({ type: 'result', id, action: new Int32Array(ACTION_INTS), ms: 1 });
    expect(runtime.disposed).toBe(false);
  });

  it('fails one decision on its own error and leaves the others alone', async () => {
    const { worker, runtime } = await loaded();
    const first = runtime.act(request(1));
    const firstId = worker.lastAct().id;
    const second = runtime.act(request(2));
    const secondId = worker.lastAct().id;
    worker.reply({ type: 'error', id: firstId, message: 'bad tensor' });
    await expect(first).rejects.toThrow('bad tensor');
    const action = new Int32Array(ACTION_INTS);
    worker.reply({ type: 'result', id: secondId, action, ms: 1 });
    await expect(second).resolves.toBe(action);
    expect(runtime.stats.failures).toBe(1);
  });

  it('takes every outstanding decision down with a crashed worker', async () => {
    const { worker, runtime } = await loaded();
    const failures: string[] = [];
    runtime.onFailure = (message) => failures.push(message);
    const a = runtime.act(request(1));
    const b = runtime.act(request(2));
    worker.crash('out of memory');
    await expect(a).rejects.toThrow('out of memory');
    await expect(b).rejects.toThrow('out of memory');
    expect(runtime.disposed).toBe(true);
    expect(worker.terminated).toBe(true);
    expect(failures).toEqual(['out of memory']);
    await expect(runtime.act(request(3))).rejects.toThrow(/disposed/);
  });

  it('disposes once, rejecting what was in flight', async () => {
    const { worker, runtime } = await loaded();
    const pending = runtime.act(request());
    runtime.dispose();
    runtime.dispose();
    await expect(pending).rejects.toThrow(/disposed/);
    expect(worker.terminated).toBe(true);
    await expect(runtime.act(request())).rejects.toThrow(/disposed/);
  });
});
