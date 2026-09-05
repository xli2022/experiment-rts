/**
 * The inference worker: ONNX Runtime Web, one model, one decision at a time.
 *
 * Runs under the WebAssembly backend on a single thread — GitHub Pages sends
 * no cross-origin-isolation headers, so `SharedArrayBuffer` and with it the
 * threaded build are unavailable — and answers each `act` with the integers
 * the graph decided. The noise the graph samples with is filled here from
 * `crypto.getRandomValues`, so the main thread never touches randomness.
 */

import * as ort from 'onnxruntime-web/wasm';
import { ENTITY_TYPE_COUNT } from '../../sim/types.js';
import type { ActMessage, FromWorker, InitMessage, ToWorker } from './messages.js';
import { fillGumbel } from './noise.js';
import {
  ACTION_INTS,
  ACTION_TYPE_COUNT,
  ENTITY_FEATURE_COUNT,
  GRID,
  GRID_CHANNEL_COUNT,
  N_ENT,
  NOISE_LEN,
  SCALAR_COUNT,
} from './spec.js';

const scope = self as unknown as {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
};

const CELLS = GRID * GRID;
let session: ort.InferenceSession | null = null;
const noise = new Float32Array(NOISE_LEN);
const words = new Uint32Array(NOISE_LEN);
const temperature = new Float32Array(1);

function feedsFor(m: ActMessage): Record<string, ort.Tensor> {
  crypto.getRandomValues(words);
  fillGumbel(noise, words);
  temperature[0] = m.temperature;
  return {
    entities: new ort.Tensor('float32', m.entities, [1, N_ENT, ENTITY_FEATURE_COUNT]),
    entity_mask: new ort.Tensor('uint8', m.entityMask, [1, N_ENT]),
    grid: new ort.Tensor('float32', m.grid, [1, GRID_CHANNEL_COUNT, GRID, GRID]),
    scalars: new ort.Tensor('float32', m.scalars, [1, SCALAR_COUNT]),
    mask_type: new ort.Tensor('uint8', m.maskType, [1, ACTION_TYPE_COUNT]),
    mask_selection: new ort.Tensor('uint8', m.maskSelection, [1, ACTION_TYPE_COUNT, N_ENT]),
    mask_target: new ort.Tensor('uint8', m.maskTarget, [1, ACTION_TYPE_COUNT, N_ENT]),
    mask_cell: new ort.Tensor('uint8', m.maskCell, [1, ACTION_TYPE_COUNT, CELLS]),
    mask_build_cell: new ort.Tensor('uint8', m.maskBuildCell, [1, ENTITY_TYPE_COUNT, CELLS]),
    mask_row_entity_type: new ort.Tensor('uint8', m.maskRowEntityType, [
      1,
      N_ENT,
      ENTITY_TYPE_COUNT,
    ]),
    mask_build_type: new ort.Tensor('uint8', m.maskBuildType, [1, ENTITY_TYPE_COUNT]),
    noise: new ort.Tensor('float32', noise, [1, NOISE_LEN]),
    temperature: new ort.Tensor('float32', temperature, [1]),
  };
}

/** An observation of nothing, to run the graph once before the match needs it. */
function blank(): ActMessage {
  const maskType = new Uint8Array(ACTION_TYPE_COUNT);
  maskType[0] = 1;
  return {
    type: 'act',
    id: 0,
    entities: new Float32Array(N_ENT * ENTITY_FEATURE_COUNT),
    entityMask: new Uint8Array(N_ENT),
    grid: new Float32Array(GRID_CHANNEL_COUNT * CELLS),
    scalars: new Float32Array(SCALAR_COUNT),
    maskType,
    maskSelection: new Uint8Array(ACTION_TYPE_COUNT * N_ENT),
    maskTarget: new Uint8Array(ACTION_TYPE_COUNT * N_ENT),
    maskCell: new Uint8Array(ACTION_TYPE_COUNT * CELLS),
    maskBuildCell: new Uint8Array(ENTITY_TYPE_COUNT * CELLS),
    maskRowEntityType: new Uint8Array(N_ENT * ENTITY_TYPE_COUNT),
    maskBuildType: new Uint8Array(ENTITY_TYPE_COUNT),
    temperature: 1,
  };
}

async function infer(m: ActMessage): Promise<Int32Array> {
  if (!session) throw new Error('the model is not loaded');
  const out = await session.run(feedsFor(m));
  const action = out.action;
  if (!action) throw new Error('the model has no `action` output');
  const data = action.data;
  if (!(data instanceof Int32Array) || data.length !== ACTION_INTS) {
    throw new Error(
      `the model returned ${data.length} values of ${typeof data}, expected ${ACTION_INTS} int32`,
    );
  }
  return data.slice();
}

async function init(m: InitMessage): Promise<void> {
  ort.env.wasm.wasmPaths = { wasm: m.wasmUrl };
  ort.env.wasm.numThreads = m.numThreads;
  session = await ort.InferenceSession.create(m.modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  if (!session.outputNames.includes('action')) {
    throw new Error(`the model's outputs are ${session.outputNames.join(', ')}, not "action"`);
  }
  const t0 = performance.now();
  await infer(blank());
  scope.postMessage({ type: 'ready', warmupMs: performance.now() - t0 });
}

async function act(m: ActMessage): Promise<void> {
  const t0 = performance.now();
  const action = await infer(m);
  scope.postMessage({ type: 'result', id: m.id, action, ms: performance.now() - t0 }, [
    action.buffer,
  ]);
}

scope.onmessage = (event) => {
  const m = event.data;
  const failed = (error: unknown): void => {
    const message = error instanceof Error ? error.message : String(error);
    if (m.type === 'act') scope.postMessage({ type: 'error', id: m.id, message });
    else scope.postMessage({ type: 'error', message });
  };
  if (m.type === 'init') init(m).catch(failed);
  else if (m.type === 'act') act(m).catch(failed);
};
