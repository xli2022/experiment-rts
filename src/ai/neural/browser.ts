/**
 * Where the models live in a build, and how the browser starts one.
 *
 * `public/models/policy-lanes.json` and the `.onnx` beside it are committed like
 * the unit art under `public/units/` and served as static files; `rtsml-export`
 * writes them. The ONNX Runtime WebAssembly binary is imported as a URL so Vite
 * hashes it and honours `BASE_PATH`, and the worker is told where it is rather
 * than left to guess a path relative to a bundle it does not know the shape of.
 *
 * There is one model per map layout, not one model. The observation tells the
 * policy which layout it is in — a `layout:*` one-hot, plus `allies` and
 * `seatInHalf` — so a single network is free to learn one map and neglect the
 * other, and that is exactly what a shared model did: 85% on Lanes and 3% on
 * Quarters, the latter never having had a checkpoint chosen for it. The layouts
 * also differ in size (128 tiles against 152), so they do not even address the
 * same part of the cell head. A caller therefore asks for the model for the map
 * it is about to play.
 */

import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { MapLayout } from '../../sim/types.js';
import { isPolicyManifest, WorkerRuntime, type PolicyManifest } from './runtime.js';
import { SPEC } from './spec.js';

export function modelBaseUrl(): string {
  return `${import.meta.env.BASE_URL}models/`;
}

/** The file stem `rtsml-export --layout` writes for a layout. */
export function modelStem(layout: MapLayout): string {
  return layout === MapLayout.Quarters ? 'policy-quarters' : 'policy-lanes';
}

/**
 * The manifest of the model this build ships for `layout`, or null when it
 * ships none this build can run.
 *
 * A model exported for an older codec counts as none. `WorkerRuntime.load`
 * refuses it either way, but the lobby asks this question to decide whether to
 * *offer* the Neural chip — so without the check here a stale model left the
 * chip lit and failed on the loading screen, after the match was agreed.
 */
export async function probeNeuralModel(layout: MapLayout): Promise<PolicyManifest | null> {
  try {
    const response = await fetch(`${modelBaseUrl()}${modelStem(layout)}.json`, {
      cache: 'no-cache',
    });
    if (!response.ok) return null;
    const manifest: unknown = await response.json();
    if (!isPolicyManifest(manifest)) return null;
    return manifest.specVersion === SPEC.version ? manifest : null;
  } catch {
    return null;
  }
}

/** Fetch the manifest for `layout`, start the worker, load the model, infer once. */
export async function loadNeuralRuntime(layout: MapLayout): Promise<WorkerRuntime> {
  const manifest = await probeNeuralModel(layout);
  if (manifest === null) {
    throw new Error(
      `This build ships no neural model for that map that it can run (public/models/${modelStem(layout)}.json is missing, or was exported for an older codec than version ${SPEC.version}). See ml/README.md for how to train and export one.`,
    );
  }
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  return WorkerRuntime.load(
    worker,
    { modelUrl: `${modelBaseUrl()}${manifest.model}`, wasmUrl, numThreads: 1 },
    manifest,
  );
}
