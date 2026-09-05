/**
 * Where the model lives in a build, and how the browser starts it.
 *
 * `public/models/policy.json` and the `.onnx` beside it are committed like the
 * unit art under `public/units/` and served as static files; `rtsml-export` writes them. The
 * ONNX Runtime WebAssembly binary is imported as a URL so Vite hashes it and
 * honours `BASE_PATH`, and the worker is told where it is rather than left to
 * guess a path relative to a bundle it does not know the shape of.
 */

import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { isPolicyManifest, WorkerRuntime, type PolicyManifest } from './runtime.js';

export function modelBaseUrl(): string {
  return `${import.meta.env.BASE_URL}models/`;
}

/** The manifest of the model this build ships, or null when it ships none. */
export async function probeNeuralModel(): Promise<PolicyManifest | null> {
  try {
    const response = await fetch(`${modelBaseUrl()}policy.json`, { cache: 'no-cache' });
    if (!response.ok) return null;
    const manifest: unknown = await response.json();
    return isPolicyManifest(manifest) ? manifest : null;
  } catch {
    return null;
  }
}

/** Fetch the manifest, start the worker, load the model, infer once. */
export async function loadNeuralRuntime(): Promise<WorkerRuntime> {
  const manifest = await probeNeuralModel();
  if (manifest === null) {
    throw new Error(
      'This build ships no neural model (public/models/policy.json is missing). See ml/README.md for how to train and export one.',
    );
  }
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  return WorkerRuntime.load(
    worker,
    { modelUrl: `${modelBaseUrl()}${manifest.model}`, wasmUrl, numThreads: 1 },
    manifest,
  );
}
