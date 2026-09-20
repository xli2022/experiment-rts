import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname } from 'node:path';

let basisPromise;

/** Use the same Basis decoder as the browser, without a GPU or source PNGs. */
export async function decodeKTX2(bytes) {
  basisPromise ??= loadBasis();
  const basis = await basisPromise;
  const file = new basis.KTX2File(new Uint8Array(bytes));
  try {
    if (!file.isValid() || file.getFaces() !== 1 || file.getLayers() > 1) {
      throw new Error('Expected a valid, single-image Basis KTX2 texture');
    }
    const width = file.getWidth();
    const height = file.getHeight();
    const format = 13; // Basis transcoder RGBA32, also used by Three's KTX2Loader.
    if (!file.startTranscoding()) throw new Error('Unable to start KTX2 transcoding');
    const data = new Uint8Array(file.getImageTranscodedSizeInBytes(0, 0, 0, format));
    if (!file.transcodeImage(data, 0, 0, 0, format, 0, -1, -1)) {
      throw new Error('Unable to decode KTX2 base level');
    }
    if (data.length !== width * height * 4) throw new Error('Unexpected RGBA buffer size');
    return { data, width, height };
  } finally {
    file.close();
    file.delete();
  }
}

async function loadBasis() {
  const require = createRequire(import.meta.url);
  const filename = require.resolve('three/examples/jsm/libs/basis/basis_transcoder.js');
  const [source, wasmBinary] = await Promise.all([
    readFile(filename, 'utf8'),
    readFile(require.resolve('three/examples/jsm/libs/basis/basis_transcoder.wasm')),
  ]);
  // Three ships this Emscripten wrapper as CommonJS inside an ESM package.
  // Evaluate the installed wrapper with its ordinary Node module bindings.
  const module = { exports: {} };
  const factory = new Function(
    'require',
    '__filename',
    '__dirname',
    'module',
    'exports',
    `${source}\nreturn module.exports;`,
  )(require, filename, dirname(filename), module, module.exports);
  const basis = await factory({ wasmBinary });
  basis.initializeBasis();
  return basis;
}
