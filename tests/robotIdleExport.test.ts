import { execFile } from 'node:child_process';
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const execute = promisify(execFile);
const root = fileURLToPath(new URL('../', import.meta.url));
const tempRoot = resolve(tmpdir());
const file = 'sword-machine.glb';
let directory: string;

interface GlbJson {
  buffers: { byteLength: number }[];
  bufferViews: { buffer: number; byteOffset: number; byteLength: number }[];
  accessors: Record<string, unknown>[];
  animations: {
    name: string;
    channels: unknown[];
    samplers: { input: number; output: number; interpolation: string }[];
    extras?: {
      sourceBufferByteLength: number;
      sourceAccessorCount: number;
      sourceBufferViewCount: number;
    };
  }[];
}

function decode(bytes: Buffer): { json: GlbJson; binary: Buffer } {
  const length = bytes.readUInt32LE(12);
  return {
    json: JSON.parse(bytes.subarray(20, 20 + length).toString('utf8')),
    binary: bytes.subarray(28 + length),
  };
}

/** Replace only JSON for the two deliberate post-export edit fixtures. */
function replaceJson(bytes: Buffer, json: GlbJson): Buffer {
  const content = Buffer.from(JSON.stringify(json));
  const length = Math.ceil(content.length / 4) * 4;
  const binaryChunk = bytes.subarray(20 + bytes.readUInt32LE(12));
  const result = Buffer.alloc(20 + length + binaryChunk.length);
  bytes.copy(result, 0, 0, 20);
  result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(length, 12);
  result.fill(0x20, 20, 20 + length);
  content.copy(result, 20);
  binaryChunk.copy(result, 20 + length);
  return result;
}

function regenerate() {
  return execute(
    process.execPath,
    [
      join(root, 'node_modules/vite-node/dist/cli.mjs'),
      join(root, 'scripts/bake-robot-idles.ts'),
      '--out',
      directory,
    ],
    { cwd: root, windowsHide: true, timeout: 15_000, maxBuffer: 64 * 1024 },
  );
}

beforeEach(async () => {
  directory = await mkdtemp(join(tempRoot, 'rts-idle-export-'));
  await copyFile(join(root, 'public/units', file), join(directory, file));
  const catalog = JSON.parse(await readFile(join(root, 'public/units/all-units.json'), 'utf8'));
  catalog.models = catalog.models.filter((entry: { file: string }) => entry.file === file);
  await writeFile(join(directory, 'all-units.json'), `${JSON.stringify(catalog, null, 2)}\n`);
});

afterEach(async () => {
  // Delete only the directory allocated by this fixture, never the temp root.
  const path = relative(tempRoot, resolve(directory));
  if (path.startsWith('rts-idle-export-') && !path.includes('..')) {
    await rm(directory, { recursive: true, force: true });
  }
});

describe('native robot idle exporter', () => {
  it('regenerates byte-identically while preserving authored animation and mesh data', async () => {
    const before = await readFile(join(directory, file));
    const catalogBefore = await readFile(join(directory, 'all-units.json'));
    const original = decode(before);
    const source = original.json.animations.find((animation) => animation.name === 'idle')!.extras!;
    await regenerate();
    const after = await readFile(join(directory, file));
    const regenerated = decode(after);
    expect(after.equals(before)).toBe(true);
    expect((await readFile(join(directory, 'all-units.json'))).equals(catalogBefore)).toBe(true);
    expect(regenerated.binary.subarray(0, source.sourceBufferByteLength)).toEqual(
      original.binary.subarray(0, source.sourceBufferByteLength),
    );
    expect(regenerated.json.accessors.slice(0, source.sourceAccessorCount)).toEqual(
      original.json.accessors.slice(0, source.sourceAccessorCount),
    );
    expect(regenerated.json.bufferViews.slice(0, source.sourceBufferViewCount)).toEqual(
      original.json.bufferViews.slice(0, source.sourceBufferViewCount),
    );
    expect(regenerated.json.animations.filter((animation) => animation.name !== 'idle')).toEqual(
      original.json.animations.filter((animation) => animation.name !== 'idle'),
    );
  });

  it.each(['accessor', 'animation reference'])(
    'rejects a later %s without modifying files',
    async (edit) => {
      const bytes = await readFile(join(directory, file));
      const { json } = decode(bytes);
      if (edit === 'accessor') {
        json.accessors.push(structuredClone(json.accessors.at(-1)!));
      } else {
        const idle = json.animations.find((animation) => animation.name === 'idle')!;
        json.animations.push({ ...structuredClone(idle), name: 'artist-added' });
      }
      const edited = replaceJson(bytes, json);
      await writeFile(join(directory, file), edited);
      const catalogBefore = await readFile(join(directory, 'all-units.json'));
      await expect(regenerate()).rejects.toMatchObject({
        stderr: expect.stringContaining('data was added after the generated idle'),
      });
      expect((await readFile(join(directory, file))).equals(edited)).toBe(true);
      expect((await readFile(join(directory, 'all-units.json'))).equals(catalogBefore)).toBe(true);
    },
  );
});
