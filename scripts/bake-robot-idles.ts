/** Append ordinary glTF idle clips without re-exporting any authored mesh or animation data. */
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createSlicebotIdleClip } from '../tools/animation/slicebotIdle.js';
import {
  createBurstbotIdleClip,
  createBoomwalkerIdleClip,
  createFirespoutIdleClip,
  createFixomaticIdleClip,
} from '../tools/animation/lightRobotIdles.js';
import {
  createPiercebotIdleClip,
  createArclightIdleClip,
  createSentryIdleClip,
  createDarkGolemIdleClip,
  createIceGolemIdleClip,
} from '../tools/animation/heavyRobotIdles.js';
import {
  createBeamdroneIdleClip,
  createPlasmodroneIdleClip,
} from '../tools/animation/flyingRobotIdles.js';

interface IdleMetadata {
  generator: string;
  sourceFileByteLength: number;
  sourceBufferByteLength: number;
  sourceAccessorCount: number;
  sourceBufferViewCount: number;
}

interface GlbAnimation {
  name: string;
  samplers: { input: number; output: number; interpolation?: string }[];
  channels: { sampler: number; target: { node: number; path: string } }[];
  extras?: IdleMetadata;
}

interface GlbJson {
  buffers: { byteLength: number; uri?: string }[];
  bufferViews: { buffer: number; byteOffset?: number; byteLength: number }[];
  accessors: {
    bufferView: number;
    componentType: number;
    count: number;
    type: string;
    min?: number[];
    max?: number[];
  }[];
  animations: GlbAnimation[];
  nodes: { extras?: { clipFrames?: Record<string, number> } }[];
}

interface IdleCatalog {
  models: {
    file: string;
    unit: string;
    clips: Record<string, { static: boolean; duration: number; frames: number; frameRate: number }>;
  }[];
}

const generators = {
  'sword-machine.glb': createSlicebotIdleClip,
  'revolver.glb': createBurstbotIdleClip,
  'bomb.glb': createBoomwalkerIdleClip,
  'flamethrower.glb': createFirespoutIdleClip,
  'healing-machine.glb': createFixomaticIdleClip,
  'ballista.glb': createPiercebotIdleClip,
  'tesla-coil.glb': createArclightIdleClip,
  'cannon.glb': createSentryIdleClip,
  'dark-golem.glb': createDarkGolemIdleClip,
  'ice-golem.glb': createIceGolemIdleClip,
  'beam-ship.glb': createBeamdroneIdleClip,
  'flying-machine.glb': createPlasmodroneIdleClip,
};
const outputArg = process.argv.indexOf('--out');
const output =
  outputArg < 0
    ? fileURLToPath(new URL('../public/units/', import.meta.url))
    : resolve(process.argv[outputArg + 1]);
const catalogPath = join(output, 'all-units.json');
const catalog: IdleCatalog = JSON.parse(await readFile(catalogPath, 'utf8'));
const pending: { path: string; bytes: Buffer }[] = [];
for (const entry of catalog.models) {
  const generate = generators[entry.file as keyof typeof generators];
  if (!generate) continue;
  const path = join(output, entry.file);
  const bytes = await readFile(path);
  if (bytes.readUInt32LE(0) !== 0x46546c67 || bytes.readUInt32LE(4) !== 2) {
    throw new Error(`${entry.file}: expected GLB 2.0`);
  }
  const jsonLength = bytes.readUInt32LE(12);
  const json: GlbJson = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString('utf8'));
  let binary = bytes.subarray(
    28 + jsonLength,
    28 + jsonLength + bytes.readUInt32LE(20 + jsonLength),
  );
  if (json.buffers.length !== 1 || json.buffers[0].uri)
    throw new Error(`${entry.file}: expected an embedded buffer`);
  let sourceFileByteLength = bytes.length;
  const existing = json.animations.find((animation) => animation.name === 'idle');
  if (existing) {
    if (existing.extras?.generator !== 'robot-idle-v1') {
      throw new Error(`${entry.file}: refusing to overwrite an idle from another author`);
    }
    const source = existing.extras;
    const ownedAccessors = new Set<number>(
      existing.samplers.flatMap((sampler) => [sampler.input, sampler.output]),
    );
    const ownedViews = new Set<number>(
      [...ownedAccessors].map((index) => json.accessors[index]?.bufferView),
    );
    const ownsTail = (indices: Set<number>, start: number, length: number): boolean =>
      Number.isSafeInteger(start) &&
      start >= 0 &&
      indices.size === length - start &&
      [...indices].every(
        (index) => Number.isSafeInteger(index) && index >= start && index < length,
      );
    const binaryEnd = Math.max(
      ...[...ownedViews].map((index) => {
        const view = json.bufferViews[index];
        return view ? (view.byteOffset ?? 0) + view.byteLength : NaN;
      }),
    );
    const sharedWithAnotherClip = json.animations.some(
      (animation) =>
        animation !== existing &&
        animation.samplers.some(
          (sampler) => ownedAccessors.has(sampler.input) || ownedAccessors.has(sampler.output),
        ),
    );
    if (
      !ownsTail(ownedAccessors, source.sourceAccessorCount, json.accessors.length) ||
      !ownsTail(ownedViews, source.sourceBufferViewCount, json.bufferViews.length) ||
      binaryEnd !== json.buffers[0].byteLength ||
      sharedWithAnotherClip
    ) {
      throw new Error(
        `${entry.file}: data was added after the generated idle; reimport the source before regenerating`,
      );
    }
    // Our animation owns only appended arrays and bytes. Strip them before
    // rebuilding, keeping repeated runs byte-identical instead of growing GLBs.
    binary = binary.subarray(0, source.sourceBufferByteLength);
    json.buffers[0].byteLength = source.sourceBufferByteLength;
    json.accessors.length = source.sourceAccessorCount;
    json.bufferViews.length = source.sourceBufferViewCount;
    json.animations = json.animations.filter((animation) => animation !== existing);
    sourceFileByteLength = source.sourceFileByteLength;
    for (const node of json.nodes) if (node.extras?.clipFrames) delete node.extras.clipFrames.idle;
  }
  binary = binary.subarray(0, json.buffers[0].byteLength);
  const source = {
    generator: 'robot-idle-v1',
    sourceFileByteLength,
    sourceBufferByteLength: binary.length,
    sourceAccessorCount: json.accessors.length,
    sourceBufferViewCount: json.bufferViews.length,
  };
  const input = encodeGlb(json, binary);
  const gltf = await new GLTFLoader().parseAsync(new Uint8Array(input).buffer, '');
  const idle = generate(gltf.scene, gltf.animations);
  const indices = new Map<string, number>();
  gltf.scene.traverse((node) => {
    const index = gltf.parser.associations.get(node)?.nodes;
    if (index !== undefined) indices.set(node.name, index);
  });

  const pieces: Buffer[] = [binary];
  let length = binary.length;
  const appendAccessor = (values: Float32Array, width: number, time = false): number => {
    const padding = (4 - (length % 4)) % 4;
    if (padding) {
      pieces.push(Buffer.alloc(padding));
      length += padding;
    }
    const data = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
    const bufferView = json.bufferViews.length;
    json.bufferViews.push({ buffer: 0, byteOffset: length, byteLength: data.length });
    pieces.push(data);
    length += data.length;
    const index = json.accessors.length;
    json.accessors.push({
      bufferView,
      componentType: 5126,
      count: values.length / width,
      type: width === 1 ? 'SCALAR' : width === 3 ? 'VEC3' : 'VEC4',
      ...(time ? { min: [values[0]], max: [values[values.length - 1]] } : {}),
    });
    return index;
  };
  const samplers = [];
  const channels = [];
  const timelines = new Map<string, number>();
  for (const track of idle.tracks) {
    const parsed = THREE.PropertyBinding.parseTrackName(track.name);
    const node = indices.get(parsed.nodeName);
    if (node === undefined) throw new Error(`${entry.file}: no glTF node for ${track.name}`);
    const targetPath = (
      { position: 'translation', quaternion: 'rotation', scale: 'scale' } as Record<string, string>
    )[parsed.propertyName];
    if (!targetPath) throw new Error(`${entry.file}: unsupported track ${track.name}`);
    const timeline = Array.from(track.times).join(',');
    let input = timelines.get(timeline);
    if (input === undefined) {
      input = appendAccessor(track.times, 1, true);
      timelines.set(timeline, input);
    }
    const output = appendAccessor(track.values, track.getValueSize());
    channels.push({ sampler: samplers.length, target: { node, path: targetPath } });
    samplers.push({ input, output, interpolation: 'LINEAR' });
  }
  json.animations.push({ name: 'idle', channels, samplers, extras: source });
  const frames = Math.round(idle.duration * 30);
  for (const node of json.nodes) if (node.extras?.clipFrames) node.extras.clipFrames.idle = frames;
  json.buffers[0].byteLength = length;
  pending.push({ path, bytes: encodeGlb(json, Buffer.concat(pieces)) });
  entry.clips.idle = { static: false, duration: idle.duration, frames, frameRate: 30 };
  console.log(`${entry.unit}: embedded idle (${idle.duration}s, ${frames} frames)`);
  gltf.scene.traverse((node) => {
    if (node instanceof THREE.SkinnedMesh) {
      node.geometry.dispose();
      node.skeleton.dispose();
    }
  });
}
// Author every clip successfully before modifying any asset.
for (const file of pending) await writeFile(file.path, file.bytes);
await writeFile(catalogPath, `${JSON.stringify(catalog, null, 2)}\n`);

function encodeGlb(json: unknown, binary: Buffer): Buffer {
  const rawJson = Buffer.from(JSON.stringify(json));
  const jsonLength = Math.ceil(rawJson.length / 4) * 4;
  const binaryLength = Math.ceil(binary.length / 4) * 4;
  const result = Buffer.alloc(28 + jsonLength + binaryLength);
  result.writeUInt32LE(0x46546c67, 0);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(jsonLength, 12);
  result.writeUInt32LE(0x4e4f534a, 16);
  result.fill(0x20, 20, 20 + jsonLength);
  rawJson.copy(result, 20);
  result.writeUInt32LE(binaryLength, 20 + jsonLength);
  result.writeUInt32LE(0x004e4942, 24 + jsonLength);
  binary.copy(result, 28 + jsonLength);
  return result;
}
