/**
 * Build a standard skeletal GLB from Unity's imported mesh and sampled local
 * transforms. This bypasses FBX export entirely: Unity owns the legacy FBX and
 * Animator semantics, while Three only serializes already-resolved glTF data.
 */

import { spawn } from 'node:child_process';
import { access, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

const UNITY_VERSION = '2022.3.62f3';
const DEFAULT_UNITY = `C:\\Program Files\\Unity\\Hub\\Editor\\${UNITY_VERSION}\\Editor\\Unity.exe`;
const CLIP_NAMES = ['run', 'attack', 'die'];
const POSITION_EPSILON = 1e-5;
const SCALE_EPSILON = 1e-7;
const QUATERNION_EPSILON = 1e-12;

installBrowserShims();

/** Run Unity in an isolated copy of the source assets, then serialize its data. */
export async function buildUnitySampledSkeletonModel(sourceRoot, spec, timings, unityOverride) {
  const configuration = spec.unitySampledSkeleton;
  if (!configuration?.controller) {
    throw new Error(`${spec.unit}: Unity sampled-skeleton controller is absent`);
  }
  const unity = await firstAvailablePath([
    unityOverride ? resolve(unityOverride) : null,
    DEFAULT_UNITY,
  ]);
  if (!unity) {
    throw new Error(
      `${spec.unit} requires Unity ${UNITY_VERSION}; install that editor or pass --unity.`,
    );
  }

  const geometrySlot = configuration.geometry ?? spec.geometry ?? 'run';
  const geometryFile =
    spec.geometryFile ?? (geometrySlot in spec.files ? spec.files[geometrySlot] : null);
  if (!geometryFile) {
    throw new Error(`${spec.unit}: sampled-skeleton geometry ${geometrySlot} is absent`);
  }
  validateTimings(timings, spec.unit);

  const tempRoot = await mkdtemp(join(tmpdir(), 'rts-athena2-unity-skeleton-'));
  try {
    const assetRoot = join(tempRoot, 'Assets', 'Source');
    const editorRoot = join(tempRoot, 'Assets', 'Editor');
    const packagesRoot = join(tempRoot, 'Packages');
    const settingsRoot = join(tempRoot, 'ProjectSettings');
    const outputRoot = join(tempRoot, 'Output');
    await Promise.all(
      [assetRoot, editorRoot, packagesRoot, settingsRoot, outputRoot].map((path) =>
        mkdir(path, { recursive: true }),
      ),
    );

    const sourceNames = [
      ...new Set(
        [...Object.values(spec.files), spec.geometryFile, configuration.controller].filter(Boolean),
      ),
    ];
    const sourceFiles = sourceNames.map((name) => join(sourceRoot, name));
    await Promise.all(sourceFiles.flatMap((path) => [access(path), access(`${path}.meta`)]));
    await Promise.all(
      sourceFiles.flatMap((path) => [
        copyFile(path, join(assetRoot, basename(path))),
        copyFile(`${path}.meta`, join(assetRoot, `${basename(path)}.meta`)),
      ]),
    );

    const sourceScript = fileURLToPath(new URL('./unity-sampled-skeleton.cs', import.meta.url));
    const samplePath = join(outputRoot, 'unity-samples.json');
    const manifestPath = join(tempRoot, 'sample-job.json');
    const manifest = {
      prefabPath: unityAssetPath(geometryFile),
      controllerPath: unityAssetPath(configuration.controller),
      outputPath: samplePath,
      clips: CLIP_NAMES.map((name) => ({
        name,
        state: configuration.states?.[name] ?? name,
        frames: timings[name].frames,
        frameRate: timings[name].frameRate,
      })),
    };
    await Promise.all([
      copyFile(sourceScript, join(editorRoot, 'RtsUnitySampledSkeleton.cs')),
      writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`),
      writeFile(
        join(packagesRoot, 'manifest.json'),
        `${JSON.stringify({ dependencies: {} }, null, 2)}\n`,
      ),
      writeFile(
        join(settingsRoot, 'ProjectVersion.txt'),
        `m_EditorVersion: ${UNITY_VERSION}\n` + `m_EditorVersionWithRevision: ${UNITY_VERSION}\n`,
      ),
    ]);

    console.log(
      `Sampling ${spec.unit} mesh and animation transforms with Unity ${UNITY_VERSION}...`,
    );
    const unityLog = join(tempRoot, 'unity.log');
    try {
      await runProcess(unity, [
        '-batchmode',
        '-nographics',
        '-quit',
        '-projectPath',
        tempRoot,
        '-executeMethod',
        'RtsUnitySampledSkeleton.Run',
        '-rtsSampleManifest',
        manifestPath,
        '-logFile',
        unityLog,
      ]);
    } catch (error) {
      let tail = '';
      try {
        const log = await readFile(unityLog, 'utf8');
        const lines = log.split(/\r?\n/);
        const highlights = lines.filter((line) =>
          /exception|error|sampled-skeleton|animator has no|executeMethod|aborting batchmode/i.test(
            line,
          ),
        );
        tail = [...highlights.slice(-40), '--- final Unity lines ---', ...lines.slice(-30)].join(
          '\n',
        );
      } catch {
        // The process error is still useful if Unity failed before opening a log.
      }
      throw new Error(
        `${spec.unit}: Unity sampled-skeleton export failed` +
          (tail ? `\n--- Unity log tail ---\n${tail}` : ''),
        { cause: error },
      );
    }
    const samples = JSON.parse(await readFile(samplePath, 'utf8'));
    validateSamples(samples, timings, spec.unit);
    const result = await samplesToGlb(samples, spec, timings);
    console.log(
      `${spec.unit}: ${result.byteLength.toLocaleString()}-byte Unity-sampled skeletal GLB`,
    );
    return result;
  } finally {
    // Unity can release its redirected log a fraction after the process exits
    // on Windows. Retrying keeps an otherwise-successful import from being
    // reported as a cleanup failure while still guaranteeing temp isolation.
    await rm(tempRoot, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 125,
    });
  }
}

async function samplesToGlb(data, spec, timings) {
  const reflection = new THREE.Matrix4().makeScale(-1, 1, 1);
  const meshData = data.mesh;
  const boneNodeSet = new Set(meshData.boneNodes);
  const objects = data.nodes.map((record, index) => {
    const object = boneNodeSet.has(index) ? new THREE.Bone() : new THREE.Group();
    const safeName = THREE.PropertyBinding.sanitizeNodeName(record.name) || 'node';
    object.name = `n${index}_${safeName}`;
    return object;
  });
  const pathToIndex = new Map(data.nodes.map((record, index) => [record.path, index]));
  const parentIndices = new Array(objects.length).fill(-1);
  for (let index = 0; index < objects.length; index++) {
    const path = data.nodes[index].path;
    const slash = path.lastIndexOf('/');
    if (index > 0) {
      const parentPath = slash < 0 ? '' : path.slice(0, slash);
      const parent = pathToIndex.get(parentPath);
      if (parent === undefined) {
        throw new Error(`${spec.unit}: Unity node ${path} has no parent`);
      }
      parentIndices[index] = parent;
      objects[parent].add(objects[index]);
    }
    applyUnityLocal(data.restTransforms, index * 10, objects[index]);
  }

  const vertexCount = meshData.vertices.length / 3;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(meshData.vertices.length);
  const normals = new Float32Array(meshData.normals.length);
  for (let index = 0; index < meshData.vertices.length; index += 3) {
    positions[index] = -meshData.vertices[index] * 100;
    positions[index + 1] = meshData.vertices[index + 1] * 100;
    positions[index + 2] = meshData.vertices[index + 2] * 100;
    normals[index] = -meshData.normals[index];
    normals[index + 1] = meshData.normals[index + 1];
    normals[index + 2] = meshData.normals[index + 2];
  }
  if (vertexCount > 65_535) {
    throw new Error(`${spec.unit}: sampled mesh exceeds 16-bit topology`);
  }
  const indices = new Uint16Array(meshData.triangles.length);
  for (let index = 0; index < indices.length; index += 3) {
    indices[index] = meshData.triangles[index];
    indices[index + 1] = meshData.triangles[index + 2];
    indices[index + 2] = meshData.triangles[index + 1];
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(meshData.uv), 2));
  geometry.setAttribute(
    'skinIndex',
    new THREE.BufferAttribute(new Uint16Array(meshData.boneIndices), 4),
  );
  geometry.setAttribute(
    'skinWeight',
    new THREE.BufferAttribute(new Float32Array(meshData.boneWeights), 4),
  );
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const mesh = new THREE.SkinnedMesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.8,
      metalness: 0,
    }),
  );
  mesh.name = `${spec.unit}Mesh`;
  mesh.userData = {
    animationFormat: 'athena2-unity-sampled-skeleton-v1',
    sourceVertexCount: vertexCount,
    clipFrames: Object.fromEntries(CLIP_NAMES.map((name) => [name, timings[name].frames])),
  };
  const exportRoot = new THREE.Group();
  exportRoot.name = `${spec.unit}UnitySampled`;
  exportRoot.add(objects[0]);
  exportRoot.add(mesh);
  // Match the model-node correction used by the normal FBX importer. Keeping
  // this above the resolved Unity hierarchy preserves every sampled local
  // transform while fixing source rigs whose FBX omits its authored up/facing
  // parent transform.
  if (spec.rotateX !== 0) {
    exportRoot.quaternion.premultiply(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), spec.rotateX),
    );
  }
  if (spec.rotateY !== 0) {
    exportRoot.quaternion.premultiply(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), spec.rotateY),
    );
  }
  exportRoot.updateMatrixWorld(true);

  const boneInverses = meshData.boneNodes.map((_, bone) => {
    const source = new THREE.Matrix4().fromArray(meshData.bindPoses, bone * 16);
    const converted = reflection.clone().multiply(source).multiply(reflection);
    converted.elements[12] *= 100;
    converted.elements[13] *= 100;
    converted.elements[14] *= 100;
    return converted;
  });
  const skeleton = new THREE.Skeleton(
    meshData.boneNodes.map((index) => objects[index]),
    boneInverses,
  );
  mesh.bind(skeleton, new THREE.Matrix4());
  mesh.normalizeSkinWeights();

  const animatedNodes = new Set(meshData.boneNodes);
  for (const boneNode of meshData.boneNodes) {
    let parent = parentIndices[boneNode];
    while (parent >= 0) {
      animatedNodes.add(parent);
      parent = parentIndices[parent];
    }
  }
  const clips = data.clips.map((sampleClip) => sampledClip(sampleClip, objects, animatedNodes));
  exportRoot.animations = clips;
  const glb = await new GLTFExporter().parseAsync(exportRoot, {
    animations: clips,
    binary: true,
    onlyVisible: false,
  });
  if (!(glb instanceof ArrayBuffer)) {
    throw new Error(`${spec.unit}: Unity sampled-skeleton exporter returned no GLB`);
  }
  return glb;
}

function sampledClip(sampleClip, objects, animatedNodes) {
  const sampleCount = sampleClip.authoredFrames + 1;
  const times = new Float32Array(sampleCount);
  for (let frame = 0; frame < sampleCount; frame++) {
    times[frame] = frame / sampleClip.frameRate;
  }
  const tracks = [];
  for (const node of animatedNodes) {
    const positions = new Float32Array(sampleCount * 3);
    const quaternions = new Float32Array(sampleCount * 4);
    const scales = new Float32Array(sampleCount * 3);
    for (let frame = 0; frame < sampleCount; frame++) {
      const offset = (frame * objects.length + node) * 10;
      positions[frame * 3] = -sampleClip.transforms[offset] * 100;
      positions[frame * 3 + 1] = sampleClip.transforms[offset + 1] * 100;
      positions[frame * 3 + 2] = sampleClip.transforms[offset + 2] * 100;
      // Reflection across X converts Unity's handedness: R * rotation * R.
      quaternions[frame * 4] = sampleClip.transforms[offset + 3];
      quaternions[frame * 4 + 1] = -sampleClip.transforms[offset + 4];
      quaternions[frame * 4 + 2] = -sampleClip.transforms[offset + 5];
      quaternions[frame * 4 + 3] = sampleClip.transforms[offset + 6];
      scales[frame * 3] = sampleClip.transforms[offset + 7];
      scales[frame * 3 + 1] = sampleClip.transforms[offset + 8];
      scales[frame * 3 + 2] = sampleClip.transforms[offset + 9];
    }
    addCompactVectorTrack(tracks, objects[node], 'position', times, positions, POSITION_EPSILON);
    addCompactQuaternionTrack(tracks, objects[node], times, quaternions);
    addCompactVectorTrack(tracks, objects[node], 'scale', times, scales, SCALE_EPSILON);
  }
  return new THREE.AnimationClip(
    sampleClip.name,
    sampleClip.authoredFrames / sampleClip.frameRate,
    tracks,
  );
}

function addCompactVectorTrack(tracks, object, property, times, values, epsilon) {
  const rest = object[property];
  let maxRest = 0;
  let maxFirst = 0;
  for (let frame = 0; frame < times.length; frame++) {
    for (let component = 0; component < 3; component++) {
      maxRest = Math.max(
        maxRest,
        Math.abs(values[frame * 3 + component] - rest.getComponent(component)),
      );
      maxFirst = Math.max(maxFirst, Math.abs(values[frame * 3 + component] - values[component]));
    }
  }
  if (maxRest <= epsilon) return;
  const trackTimes =
    maxFirst <= epsilon ? new Float32Array([times[0], times[times.length - 1]]) : times;
  const trackValues =
    maxFirst <= epsilon ? new Float32Array([...values.slice(0, 3), ...values.slice(0, 3)]) : values;
  tracks.push(new THREE.VectorKeyframeTrack(`${object.name}.${property}`, trackTimes, trackValues));
}

function addCompactQuaternionTrack(tracks, object, times, values) {
  const rest = object.quaternion;
  const first = new THREE.Quaternion().fromArray(values, 0);
  const current = new THREE.Quaternion();
  const previous = new THREE.Quaternion().copy(first);
  let maxRest = 0;
  let maxFirst = 0;
  for (let frame = 0; frame < times.length; frame++) {
    current.fromArray(values, frame * 4);
    if (frame > 0 && current.dot(previous) < 0) {
      for (let component = 0; component < 4; component++) {
        values[frame * 4 + component] *= -1;
      }
      current.fromArray(values, frame * 4);
    }
    maxRest = Math.max(maxRest, 1 - Math.abs(current.dot(rest)));
    maxFirst = Math.max(maxFirst, 1 - Math.abs(current.dot(first)));
    previous.copy(current);
  }
  if (maxRest <= QUATERNION_EPSILON) return;
  const trackTimes =
    maxFirst <= QUATERNION_EPSILON ? new Float32Array([times[0], times[times.length - 1]]) : times;
  const trackValues =
    maxFirst <= QUATERNION_EPSILON
      ? new Float32Array([...values.slice(0, 4), ...values.slice(0, 4)])
      : values;
  tracks.push(
    new THREE.QuaternionKeyframeTrack(`${object.name}.quaternion`, trackTimes, trackValues),
  );
}

function applyUnityLocal(values, offset, object) {
  object.position.set(-values[offset] * 100, values[offset + 1] * 100, values[offset + 2] * 100);
  object.quaternion.set(
    values[offset + 3],
    -values[offset + 4],
    -values[offset + 5],
    values[offset + 6],
  );
  object.scale.set(values[offset + 7], values[offset + 8], values[offset + 9]);
}

function validateTimings(timings, unit) {
  for (const name of CLIP_NAMES) {
    const timing = timings?.[name];
    if (!(timing?.frames > 1) || !(timing.frameRate > 0)) {
      throw new Error(`${unit}: invalid authored ${name} timing`);
    }
  }
}

function validateSamples(samples, timings, unit) {
  if (samples.unityVersion !== UNITY_VERSION) {
    throw new Error(
      `${unit}: sampled with Unity ${samples.unityVersion}, expected ${UNITY_VERSION}`,
    );
  }
  if (!Array.isArray(samples.nodes) || samples.nodes.length === 0) {
    throw new Error(`${unit}: Unity returned no sampled hierarchy`);
  }
  const mesh = samples.mesh;
  const vertexCount = (mesh?.vertices?.length ?? 0) / 3;
  if (
    !Number.isInteger(vertexCount) ||
    vertexCount <= 0 ||
    mesh.normals.length !== vertexCount * 3 ||
    mesh.uv.length !== vertexCount * 2 ||
    mesh.boneIndices.length !== vertexCount * 4 ||
    mesh.boneWeights.length !== vertexCount * 4 ||
    mesh.bindPoses.length !== mesh.boneNodes.length * 16
  ) {
    throw new Error(`${unit}: Unity returned malformed sampled mesh data`);
  }
  for (const name of CLIP_NAMES) {
    const clip = samples.clips?.find((candidate) => candidate.name === name);
    const timing = timings[name];
    if (
      clip?.authoredFrames !== timing.frames ||
      clip?.frameRate !== timing.frameRate ||
      clip?.transforms?.length !== (timing.frames + 1) * samples.nodes.length * 10
    ) {
      throw new Error(`${unit}: Unity returned malformed ${name} samples`);
    }
  }
}

function unityAssetPath(name) {
  return `Assets/Source/${basename(name)}`.replaceAll('\\', '/');
}

async function firstAvailablePath(candidates) {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      await access(candidate);
      return candidate;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  return null;
}

async function runProcess(command, args) {
  await new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      stdio: ['ignore', 'inherit', 'inherit'],
      windowsHide: true,
    });
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) resolvePromise();
      else {
        reject(new Error(`${command} exited with ${code ?? `signal ${signal ?? 'unknown'}`}`));
      }
    });
  });
}

function installBrowserShims() {
  globalThis.window ??= globalThis;
  globalThis.document ??= {
    createElementNS() {
      const listeners = new Map();
      const image = {
        addEventListener(type, listener) {
          listeners.set(type, listener);
        },
        removeEventListener(type) {
          listeners.delete(type);
        },
      };
      Object.defineProperty(image, 'src', {
        set() {
          queueMicrotask(() => listeners.get('load')?.());
        },
      });
      return image;
    },
  };
  globalThis.FileReader ??= class FileReader {
    result = null;
    onloadend = null;

    readAsArrayBuffer(blob) {
      void blob.arrayBuffer().then((result) => {
        this.result = result;
        this.onloadend?.({ target: this });
      });
    }

    readAsDataURL(blob) {
      void blob.arrayBuffer().then((result) => {
        this.result = `data:${blob.type || 'application/octet-stream'};base64,${Buffer.from(result).toString('base64')}`;
        this.onloadend?.({ target: this });
      });
    }
  };
}
