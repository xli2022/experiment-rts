import { readFile } from 'node:fs/promises';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { EntityType } from '../src/sim/types.js';
import { loadAnimatedModel, type AnimatedModel } from '../src/render/models/animated.js';
import { UNIT_MODELS } from '../src/render/models/unitModels.js';

// FileLoader constructs progress events even when fetching an in-memory GLB.
if (typeof ProgressEvent === 'undefined') {
  Object.defineProperty(globalThis, 'ProgressEvent', { value: class extends Event {} });
}

const FLYING = new Set([EntityType.Beamdrone, EntityType.Plasmodrone]);

describe.each(UNIT_MODELS)('$file embedded robot idle', (spec) => {
  let gltf: GLTF;
  let mesh: THREE.SkinnedMesh;
  let idle: THREE.AnimationClip;
  let authored: AnimatedModel;
  let withIdle: AnimatedModel;

  beforeAll(async () => {
    const bytes = await readFile(new URL(`../public/units/${spec.file}`, import.meta.url));
    const url = `data:model/gltf-binary;base64,${bytes.toString('base64')}`;
    [gltf, authored, withIdle] = await Promise.all([
      new GLTFLoader().parseAsync(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        '',
      ),
      loadAnimatedModel(url, { clips: ['run', 'attack', 'die'], boundsClip: 'run' }),
      loadAnimatedModel(url, { boundsClip: 'run' }),
    ]);
    idle = gltf.animations.find((clip) => clip.name === 'idle')!;
    gltf.scene.traverse((node) => {
      if ((node as THREE.SkinnedMesh).isSkinnedMesh) mesh = node as THREE.SkinnedMesh;
    });
  });

  afterAll(() => {
    for (const model of [authored, withIdle]) {
      model?.geometry.dispose();
      model?.boneTexture.dispose();
    }
    mesh?.geometry.dispose();
  });

  it('loads a finite, closed idle loop directly from the GLB', () => {
    expect(idle, 'native idle clip').toBeDefined();
    expect(idle.duration).toBeGreaterThan(0);
    expect(idle.tracks.length).toBeGreaterThan(0);
    for (const track of idle.tracks) {
      const width = track.getValueSize();
      expect(track.times[0], track.name).toBe(0);
      expect(track.times.at(-1), track.name).toBe(idle.duration);
      expect(Array.from(track.values).every(Number.isFinite), track.name).toBe(true);
      expect(Array.from(track.values.slice(-width)), `${track.name} loop seam`).toEqual(
        Array.from(track.values.slice(0, width)),
      );
    }
    expect(withIdle.clips.get('idle')?.frameCount).toBeGreaterThan(1);
    expect((withIdle.boneTexture.image.data as Float32Array).every(Number.isFinite)).toBe(true);
  });

  it('preserves authored animation frames, geometry, and run framing when idle is baked', () => {
    const original = authored.boneTexture.image.data as Float32Array;
    const extended = withIdle.boneTexture.image.data as Float32Array;
    const stride = authored.boneCount * 16;
    for (const [name, before] of authored.clips) {
      const after = withIdle.clips.get(name)!;
      expect(after.duration, name).toBe(before.duration);
      expect(after.frameCount, name).toBe(before.frameCount);
      expect(
        extended.slice(after.startFrame * stride, (after.startFrame + after.frameCount) * stride),
        `${name} baked bone matrices`,
      ).toEqual(
        original.slice(
          before.startFrame * stride,
          (before.startFrame + before.frameCount) * stride,
        ),
      );
    }
    for (const name of Object.keys(authored.geometry.attributes)) {
      expect(withIdle.geometry.getAttribute(name).array, name).toEqual(
        authored.geometry.getAttribute(name).array,
      );
    }
    expect(withIdle.geometry.index?.array).toEqual(authored.geometry.index?.array);
    expect(withIdle.animatedBounds.equals(authored.animatedBounds)).toBe(true);
    expect(withIdle.firstFrameBounds.equals(authored.firstFrameBounds)).toBe(true);
    expect(withIdle.lowestY).toBe(authored.lowestY);
    expect(withIdle.nodeMatrix.equals(authored.nodeMatrix)).toBe(true);
    expect(withIdle.bindMatrix.equals(authored.bindMatrix)).toBe(true);
  });

  it('moves the visible skin while keeping grounded contact points planted', () => {
    expect(idle, 'native idle clip').toBeDefined();
    const mixer = new THREE.AnimationMixer(gltf.scene);
    mixer.clipAction(idle).play();
    const sample = (time: number) => {
      mixer.setTime(time);
      gltf.scene.updateMatrixWorld(true);
      mesh.skeleton.update();
    };
    sample(0);
    const start = Array.from({ length: mesh.geometry.getAttribute('position').count }, (_, i) =>
      mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld),
    );
    const bounds = new THREE.Box3().setFromPoints(start);
    const size = Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
    // Test the actual lowest surface vertices, since many rigs have unnamed
    // feet or separate control bones that do not deform their visible skin.
    const contact = start.map((v) => v.y <= bounds.min.y + size * 0.005);
    const vertex = new THREE.Vector3();
    const vertexMotion = new Float64Array(start.length);
    let visibleMotion = 0;
    let contactMotion = 0;
    try {
      for (let frame = 1; frame < 24; frame++) {
        sample((frame / 24) * idle.duration);
        for (let i = 0; i < start.length; i++) {
          mesh.getVertexPosition(i, vertex).applyMatrix4(mesh.matrixWorld);
          const distance = vertex.distanceTo(start[i]);
          visibleMotion = Math.max(visibleMotion, distance);
          vertexMotion[i] = Math.max(vertexMotion[i], distance);
          if (contact[i]) contactMotion = Math.max(contactMotion, distance);
        }
      }
      expect(visibleMotion / size, 'visible movement relative to unit size').toBeGreaterThan(0.04);
      // A moving weapon tip alone must not make an otherwise frozen model pass.
      // At least a tenth of the visible skin should have a readable idle motion.
      vertexMotion.sort();
      expect(
        vertexMotion[Math.floor(vertexMotion.length * 0.9)] / size,
        '90th-percentile visible movement relative to unit size',
      ).toBeGreaterThan(0.015);
      if (!FLYING.has(spec.type)) {
        expect(contactMotion / size, 'ground contact drift relative to unit size').toBeLessThan(
          0.0001,
        );
      }
    } finally {
      mixer.stopAllAction();
      mixer.uncacheRoot(gltf.scene);
    }
  });
});
