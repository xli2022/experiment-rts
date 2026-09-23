import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { loadAnimatedModel } from '../src/render/models/animated.js';
import {
  createArclightIdleClip,
  createDarkGolemIdleClip,
  createIceGolemIdleClip,
  createPiercebotIdleClip,
  createSentryIdleClip,
} from '../tools/animation/heavyRobotIdles.js';

const CASES = [
  {
    name: 'Piercebot',
    file: 'ballista.glb',
    create: createPiercebotIdleClip,
    planted: ['n13_Bone008', 'n19_Bone014', 'n16_Bone011', 'n10_Bone005'],
    moving: 'n6_Bone001',
  },
  {
    name: 'Arclight',
    file: 'tesla-coil.glb',
    create: createArclightIdleClip,
    planted: ['n8_Bip001_L_Foot', 'n13_Bip001_R_Foot'],
    moving: 'n16_Bip001_Spine1',
  },
  {
    name: 'Sentry',
    file: 'cannon.glb',
    create: createSentryIdleClip,
    planted: ['Bone001', 'Dummy004', 'Dummy005', 'Dummy006', 'Dummy007'],
    moving: 'Bone007',
  },
  {
    name: 'Dark Golem',
    file: 'dark-golem.glb',
    create: createDarkGolemIdleClip,
    planted: ['n9_Bip001_L_Foot', 'n13_Bip001_R_Foot'],
    moving: 'n15_Bip001_Spine1',
  },
  {
    name: 'Ice Golem',
    file: 'ice-golem.glb',
    create: createIceGolemIdleClip,
    planted: ['Bip001_L_Foot', 'Bip001_R_Foot'],
    moving: 'Bip001_Head',
  },
];

afterEach(() => vi.restoreAllMocks());

describe('heavy robot idle assets', () => {
  it.each(CASES)(
    '$name keeps its ready stance planted on the run floor in a closed loop',
    async (spec) => {
      const bytes = await readFile(new URL(`../public/units/${spec.file}`, import.meta.url));
      const gltf = await new GLTFLoader().parseAsync(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        '',
      );
      const snapshot = () => {
        const transforms: number[][] = [];
        gltf.scene.traverse((node) => {
          transforms.push([
            ...node.position.toArray(),
            ...node.quaternion.toArray(),
            ...node.scale.toArray(),
          ]);
        });
        return transforms;
      };
      const before = snapshot();
      const clipsBefore = JSON.stringify(gltf.animations.map((clip) => clip.toJSON()));
      const idle = spec.create(gltf.scene, gltf.animations);
      expect(snapshot()).toEqual(before);
      expect(JSON.stringify(gltf.animations.map((clip) => clip.toJSON()))).toBe(clipsBefore);
      expect(idle.name).toBe('idle');
      expect(idle.duration).toBeGreaterThanOrEqual(3);
      expect(idle.duration).toBeLessThanOrEqual(4);
      for (const track of idle.tracks) {
        const width = track.getValueSize();
        expect(track.times[0]).toBe(0);
        expect(track.times.at(-1)).toBeCloseTo(idle.duration, 5);
        expect(Array.from(track.values.slice(-width))).toEqual(
          Array.from(track.values.slice(0, width)),
        );
      }

      // Use the same framing bake as the renderer to catch feet floating after
      // a change to either the run envelope or the authored idle root offset.
      vi.spyOn(GLTFLoader.prototype, 'loadAsync').mockResolvedValueOnce(gltf);
      const run = await loadAnimatedModel(spec.file, 'run');
      const mixer = new THREE.AnimationMixer(gltf.scene);
      try {
        mixer.clipAction(idle).play();
        mixer.setTime(0);
        gltf.scene.updateMatrixWorld(true);
        const planted = spec.planted.map((name) => gltf.scene.getObjectByName(name)!);
        const matrices = planted.map((node) => node.matrixWorld.clone());
        const moving = gltf.scene.getObjectByName(spec.moving)!;
        const rotation = moving.getWorldQuaternion(new THREE.Quaternion());
        let motion = 0;
        let mesh: THREE.SkinnedMesh | undefined;
        gltf.scene.traverse((node) => {
          if ((node as THREE.SkinnedMesh).isSkinnedMesh) mesh = node as THREE.SkinnedMesh;
        });
        const vertex = new THREE.Vector3();
        for (let frame = 0; frame < 12; frame++) {
          mixer.setTime((frame / 12) * idle.duration);
          gltf.scene.updateMatrixWorld(true);
          planted.forEach((node, index) => {
            node.matrixWorld.elements.forEach((value, element) =>
              expect(value).toBeCloseTo(matrices[index].elements[element], 8),
            );
          });
          motion = Math.max(
            motion,
            rotation.angleTo(moving.getWorldQuaternion(new THREE.Quaternion())),
          );
          let floor = Infinity;
          for (let i = 0; i < mesh!.geometry.getAttribute('position').count; i++) {
            mesh!.getVertexPosition(i, vertex).applyMatrix4(mesh!.matrixWorld);
            floor = Math.min(floor, vertex.y);
          }
          expect(Math.abs(floor - run.lowestY)).toBeLessThan(0.03);
        }
        expect(motion).toBeGreaterThan(0.01);
      } finally {
        mixer.stopAllAction();
        mixer.uncacheRoot(gltf.scene);
        run.geometry.dispose();
        run.boneTexture.dispose();
      }
    },
  );
});
