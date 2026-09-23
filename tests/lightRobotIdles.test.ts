import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
  createBoomwalkerIdleClip,
  createBurstbotIdleClip,
  createFirespoutIdleClip,
  createFixomaticIdleClip,
} from '../tools/animation/lightRobotIdles.js';

const CASES = [
  {
    file: 'revolver',
    createIdle: createBurstbotIdleClip,
    feet: ['Bone009', 'Bone013', 'Bone017', 'Bone021'],
    moving: 'Bone023',
    floor: -0.9886,
  },
  {
    file: 'bomb',
    createIdle: createBoomwalkerIdleClip,
    feet: ['Bone005', 'Bone005(mirrored)'],
    moving: 'Bone002',
    floor: -0.8679,
  },
  {
    file: 'flamethrower',
    createIdle: createFirespoutIdleClip,
    feet: ['n12_bone14', 'n20_bone12', 'n28_bone15', 'n36_bone09'],
    moving: 'n5_bone01',
    floor: -0.3814,
  },
  {
    file: 'healing-machine',
    createIdle: createFixomaticIdleClip,
    feet: ['n7_Bip001_L_Calf', 'n12_Bip001_R_Calf'],
    moving: 'n18_Bip001_Head',
    floor: 1.081,
  },
];

describe.each(CASES)('$file idle', ({ file, createIdle, feet, moving, floor }) => {
  let gltf: GLTF;

  beforeAll(async () => {
    const bytes = await readFile(new URL(`../public/units/${file}.glb`, import.meta.url));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    gltf = await new GLTFLoader().parseAsync(buffer, '');
  });

  it('returns a seamless clip without changing the source transforms or authored animations', () => {
    const snapshot = () => {
      const result: number[][] = [];
      gltf.scene.traverse((node) => {
        result.push([
          ...node.position.toArray(),
          ...node.quaternion.toArray(),
          ...node.scale.toArray(),
        ]);
      });
      return result;
    };
    const transforms = snapshot();
    const source = JSON.stringify(gltf.animations.map((clip) => THREE.AnimationClip.toJSON(clip)));
    const idle = createIdle(gltf.scene, gltf.animations);
    expect(snapshot()).toEqual(transforms);
    expect(JSON.stringify(gltf.animations.map((clip) => THREE.AnimationClip.toJSON(clip)))).toBe(
      source,
    );
    expect(idle.name).toBe('idle');
    expect(idle.duration).toBeGreaterThanOrEqual(3);
    expect(idle.duration).toBeLessThanOrEqual(4);
    for (const track of idle.tracks) {
      const width = track.getValueSize();
      expect(Array.from(track.values.slice(0, width))).toEqual(
        Array.from(track.values.slice(-width)),
      );
    }
  });

  it('moves its upper assembly while keeping every supporting leg and the floor stable', () => {
    const idle = createIdle(gltf.scene, gltf.animations);
    const mixer = new THREE.AnimationMixer(gltf.scene);
    mixer.clipAction(idle).play();
    mixer.setTime(0);
    gltf.scene.updateMatrixWorld(true);
    const supports = feet.map((name) => gltf.scene.getObjectByName(name)!);
    const planted = supports.map((node) => node.matrixWorld.clone());
    const assembly = gltf.scene.getObjectByName(moving)!;
    const start = assembly.getWorldQuaternion(new THREE.Quaternion());
    let movement = 0;
    let mesh: THREE.SkinnedMesh | undefined;
    gltf.scene.traverse((node) => {
      if ((node as THREE.SkinnedMesh).isSkinnedMesh) mesh = node as THREE.SkinnedMesh;
    });
    const vertex = new THREE.Vector3();
    for (let frame = 0; frame < 24; frame++) {
      mixer.setTime((frame / 24) * idle.duration);
      gltf.scene.updateMatrixWorld(true);
      supports.forEach((node, index) => {
        node.matrixWorld.elements.forEach((value, element) =>
          expect(value).toBeCloseTo(planted[index].elements[element], 8),
        );
      });
      movement = Math.max(
        movement,
        start.angleTo(assembly.getWorldQuaternion(new THREE.Quaternion())),
      );
      let lowestY = Infinity;
      for (let i = 0; i < mesh!.geometry.getAttribute('position').count; i++) {
        mesh!.getVertexPosition(i, vertex).applyMatrix4(mesh!.matrixWorld);
        lowestY = Math.min(lowestY, vertex.y);
      }
      expect(Math.abs(lowestY - floor)).toBeLessThan(0.01);
    }
    expect(movement).toBeGreaterThan(0.02);
    mixer.stopAllAction();
    mixer.uncacheRoot(gltf.scene);
  });
});
