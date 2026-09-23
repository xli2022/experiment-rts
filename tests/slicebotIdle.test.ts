import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createSlicebotIdleClip } from '../tools/animation/slicebotIdle.js';

let gltf: GLTF;
let idle: THREE.AnimationClip;

beforeAll(async () => {
  const bytes = await readFile(new URL('../public/units/sword-machine.glb', import.meta.url));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  gltf = await new GLTFLoader().parseAsync(buffer, '');
  idle = createSlicebotIdleClip(gltf.scene, gltf.animations);
});

describe('Slicebot procedural idle', () => {
  it('closes a three-second loop with identical endpoints', () => {
    expect(idle.name).toBe('idle');
    expect(idle.duration).toBe(3);
    for (const track of idle.tracks) {
      const width = track.getValueSize();
      expect(track.times[0]).toBe(0);
      expect(track.times.at(-1)).toBe(3);
      expect(Array.from(track.values.slice(-width))).toEqual(
        Array.from(track.values.slice(0, width)),
      );
    }
  });

  it('keeps the actual foot transforms planted while its head and sword arms move', () => {
    const mixer = new THREE.AnimationMixer(gltf.scene);
    mixer.clipAction(idle).play();
    const feet = ['Bip001_L_Foot', 'Bip001_R_Foot', 'Bip001_L_Toe0', 'Bip001_R_Toe0'].map((name) =>
      gltf.scene.getObjectByName(name)!,
    );
    const head = gltf.scene.getObjectByName('Bip001_Head')!;
    const hand = gltf.scene.getObjectByName('Bip001_R_Hand')!;
    mixer.setTime(0);
    gltf.scene.updateMatrixWorld(true);
    const planted = feet.map((foot) => foot.matrixWorld.clone());
    const headStart = head.getWorldQuaternion(new THREE.Quaternion());
    const handStart = hand.getWorldPosition(new THREE.Vector3());
    let headMotion = 0;
    let handMotion = 0;
    for (let frame = 1; frame < 90; frame++) {
      mixer.setTime(frame / 30);
      gltf.scene.updateMatrixWorld(true);
      feet.forEach((foot, index) => {
        foot.matrixWorld.elements.forEach((value, element) =>
          expect(value).toBeCloseTo(planted[index].elements[element], 10),
        );
      });
      headMotion = Math.max(
        headMotion,
        headStart.angleTo(head.getWorldQuaternion(new THREE.Quaternion())),
      );
      handMotion = Math.max(
        handMotion,
        handStart.distanceTo(hand.getWorldPosition(new THREE.Vector3())),
      );
    }
    expect(headMotion).toBeGreaterThan(0.04);
    expect(handMotion).toBeGreaterThan(0.5);
    // Both toes rest at the same floor height in the authored attack ready pose.
    expect(Math.abs(planted[2].elements[13] - planted[3].elements[13])).toBeLessThan(0.01);
    let mesh: THREE.SkinnedMesh | undefined;
    gltf.scene.traverse((node) => {
      if ((node as THREE.SkinnedMesh).isSkinnedMesh) mesh = node as THREE.SkinnedMesh;
    });
    const vertex = new THREE.Vector3();
    let lowestY = Infinity;
    for (let i = 0; i < mesh!.geometry.getAttribute('position').count; i++) {
      mesh!.getVertexPosition(i, vertex).applyMatrix4(mesh!.matrixWorld);
      lowestY = Math.min(lowestY, vertex.y);
    }
    // The run envelope used to place Slicebot on the floor has Y=-0.1683.
    expect(Math.abs(lowestY - -0.1683)).toBeLessThan(0.01);
    mixer.stopAllAction();
    mixer.uncacheRoot(gltf.scene);
  });

  it('leaves the source rig and authored clips unchanged', () => {
    const transforms: number[][] = [];
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
    transforms.push(...snapshot());
    const clipsBefore = JSON.stringify(
      gltf.animations.map((clip) => THREE.AnimationClip.toJSON(clip)),
    );
    createSlicebotIdleClip(gltf.scene, gltf.animations);
    expect(snapshot()).toEqual(transforms);
    expect(JSON.stringify(gltf.animations.map((clip) => THREE.AnimationClip.toJSON(clip)))).toBe(
      clipsBefore,
    );
  });
});
