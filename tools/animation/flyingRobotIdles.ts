import * as THREE from 'three';
import { createRobotIdleClip } from './robotIdle.js';

/** Clear hovering and alternating flight-surface corrections at game scale. */
export function createBeamdroneIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Beamdrone',
    baseClip: 'run',
    duration: 4,
    rotations: {
      n3_Bone001: (p) => [0.06 * Math.sin(p), 0.08 * Math.sin(p), 0],
      n4_Bone002: (p) => [0, 0.12 * Math.sin(p - 0.4), 0],
      n5_Bone003: (p) => [0.14 * Math.sin(p + 0.6), 0, 0],
      n6_Bone004: (p) => [-0.14 * Math.sin(p + 0.6), 0, 0],
    },
    positions: { n3_Bone001: (p) => [0, 0, 3.5 * Math.sin(p)] },
  });
}

/** A heavier, slower hover, with the segmented tail and stabilizers alive. */
export function createPlasmodroneIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Plasmodrone',
    baseClip: 'run',
    duration: 4.5,
    rotations: {
      n3_Bone001: (p) => [0.06 * Math.sin(p), 0, 0.09 * Math.sin(p)],
      n4_Bone003: (p) => [0, 0.07 * Math.sin(p - 0.3), 0],
      n5_Bone004: (p) => [0, 0.09 * Math.sin(p - 0.6), 0],
      n6_Bone005: (p) => [0, 0.1 * Math.sin(p - 0.9), 0],
      n7_Bone006: (p) => [0, 0.12 * Math.sin(p - 1.2), 0],
      n8_Bone0010: (p) => [0.125 * Math.sin(p + 0.4), 0, 0],
      n9_Bone0011: (p) => [-0.125 * Math.sin(p + 0.4), 0, 0],
    },
    positions: { n3_Bone001: (p) => [0, 0, 4 * Math.sin(p)] },
  });
}
