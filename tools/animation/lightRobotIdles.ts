import type * as THREE from 'three';
import { createRobotIdleClip } from './robotIdle.js';

export function createBurstbotIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Burstbot',
    duration: 3.6,
    // Dummy001 carries the gun assembly; its sibling Dummy007 carries all four
    // legs. Give the turret a readable search sweep and barrel dip while the
    // attack-ready feet remain planted.
    rotations: {
      Dummy001: (p) => [0.045 * Math.sin(p + 0.4), 0, 0.22 * Math.sin(p)],
      Bone023: (p) => [0, 0, 0.04 * Math.sin(p - 0.3)],
    },
    // Align the attack-ready soles (-0.2246) with the run floor (-0.9886).
    positionOffsets: { root: [0, -0.764, 0] },
  });
}

export function createBoomwalkerIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Boomwalker',
    duration: 3.2,
    // Bone002 is the bomb payload, a sibling of the hips/legs under Dummy007.
    // Let the tall payload sway and settle on its suspension without shifting feet.
    rotations: {
      Bone002: (p) => [0.13 * Math.sin(p), 0.07 * Math.sin(p + 0.4), 0.11 * Math.sin(p)],
    },
    positions: { Bone002: (p) => [0, 1.5 * Math.sin(p - 0.3), 0] },
    // Attack-ready floor -0.0539; existing run floor -0.8679.
    positionOffsets: { root: [0, -0.814, 0] },
  });
}

export function createFirespoutIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Firespout',
    duration: 4,
    // The direct Unity rig keeps the turret under n4_bone00 and the four legs
    // under sibling n8_Dummy006. Local X is vertical for the turret's joints.
    rotations: {
      n4_bone00: (p) => [0.075 * Math.sin(p), 0, 0.032 * Math.sin(p + 0.5)],
      n5_bone01: (p) => [0.18 * Math.sin(p - 0.35), 0.042 * Math.sin(p), 0],
    },
    // Attack-ready floor 0.1654; existing run floor -0.3814.
    positionOffsets: { n0_SampledSkeleton: [0, -0.547, 0] },
  });
}

export function createFixomaticIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, {
    label: 'Fixomatic',
    duration: 3.8,
    // As with Slicebot, the lower Spine also parents the legs. Move Spine1 and
    // its descendants only, looking around while flexing the tool arms and wrists.
    rotations: {
      n16_Bip001_Spine1: (p) => [0.05 * Math.sin(p), 0.024 * Math.sin(p), 0.055 * Math.sin(p)],
      n18_Bip001_Head: (p) => [0.26 * Math.sin(p + 0.45), 0, -0.06 * Math.sin(p)],
      n21_Bip001_L_UpperArm: (p) => [0, 0.072 * Math.sin(p - 0.25), -0.12 * Math.sin(p - 0.25)],
      n31_Bip001_R_UpperArm: (p) => [0, -0.072 * Math.sin(p - 0.25), 0.12 * Math.sin(p - 0.25)],
      n22_Bip001_L_Forearm: (p) => [0, 0, 0.135 * Math.sin(p - 0.55)],
      n32_Bip001_R_Forearm: (p) => [0, 0, -0.135 * Math.sin(p - 0.55)],
      n23_Bip001_L_Hand: (p) => [0.12 * Math.sin(p - 0.8), 0, 0],
      n33_Bip001_R_Hand: (p) => [-0.12 * Math.sin(p - 0.8), 0, 0],
    },
    // Attack-ready floor -0.5447; existing run floor 1.0810.
    positionOffsets: { n0_SampledSkeleton: [0, 1.626, 0] },
  });
}
