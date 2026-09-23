import * as THREE from 'three';
import { createRobotIdleClip, type RobotIdleSpec } from './robotIdle.js';

// These offsets align the attack-ready pose with the run envelope used to
// ground each model. All five rigs use local Z as their vertical root axis.
// Keep the root and leg chains still; only independent weapon/upper branches
// move so the ready stance never slides across the terrain.
const PIERCEBOT: RobotIdleSpec = {
  label: 'Piercebot',
  duration: 3.4,
  positionOffsets: { n4_Dummy001: [0, 0, -0.234202] },
  rotations: {
    // Bone002 carries all four legs. Bone001 scans the raised crossbow around
    // its local Z (world-up) axis, with a smaller pitch adjustment.
    n6_Bone001: (p) => [0.04 * Math.sin(p), 0.055 * Math.sin(2 * p), 0.2 * Math.sin(p + 0.4)],
    n7_Dummy002: (p) => [0, 0.075 * Math.sin(p - 0.3), 0],
  },
};

const ARCLIGHT: RobotIdleSpec = {
  label: 'Arclight',
  duration: 3.2,
  positionOffsets: { n2_Bip001: [0, 0, -10.846217] },
  rotations: {
    // The lower Spine owns both thighs; Spine1 starts the free upper body.
    n16_Bip001_Spine1: (p) => [0.03 * Math.sin(p), 0.04 * Math.sin(p), 0.06 * Math.sin(p)],
    n17_Bip001_Neck: (p) => [0.025 * Math.sin(p), 0.1 * Math.sin(p + 0.35), 0],
    n21_Bip001_L_UpperArm: (p) => [0, 0.06 * Math.sin(p - 0.2), -0.12 * Math.sin(p - 0.2)],
    n27_Bip001_R_UpperArm: (p) => [0, -0.06 * Math.sin(p - 0.2), 0.12 * Math.sin(p - 0.2)],
    // The paired coils settle in alternating pulses above the shoulders.
    n32_Bone001: (p) => [0, 0, 0.14 * Math.sin(2 * p)],
    'n34_Bone001(mirrored)': (p) => [0, 0, -0.14 * Math.sin(2 * p)],
  },
};

const SENTRY: RobotIdleSpec = {
  label: 'Sentry',
  duration: 3.6,
  // Attack frame zero is its deployed stance, resting on the main body with
  // the walking legs retracted. Retain that authored transformation in idle.
  positionOffsets: { Dummy001: [0, 0, 3.040479] },
  rotations: {
    // The gun's local Y is world-up; sweep it rather than rolling the barrel.
    Bone007: (p) => [0.03 * Math.sin(p), 0.2 * Math.sin(p), 0.055 * Math.sin(2 * p)],
    Bone003: (p) => [0.09 * Math.sin(p + 0.4), 0, 0],
  },
  positions: {
    Bone005: (p) => [1.2 * Math.sin(2 * p - 0.2), 0, 0],
  },
};

const DARK_GOLEM: RobotIdleSpec = {
  label: 'Dark Golem',
  duration: 4,
  positionOffsets: { n3_Bip001: [0, 0, -9.382183] },
  rotations: {
    n15_Bip001_Spine1: (p) => [0.04 * Math.sin(p), 0.02 * Math.sin(p), 0.075 * Math.sin(p)],
    n19_Bip001_L_UpperArm: (p) => [0, 0.05 * Math.sin(p - 0.25), -0.08 * Math.sin(p - 0.25)],
    n26_Bip001_R_UpperArm: (p) => [0, -0.05 * Math.sin(p - 0.25), 0.08 * Math.sin(p - 0.25)],
    n20_Bip001_L_Forearm: (p) => [0, 0, 0.1 * Math.sin(p - 0.5)],
    n27_Bip001_R_Forearm: (p) => [0, 0, -0.1 * Math.sin(p - 0.5)],
    n32_Bone001: (p) => [0, 0, 0.09 * Math.sin(2 * p)],
    'n33_Bone001(mirrored)': (p) => [0, 0, -0.09 * Math.sin(2 * p)],
  },
};

const ICE_GOLEM: RobotIdleSpec = {
  label: 'Ice Golem',
  duration: 3.8,
  positionOffsets: { Bip001: [0, 0, -3.864234] },
  rotations: {
    Bip001_Spine1: (p) => [0.048 * Math.sin(p), 0.03 * Math.sin(p), 0.06 * Math.sin(p)],
    Bip001_Head: (p) => [0.04 * Math.sin(p), 0.17 * Math.sin(p + 0.5), -0.04 * Math.sin(p)],
    Bip001_L_Clavicle: (p) => [0, 0.06 * Math.sin(p - 0.2), -0.1 * Math.sin(p - 0.2)],
    Bip001_R_Clavicle: (p) => [0, -0.06 * Math.sin(p - 0.2), 0.1 * Math.sin(p - 0.2)],
    Bip001_L_Forearm: (p) => [0, 0, 0.09 * Math.sin(p - 0.5)],
    Bip001_R_Forearm: (p) => [0, 0, -0.09 * Math.sin(p - 0.5)],
  },
};

export function createPiercebotIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, PIERCEBOT);
}

export function createArclightIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, ARCLIGHT);
}

export function createSentryIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, SENTRY);
}

export function createDarkGolemIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, DARK_GOLEM);
}

export function createIceGolemIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  return createRobotIdleClip(scene, clips, ICE_GOLEM);
}
