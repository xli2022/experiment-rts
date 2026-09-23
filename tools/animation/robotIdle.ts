import * as THREE from 'three';

type Triple = readonly [number, number, number];
export type IdleMotion = (phase: number) => Triple;

export interface RobotIdleSpec {
  label: string;
  baseClip?: string;
  baseTime?: number;
  duration?: number;
  /** Local Euler offsets from the held ready pose. */
  rotations: Readonly<Record<string, IdleMotion>>;
  positions?: Readonly<Record<string, IdleMotion>>;
  positionOffsets?: Readonly<Record<string, Triple>>;
  rotationOffsets?: Readonly<Record<string, Triple>>;
}

/** Author a closed idle from a held pose without altering the source asset. */
export function createRobotIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
  spec: RobotIdleSpec,
): THREE.AnimationClip {
  const source = clips.find((clip) => clip.name === (spec.baseClip ?? 'attack'));
  if (!source) throw new Error(`${spec.label} idle is missing its ready-pose clip`);
  const duration = spec.duration ?? 3.6;
  const frameCount = Math.round(duration * 30);
  const nodes = new Map<string, THREE.Object3D>();
  const original: {
    node: THREE.Object3D;
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    scale: THREE.Vector3;
  }[] = [];
  scene.traverse((node) => {
    nodes.set(node.name, node);
    original.push({
      node,
      position: node.position.clone(),
      rotation: node.quaternion.clone(),
      scale: node.scale.clone(),
    });
  });

  // Include sparse properties from other clips so an idle never inherits a
  // recoil, collapsed effect, or death pose from whichever clip baked before it.
  const targets = new Map<
    string,
    { node: THREE.Object3D; property: 'position' | 'quaternion' | 'scale' }
  >();
  const add = (name: string, property: string): void => {
    if (property !== 'position' && property !== 'quaternion' && property !== 'scale') {
      throw new Error(`${spec.label} idle cannot freeze ${name}.${property}`);
    }
    const node = nodes.get(name);
    if (!node) throw new Error(`${spec.label} idle is missing joint ${name}`);
    targets.set(`${name}.${property}`, { node, property });
  };
  for (const clip of clips) {
    if (clip.name === 'idle') continue;
    for (const track of clip.tracks) {
      const parsed = THREE.PropertyBinding.parseTrackName(track.name);
      add(parsed.nodeName, parsed.propertyName);
    }
  }
  for (const name of Object.keys({ ...spec.rotations, ...spec.rotationOffsets }))
    add(name, 'quaternion');
  for (const name of Object.keys({ ...spec.positions, ...spec.positionOffsets }))
    add(name, 'position');

  const mixer = new THREE.AnimationMixer(scene);
  const tracks: THREE.KeyframeTrack[] = [];
  try {
    const action = mixer.clipAction(source);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
    mixer.setTime(spec.baseTime ?? 0);
    scene.updateMatrixWorld(true);

    for (const [name, { node, property }] of targets) {
      const motion =
        property === 'quaternion'
          ? spec.rotations[node.name]
          : property === 'position'
            ? spec.positions?.[node.name]
            : undefined;
      const count = motion ? frameCount : 1;
      const times = new Float32Array(count + 1);
      const width = property === 'quaternion' ? 4 : 3;
      const values = new Float32Array((count + 1) * width);
      const base = node[property].clone();
      const positionOffset = spec.positionOffsets?.[node.name];
      const rotationOffset = spec.rotationOffsets?.[node.name];
      if (property === 'position' && positionOffset) {
        (base as THREE.Vector3).add(new THREE.Vector3(...positionOffset));
      }
      if (property === 'quaternion' && rotationOffset) {
        (base as THREE.Quaternion).multiply(
          new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotationOffset)),
        );
      }
      const q = new THREE.Quaternion();
      const euler = new THREE.Euler();
      const v = new THREE.Vector3();
      for (let frame = 0; frame <= count; frame++) {
        times[frame] = (frame / count) * duration;
        const phase = frame === count ? 0 : (frame / count) * Math.PI * 2;
        const offset = motion?.(phase) ?? [0, 0, 0];
        if (property === 'quaternion') {
          q.setFromEuler(euler.set(...offset))
            .premultiply(base as THREE.Quaternion)
            .normalize()
            .toArray(values, frame * width);
        } else {
          v.copy(base as THREE.Vector3)
            .add(new THREE.Vector3(...offset))
            .toArray(values, frame * width);
        }
      }
      const Track =
        property === 'quaternion' ? THREE.QuaternionKeyframeTrack : THREE.VectorKeyframeTrack;
      tracks.push(new Track(name, times, values));
    }
  } finally {
    mixer.stopAllAction();
    mixer.uncacheRoot(scene);
    for (const saved of original) {
      saved.node.position.copy(saved.position);
      saved.node.quaternion.copy(saved.rotation);
      saved.node.scale.copy(saved.scale);
    }
    scene.updateMatrixWorld(true);
  }
  return new THREE.AnimationClip('idle', duration, tracks);
}
