import * as THREE from 'three';

const IDLE_DURATION = 3;
const IDLE_SAMPLES = 60;

/** Local rotation in radians around the ready pose's joint axes. */
type JointMotion = (phase: number) => readonly [number, number, number];

// The imported Biped's lower Spine also parents its legs. Only Spine1 and
// joints above it may move here: animating the pelvis or Spine slides the feet.
const JOINT_MOTION: Readonly<Record<string, JointMotion>> = {
  Bip001_Spine1: (p) => [0.075 * Math.sin(p), 0.035 * Math.sin(p), 0.095 * Math.sin(p)],
  Bip001_Head: (p) => [0.22 * Math.sin(p + 0.45), 0, -0.055 * Math.sin(p)],
  Bip001_L_UpperArm: (p) => [0, 0.1 * Math.sin(p - 0.25), -0.168 * Math.sin(p - 0.25)],
  Bip001_R_UpperArm: (p) => [0, -0.1 * Math.sin(p - 0.25), 0.168 * Math.sin(p - 0.25)],
  Bip001_L_Forearm: (p) => [0, 0, 0.115 * Math.sin(p - 0.55)],
  Bip001_R_Forearm: (p) => [0, 0, -0.115 * Math.sin(p - 0.55)],
};

/**
 * A three-second ready stance with visible blade adjustments and head scans.
 *
 * Its attack starts with both feet planted; the run/rest pose is mid-stride.
 * Keep that attack pose's root, legs, translations, and scales fixed, adding
 * chest, head, and arm rotations large enough to read at game scale. The clip lets
 * the existing bone-texture baker share this work across every Slicebot.
 * Neither the scene nor the source clips are modified.
 */
export function createSlicebotIdleClip(
  scene: THREE.Group,
  clips: readonly THREE.AnimationClip[],
): THREE.AnimationClip {
  const ready = clips.find((clip) => clip.name === 'attack');
  if (!ready) throw new Error('Slicebot idle requires its attack ready pose');

  for (const name of Object.keys(JOINT_MOTION)) {
    if (
      !scene.getObjectByName(name) ||
      !ready.tracks.some((track) => track.name === `${name}.quaternion`)
    ) {
      throw new Error(`Slicebot idle requires the ${name} joint`);
    }
  }

  const tracks = ready.tracks.map((source) => {
    const width = source.getValueSize();
    const first = Array.from(source.values.slice(0, width));
    if (source.name === 'Bip001.position') {
      // The ready pose's soles are 0.246 authored units below the sampled run
      // floor used by the renderer. Lift its Z-up root to match that floor.
      first[2] += 0.25;
    }
    const track = source.clone();
    const node = THREE.PropertyBinding.parseTrackName(source.name);
    const motion = node.propertyName === 'quaternion' ? JOINT_MOTION[node.nodeName] : undefined;

    if (!motion) {
      track.times = new Float32Array([0, IDLE_DURATION]);
      track.values = new Float32Array([...first, ...first]);
      return track;
    }

    const times = new Float32Array(IDLE_SAMPLES + 1);
    const values = new Float32Array((IDLE_SAMPLES + 1) * 4);
    const rest = new THREE.Quaternion().fromArray(first).normalize();
    const offset = new THREE.Quaternion();
    const rotation = new THREE.Euler();
    const sample = new THREE.Quaternion();
    for (let frame = 0; frame <= IDLE_SAMPLES; frame++) {
      times[frame] = (frame / IDLE_SAMPLES) * IDLE_DURATION;
      // Use exactly the first sample at the endpoint, including its phase
      // offsets, so interpolation closes the loop without a seam.
      const phase = frame === IDLE_SAMPLES ? 0 : (frame / IDLE_SAMPLES) * Math.PI * 2;
      const [x, y, z] = motion(phase);
      offset.setFromEuler(rotation.set(x, y, z));
      sample
        .copy(rest)
        .multiply(offset)
        .normalize()
        .toArray(values, frame * 4);
    }
    track.times = times;
    track.values = values;
    return track;
  });

  return new THREE.AnimationClip('idle', IDLE_DURATION, tracks);
}
