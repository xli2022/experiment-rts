/**
 * Loading a skinned model and baking its animation into a texture.
 *
 * ## Why not just use SkinnedMesh
 *
 * Three.js animates a `SkinnedMesh` by walking its bone hierarchy on the CPU
 * every frame and uploading the result. That is fine for a hero character and
 * wrong for an RTS: a hundred Slicebots would be a hundred skeletons updated per
 * frame and a hundred draw calls, against the one draw call per unit type the
 * rest of the renderer manages.
 *
 * So the animation is *baked*. Every clip is sampled at a fixed rate, and each
 * sample's bone matrices are written into a small floating-point texture — 19
 * bones by 95 frames is about 115 KB. At draw time a single `InstancedMesh`
 * carries every unit of that type, each instance holding one number: which row
 * of the texture it is on. The vertex shader reads its four bone matrices from
 * that row and skins the vertex.
 *
 * The cost is that animation is quantised to the bake rate and cannot blend
 * between clips. At 30 samples a second, on units a centimetre tall on screen,
 * neither is visible.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/** Samples per second taken when baking. */
export const BAKE_FPS = 30;

export interface BakedClip {
  /** First row of this clip in the bone texture. */
  startFrame: number;
  frameCount: number;
  /** Seconds the clip runs for at its authored speed. */
  duration: number;
}

export interface AnimatedModel {
  /** Skinned geometry: position, normal, uv, skinIndex, skinWeight. */
  geometry: THREE.BufferGeometry;
  /** Bone matrices for every baked frame, laid out 4 texels per bone. */
  boneTexture: THREE.DataTexture;
  boneCount: number;
  totalFrames: number;
  bindMatrix: THREE.Matrix4;
  bindMatrixInverse: THREE.Matrix4;
  /**
   * The transform the asset carries above its mesh.
   *
   * Three.js applies this as the model matrix when it draws a `SkinnedMesh`, so
   * anything drawing the skinned geometry by hand has to apply it too. It is not
   * a formality: an FBX exported Z-up arrives here as a 90-degree rotation, and
   * without it every unit is skinned correctly and then drawn lying on its back.
   */
  nodeMatrix: THREE.Matrix4;
  clips: Map<string, BakedClip>;
  /** Bounding box of the bind pose, in the model's own units. */
  bounds: THREE.Box3;
  /**
   * Bounding box containing every sampled frame of the clips baked into this
   * model, after applying the asset's node transform.
   *
   * A caller that requests one clip gets that clip's envelope. This lets a
   * model gallery fit a looping run without letting an unrelated attack or
   * death pose determine its scale and ground position.
   */
  animatedBounds: THREE.Box3;
  /** Exact bounds of the first sampled frame of the first baked clip. */
  firstFrameBounds: THREE.Box3;
  /**
   * Lowest point the model reaches across every baked frame, in world units.
   *
   * Measured rather than guessed, because a run cycle crouches well below the
   * bind pose: offsetting by the bind pose alone leaves the unit's feet buried
   * for most of its stride.
   */
  lowestY: number;
  /**
   * Size of the bind pose in world units, per axis.
   *
   * Deliberately the bind pose rather than the extent across every frame: an
   * attack clip that swings a sword overhead would otherwise scale the unit by
   * its weapon and leave the body half the size it should be. Which axis a model
   * is fitted on is the caller's choice — a walker is sized by its height, an
   * aircraft by its wingspan.
   */
  bindSize: THREE.Vector3;
}

/**
 * Load a GLB and bake its animation clips, or only one requested clip.
 *
 * The GLB is an ordinary asset — it opens in Blender, and nothing here is a
 * bespoke format. All the specialisation happens at load, so replacing the model
 * is dropping in a different file.
 */
export async function loadAnimatedModel(
  url: string,
  requestedClip?: string,
): Promise<AnimatedModel> {
  const gltf = await new GLTFLoader().loadAsync(url);
  const animations =
    requestedClip === undefined
      ? gltf.animations
      : gltf.animations.filter((clip) => clip.name === requestedClip);
  if (requestedClip !== undefined && animations.length === 0) {
    throw new Error(`${url} contains no ${requestedClip} animation`);
  }

  let skinned: THREE.SkinnedMesh | null = null;
  gltf.scene.traverse((o) => {
    if (!skinned && (o as THREE.SkinnedMesh).isSkinnedMesh)
      skinned = o as THREE.SkinnedMesh;
  });
  if (!skinned) throw new Error(`${url} contains no skinned mesh`);
  const mesh = skinned as THREE.SkinnedMesh;
  const skeleton = mesh.skeleton;
  const boneCount = skeleton.bones.length;

  // The mesh's own transform, taken from the rest pose before any clip is
  // applied — sampling it after baking would capture whatever pose the last
  // frame happened to leave behind.
  gltf.scene.updateMatrixWorld(true);
  const nodeMatrix = mesh.matrixWorld.clone();
  const nodeMatrixInverse = nodeMatrix.clone().invert();
  const bindMatrix = mesh.bindMatrix.clone();
  const bindMatrixInverse = mesh.bindMatrixInverse.clone();
  const bindMatrixInverseInverse = bindMatrixInverse.clone().invert();

  // Frame budget per clip, and where each one starts in the texture.
  const clips = new Map<string, BakedClip>();
  let totalFrames = 0;
  for (const clip of animations) {
    const frameCount = Math.max(1, Math.round(clip.duration * BAKE_FPS));
    clips.set(clip.name, {
      startFrame: totalFrames,
      frameCount,
      duration: clip.duration,
    });
    totalFrames += frameCount;
  }
  if (totalFrames === 0) throw new Error(`${url} contains no animations`);

  // Four texels hold one mat4, so a row is every bone for one instant in time.
  const width = boneCount * 4;
  const data = new Float32Array(width * totalFrames * 4);
  const nodeCorrection = new THREE.Matrix4();
  const correctedBone = new THREE.Matrix4();

  // The mixer is rooted at the *scene*, not the mesh.
  //
  // Rooted at the mesh, three.js resolves a track's target either as one of the
  // skeleton's bones or as a descendant of the mesh — and an FBX rig's helper
  // nodes are neither. A 3ds Max Biped puts `Bip001` above the bones as the
  // rig root, so binding from the mesh silently dropped the root motion of
  // every clip, and for the five-bone aircraft that was most of the animation.
  const mixer = new THREE.AnimationMixer(gltf.scene);
  for (const clip of animations) {
    const baked = clips.get(clip.name)!;
    const action = mixer.clipAction(clip);
    mixer.stopAllAction();
    action.play();

    for (let f = 0; f < baked.frameCount; f++) {
      // Sample at the *start* of each frame's interval. Sampling at the end
      // would drop the pose a looping clip returns to.
      const t = (f / baked.frameCount) * clip.duration;
      mixer.setTime(0);
      mixer.setTime(t);
      // From the scene root, not the mesh: bones are usually siblings of the
      // mesh rather than its children, and updating from the mesh down leaves
      // every bone's world matrix stale.
      gltf.scene.updateMatrixWorld(true);
      skeleton.update();
      const matrices = skeleton.boneMatrices;
      if (matrices) {
        // A SkinnedMesh can live below an animated joint. Three.js normally
        // combines that changing mesh.matrixWorld with a bindMatrixInverse that
        // it also updates every frame. The RTS renderer needs one stable pair
        // for every instance, so fold both relative transforms into the stored
        // bone matrix:
        //
        //   nodeRest * bindInvRest * correctedBone * bind
        //     = nodeFrame * bindInvFrame * bone * bind
        //
        // Most rigs make this correction the identity. Rigs that parent the
        // renderer to an animated joint need it to retain that motion.
        nodeCorrection
          .copy(bindMatrixInverseInverse)
          .multiply(nodeMatrixInverse)
          .multiply(mesh.matrixWorld)
          .multiply(mesh.bindMatrixInverse);
        const frameOffset = (baked.startFrame + f) * width * 4;
        for (let bone = 0; bone < boneCount; bone++) {
          correctedBone
            .fromArray(matrices, bone * 16)
            .premultiply(nodeCorrection)
            .toArray(data, frameOffset + bone * 16);
        }
      }
    }
    mixer.uncacheAction(clip);
  }
  mixer.stopAllAction();

  const boneTexture = new THREE.DataTexture(
    data,
    width,
    totalFrames,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  // Exact texel reads: interpolating between two bones' matrices, or between two
  // frames, produces garbage rather than a smoother animation.
  boneTexture.magFilter = THREE.NearestFilter;
  boneTexture.minFilter = THREE.NearestFilter;
  boneTexture.generateMipmaps = false;
  boneTexture.needsUpdate = true;

  const geometry = mesh.geometry;
  const position = geometry.getAttribute("position");
  if (!position) throw new Error(`${url} contains no vertex positions`);
  const boundsVertexCount = validatedBoundsVertexCount(
    mesh.userData.boundsVertexCount,
    position.count,
    url,
  );
  const bounds = vertexBounds(position, boundsVertexCount);

  const extent = animatedExtent(
    geometry,
    data,
    boneCount,
    totalFrames,
    bindMatrix,
    bindMatrixInverse,
    nodeMatrix,
    boundsVertexCount,
  );

  return {
    geometry,
    boneTexture,
    boneCount,
    totalFrames,
    bindMatrix,
    bindMatrixInverse,
    nodeMatrix,
    clips,
    bounds,
    animatedBounds: extent.bounds,
    firstFrameBounds: extent.firstFrameBounds,
    lowestY: extent.lowestY,
    bindSize: bounds
      .clone()
      .applyMatrix4(nodeMatrix)
      .getSize(new THREE.Vector3()),
  };
}

/**
 * Bounds of the model over the whole baked animation set.
 *
 * Skins a strided sample of vertices against every baked frame on the CPU. That
 * is a few tens of thousands of operations once at load, and it is what lets the
 * model sit exactly on the ground without a hand-tuned offset that would have to
 * be re-tuned for every new asset.
 */
export function animatedExtent(
  geometry: THREE.BufferGeometry,
  bones: Float32Array,
  boneCount: number,
  totalFrames: number,
  bindMatrix: THREE.Matrix4,
  bindMatrixInverse: THREE.Matrix4,
  nodeMatrix: THREE.Matrix4,
  boundsVertexCount: number,
): {
  bounds: THREE.Box3;
  firstFrameBounds: THREE.Box3;
  lowestY: number;
  height: number;
} {
  const position = geometry.getAttribute("position");
  const skinIndex = geometry.getAttribute("skinIndex");
  const skinWeight = geometry.getAttribute("skinWeight");
  if (!position || !skinIndex || !skinWeight) {
    return {
      bounds: new THREE.Box3(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 1, 0),
      ),
      firstFrameBounds: new THREE.Box3(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 1, 0),
      ),
      lowestY: 0,
      height: 1,
    };
  }

  const index = geometry.getIndex();
  // Indexed Unity exports compact duplicate triangle corners into one POSITION
  // accessor. Sample their draw order so the representative envelope is
  // invariant between an indexed mesh and its expanded equivalent. The exact
  // first frame still visits each unique POSITION once, and unindexed prefix
  // semantics (notably Treant's body-only bounds) remain unchanged.
  const animatedVertexCount = index?.count ?? boundsVertexCount;
  const stride = Math.max(1, Math.floor(animatedVertexCount / 600));
  const floatsPerFrame = boneCount * 16;
  const blended = new THREE.Matrix4();
  const bone = new THREE.Matrix4();
  const point = new THREE.Vector3();

  const bounds = new THREE.Box3();
  const firstFrameBounds = new THREE.Box3();
  for (let f = 0; f < totalFrames; f++) {
    const frameBase = f * floatsPerFrame;
    // The first frame is also the same-pose size reference used to preserve
    // Athena2's relative unit scale, so measure it exactly. Later frames only
    // need a representative envelope for centering and grounding.
    const firstFrame = f === 0;
    const frameStride = firstFrame ? 1 : stride;
    const frameVertexCount = firstFrame
      ? boundsVertexCount
      : animatedVertexCount;
    for (let ordinal = 0; ordinal < frameVertexCount; ordinal += frameStride) {
      const v = firstFrame || !index ? ordinal : index.getX(ordinal);
      if (v >= boundsVertexCount) continue;
      const e = blended.elements;
      for (let k = 0; k < 16; k++) e[k] = 0;
      for (let k = 0; k < 4; k++) {
        const weight = skinWeight.getComponent(v, k);
        if (weight === 0) continue;
        bone.fromArray(bones, frameBase + skinIndex.getComponent(v, k) * 16);
        for (let j = 0; j < 16; j++) e[j] += bone.elements[j]! * weight;
      }
      // bindMatrixInverse * blended * bindMatrix, the same product the shader forms.
      blended.multiply(bindMatrix).premultiply(bindMatrixInverse);
      point.fromBufferAttribute(position as THREE.BufferAttribute, v);
      point.applyMatrix4(blended).applyMatrix4(nodeMatrix);
      bounds.expandByPoint(point);
      if (firstFrame) firstFrameBounds.expandByPoint(point);
    }
  }
  return {
    bounds,
    firstFrameBounds,
    lowestY: bounds.min.y,
    height: bounds.max.y - bounds.min.y,
  };
}

function validatedBoundsVertexCount(
  configured: unknown,
  vertexCount: number,
  url: string,
): number {
  if (configured === undefined) return vertexCount;
  if (
    typeof configured !== "number" ||
    !Number.isSafeInteger(configured) ||
    configured <= 0 ||
    configured > vertexCount
  ) {
    throw new Error(
      `${url} has invalid boundsVertexCount ${String(configured)} for ${vertexCount} vertices`,
    );
  }
  return configured;
}

function vertexBounds(
  position: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
  vertexCount: number,
): THREE.Box3 {
  const bounds = new THREE.Box3();
  const point = new THREE.Vector3();
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    point.fromBufferAttribute(position, vertex);
    bounds.expandByPoint(point);
  }
  return bounds;
}
