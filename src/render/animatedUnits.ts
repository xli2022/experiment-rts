/**
 * Animated units, instanced.
 *
 * One `InstancedMesh` carries every unit of a type for one player, and each
 * instance holds a single extra number: the row of the baked bone texture it is
 * currently posed on. The vertex shader reads that row's bone matrices and skins
 * the vertex, so a hundred Slicebots mid-swing cost the same one draw call a
 * hundred static boxes would.
 *
 * The shader is the stock Lambert vertex shader with skinning spliced in.
 * Three.js only compiles its own skinning chunks for a real `SkinnedMesh`, so
 * they are written out here instead — about a dozen lines, and it keeps the
 * whole thing on the instanced path.
 */

import * as THREE from 'three';
import type { AnimatedModel } from './models/animated.js';

/**
 * Injected declarations: the bone texture and how to read a matrix out of it.
 *
 * Each instance names *two* rows and a weight between them. One mechanism, two
 * jobs: between consecutive frames of one clip it smooths the 30Hz bake up to
 * display rate, and between frames of two different clips it cross-fades, so a
 * unit that starts swinging eases into the swing instead of snapping to it.
 *
 * The lerp is componentwise on the matrices, which is an approximation — it
 * shortens a bone when the two poses differ by a large rotation, and a
 * rotation-correct blend would need a decomposition per bone. Measured by
 * skinning the real geometry both ways: 0.0% shortening between neighbouring
 * frames, and 1.3% at worst for the idle, which mixes opposite ends of a
 * stride. Two texture reads buys that.
 *
 * Note this is not the same thing the texture's `NearestFilter` forbids. That
 * prevents the *sampler* blending adjacent texels, which would mix one bone's
 * matrix into its neighbour's and produce nonsense. Here two whole matrices are
 * read exactly and then mixed deliberately.
 */
const SKIN_HEADER = /* glsl */ `
  attribute vec4 skinIndex;
  attribute vec4 skinWeight;
  attribute float aFrame;
  attribute float aFrameTo;
  attribute float aBlend;
  uniform sampler2D boneTexture;
  uniform vec2 boneTextureSize;
  uniform mat4 bindMatrix;
  uniform mat4 bindMatrixInverse;

  mat4 readBone( const in float boneIdx, const in float frame ) {
    float du = 1.0 / boneTextureSize.x;
    float u = ( boneIdx * 4.0 + 0.5 ) * du;
    float v = ( frame + 0.5 ) / boneTextureSize.y;
    return mat4(
      texture2D( boneTexture, vec2( u, v ) ),
      texture2D( boneTexture, vec2( u + du, v ) ),
      texture2D( boneTexture, vec2( u + du * 2.0, v ) ),
      texture2D( boneTexture, vec2( u + du * 3.0, v ) )
    );
  }

  mat4 readBoneBlended( const in float boneIdx ) {
    mat4 a = readBone( boneIdx, aFrame );
    if ( aBlend <= 0.0 ) return a;
    mat4 b = readBone( boneIdx, aFrameTo );
    return mat4(
      mix( a[0], b[0], aBlend ),
      mix( a[1], b[1], aBlend ),
      mix( a[2], b[2], aBlend ),
      mix( a[3], b[3], aBlend )
    );
  }
`;

/**
 * Blend this vertex's four bone matrices.
 *
 * Spliced in at `beginnormal_vertex` rather than alongside the position work,
 * because three.js computes normals earlier in the shader than positions and
 * both need the same matrix.
 */
const SKIN_MATRIX = /* glsl */ `
  mat4 bakedSkinMatrix =
      skinWeight.x * readBoneBlended( skinIndex.x )
    + skinWeight.y * readBoneBlended( skinIndex.y )
    + skinWeight.z * readBoneBlended( skinIndex.z )
    + skinWeight.w * readBoneBlended( skinIndex.w );
  bakedSkinMatrix = bindMatrixInverse * bakedSkinMatrix * bindMatrix;
`;

const SKIN_NORMAL = /* glsl */ `
  objectNormal = mat3( bakedSkinMatrix ) * objectNormal;
`;

const SKIN_POSITION = /* glsl */ `
  transformed = ( bakedSkinMatrix * vec4( transformed, 1.0 ) ).xyz;
`;

/** Scratch for composing an instance's placement with the asset's node transform. */
const scratch = new THREE.Matrix4();

export class AnimatedUnitPool {
  readonly mesh: THREE.InstancedMesh;

  private readonly frames: THREE.InstancedBufferAttribute;
  private readonly framesTo: THREE.InstancedBufferAttribute;
  private readonly blends: THREE.InstancedBufferAttribute;
  private readonly disposables: { dispose(): void }[] = [];
  private used = 0;

  constructor(
    private readonly model: AnimatedModel,
    material: THREE.MeshLambertMaterial,
    capacity: number,
  ) {
    // Share the heavy attributes with every other pool of this model; only the
    // per-instance frame is private.
    const geometry = new THREE.BufferGeometry();
    for (const name of ['position', 'normal', 'uv', 'skinIndex', 'skinWeight']) {
      const attr = model.geometry.getAttribute(name);
      if (attr) geometry.setAttribute(name, attr);
    }
    const index = model.geometry.getIndex();
    if (index) geometry.setIndex(index);

    this.frames = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    this.frames.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aFrame', this.frames);
    this.framesTo = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    this.framesTo.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aFrameTo', this.framesTo);
    this.blends = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    this.blends.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aBlend', this.blends);

    patchMaterial(material, model);
    const depthMaterial = new THREE.MeshDepthMaterial();
    patchDepthMaterial(depthMaterial, model);

    this.mesh = new THREE.InstancedMesh(geometry, material, capacity);
    this.mesh.count = 0;
    // Skinning moves vertices well outside the bind pose, and every other mesh
    // in this renderer is unculled anyway.
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    // Three's shadow pass substitutes a depth material. Give it the same baked
    // skinning transform or an animated unit casts its undeformed bind pose.
    this.mesh.customDepthMaterial = depthMaterial;
    this.disposables.push(geometry, material, depthMaterial);
  }

  /** Start a frame. Instances are re-emitted from scratch every time. */
  begin(): void {
    this.used = 0;
  }

  /**
   * Place one unit, posed between two rows of the bone texture.
   *
   * `matrix` positions the unit in the world; the asset's own node transform is
   * folded in here, exactly as three.js would when drawing the skinned mesh.
   */
  add(matrix: THREE.Matrix4, frame: number, frameTo = frame, blend = 0): void {
    if (this.used >= this.mesh.instanceMatrix.count) return;
    scratch.multiplyMatrices(matrix, this.model.nodeMatrix);
    this.mesh.setMatrixAt(this.used, scratch);
    this.frames.setX(this.used, frame);
    this.framesTo.setX(this.used, frameTo);
    this.blends.setX(this.used, blend);
    this.used++;
  }

  /** Publish this frame's instances. */
  commit(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.frames.needsUpdate = true;
    this.framesTo.needsUpdate = true;
    this.blends.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.dispose();
    for (const d of this.disposables) d.dispose();
  }

  /** Row of the bone texture for a clip at `t` seconds, looping or held. */
  static frameFor(model: AnimatedModel, clip: string, t: number, loop: boolean): number {
    const baked = model.clips.get(clip);
    if (!baked) return 0;
    const raw = Math.floor(t * (baked.frameCount / baked.duration));
    const local = loop
      ? ((raw % baked.frameCount) + baked.frameCount) % baked.frameCount
      : Math.min(Math.max(raw, 0), baked.frameCount - 1);
    return baked.startFrame + local;
  }

  /**
   * The two rows straddling `t`, and how far between them it falls.
   *
   * The bake is 30Hz and the screen is not, so playing a clip by its nearest
   * whole frame shows every pose twice at 60fps and reads as a slight judder.
   * This is what removes it — and it is the same pair-and-weight the cross-fade
   * uses, which is why the shader needs only one mechanism.
   */
  static framePairFor(
    model: AnimatedModel,
    clip: string,
    t: number,
    loop: boolean,
  ): { from: number; to: number; blend: number } {
    const baked = model.clips.get(clip);
    if (!baked) return { from: 0, to: 0, blend: 0 };
    const exact = t * (baked.frameCount / baked.duration);
    const whole = Math.floor(exact);
    const frac = exact - whole;
    const wrap = (n: number): number =>
      ((n % baked.frameCount) + baked.frameCount) % baked.frameCount;
    const clamp = (n: number): number =>
      n < 0 ? 0 : n > baked.frameCount - 1 ? baked.frameCount - 1 : n;
    const a = loop ? wrap(whole) : clamp(whole);
    const b = loop ? wrap(whole + 1) : clamp(whole + 1);
    return {
      from: baked.startFrame + a,
      to: baked.startFrame + b,
      blend: frac,
    };
  }
}

/**
 * Splice skinning into a stock material.
 *
 * `onBeforeCompile` runs once per program, so the uniforms are wired here and
 * the shader source is rewritten around three.js's own include points.
 */
function patchMaterial(material: THREE.MeshLambertMaterial, model: AnimatedModel): void {
  const size = new THREE.Vector2(model.boneCount * 4, model.totalFrames);
  material.onBeforeCompile = (shader) => {
    shader.uniforms.boneTexture = { value: model.boneTexture };
    shader.uniforms.boneTextureSize = { value: size };
    shader.uniforms.bindMatrix = { value: model.bindMatrix };
    shader.uniforms.bindMatrixInverse = { value: model.bindMatrixInverse };

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${SKIN_HEADER}`)
      .replace(
        '#include <beginnormal_vertex>',
        `#include <beginnormal_vertex>\n${SKIN_MATRIX}\n${SKIN_NORMAL}`,
      )
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${SKIN_POSITION}`);
  };
  // Distinguishes this program from an unpatched Lambert in three's cache.
  material.customProgramCacheKey = () => 'baked-skin';
}

/** Apply baked skinning to the directional-light depth pass as well. */
function patchDepthMaterial(material: THREE.MeshDepthMaterial, model: AnimatedModel): void {
  const size = new THREE.Vector2(model.boneCount * 4, model.totalFrames);
  material.onBeforeCompile = (shader) => {
    shader.uniforms.boneTexture = { value: model.boneTexture };
    shader.uniforms.boneTextureSize = { value: size };
    shader.uniforms.bindMatrix = { value: model.bindMatrix };
    shader.uniforms.bindMatrixInverse = { value: model.bindMatrixInverse };

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${SKIN_HEADER}`)
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>\n${SKIN_MATRIX}\n${SKIN_POSITION}`,
      );
  };
  material.customProgramCacheKey = () => 'baked-skin-depth';
}
