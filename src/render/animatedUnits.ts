/**
 * Animated units, instanced.
 *
 * One `InstancedMesh` carries every unit of a type for one player, and each
 * instance holds a single extra number: the row of the baked bone texture it is
 * currently posed on. The vertex shader reads that row's bone matrices and skins
 * the vertex, so a hundred brawlers mid-swing cost the same one draw call a
 * hundred static boxes would.
 *
 * The shader is the stock Lambert vertex shader with skinning spliced in.
 * Three.js only compiles its own skinning chunks for a real `SkinnedMesh`, so
 * they are written out here instead — about a dozen lines, and it keeps the
 * whole thing on the instanced path.
 */

import * as THREE from 'three';
import type { AnimatedModel } from './models/animated.js';

/** Injected declarations: the bone texture and how to read a matrix out of it. */
const SKIN_HEADER = /* glsl */ `
  attribute vec4 skinIndex;
  attribute vec4 skinWeight;
  attribute float aFrame;
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
`;

/**
 * Blend this vertex's four bone matrices.
 *
 * Spliced in at `beginnormal_vertex` rather than alongside the position work,
 * because three.js computes normals earlier in the shader than positions and
 * both need the same matrix.
 */
const SKIN_NORMAL = /* glsl */ `
  mat4 bakedSkinMatrix =
      skinWeight.x * readBone( skinIndex.x, aFrame )
    + skinWeight.y * readBone( skinIndex.y, aFrame )
    + skinWeight.z * readBone( skinIndex.z, aFrame )
    + skinWeight.w * readBone( skinIndex.w, aFrame );
  bakedSkinMatrix = bindMatrixInverse * bakedSkinMatrix * bindMatrix;
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

    patchMaterial(material, model);

    this.mesh = new THREE.InstancedMesh(geometry, material, capacity);
    this.mesh.count = 0;
    // Skinning moves vertices well outside the bind pose, and every other mesh
    // in this renderer is unculled anyway.
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = true;
    this.disposables.push(geometry, material);
  }

  /** Start a frame. Instances are re-emitted from scratch every time. */
  begin(): void {
    this.used = 0;
  }

  /**
   * Place one unit, posed on `frame` of the bone texture.
   *
   * `matrix` positions the unit in the world; the asset's own node transform is
   * folded in here, exactly as three.js would when drawing the skinned mesh.
   */
  add(matrix: THREE.Matrix4, frame: number): void {
    if (this.used >= this.mesh.instanceMatrix.count) return;
    scratch.multiplyMatrices(matrix, this.model.nodeMatrix);
    this.mesh.setMatrixAt(this.used, scratch);
    this.frames.setX(this.used, frame);
    this.used++;
  }

  /** Publish this frame's instances. */
  commit(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.frames.needsUpdate = true;
  }

  dispose(): void {
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
      .replace('#include <beginnormal_vertex>', `#include <beginnormal_vertex>\n${SKIN_NORMAL}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${SKIN_POSITION}`);
  };
  // Distinguishes this program from an unpatched Lambert in three's cache.
  material.customProgramCacheKey = () => 'baked-skin';
}
