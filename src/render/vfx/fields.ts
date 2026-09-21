/**
 * The three draw primitives every effect in the game is built from.
 *
 * Each is one `InstancedMesh` over a unit quad, with a small shader that
 * reshapes that quad per instance:
 *
 * - `SpriteField` turns it to face the camera — flashes, fire, smoke, embers.
 * - `BeamField` stretches it between two world points and rolls it to face the
 *   camera around that axis — beams, tracers, lightning, sparks.
 * - `GroundField` lays it flat on the ground — shockwaves, scorch, order rings.
 *
 * ## Why a shader rather than `setMatrixAt` and `setColorAt`
 *
 * The placeholder effects had to animate a fade by *shrinking* the mesh,
 * because a shared material has one opacity and instances cannot override it.
 * That is the single largest thing standing between these effects and looking
 * finished: fire, smoke and every kind of afterglow are shapes that hold their
 * size while they fade, and none of them can be drawn by a system that can only
 * fade by vanishing. An instanced attribute costs one float per particle and
 * buys the entire vocabulary.
 *
 * Billboarding on the GPU is the other half. Working out a camera-facing
 * quaternion per particle on the CPU is a square root, a cross product and a
 * matrix compose each; in view space the camera faces -Z by construction, so
 * the vertex shader gets it for two adds.
 *
 * ## Frame ownership
 *
 * A field holds no state between frames. Every frame is `begin()`, some number
 * of `push()` calls in any order, then `commit()`. Which particles exist and
 * how old they are is the caller's business — this is the buffer they are
 * drawn into, and nothing more.
 */

import * as THREE from 'three';

/**
 * Anything past this in one frame is dropped.
 *
 * Silently, and it does not matter: the cap is per effect kind and sits far
 * above what a full-screen battle produces, so the frame where it bites is one
 * where several hundred sprites are already overlapping.
 */
export type FieldBlending = 'additive' | 'normal';

/** Shared plumbing: geometry, attribute allocation and the per-frame cursor. */
abstract class Field {
  readonly mesh: THREE.InstancedMesh;
  protected count = 0;
  protected readonly geometry: THREE.PlaneGeometry;
  private readonly material: THREE.ShaderMaterial;
  private readonly attributes: THREE.InstancedBufferAttribute[] = [];

  protected constructor(
    readonly capacity: number,
    texture: THREE.Texture,
    vertexShader: string,
    fragmentShader: string,
    blending: FieldBlending,
    depthTest: boolean,
    renderOrder: number,
  ) {
    this.geometry = new THREE.PlaneGeometry(1, 1);
    this.material = new THREE.ShaderMaterial({
      uniforms: { map: { value: texture } },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      depthTest,
      blending: blending === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, capacity);
    // Effects are scattered over the whole map and the instance matrices are
    // never written, so three.js has no bounding volume worth culling against.
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.mesh.renderOrder = renderOrder;
  }

  /**
   * Allocate a per-instance attribute and hand back its backing array.
   *
   * The array is written directly by `push`, which is why these classes keep
   * the `Float32Array` rather than going through `BufferAttribute.setXYZ`.
   */
  protected attribute(name: string, itemSize: number): Float32Array {
    const array = new Float32Array(this.capacity * itemSize);
    const attribute = new THREE.InstancedBufferAttribute(array, itemSize);
    attribute.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute(name, attribute);
    this.attributes.push(attribute);
    return array;
  }

  /** Start a frame. Everything pushed before this is forgotten. */
  begin(): void {
    this.count = 0;
  }

  /** True while there is room for another instance this frame. */
  protected get hasRoom(): boolean {
    return this.count < this.capacity;
  }

  /** Publish this frame's instances to the GPU. */
  commit(): void {
    this.mesh.count = this.count;
    // An empty field uploads nothing: the draw is skipped, and last frame's
    // buffer contents are unreachable behind a zero instance count.
    if (this.count === 0) return;
    for (const attribute of this.attributes) attribute.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * Shared fragment stage.
 *
 * The mask's alpha is the intensity, so one white texture serves every colour.
 * Tone mapping and the output colour-space encode are the same two chunks
 * three.js appends to its own materials, included here so an effect sits in the
 * same colour space as the units it is drawn over.
 */
const MASK_FRAGMENT = /* glsl */ `
uniform sampler2D map;
varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;

void main() {
  vec4 texel = texture2D(map, vUv);
  float alpha = texel.a * vAlpha;
  if (alpha < 0.003) discard;
  gl_FragColor = vec4(texel.rgb * vColor, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const SPRITE_VERTEX = /* glsl */ `
attribute vec3 aPos;
attribute float aSize;
attribute float aRot;
attribute vec3 aColor;
attribute float aAlpha;

varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;

void main() {
  vUv = uv;
  vColor = aColor;
  vAlpha = aAlpha;

  // View space has the camera at the origin looking down -Z, so a quad that
  // ignores the view rotation entirely is already facing it.
  vec4 centre = modelViewMatrix * vec4(aPos, 1.0);
  vec2 corner = position.xy * aSize;
  float s = sin(aRot);
  float c = cos(aRot);
  centre.xy += vec2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
  gl_Position = projectionMatrix * centre;
}
`;

/** Camera-facing soft quads: flashes, fire, smoke, embers, motes. */
export class SpriteField extends Field {
  private readonly pos: Float32Array;
  private readonly size: Float32Array;
  private readonly rot: Float32Array;
  private readonly col: Float32Array;
  private readonly alpha: Float32Array;

  constructor(texture: THREE.Texture, capacity: number, blending: FieldBlending, renderOrder = 9) {
    super(capacity, texture, SPRITE_VERTEX, MASK_FRAGMENT, blending, false, renderOrder);
    this.pos = this.attribute('aPos', 3);
    this.size = this.attribute('aSize', 1);
    this.rot = this.attribute('aRot', 1);
    this.col = this.attribute('aColor', 3);
    this.alpha = this.attribute('aAlpha', 1);
  }

  /**
   * `size` is the sprite's full width and height in world units.
   *
   * `colour` is read immediately, so callers pass a scratch `Color` rather than
   * making one per particle.
   */
  push(
    x: number,
    y: number,
    z: number,
    size: number,
    rotation: number,
    colour: THREE.Color,
    alpha: number,
  ): void {
    if (!this.hasRoom || alpha <= 0 || size <= 0) return;
    const i = this.count++;
    this.pos[i * 3] = x;
    this.pos[i * 3 + 1] = y;
    this.pos[i * 3 + 2] = z;
    this.size[i] = size;
    this.rot[i] = rotation;
    this.col[i * 3] = colour.r;
    this.col[i * 3 + 1] = colour.g;
    this.col[i * 3 + 2] = colour.b;
    this.alpha[i] = alpha;
  }
}

const BEAM_VERTEX = /* glsl */ `
attribute vec3 aStart;
attribute vec3 aEnd;
attribute float aWidth;
attribute vec2 aShape;
attribute vec3 aColor;
attribute float aAlpha;

varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;
varying vec2 vShape;

void main() {
  vUv = uv;
  vColor = aColor;
  vAlpha = aAlpha;
  vShape = aShape;

  vec3 head = (modelViewMatrix * vec4(aStart, 1.0)).xyz;
  vec3 tail = (modelViewMatrix * vec4(aEnd, 1.0)).xyz;
  vec3 point = mix(head, tail, uv.y);

  vec3 axis = tail - head;
  float span = length(axis);
  vec3 dir = span > 1e-5 ? axis / span : vec3(0.0, 1.0, 0.0);
  // Roll the ribbon about its own axis until its face is square to the eye.
  // A beam aimed straight at the camera has no such roll; it collapses to a
  // point either way, so the fallback never shows.
  vec3 side = cross(dir, normalize(-point));
  float sideLength = length(side);
  side = sideLength > 1e-4 ? side / sideLength : vec3(1.0, 0.0, 0.0);

  point += side * (uv.x - 0.5) * aWidth;
  gl_Position = projectionMatrix * vec4(point, 1.0);
}
`;

/**
 * Fade along the ribbon, as a fraction of its length.
 *
 * `smoothstep(0.0, e, t)` is undefined at `e == 0`, so a hard end is a very
 * short ramp rather than none.
 */
const BEAM_FRAGMENT = /* glsl */ `
uniform sampler2D map;
varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;
varying vec2 vShape;

void main() {
  float along =
    smoothstep(0.0, vShape.x, vUv.y) * smoothstep(0.0, vShape.y, 1.0 - vUv.y);
  float alpha = texture2D(map, vec2(vUv.x, 0.5)).a * vAlpha * along;
  if (alpha < 0.003) discard;
  gl_FragColor = vec4(vColor, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** The smallest fade either end of a ribbon may have, in length fractions. */
const MIN_TAPER = 0.002;

/** Ribbons between two world points: beams, tracers, lightning, sparks. */
export class BeamField extends Field {
  private readonly start: Float32Array;
  private readonly end: Float32Array;
  private readonly width: Float32Array;
  private readonly shape: Float32Array;
  private readonly col: Float32Array;
  private readonly alpha: Float32Array;

  constructor(texture: THREE.Texture, capacity: number, renderOrder = 9) {
    super(capacity, texture, BEAM_VERTEX, BEAM_FRAGMENT, 'additive', false, renderOrder);
    this.start = this.attribute('aStart', 3);
    this.end = this.attribute('aEnd', 3);
    this.width = this.attribute('aWidth', 1);
    this.shape = this.attribute('aShape', 2);
    this.col = this.attribute('aColor', 3);
    this.alpha = this.attribute('aAlpha', 1);
  }

  /**
   * `headFade` and `tailFade` are how much of the ribbon's length is spent
   * ramping up from the start point and down to the end point. A beam holds its
   * brightness end to end; a tracer fades out behind its head.
   */
  push(
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    width: number,
    colour: THREE.Color,
    alpha: number,
    headFade = MIN_TAPER,
    tailFade = MIN_TAPER,
  ): void {
    if (!this.hasRoom || alpha <= 0 || width <= 0) return;
    const i = this.count++;
    this.start[i * 3] = x0;
    this.start[i * 3 + 1] = y0;
    this.start[i * 3 + 2] = z0;
    this.end[i * 3] = x1;
    this.end[i * 3 + 1] = y1;
    this.end[i * 3 + 2] = z1;
    this.width[i] = width;
    this.shape[i * 2] = headFade < MIN_TAPER ? MIN_TAPER : headFade;
    this.shape[i * 2 + 1] = tailFade < MIN_TAPER ? MIN_TAPER : tailFade;
    this.col[i * 3] = colour.r;
    this.col[i * 3 + 1] = colour.g;
    this.col[i * 3 + 2] = colour.b;
    this.alpha[i] = alpha;
  }
}

const GROUND_VERTEX = /* glsl */ `
attribute vec3 aPos;
attribute float aRadius;
attribute float aRot;
attribute vec3 aColor;
attribute float aAlpha;

varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;

void main() {
  vUv = uv;
  vColor = aColor;
  vAlpha = aAlpha;

  vec2 corner = position.xy * (aRadius * 2.0);
  float s = sin(aRot);
  float c = cos(aRot);
  vec3 world = aPos + vec3(corner.x * c - corner.y * s, 0.0, corner.x * s + corner.y * c);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
}
`;

/**
 * Flat quads lying on the ground: shockwaves, scorch marks, order rings.
 *
 * `depthTest` is the interesting knob. A blast ring is part of the world and
 * should disappear under the tank standing on it, so it tests depth; the ring
 * that confirms a click is feedback about an order and has to be visible
 * whatever is standing there, so it does not.
 */
export class GroundField extends Field {
  private readonly pos: Float32Array;
  private readonly rad: Float32Array;
  private readonly rot: Float32Array;
  private readonly col: Float32Array;
  private readonly alpha: Float32Array;

  constructor(
    texture: THREE.Texture,
    capacity: number,
    blending: FieldBlending,
    depthTest: boolean,
    renderOrder: number,
  ) {
    super(capacity, texture, GROUND_VERTEX, MASK_FRAGMENT, blending, depthTest, renderOrder);
    this.pos = this.attribute('aPos', 3);
    this.rad = this.attribute('aRadius', 1);
    this.rot = this.attribute('aRot', 1);
    this.col = this.attribute('aColor', 3);
    this.alpha = this.attribute('aAlpha', 1);
  }

  push(
    x: number,
    y: number,
    z: number,
    radius: number,
    rotation: number,
    colour: THREE.Color,
    alpha: number,
  ): void {
    if (!this.hasRoom || alpha <= 0 || radius <= 0) return;
    const i = this.count++;
    this.pos[i * 3] = x;
    this.pos[i * 3 + 1] = y;
    this.pos[i * 3 + 2] = z;
    this.rad[i] = radius;
    this.rot[i] = rotation;
    this.col[i * 3] = colour.r;
    this.col[i * 3 + 1] = colour.g;
    this.col[i * 3 + 2] = colour.b;
    this.alpha[i] = alpha;
  }
}
