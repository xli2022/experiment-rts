/**
 * The model seam.
 *
 * Rendering asks for a `ModelSpec` and never cares where it came from. Today the
 * only implementation builds units out of Three.js primitives; swapping in
 * loaded GLTF assets later means writing a second provider that returns the same
 * shape, with no change to gameplay, instancing, or selection code.
 *
 * A spec is a flat list of parts rather than a scene graph on purpose. Every
 * part becomes one `InstancedMesh`, so a hundred units of the same type cost one
 * draw call per part instead of a hundred draw calls each.
 */

import type * as THREE from 'three';
import type { EntityType } from '../../sim/types.js';

/**
 * Which colour a part takes.
 *
 * Parts are tinted at render time rather than baked, so the same geometry serves
 * both players. `player` is the team colour — the single most important
 * readability cue in an RTS, and the reason team colour is a material role
 * rather than a per-model decision.
 */
export type PartRole = 'player' | 'accent' | 'dark' | 'resource';

export interface ModelPart {
  geometry: THREE.BufferGeometry;
  role: PartRole;
  /** Offset from the entity origin, in world units. Y is up. */
  offset: [number, number, number];
  rotation?: [number, number, number];
}

export interface ModelSpec {
  parts: ModelPart[];
  /**
   * Radius used for the selection ring.
   *
   * Should track the simulation's collision radius for the same type — if the
   * ring is visibly larger or smaller than where the unit actually collides,
   * players read the difference as the game being unresponsive.
   */
  radius: number;
  /** Height above ground, for health bar placement. */
  height: number;
}

export interface ModelProvider {
  get(type: EntityType): ModelSpec;
  /** Release GPU resources. */
  dispose(): void;
}
