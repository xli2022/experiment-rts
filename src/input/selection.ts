/**
 * Selection and world picking.
 *
 * Selection is *presentation state*, deliberately outside the simulation. Two
 * players have different units selected at all times, so if selection lived in
 * the simulation the two peers would immediately hold different state. Commands
 * therefore carry explicit entity id lists rather than "my selection".
 *
 * Picking projects entity positions to screen space rather than raycasting
 * against the instanced meshes. With instancing, a raycast hit gives an
 * `instanceId` that has to be mapped back through the per-frame packing order,
 * and that packing changes every frame as units die. Projecting positions is
 * both simpler and immune to that.
 */

import * as THREE from 'three';
import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import { EntityType, NEUTRAL, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';

/** Selecting more than this is unwieldy and slows command packets down. */
export const MAX_SELECTION = 24;

/**
 * Selection ring size relative to collision radius.
 *
 * Must match the renderer's own constant: the ring is both what the player sees
 * and what they click, so the two drifting apart makes the game feel like it is
 * ignoring accurate clicks.
 */
export const RING_OVERSIZE = 1.15;

/** How far outside a building's footprint still counts as clicking it. */
const BUILDING_PICK_MARGIN = 0.35;

export class Selection {
  /** Entity slot indices currently selected. */
  readonly indices = new Set<number>();
  private readonly groups = new Map<number, number[]>();

  constructor(private readonly localPlayer: PlayerId) {}

  clear(): void {
    this.indices.clear();
  }

  /** Drop anything that has died since the last frame. */
  prune(world: World): void {
    for (const i of [...this.indices]) {
      if (world.pool.alive[i] !== 1) this.indices.delete(i);
    }
  }

  /** Entity handles for the current selection, for building a command. */
  ids(world: World): number[] {
    const out: number[] = [];
    for (const i of this.indices) {
      if (world.pool.alive[i] === 1) out.push(world.pool.idAt(i));
    }
    return out;
  }

  /** True if the selection contains at least one unit we own and can order. */
  hasOwnUnits(world: World): boolean {
    for (const i of this.indices) {
      if (world.pool.owner[i] === this.localPlayer && !isBuilding(world, i)) return true;
    }
    return false;
  }

  /** The single selected entity, or -1 when the selection is empty or mixed. */
  single(): number {
    return this.indices.size === 1 ? [...this.indices][0]! : -1;
  }

  set(indices: number[]): void {
    this.indices.clear();
    for (const i of indices.slice(0, MAX_SELECTION)) this.indices.add(i);
  }

  toggle(index: number): void {
    if (this.indices.has(index)) this.indices.delete(index);
    else if (this.indices.size < MAX_SELECTION) this.indices.add(index);
  }

  add(indices: number[]): void {
    for (const i of indices) {
      if (this.indices.size >= MAX_SELECTION) break;
      this.indices.add(i);
    }
  }

  assignGroup(key: number): void {
    this.groups.set(key, [...this.indices]);
  }

  recallGroup(key: number, world: World): boolean {
    const stored = this.groups.get(key);
    if (!stored) return false;
    const live = stored.filter((i) => world.pool.alive[i] === 1);
    if (live.length === 0) return false;
    this.set(live);
    return true;
  }
}

function isBuilding(world: World, index: number): boolean {
  return defOf(world.pool.type[index]! as EntityType).isBuilding;
}

/**
 * Project a world position to normalised device coordinates.
 * Returns null when the point is behind the camera.
 */
function project(
  camera: THREE.Camera,
  x: number,
  z: number,
  out: THREE.Vector3,
): THREE.Vector3 | null {
  out.set(x, 0.4, z).project(camera);
  if (out.z > 1) return null;
  return out;
}

const scratch = new THREE.Vector3();

/**
 * Entity nearest to a click, within a screen-space tolerance.
 *
 * Prefers our own units over buildings and over enemies, because clicking near a
 * mixed clump should select the thing the player can actually give orders to.
 */
export function pickAt(
  world: World,
  camera: THREE.Camera,
  ndcX: number,
  ndcY: number,
  localPlayer: PlayerId,
): number {
  const pool = world.pool;
  const ground = groundPointAt(camera, ndcX, ndcY);
  if (!ground) return -1;

  let best = -1;
  let bestScore = Number.POSITIVE_INFINITY;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;
    const def = defOf(type);

    const ex = toFloat(pool.posX[i]!);
    const ez = toFloat(pool.posY[i]!);
    let dist: number;

    if (def.footprint > 0) {
      // Buildings are squares, so hit-test the square. Distance is zero
      // anywhere inside the footprint, which is what makes the middle of a
      // Command Post as clickable as its edge.
      const half = def.footprint / 2;
      const dx = Math.max(0, Math.abs(ground.x - ex) - half);
      const dz = Math.max(0, Math.abs(ground.z - ez) - half);
      dist = Math.hypot(dx, dz);
      if (dist > BUILDING_PICK_MARGIN) continue;
    } else {
      // Exactly the selection ring, because the ring is drawn on the ground at
      // this radius and a hit area that does not match what is drawn is a hit
      // area the player cannot see. A screen-space tolerance was tried and is
      // worse: it does not shrink when you zoom out, so a distant clump becomes
      // a lottery.
      dist = Math.hypot(ground.x - ex, ground.z - ez);
      if (dist > toFloat(def.radius) * RING_OVERSIZE) continue;
    }

    // Rank: own units first, then own buildings, then anything else.
    const owned = pool.owner[i] === localPlayer;
    const priority = owned && !def.isBuilding ? 0 : owned ? 1 : 2;
    const score = priority * 1000 + dist;
    if (score < bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

/**
 * Everything inside a screen-space rectangle.
 *
 * Box-select returns only the player's own units when it catches any, matching
 * every RTS: dragging across a battle should grab your army, not a mix of both
 * sides and the scenery.
 */
export function pickInBox(
  world: World,
  camera: THREE.Camera,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  localPlayer: PlayerId,
): number[] {
  const pool = world.pool;
  const own: number[] = [];
  const others: number[] = [];

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const p = project(camera, toFloat(pool.posX[i]!), toFloat(pool.posY[i]!), scratch);
    if (!p) continue;
    if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY) continue;

    const def = defOf(pool.type[i]! as EntityType);
    if (pool.owner[i] === localPlayer && !def.isBuilding) own.push(i);
    else if (pool.owner[i] !== NEUTRAL) others.push(i);
  }

  return own.length > 0 ? own : others;
}

/**
 * Where a screen ray meets the ground plane, in world units.
 * Returns null if the ray points at the sky.
 */
export function groundPointAt(
  camera: THREE.Camera,
  ndcX: number,
  ndcY: number,
): { x: number; z: number } | null {
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hit = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(plane, hit)) return null;
  return { x: hit.x, z: hit.z };
}
