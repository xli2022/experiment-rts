/**
 * Fog of war.
 *
 * Three states per tile: never seen, seen before but not currently observed, and
 * currently observed. Unexplored is fully black; explored-but-unobserved keeps
 * the terrain visible but dimmed, which is what lets a player remember the map
 * without being told what is happening on it right now.
 *
 * ## Why this lives in the renderer
 *
 * Visibility is *per side*, so it cannot live in the simulation — two peers
 * would immediately hold different state and desync. It is derived fresh each
 * tick from checksummed simulation data (unit positions and sight ranges), so
 * every peer could compute every side's fog identically if it wanted to; each
 * simply computes its own.
 *
 * Vision is shared across a team. Partners in co-op scout for each other, which
 * is most of what makes playing together feel like one side rather than two
 * players who happen to share an enemy.
 *
 * This is presentation, not secrecy. Peer-to-peer lockstep gives every client the
 * full game state by construction, so a modified client could always draw
 * through fog — the same is true of the genre's originals. What fog buys is the
 * thing that actually matters for play: scouting, map control, and not knowing
 * what is coming.
 */

import * as THREE from 'three';
import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import type { GameMap } from '../sim/map.js';
import { EntityType, NEUTRAL, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';

export const UNEXPLORED = 0;
export const EXPLORED = 1;
export const VISIBLE = 2;

/** Alpha of the shroud over ground that has been seen but is not observed now. */
const EXPLORED_ALPHA = 140;
/** Alpha over ground never seen. Not fully opaque, so the map edge stays legible. */
const UNEXPLORED_ALPHA = 244;

export class FogRenderer {
  readonly mesh: THREE.Mesh;

  /** One byte per tile: UNEXPLORED / EXPLORED / VISIBLE. */
  readonly state: Uint8Array;

  private readonly width: number;
  private readonly height: number;
  private readonly texture: THREE.DataTexture;
  private readonly pixels: Uint8Array;
  private readonly disposables: { dispose(): void }[] = [];
  private dirty = true;

  /**
   * Bumped every time visibility changes.
   *
   * The terrain renderer shades its cliffs from this state and needs to know
   * when to bother — comparing a counter is cheaper than diffing 16k tiles.
   */
  version = 0;

  constructor(map: GameMap) {
    this.width = map.width;
    this.height = map.height;
    this.state = new Uint8Array(map.width * map.height);

    // RGBA, black with a varying alpha. A data texture rather than a canvas
    // because this is rewritten every tick and there is no drawing to do.
    this.pixels = new Uint8Array(map.width * map.height * 4);
    for (let i = 0; i < map.width * map.height; i++) {
      this.pixels[i * 4 + 3] = UNEXPLORED_ALPHA;
    }
    this.texture = new THREE.DataTexture(this.pixels, map.width, map.height);
    // Linear filtering blurs the tile grid into soft edges, which is most of
    // what makes fog look like fog rather than a checkerboard.
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(map.width, map.height);
    geometry.rotateX(-Math.PI / 2);
    geometry.translate(map.width / 2, 0, map.height / 2);

    const material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthWrite: false,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    // Above the ground and the selection rings, below the units — units are
    // hidden by being skipped entirely, not by being covered.
    this.mesh.position.y = 0.12;
    this.mesh.renderOrder = 5;
    this.mesh.frustumCulled = false;
    this.disposables.push(geometry, material, this.texture);
  }

  /**
   * Recompute what the local player can currently see.
   *
   * Call once per simulation tick. Everything currently visible decays to
   * "explored" first, then every owned entity re-lights the tiles in its sight
   * radius — so losing a unit correctly darkens the ground it was watching.
   */
  update(world: World, viewer: PlayerId): void {
    const state = this.state;
    for (let i = 0; i < state.length; i++) {
      if (state[i] === VISIBLE) state[i] = EXPLORED;
    }

    const pool = world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (!world.areAllied(pool.owner[i]!, viewer)) continue;

      const def = defOf(pool.type[i]! as EntityType);
      const sight = toFloat(def.sightRange);
      if (sight <= 0) continue;

      this.reveal(toFloat(pool.posX[i]!), toFloat(pool.posY[i]!), sight);
    }

    this.dirty = true;
    this.version++;
  }

  /** Mark every tile within `radius` of a world position as currently visible. */
  private reveal(x: number, z: number, radius: number): void {
    const r = Math.ceil(radius);
    const cx = Math.floor(x);
    const cz = Math.floor(z);
    const rSq = radius * radius;

    const minX = Math.max(0, cx - r);
    const maxX = Math.min(this.width - 1, cx + r);
    const minZ = Math.max(0, cz - r);
    const maxZ = Math.min(this.height - 1, cz + r);

    for (let tz = minZ; tz <= maxZ; tz++) {
      const dz = tz + 0.5 - z;
      for (let tx = minX; tx <= maxX; tx++) {
        const dx = tx + 0.5 - x;
        if (dx * dx + dz * dz > rSq) continue;
        this.state[tz * this.width + tx] = VISIBLE;
      }
    }
  }

  /**
   * Push the visibility state to the GPU. Call once per rendered frame.
   *
   * Rows are written bottom-up. Texture V runs opposite to tile Y on this plane,
   * and unlike the canvas-backed ground texture a `DataTexture` is uploaded
   * without three.js's default vertical flip — so writing rows in the obvious
   * order mirrors the fog about the map's centre line. That is a quiet bug: the
   * shroud still looks like a shroud, it just lifts over the wrong corner, and
   * in a mirror match each player ends up watching the other's half.
   */
  refresh(): void {
    if (!this.dirty) return;
    this.dirty = false;
    const state = this.state;
    const px = this.pixels;
    for (let tz = 0; tz < this.height; tz++) {
      const src = tz * this.width;
      const dst = (this.height - 1 - tz) * this.width;
      for (let tx = 0; tx < this.width; tx++) {
        const s = state[src + tx]!;
        px[(dst + tx) * 4 + 3] =
          s === VISIBLE ? 0 : s === EXPLORED ? EXPLORED_ALPHA : UNEXPLORED_ALPHA;
      }
    }
    this.texture.needsUpdate = true;
  }

  /** True when the tile under this world position is currently observed. */
  isVisibleAt(x: number, z: number): boolean {
    const tx = Math.floor(x);
    const tz = Math.floor(z);
    if (tx < 0 || tz < 0 || tx >= this.width || tz >= this.height) return false;
    return this.state[tz * this.width + tx] === VISIBLE;
  }

  /** True when this tile has ever been seen. */
  isExploredAt(tx: number, tz: number): boolean {
    if (tx < 0 || tz < 0 || tx >= this.width || tz >= this.height) return false;
    return this.state[tz * this.width + tx] !== UNEXPLORED;
  }

  /**
   * Should this entity be drawn?
   *
   * Friendly and neutral entities are always drawn — hiding your own units, or
   * a partner's, would be absurd, and mineral patches are terrain. Enemies are
   * drawn only while something on your side is actually watching them, which is
   * the entire point.
   */
  shouldDraw(world: World, index: number, viewer: PlayerId): boolean {
    const owner = world.pool.owner[index]!;
    if (world.areAllied(owner, viewer)) return true;
    if (owner === NEUTRAL) {
      // Resources stay on screen once discovered, like the terrain they sit on.
      return this.isExploredAt(
        Math.floor(toFloat(world.pool.posX[index]!)),
        Math.floor(toFloat(world.pool.posY[index]!)),
      );
    }
    return this.isVisibleAt(toFloat(world.pool.posX[index]!), toFloat(world.pool.posY[index]!));
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
  }
}
