/**
 * Fog of war: the picture of what a side can see.
 *
 * Unexplored is fully black; explored-but-unobserved keeps the terrain visible
 * but dimmed, which is what lets a player remember the map without being told
 * what is happening on it right now.
 *
 * The visibility itself — which tiles are seen, and by whom — is computed in
 * `src/vision/visibility.ts`, with no dependency on the renderer, because the
 * neural bot reads the same thing: it is allowed to see exactly what a human
 * is, and nothing more. This class owns one and draws it.
 */

import * as THREE from 'three';
import type { GameMap } from '../sim/map.js';
import type { PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { EXPLORED, UNEXPLORED, VISIBLE, Visibility } from '../vision/visibility.js';

export { EXPLORED, UNEXPLORED, VISIBLE };

/** Alpha of the shroud over ground that has been seen but is not observed now. */
const EXPLORED_ALPHA = 140;
/** Alpha over ground never seen. Not fully opaque, so the map edge stays legible. */
const UNEXPLORED_ALPHA = 244;

export class FogRenderer {
  readonly mesh: THREE.Mesh;

  /** The visibility being drawn. Shared, not copied: writes to `state` are seen here. */
  readonly visibility: Visibility;

  private readonly width: number;
  private readonly height: number;
  private readonly texture: THREE.DataTexture;
  private readonly pixels: Uint8Array;
  private readonly disposables: { dispose(): void }[] = [];
  private dirty = true;

  /** One byte per tile: UNEXPLORED / EXPLORED / VISIBLE. */
  get state(): Uint8Array {
    return this.visibility.state;
  }

  /** Bumped every time visibility changes; see `Visibility.version`. */
  get version(): number {
    return this.visibility.version;
  }

  constructor(map: GameMap) {
    this.width = map.width;
    this.height = map.height;
    this.visibility = new Visibility(map);

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
   * Recompute what the local player can currently see. Call once per
   * simulation tick; see `Visibility.update`.
   */
  update(world: World, viewer: PlayerId): void {
    this.visibility.update(world, viewer);
    this.dirty = true;
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
    return this.visibility.isVisibleAt(x, z);
  }

  /** True when this tile has ever been seen. */
  isExploredAt(tx: number, tz: number): boolean {
    return this.visibility.isExploredAt(tx, tz);
  }

  /**
   * Should this entity be drawn? Exactly when the side can see it — see
   * `Visibility.canSee`; enemies are hidden by being skipped, not covered.
   */
  shouldDraw(world: World, index: number, viewer: PlayerId): boolean {
    return this.visibility.canSee(world, index, viewer);
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
  }
}
