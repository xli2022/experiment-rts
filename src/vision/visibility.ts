/**
 * What a side can see.
 *
 * Three states per tile: never seen, seen before but not currently observed,
 * and currently observed. Recomputed from checksummed simulation data every
 * tick — unit positions and sight ranges — so every peer could compute every
 * side's visibility identically if it wanted to; each simply computes its own.
 *
 * Visibility is *per side*, so it cannot live in the simulation: two peers
 * would immediately hold different state and desync. It lives here, with no
 * dependency on the renderer, because two things read it — the fog the player
 * sees, and the neural bot, which is allowed to see exactly what a human is.
 * Vision is shared across a team: partners scout for each other.
 *
 * This is presentation, not secrecy. Peer-to-peer lockstep gives every client
 * the full game state by construction, so a modified client could always look
 * through fog — the same is true of the genre's originals. What it buys is the
 * thing that matters for play: scouting, map control, and not knowing what is
 * coming.
 */

import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import type { GameMap } from '../sim/map.js';
import { EntityType, NEUTRAL, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';

export const UNEXPLORED = 0;
export const EXPLORED = 1;
export const VISIBLE = 2;

export class Visibility {
  /** One byte per tile: UNEXPLORED / EXPLORED / VISIBLE. */
  readonly state: Uint8Array;
  readonly width: number;
  readonly height: number;

  /**
   * Bumped every time visibility is recomputed.
   *
   * The terrain renderer shades its cliffs from this state and needs to know
   * when to bother — comparing a counter is cheaper than diffing 16k tiles.
   */
  version = 0;

  constructor(map: GameMap) {
    this.width = map.width;
    this.height = map.height;
    this.state = new Uint8Array(map.width * map.height);
  }

  /**
   * Recompute what `viewer`'s side can currently see.
   *
   * Call once per simulation tick. Everything currently visible decays to
   * "explored" first, then every allied entity re-lights the tiles in its sight
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

  /** True when the tile with this index is currently observed. */
  isVisibleTile(tile: number): boolean {
    return tile >= 0 && tile < this.state.length && this.state[tile] === VISIBLE;
  }

  /** True when the tile with this index has ever been seen. */
  isExploredTile(tile: number): boolean {
    return tile >= 0 && tile < this.state.length && this.state[tile] !== UNEXPLORED;
  }

  /**
   * Can this side see this entity?
   *
   * Friendly entities always; mineral patches once their ground has been
   * discovered, like the terrain they sit on; enemies only while something on
   * this side is actually watching them, which is the entire point.
   */
  canSee(world: World, index: number, viewer: PlayerId): boolean {
    const owner = world.pool.owner[index]!;
    if (world.areAllied(owner, viewer)) return true;
    if (owner === NEUTRAL) {
      return this.isExploredAt(
        Math.floor(toFloat(world.pool.posX[index]!)),
        Math.floor(toFloat(world.pool.posY[index]!)),
      );
    }
    return this.isVisibleAt(toFloat(world.pool.posX[index]!), toFloat(world.pool.posY[index]!));
  }
}
