/**
 * Terrain grid and navigation.
 *
 * The map is a square tile grid. One tile is one world unit, so a position in
 * Q16.16 converts to a tile with a single shift.
 *
 * ## Symmetry
 *
 * Generation enforces exact 180-degree rotational symmetry: tile (x, y) always
 * matches tile (W-1-x, H-1-y). In a mirror matchup that is the difference
 * between a fair game and one decided by the spawn roll, and it costs nothing —
 * we generate one half and reflect it.
 *
 * ## Height is cosmetic
 *
 * `height` exists so the renderer can extrude cliffs into something that reads
 * as terrain. The simulation never consults it: movement is decided purely by
 * the walkability grid. Multi-level terrain with ramps would change that, and is
 * deliberately out of scope for v1.
 */

import { checksumArray, checksumInit, checksumU32 } from './checksum.js';
import { toInt, type Fix } from './fixed.js';
import { Rng } from './rng.js';
import { Tile } from './types.js';

export const MAP_SIZE = 128;

/** Where each player's first Command Post is centred, in tiles. */
export interface StartLocation {
  tileX: number;
  tileY: number;
}

export class GameMap {
  readonly width: number;
  readonly height: number;

  /** Terrain classification per tile. */
  readonly tiles: Uint8Array;
  /** Cosmetic elevation per tile, for the renderer only. */
  readonly elevation: Uint8Array;
  /**
   * Tiles blocked by a building footprint. Kept separate from `tiles` so a
   * building can be destroyed without having to remember what terrain it stood
   * on.
   */
  readonly occupied: Uint8Array;

  readonly starts: StartLocation[] = [];

  constructor(size = MAP_SIZE) {
    this.width = size;
    this.height = size;
    const n = size * size;
    this.tiles = new Uint8Array(n);
    this.elevation = new Uint8Array(n);
    this.occupied = new Uint8Array(n);
  }

  index(tx: number, ty: number): number {
    return ty * this.width + tx;
  }

  tileXOf(index: number): number {
    return index % this.width;
  }

  tileYOf(index: number): number {
    return (index / this.width) | 0;
  }

  inBounds(tx: number, ty: number): boolean {
    return tx >= 0 && ty >= 0 && tx < this.width && ty < this.height;
  }

  /** World position (Q16.16) to tile coordinate. */
  tileOfPos(x: Fix, y: Fix): number {
    const tx = toInt(x);
    const ty = toInt(y);
    if (!this.inBounds(tx, ty)) return -1;
    return this.index(tx, ty);
  }

  /** Terrain alone permits walking here. */
  isGroundWalkable(tx: number, ty: number): boolean {
    if (!this.inBounds(tx, ty)) return false;
    return this.tiles[this.index(tx, ty)] !== Tile.Cliff;
  }

  /** Terrain and buildings both permit walking here. */
  isWalkable(tx: number, ty: number): boolean {
    if (!this.inBounds(tx, ty)) return false;
    const i = this.index(tx, ty);
    return this.tiles[i] !== Tile.Cliff && this.occupied[i] === 0;
  }

  /** A building of `footprint` size may be placed with its top-left here. */
  canPlace(tx: number, ty: number, footprint: number): boolean {
    for (let y = ty; y < ty + footprint; y++) {
      for (let x = tx; x < tx + footprint; x++) {
        if (!this.inBounds(x, y)) return false;
        const i = this.index(x, y);
        if (this.tiles[i] !== Tile.Ground) return false;
        if (this.occupied[i] !== 0) return false;
      }
    }
    return true;
  }

  /**
   * Bumped whenever a building appears or is destroyed.
   *
   * Cached flow fields carry the version they were built against, so a route
   * computed before a wall existed is rebuilt rather than walked into. Derived
   * from building state, so it is not separately checksummed.
   */
  occupancyVersion = 0;

  setOccupied(tx: number, ty: number, footprint: number, value: number): void {
    for (let y = ty; y < ty + footprint; y++) {
      for (let x = tx; x < tx + footprint; x++) {
        if (this.inBounds(x, y)) this.occupied[this.index(x, y)] = value;
      }
    }
    this.occupancyVersion++;
  }

  /**
   * Hash of the generated terrain, computed once.
   *
   * Terrain never changes after generation, so hashing all 16k tiles on every
   * tick would burn most of the checksum budget re-proving something that cannot
   * have moved. It is computed once by `sealTerrain()` and folded in as a single
   * value thereafter.
   */
  private terrainHash = 0;

  /**
   * Freeze the terrain hash. Called at the end of generation; must be called
   * before the first checksum.
   */
  sealTerrain(): void {
    this.terrainHash = checksumArray(checksumInit(), this.tiles, this.tiles.length);
  }

  /**
   * Fold the map into the world checksum.
   *
   * `occupied` is deliberately excluded: it is derived state, written only by
   * `setOccupied` from building placement and destruction, and the entity
   * checksum already covers every building's type, footprint tile, and liveness.
   * Hashing it too would cost 16k operations a tick to detect nothing the entity
   * pass would miss.
   */
  checksum(h: number): number {
    let x = checksumU32(h, this.width);
    x = checksumU32(x, this.terrainHash);
    return x;
  }
}

/**
 * Generate a symmetric two-player map from a seed.
 *
 * Both peers run this with the same seed during the lobby handshake and must
 * produce byte-identical terrain, so it uses only the seeded RNG.
 */
export function generateMap(seed: number, size = MAP_SIZE): GameMap {
  const map = new GameMap(size);
  const rng = new Rng(seed ^ 0x5f3a7c1d);
  const w = map.width;
  const h = map.height;

  // Start locations, placed symmetrically on the diagonal.
  const inset = Math.floor(size * 0.16);
  const a: StartLocation = { tileX: inset, tileY: inset };
  const b: StartLocation = { tileX: w - 1 - inset, tileY: h - 1 - inset };
  map.starts.push(a, b);

  /** Reflect a tile through the map centre. */
  const mirror = (tx: number, ty: number): [number, number] => [w - 1 - tx, h - 1 - ty];

  const setCliff = (tx: number, ty: number): void => {
    if (!map.inBounds(tx, ty)) return;
    const i = map.index(tx, ty);
    map.tiles[i] = Tile.Cliff;
    map.elevation[i] = 1;
  };

  // Scatter rock clusters across one half; every cluster is mirrored, so the
  // two halves are identical under 180-degree rotation.
  const clusters = 26;
  for (let c = 0; c < clusters; c++) {
    const cx = rng.nextInt(w);
    const cy = rng.nextInt(h);
    // Only seed clusters from one half; the mirror fills the other.
    if (cy * w + cx > ((h * w) >> 1)) continue;

    const blobRadius = rng.nextRange(2, 5);
    for (let dy = -blobRadius; dy <= blobRadius; dy++) {
      for (let dx = -blobRadius; dx <= blobRadius; dx++) {
        if (dx * dx + dy * dy > blobRadius * blobRadius) continue;
        // Ragged edges, so blobs do not read as circles.
        if (rng.nextInt(100) < 18) continue;
        const tx = cx + dx;
        const ty = cy + dy;
        setCliff(tx, ty);
        const [mx, my] = mirror(tx, ty);
        setCliff(mx, my);
      }
    }
  }

  // Carve the start areas clear again, generously, so bases always have room.
  const clearRadius = 12;
  for (const s of map.starts) {
    for (let dy = -clearRadius; dy <= clearRadius; dy++) {
      for (let dx = -clearRadius; dx <= clearRadius; dx++) {
        if (dx * dx + dy * dy > clearRadius * clearRadius) continue;
        const tx = s.tileX + dx;
        const ty = s.tileY + dy;
        if (!map.inBounds(tx, ty)) continue;
        const i = map.index(tx, ty);
        map.tiles[i] = Tile.Ground;
        map.elevation[i] = 0;
      }
    }
  }

  ensureConnected(map, a, b);
  // Terrain is final from here on; freeze its hash so per-tick checksums are cheap.
  map.sealTerrain();
  return map;
}

/**
 * Guarantee the two starts can reach each other.
 *
 * Random blobs can seal a base off entirely, which would produce a match nobody
 * can win. Rather than rejecting and regenerating — which burns RNG draws and
 * makes the seed-to-map relationship hard to reason about — we flood fill from
 * one start and, if the other is unreachable, carve a direct corridor. Both
 * peers run the identical procedure, so the repair is deterministic too.
 */
function ensureConnected(map: GameMap, a: StartLocation, b: StartLocation): void {
  const w = map.width;
  const h = map.height;
  const seen = new Uint8Array(w * h);
  const startIndex = map.index(a.tileX, a.tileY);
  const goalIndex = map.index(b.tileX, b.tileY);

  // Breadth-first flood over walkable ground, using an array as a queue with a
  // moving head so ordering is fully determined.
  const queue: number[] = [startIndex];
  seen[startIndex] = 1;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    if (cur === goalIndex) return; // connected already
    const cx = cur % w;
    const cy = (cur / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
      const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ni = ny * w + nx;
      if (seen[ni] === 1) continue;
      if (map.tiles[ni] === Tile.Cliff) continue;
      seen[ni] = 1;
      queue.push(ni);
    }
  }

  // Unreachable: cut a three-tile-wide corridor along the line between starts.
  let x = a.tileX;
  let y = a.tileY;
  while (x !== b.tileX || y !== b.tileY) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const tx = x + dx;
        const ty = y + dy;
        if (!map.inBounds(tx, ty)) continue;
        const i = map.index(tx, ty);
        map.tiles[i] = Tile.Ground;
        map.elevation[i] = 0;
      }
    }
    if (x !== b.tileX) x += x < b.tileX ? 1 : -1;
    if (y !== b.tileY) y += y < b.tileY ? 1 : -1;
  }
}
