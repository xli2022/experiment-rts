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
import { carveLayout, layoutSize } from './mapgen.js';
import { MapLayout, Tile } from './types.js';

export const MAP_SIZE = 128;

/** Nothing stands here. */
export const UNOCCUPIED = 0;
/**
 * A structure stands here: it blocks both movement and further building.
 */
export const OCCUPIED_SOLID = 1;
/**
 * Something stands here that blocks building but not movement.
 *
 * Mineral patches are the case. A patch is scenery you mine, not a wall — units
 * walk over one rather than round it, and a mineral line stops being a fence
 * that a worker has to path all the way around to reach the far side of. It
 * still cannot be built on, or a base could be walled in with its own economy
 * underneath it.
 */
export const OCCUPIED_RESERVED = 2;

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

  /**
   * Where each player's first Command Post is centred.
   *
   * In mirrored halves: `starts[i]` and `starts[i + n/2]` are exact 180-degree
   * rotations of one another, so the first half is one team's side of the map
   * and the second half is the other's. See `Layout.bases`.
   */
  readonly starts: StartLocation[] = [];

  /**
   * Second base sites: clear ground with a mineral line already on it, which a
   * player can claim by building a Command Post.
   *
   * Terrain, not ownership — nobody holds an expansion until they build on it,
   * and the simulation treats a Command Post here exactly like the starting
   * one. Stored in mirrored halves, like `starts`, so neither team is nearer to
   * more of them.
   */
  readonly expansions: StartLocation[] = [];

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
    return this.tiles[i] !== Tile.Cliff && this.occupied[i] !== OCCUPIED_SOLID;
  }

  /**
   * A building of `footprint` size may be placed with its top-left here.
   *
   * Stricter than `isWalkable`: reserved tiles are walkable but not buildable.
   */
  canPlace(tx: number, ty: number, footprint: number): boolean {
    for (let y = ty; y < ty + footprint; y++) {
      for (let x = tx; x < tx + footprint; x++) {
        if (!this.inBounds(x, y)) return false;
        const i = this.index(x, y);
        if (this.tiles[i] !== Tile.Ground) return false;
        if (this.occupied[i] !== UNOCCUPIED) return false;
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
 * Generate a symmetric map from a seed and a layout.
 *
 * The layout is carved rather than scattered — see `mapgen.ts` for the shapes
 * and why. Both peers run this with the same seed and layout, agreed in the
 * lobby before either side sends anything, and must produce byte-identical
 * terrain, so it uses only the seeded RNG.
 */
export function generateMap(
  seed: number,
  size = MAP_SIZE,
  kind: MapLayout = MapLayout.Lanes,
): GameMap {
  const map = new GameMap(size);

  const { bases, expansions } = carveLayout(map.tiles, map.elevation, size, seed, kind);
  for (const b of bases) map.starts.push({ tileX: b.x, tileY: b.y });
  for (const e of expansions) map.expansions.push({ tileX: e.x, tileY: e.y });

  // The lanes are carved to connect by construction, but a jitter that pinched a
  // corridor shut would be a silent, unwinnable match — so it is still checked,
  // and repaired if it ever happens.
  //
  // Every start must reach the first one, which on a map where reachability is
  // symmetric makes them all mutually reachable. Four starts need this more than
  // two did: a corridor pinched between two allied bases is not a lost match,
  // just a partner who can never be helped, which is worse for being subtle.
  for (let i = 1; i < map.starts.length; i++) {
    ensureConnected(map, map.starts[0]!, map.starts[i]!);
  }
  // Terrain is final from here on; freeze its hash so per-tick checksums are cheap.
  map.sealTerrain();
  return map;
}

/** The map size a layout is designed for. */
export function mapSizeFor(kind: MapLayout): number {
  return layoutSize(kind);
}

/**
 * Guarantee two starts can reach each other.
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
  // Mirrored as it goes, like every other write to the grid — a repair that
  // opened one side only would hand a player a shortcut nobody designed.
  let x = a.tileX;
  let y = a.tileY;
  while (x !== b.tileX || y !== b.tileY) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        openBoth(map, x + dx, y + dy);
      }
    }
    if (x !== b.tileX) x += x < b.tileX ? 1 : -1;
    if (y !== b.tileY) y += y < b.tileY ? 1 : -1;
  }
}

function openBoth(map: GameMap, tx: number, ty: number): void {
  const w = map.width;
  const h = map.height;
  if (tx < 0 || ty < 0 || tx >= w || ty >= h) return;
  for (let k = 0; k < 2; k++) {
    const x = k === 0 ? tx : w - 1 - tx;
    const y = k === 0 ? ty : h - 1 - ty;
    const i = map.index(x, y);
    map.tiles[i] = Tile.Ground;
    map.elevation[i] = 0;
  }
}
