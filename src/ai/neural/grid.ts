/**
 * The map, indexed the way the observation reads it.
 *
 * Which cell a tile falls in, which tiles are terrain-walkable, and which
 * world tile a (cell, sub-cell, footprint) names for the viewer's frame are
 * all fixed for the life of a match. They were being recomputed on every
 * decision — a hundred thousand small allocations a decision, most of the
 * codec's cost — so they are computed once per map and frame and shared.
 */

import { defOf } from '../../config/rules.js';
import type { GameMap } from '../../sim/map.js';
import { EntityType, Tile } from '../../sim/types.js';
import { canonTileCoord, cellOf, tileOfCell, uncanonTopLeft } from './frame.js';
import { GRID, SUB } from './spec.js';

const CELLS = GRID * GRID;

/** The building footprints a `Build` may name. */
export const BUILD_FOOTPRINTS: readonly { type: EntityType; footprint: number }[] = [
  EntityType.CommandPost,
  EntityType.Depot,
  EntityType.Barracks,
  EntityType.Turret,
].map((type) => ({ type, footprint: defOf(type).footprint }));

export class GridIndex {
  /** Canonical cell of each world tile, or -1 off the grid. */
  readonly tileCell: Int32Array;
  /** Tiles in each cell. */
  readonly cellTiles: Int32Array;
  /** Cells with at least one tile that is not a cliff. */
  readonly walkableCell: Uint8Array;
  /** World tile a tile is not a cliff. */
  readonly notCliff: Uint8Array;
  /** World tile is Ground. */
  readonly ground: Uint8Array;
  /**
   * Per building type: the world top-left tile index a (cell, sub) names for
   * that footprint, or -1 where the footprint would leave the map.
   */
  private readonly topLefts = new Map<EntityType, Int32Array>();

  constructor(map: GameMap, flip: boolean) {
    const W = map.width;
    const H = map.height;
    const tiles = W * H;
    this.tileCell = new Int32Array(tiles);
    this.cellTiles = new Int32Array(CELLS);
    this.walkableCell = new Uint8Array(CELLS);
    this.notCliff = new Uint8Array(tiles);
    this.ground = new Uint8Array(tiles);
    for (let t = 0; t < tiles; t++) {
      const ci = map.canonicalIndex(t, flip);
      const cell = cellOf(ci % W, Math.floor(ci / W));
      this.tileCell[t] = cell;
      this.notCliff[t] = map.tiles[t] !== Tile.Cliff ? 1 : 0;
      this.ground[t] = map.tiles[t] === Tile.Ground ? 1 : 0;
      if (cell < 0) continue;
      this.cellTiles[cell]!++;
      if (this.notCliff[t] === 1) this.walkableCell[cell] = 1;
    }
    for (const { type, footprint } of BUILD_FOOTPRINTS) {
      const table = new Int32Array(CELLS * SUB).fill(-1);
      for (let cell = 0; cell < CELLS; cell++) {
        for (let s = 0; s < SUB; s++) {
          const { tx, ty } = tileOfCell(cell, s);
          const wx = uncanonTopLeft(tx, W, footprint, flip);
          const wy = uncanonTopLeft(ty, H, footprint, flip);
          if (wx < 0 || wy < 0 || wx + footprint > W || wy + footprint > H) continue;
          table[cell * SUB + s] = wy * W + wx;
        }
      }
      this.topLefts.set(type, table);
    }
    void canonTileCoord;
  }

  topLeftsFor(type: EntityType): Int32Array {
    const table = this.topLefts.get(type);
    if (!table) throw new Error(`no footprint table for entity type ${type}`);
    return table;
  }
}

const cache = new WeakMap<GameMap, [GridIndex | null, GridIndex | null]>();

/** The index for a map and frame, built on first use. */
export function gridIndexFor(map: GameMap, flip: boolean): GridIndex {
  let pair = cache.get(map);
  if (!pair) {
    pair = [null, null];
    cache.set(map, pair);
  }
  const k = flip ? 1 : 0;
  let index = pair[k];
  if (!index) {
    index = new GridIndex(map, flip);
    pair[k] = index;
  }
  return index;
}
