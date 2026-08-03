/**
 * Uniform grid for proximity queries (target acquisition, unit separation).
 *
 * Rebuilt from scratch every tick by counting sort. That sounds wasteful, but it
 * is a linear pass over a few thousand int32s with no allocation and no hashing,
 * and — more importantly — it produces a bucket layout that depends only on
 * entity positions, never on insertion history. An incrementally maintained
 * structure would encode the order things were added, which is exactly the kind
 * of hidden state that diverges between peers.
 *
 * Within a bucket, entities always appear in ascending index order, because the
 * scatter pass walks entities in that order. Every consumer therefore iterates
 * neighbours identically.
 */

import { toInt, type Fix } from './fixed.js';
import { ENTITY_CAPACITY, type EntityPool } from './entities.js';

/** Bucket edge length in world units. Roughly the widest common query radius. */
export const CELL_SIZE = 8;

export class SpatialGrid {
  readonly cellsX: number;
  readonly cellsY: number;
  private readonly cellStart: Int32Array;
  private readonly items: Int32Array;
  private readonly counts: Int32Array;

  constructor(mapSize: number) {
    this.cellsX = Math.ceil(mapSize / CELL_SIZE);
    this.cellsY = Math.ceil(mapSize / CELL_SIZE);
    const n = this.cellsX * this.cellsY;
    this.cellStart = new Int32Array(n + 1);
    this.counts = new Int32Array(n);
    this.items = new Int32Array(ENTITY_CAPACITY);
  }

  private cellOf(x: Fix, y: Fix): number {
    let cx = (toInt(x) / CELL_SIZE) | 0;
    let cy = (toInt(y) / CELL_SIZE) | 0;
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= this.cellsX) cx = this.cellsX - 1;
    if (cy >= this.cellsY) cy = this.cellsY - 1;
    return cy * this.cellsX + cx;
  }

  /** Rebuild the index for the current entity positions. */
  rebuild(pool: EntityPool): void {
    const nCells = this.cellsX * this.cellsY;
    this.counts.fill(0);

    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      this.counts[this.cellOf(pool.posX[i]!, pool.posY[i]!)]! += 1;
    }

    let running = 0;
    for (let c = 0; c < nCells; c++) {
      this.cellStart[c] = running;
      running += this.counts[c]!;
    }
    this.cellStart[nCells] = running;

    // Reuse `counts` as a per-cell write cursor during the scatter pass.
    this.counts.fill(0);
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const c = this.cellOf(pool.posX[i]!, pool.posY[i]!);
      this.items[this.cellStart[c]! + this.counts[c]!] = i;
      this.counts[c]! += 1;
    }
  }

  /**
   * Visit every entity within `radius` of (x, y), in a deterministic order.
   *
   * The callback receives entity slot indices, not handles — callers are already
   * inside the simulation and iterating the pool directly. Cells are visited row
   * by row and entities within a cell in ascending index order, so two peers see
   * neighbours in exactly the same sequence. That matters for anything that
   * picks "the first valid target".
   */
  forEachNear(x: Fix, y: Fix, radius: Fix, fn: (index: number) => void): void {
    const r = toInt(radius) + 1;
    const minX = Math.max(0, ((toInt(x) - r) / CELL_SIZE) | 0);
    const maxX = Math.min(this.cellsX - 1, ((toInt(x) + r) / CELL_SIZE) | 0);
    const minY = Math.max(0, ((toInt(y) - r) / CELL_SIZE) | 0);
    const maxY = Math.min(this.cellsY - 1, ((toInt(y) + r) / CELL_SIZE) | 0);

    for (let cy = minY; cy <= maxY; cy++) {
      for (let cx = minX; cx <= maxX; cx++) {
        const c = cy * this.cellsX + cx;
        const start = this.cellStart[c]!;
        const end = this.cellStart[c + 1]!;
        for (let k = start; k < end; k++) {
          fn(this.items[k]!);
        }
      }
    }
  }
}
