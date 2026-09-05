/**
 * The canonical frame, and what an observation remembers about itself.
 *
 * The map is symmetric under a 180-degree rotation and so is the simulation,
 * so a policy written for the first seat plays every seat: an observation is
 * built with the viewer's half rotated to the front (`World.flipOf`), and a
 * decision is rotated back on the way out. The seat is invisible to the model.
 *
 * A `Frame` is what the decoder needs to turn a decision back into a command:
 * which entity each row of the table stood for, and which way the frame was
 * facing. Handles are generation-checked, so a frame a few ticks old still
 * names the right units, and a dead one is simply dropped.
 */

import { fromInt, toInt, type Fix } from '../../sim/fixed.js';
import { mirrorTile } from '../../sim/map.js';
import { NO_ENTITY, type EntityId, type PlayerId } from '../../sim/types.js';
import { CELL_TILES, GRID, N_ENT } from './spec.js';

export enum RowKind {
  Empty = 0,
  OwnUnit = 1,
  OwnBuilding = 2,
  Ally = 3,
  EnemyVisible = 4,
  EnemyRemembered = 5,
  Patch = 6,
}

export interface Frame {
  tick: number;
  viewer: PlayerId;
  flip: boolean;
  width: number;
  height: number;
  /** The entity each row stands for, or NO_ENTITY. */
  readonly rows: Int32Array;
  readonly rowKind: Uint8Array;
  /** Row by entity handle, for encoding a command into a decision. */
  readonly rowOf: Map<EntityId, number>;
}

export function allocFrame(): Frame {
  return {
    tick: 0,
    viewer: 0,
    flip: false,
    width: 0,
    height: 0,
    rows: new Int32Array(N_ENT).fill(NO_ENTITY),
    rowKind: new Uint8Array(N_ENT),
    rowOf: new Map(),
  };
}

/** A world coordinate in the canonical frame: rotated when the viewer sits in the second half. */
export function canonFix(x: Fix, size: number, flip: boolean): Fix {
  return flip ? fromInt(size) - x : x;
}

/** The canonical tile coordinate of a world tile coordinate. */
export function canonTileCoord(t: number, size: number, flip: boolean): number {
  return flip ? size - 1 - t : t;
}

/** The world top-left of a footprint whose canonical top-left is `t`, and back. */
export function uncanonTopLeft(t: number, size: number, footprint: number, flip: boolean): number {
  return flip ? mirrorTile(size, t, footprint) : t;
}

/** The canonical tile a canonical position falls in. */
export function canonTileOf(cx: Fix): number {
  return toInt(cx);
}

/** The cell a canonical tile coordinate pair belongs to, or -1 outside the grid. */
export function cellOf(tx: number, ty: number): number {
  const cx = Math.floor(tx / CELL_TILES);
  const cy = Math.floor(ty / CELL_TILES);
  if (cx < 0 || cy < 0 || cx >= GRID || cy >= GRID) return -1;
  return cy * GRID + cx;
}

/** The sub-cell index of a canonical tile inside its cell. */
export function subOf(tx: number, ty: number): number {
  return (
    (ty - Math.floor(ty / CELL_TILES) * CELL_TILES) * CELL_TILES +
    (tx - Math.floor(tx / CELL_TILES) * CELL_TILES)
  );
}

/** The canonical tile a cell and sub-cell name. */
export function tileOfCell(cell: number, sub: number): { tx: number; ty: number } {
  const cx = cell % GRID;
  const cy = Math.floor(cell / GRID);
  return {
    tx: cx * CELL_TILES + (sub % CELL_TILES),
    ty: cy * CELL_TILES + Math.floor(sub / CELL_TILES),
  };
}
