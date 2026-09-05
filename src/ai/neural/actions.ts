/**
 * What the neural bot can say, and how it becomes a command.
 *
 * A decision is a handful of choices made in order — what to do, with whom,
 * what kind, at what, where, exactly where — each drawn from a small fixed
 * vocabulary and each masked to what is legal *right now*, so a decision the
 * simulation would silently drop is never made. The vocabulary is the human
 * one: every command a player's UI can produce except Surrender, with the
 * UI's own limits — one building per Train, cancel slot zero only, a worker
 * and a top-left tile for Build, at most `SELECTION_MAX` units.
 *
 * `decode` turns a decision plus the frame it was made in back into one
 * `Command`, rotated out of the canonical frame; `encode` does the reverse for
 * imitation, turning a command into the decision that would have produced it.
 * The masks are computed here too, from the same frame, so training and play
 * agree about legality to the byte.
 */

import { defOf, MAX_PRODUCTION_QUEUE } from '../../config/rules.js';
import { CommandType, type Command } from '../../sim/commands.js';
import { idIndex } from '../../sim/entities.js';
import { FIX_HALF, fromInt, toInt } from '../../sim/fixed.js';
import { UNOCCUPIED } from '../../sim/map.js';
import {
  BuildState,
  EntityType,
  ENTITY_TYPE_COUNT,
  NO_ENTITY,
  type EntityId,
} from '../../sim/types.js';
import type { World } from '../../sim/world.js';
import { VISIBLE, type Visibility } from '../../vision/visibility.js';
import {
  canonFix,
  canonTileCoord,
  canonTileOf,
  cellOf,
  RowKind,
  subOf,
  tileOfCell,
  uncanonTopLeft,
  type Frame,
} from './frame.js';
import { BUILD_FOOTPRINTS, gridIndexFor } from './grid.js';
import type { EntityMemory } from './memory.js';
import {
  ACTION_INTS,
  ACTION_TYPE_COUNT,
  ActionType,
  CELL_TILES,
  GRID,
  N_ENT,
  SELECTION_MAX,
  SUB,
} from './spec.js';

const CELLS = GRID * GRID;

export interface Action {
  type: number;
  entityType: number;
  target: number;
  cell: number;
  sub: number;
  /** Rows, -1 padded. A single-actor decision uses the first entry. */
  readonly selection: Int32Array;
}

export function allocAction(): Action {
  return {
    type: 0,
    entityType: -1,
    target: -1,
    cell: -1,
    sub: -1,
    selection: new Int32Array(SELECTION_MAX).fill(-1),
  };
}

export function clearAction(action: Action): void {
  action.type = ActionType.Noop;
  action.entityType = -1;
  action.target = -1;
  action.cell = -1;
  action.sub = -1;
  action.selection.fill(-1);
}

/** The flat form shared with Python and the worker: type, entityType, target, cell, sub, rows. */
export function actionToInts(action: Action, out: Int32Array): void {
  if (out.length < ACTION_INTS) throw new Error(`action needs ${ACTION_INTS} ints`);
  out[0] = action.type;
  out[1] = action.entityType;
  out[2] = action.target;
  out[3] = action.cell;
  out[4] = action.sub;
  for (let k = 0; k < SELECTION_MAX; k++) out[5 + k] = action.selection[k]!;
}

export function actionFromInts(ints: ArrayLike<number>, out: Action): void {
  out.type = ints[0]!;
  out.entityType = ints[1]!;
  out.target = ints[2]!;
  out.cell = ints[3]!;
  out.sub = ints[4]!;
  for (let k = 0; k < SELECTION_MAX; k++) out.selection[k] = ints[5 + k]!;
}

/** Which heads a type uses. */
export function selectsMany(type: number): boolean {
  return (
    type === ActionType.Move ||
    type === ActionType.AttackMove ||
    type === ActionType.Attack ||
    type === ActionType.Harvest ||
    type === ActionType.Stop ||
    type === ActionType.Hold
  );
}
export function selectsOne(type: number): boolean {
  return (
    type === ActionType.Build ||
    type === ActionType.Train ||
    type === ActionType.CancelTrain ||
    type === ActionType.SetRally
  );
}
export function usesTarget(type: number): boolean {
  return type === ActionType.Attack || type === ActionType.Harvest;
}
export function usesLocation(type: number): boolean {
  return (
    type === ActionType.Move ||
    type === ActionType.AttackMove ||
    type === ActionType.Build ||
    type === ActionType.SetRally
  );
}
export function usesEntityType(type: number): boolean {
  return type === ActionType.Build || type === ActionType.Train;
}

export interface Masks {
  /** [ACTION_TYPE_COUNT] */
  readonly type: Uint8Array;
  /** [ACTION_TYPE_COUNT × N_ENT] rows a type may select. */
  readonly selection: Uint8Array;
  /** [ACTION_TYPE_COUNT × N_ENT] rows a type may target. */
  readonly target: Uint8Array;
  /** [ACTION_TYPE_COUNT × CELLS] cells a type may point at. Build uses `buildCell`. */
  readonly cell: Uint8Array;
  /** [ENTITY_TYPE_COUNT × CELLS] cells with a legal, visible top-left for a building type. */
  readonly buildCell: Uint8Array;
  /** [N_ENT × ENTITY_TYPE_COUNT] what a production-building row may train now. */
  readonly rowEntityType: Uint8Array;
  /** [ENTITY_TYPE_COUNT] building types affordable with somewhere to go. */
  readonly buildType: Uint8Array;
}

export function allocMasks(): Masks {
  return {
    type: new Uint8Array(ACTION_TYPE_COUNT),
    selection: new Uint8Array(ACTION_TYPE_COUNT * N_ENT),
    target: new Uint8Array(ACTION_TYPE_COUNT * N_ENT),
    cell: new Uint8Array(ACTION_TYPE_COUNT * CELLS),
    buildCell: new Uint8Array(ENTITY_TYPE_COUNT * CELLS),
    rowEntityType: new Uint8Array(N_ENT * ENTITY_TYPE_COUNT),
    buildType: new Uint8Array(ENTITY_TYPE_COUNT),
  };
}

/** The buildings a worker can raise, in EntityType order. */
export const BUILDINGS: readonly EntityType[] = [
  EntityType.CommandPost,
  EntityType.Depot,
  EntityType.Barracks,
  EntityType.Turret,
];

/** Scratch for the placement scan, reused across calls. */
let freeScratch = new Uint8Array(0);
let sumScratch = new Int32Array(0);
let visibleScratch = new Uint8Array(0);

function any(mask: Uint8Array, offset: number, length: number): boolean {
  for (let k = 0; k < length; k++) if (mask[offset + k] === 1) return true;
  return false;
}

/**
 * Fill `out` with what is legal for the viewer in the frame's tick.
 *
 * Legality follows the simulation's own rules (`orders.ts`) and the UI's caps,
 * plus one restriction the UI does not have: a building may only be placed in
 * a cell the side can currently see. Placement legality depends on occupancy,
 * and offering it for ground in the fog would tell the bot what stands there.
 */
export function computeMasks(
  world: World,
  frame: Frame,
  vis: Visibility,
  mem: EntityMemory,
  out: Masks,
): void {
  const pool = world.pool;
  const map = world.map;
  const viewer = frame.viewer;
  const flip = frame.flip;
  const W = map.width;
  const H = map.height;
  const minerals = world.player(viewer).minerals;

  out.type.fill(0);
  out.selection.fill(0);
  out.target.fill(0);
  out.cell.fill(0);
  out.buildCell.fill(0);
  out.rowEntityType.fill(0);
  out.buildType.fill(0);

  // --- rows -----------------------------------------------------------------
  for (let r = 0; r < N_ENT; r++) {
    const kind = frame.rowKind[r]!;
    if (kind === RowKind.Empty) continue;
    const id = frame.rows[r]!;
    if (kind === RowKind.OwnUnit) {
      if (!pool.isAlive(id)) continue;
      const i = idIndex(id);
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      for (const t of [ActionType.Move, ActionType.AttackMove, ActionType.Stop, ActionType.Hold]) {
        out.selection[t * N_ENT + r] = 1;
      }
      if (def.attackRange !== 0) out.selection[ActionType.Attack * N_ENT + r] = 1;
      if (type === EntityType.Worker) {
        out.selection[ActionType.Harvest * N_ENT + r] = 1;
        out.selection[ActionType.Build * N_ENT + r] = 1;
      }
    } else if (kind === RowKind.OwnBuilding) {
      if (!pool.isAlive(id)) continue;
      const i = idIndex(id);
      const def = defOf(pool.type[i]! as EntityType);
      if (pool.buildState[i] === BuildState.Complete && def.produces.length > 0) {
        out.selection[ActionType.SetRally * N_ENT + r] = 1;
        if (pool.prodCount[i]! < MAX_PRODUCTION_QUEUE) {
          for (const unit of def.produces) {
            if (minerals >= defOf(unit).mineralCost)
              out.rowEntityType[r * ENTITY_TYPE_COUNT + unit] = 1;
          }
          if (any(out.rowEntityType, r * ENTITY_TYPE_COUNT, ENTITY_TYPE_COUNT)) {
            out.selection[ActionType.Train * N_ENT + r] = 1;
          }
        }
      }
      if (pool.prodCount[i]! > 0) out.selection[ActionType.CancelTrain * N_ENT + r] = 1;
    } else if (kind === RowKind.EnemyVisible) {
      out.target[ActionType.Attack * N_ENT + r] = 1;
    } else if (kind === RowKind.Patch) {
      const entry = mem.get(id);
      if (entry && entry.resourceAmount > 0) out.target[ActionType.Harvest * N_ENT + r] = 1;
    }
  }

  // --- cells: anywhere the terrain allows walking, for a point ----------------
  const index = gridIndexFor(map, flip);
  const walkable = index.walkableCell;
  if (visibleScratch.length < CELLS) visibleScratch = new Uint8Array(CELLS);
  const visibleCell = visibleScratch;
  visibleCell.fill(0);
  const state = vis.state;
  const tileCell = index.tileCell;
  for (let t = 0; t < W * H; t++) {
    if (state[t] === VISIBLE) {
      const cell = tileCell[t]!;
      if (cell >= 0) visibleCell[cell] = 1;
    }
  }
  for (const t of [ActionType.Move, ActionType.AttackMove, ActionType.SetRally]) {
    out.cell.set(walkable, t * CELLS);
  }

  // --- cells a building can go in -----------------------------------------------
  // Legal top-lefts per footprint via a summed-area table over free tiles, in
  // the world frame; then each canonical cell asks whether any of its tiles,
  // rotated back, is one — through a table built once per map and frame.
  const tiles = W * H;
  if (freeScratch.length < tiles) {
    freeScratch = new Uint8Array(tiles);
    sumScratch = new Int32Array((W + 1) * (H + 1));
  }
  const free = freeScratch;
  const sum = sumScratch;
  const occupied = map.occupied;
  for (let t = 0; t < tiles; t++) {
    free[t] = index.ground[t] === 1 && occupied[t] === UNOCCUPIED ? 1 : 0;
  }
  const stride = W + 1;
  for (let y = 0; y <= H; y++) sum[y * stride] = 0;
  for (let x = 0; x <= W; x++) sum[x] = 0;
  for (let y = 1; y <= H; y++) {
    let rowSum = 0;
    for (let x = 1; x <= W; x++) {
      rowSum += free[(y - 1) * W + (x - 1)]!;
      sum[y * stride + x] = sum[(y - 1) * stride + x]! + rowSum;
    }
  }
  // Only cells in view can take a building, and they are a small part of the
  // map, so legality is asked of the table cell by cell rather than computed
  // for every tile first.
  for (const { type: building, footprint: f } of BUILD_FOOTPRINTS) {
    const affordable = minerals >= defOf(building).mineralCost;
    const topLefts = index.topLeftsFor(building);
    const area = f * f;
    for (let cell = 0; cell < CELLS; cell++) {
      if (visibleCell[cell] !== 1) continue;
      let ok = false;
      const base = cell * SUB;
      for (let s = 0; s < SUB && !ok; s++) {
        const t = topLefts[base + s]!;
        if (t < 0) continue;
        const x = t % W;
        const y = (t - x) / W;
        const covered =
          sum[(y + f) * stride + (x + f)]! -
          sum[y * stride + (x + f)]! -
          sum[(y + f) * stride + x]! +
          sum[y * stride + x]!;
        if (covered === area) ok = true;
      }
      if (ok) out.buildCell[building * CELLS + cell] = 1;
    }
    if (affordable && any(out.buildCell, building * CELLS, CELLS)) out.buildType[building] = 1;
  }

  // --- types ---------------------------------------------------------------------
  out.type[ActionType.Noop] = 1;
  const anyWalkable = any(walkable, 0, CELLS);
  const canBuild = any(out.buildType, 0, ENTITY_TYPE_COUNT);
  for (let t = 1; t < ACTION_TYPE_COUNT; t++) {
    let ok = any(out.selection, t * N_ENT, N_ENT);
    if (ok && usesTarget(t)) ok = any(out.target, t * N_ENT, N_ENT);
    if (ok && t === ActionType.Build) ok = canBuild;
    if (ok && usesLocation(t) && t !== ActionType.Build) ok = anyWalkable;
    out.type[t] = ok ? 1 : 0;
  }
}

/**
 * Is a decision legal under these masks? Illegal selection rows are removed;
 * a decision that keeps no legal choice on a head it uses is not legal.
 *
 * A teacher's command is encoded into a decision after the fact, and the
 * teacher — which sees everything and decided earlier — may have said what
 * the student could not: a Train the bank no longer covers, a Build in a cell
 * the side cannot see. Such a label is dropped rather than taught.
 */
export function legalise(action: Action, masks: Masks): boolean {
  const type = action.type;
  if (type < 0 || type >= ACTION_TYPE_COUNT || masks.type[type] !== 1) return false;
  if (type === ActionType.Noop) return true;
  const rows = selectsMany(type) ? SELECTION_MAX : selectsOne(type) ? 1 : 0;
  let kept = 0;
  for (let k = 0; k < rows; k++) {
    const r = action.selection[k]!;
    if (r >= 0 && r < N_ENT && masks.selection[type * N_ENT + r] === 1) {
      action.selection[kept++] = r;
    }
  }
  for (let k = kept; k < SELECTION_MAX; k++) action.selection[k] = -1;
  if (rows > 0 && kept === 0) return false;
  if (usesEntityType(type)) {
    const e = action.entityType;
    if (e < 0 || e >= ENTITY_TYPE_COUNT) return false;
    if (type === ActionType.Build) {
      if (masks.buildType[e] !== 1) return false;
    } else if (masks.rowEntityType[action.selection[0]! * ENTITY_TYPE_COUNT + e] !== 1) {
      return false;
    }
  }
  if (usesTarget(type)) {
    const t = action.target;
    if (t < 0 || t >= N_ENT || masks.target[type * N_ENT + t] !== 1) return false;
  }
  if (usesLocation(type)) {
    const c = action.cell;
    if (c < 0 || c >= CELLS || action.sub < 0 || action.sub >= SUB) return false;
    const legal =
      type === ActionType.Build
        ? masks.buildCell[action.entityType * CELLS + c]
        : masks.cell[type * CELLS + c];
    if (legal !== 1) return false;
  }
  return true;
}

/** A canonical tile's centre as a world position. */
function pointAt(tx: number, ty: number, frame: Frame): { x: number; y: number } {
  return {
    x: canonFix(fromInt(tx) + FIX_HALF, frame.width, frame.flip),
    y: canonFix(fromInt(ty) + FIX_HALF, frame.height, frame.flip),
  };
}

/**
 * The command a decision means, or null for a Noop or a decision whose
 * entities have gone.
 *
 * A Build whose exact tile is no longer legal is snapped to the first legal
 * top-left in its cell, in sub-cell order; the human UI refuses instead, but a
 * bot has no cursor to nudge, and the cell is what it chose.
 */
export function decode(action: Action, world: World, frame: Frame): Command | null {
  const pool = world.pool;
  const player = frame.viewer;
  const type = action.type as ActionType;
  if (type === ActionType.Noop) return null;

  const units: EntityId[] = [];
  const rows = selectsMany(type) ? SELECTION_MAX : selectsOne(type) ? 1 : 0;
  for (let k = 0; k < rows; k++) {
    const r = action.selection[k]!;
    if (r < 0 || r >= N_ENT) continue;
    const id = frame.rows[r]!;
    if (id === NO_ENTITY || !pool.isAlive(id)) continue;
    units.push(id);
  }
  if (rows > 0 && units.length === 0) return null;

  let target = NO_ENTITY;
  if (usesTarget(type)) {
    if (action.target < 0 || action.target >= N_ENT) return null;
    target = frame.rows[action.target]!;
    if (target === NO_ENTITY) return null;
  }

  let tx = -1;
  let ty = -1;
  if (usesLocation(type)) {
    if (action.cell < 0 || action.cell >= CELLS || action.sub < 0 || action.sub >= SUB) return null;
    const tile = tileOfCell(action.cell, action.sub);
    tx = tile.tx;
    ty = tile.ty;
  }

  switch (type) {
    case ActionType.Move:
    case ActionType.AttackMove: {
      const p = pointAt(tx, ty, frame);
      return {
        type: type === ActionType.Move ? CommandType.Move : CommandType.AttackMove,
        player,
        units,
        x: p.x,
        y: p.y,
      };
    }
    case ActionType.Attack:
      return { type: CommandType.Attack, player, units, target };
    case ActionType.Harvest:
      return { type: CommandType.Harvest, player, units, target };
    case ActionType.Stop:
      return { type: CommandType.Stop, player, units };
    case ActionType.Hold:
      return { type: CommandType.Hold, player, units };
    case ActionType.Build: {
      const building = action.entityType as EntityType;
      if (!BUILDINGS.includes(building)) return null;
      const f = defOf(building).footprint;
      const place = (ctx: number, cty: number): { x: number; y: number } | null => {
        const wx = uncanonTopLeft(ctx, frame.width, f, frame.flip);
        const wy = uncanonTopLeft(cty, frame.height, f, frame.flip);
        return world.map.canPlace(wx, wy, f) ? { x: wx, y: wy } : null;
      };
      let spot = place(tx, ty);
      for (let s = 0; s < SUB && spot === null; s++) {
        const alt = tileOfCell(action.cell, s);
        spot = place(alt.tx, alt.ty);
      }
      if (spot === null) return null;
      return {
        type: CommandType.Build,
        player,
        worker: units[0]!,
        building,
        tileX: spot.x,
        tileY: spot.y,
      };
    }
    case ActionType.Train:
      if (action.entityType < 0 || action.entityType >= ENTITY_TYPE_COUNT) return null;
      return {
        type: CommandType.Train,
        player,
        building: units[0]!,
        unit: action.entityType as EntityType,
      };
    case ActionType.CancelTrain:
      return { type: CommandType.CancelTrain, player, building: units[0]!, slot: 0 };
    case ActionType.SetRally: {
      const p = pointAt(tx, ty, frame);
      return { type: CommandType.SetRally, player, building: units[0]!, x: p.x, y: p.y };
    }
    default:
      return null;
  }
}

/**
 * The decision that would have produced `command` in `frame`, or null when
 * the frame cannot express it — a unit or target not in the table, a point
 * off the grid, a Surrender.
 *
 * Units the table does not hold are dropped rather than failing the whole
 * label; a teacher that sees more than the student can only be imitated as
 * far as the student can see.
 */
export function encode(command: Command, frame: Frame, out: Action): boolean {
  clearAction(out);
  const rowsOf = (ids: readonly EntityId[]): number[] => {
    const rows: number[] = [];
    for (const id of ids) {
      const r = frame.rowOf.get(id);
      if (r !== undefined) rows.push(r);
    }
    return rows;
  };
  const locate = (x: number, y: number): boolean => {
    const tx = canonTileOf(canonFix(x, frame.width, frame.flip));
    const ty = canonTileOf(canonFix(y, frame.height, frame.flip));
    // A rotated position on a tile boundary lands one tile over; a point is
    // only ever a tile centre here, so the floor is exact either way.
    const cell = cellOf(tx, ty);
    if (cell < 0) return false;
    out.cell = cell;
    out.sub = subOf(tx, ty);
    return true;
  };
  const single = (id: EntityId): boolean => {
    const r = frame.rowOf.get(id);
    if (r === undefined) return false;
    out.selection[0] = r;
    return true;
  };
  const many = (ids: readonly EntityId[]): boolean => {
    const rows = rowsOf(ids);
    if (rows.length === 0) return false;
    for (let k = 0; k < Math.min(rows.length, SELECTION_MAX); k++) out.selection[k] = rows[k]!;
    return true;
  };

  switch (command.type) {
    case CommandType.Move:
    case CommandType.AttackMove:
      out.type = command.type === CommandType.Move ? ActionType.Move : ActionType.AttackMove;
      return many(command.units) && locate(command.x, command.y);
    case CommandType.Attack:
    case CommandType.Harvest: {
      out.type = command.type === CommandType.Attack ? ActionType.Attack : ActionType.Harvest;
      const t = frame.rowOf.get(command.target);
      if (t === undefined) return false;
      out.target = t;
      return many(command.units);
    }
    case CommandType.Stop:
      out.type = ActionType.Stop;
      return many(command.units);
    case CommandType.Hold:
      out.type = ActionType.Hold;
      return many(command.units);
    case CommandType.Build: {
      out.type = ActionType.Build;
      out.entityType = command.building;
      const f = defOf(command.building).footprint;
      const tx = uncanonTopLeft(command.tileX, frame.width, f, frame.flip);
      const ty = uncanonTopLeft(command.tileY, frame.height, f, frame.flip);
      const cell = cellOf(tx, ty);
      if (cell < 0) return false;
      out.cell = cell;
      out.sub = subOf(tx, ty);
      return single(command.worker);
    }
    case CommandType.Train:
      out.type = ActionType.Train;
      out.entityType = command.unit;
      return single(command.building);
    case CommandType.CancelTrain:
      out.type = ActionType.CancelTrain;
      return single(command.building);
    case CommandType.SetRally:
      out.type = ActionType.SetRally;
      return single(command.building) && locate(command.x, command.y);
    case CommandType.Surrender:
      return false;
    default:
      return false;
  }
}

/** For tests: the world tile a canonical tile coordinate names. */
export function worldTileCoord(t: number, size: number, flip: boolean): number {
  return canonTileCoord(t, size, flip);
}

export { CELL_TILES, toInt };
