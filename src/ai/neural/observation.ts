/**
 * What the neural bot sees: a fog-limited, canonical-frame picture of the world.
 *
 * Three parts, all written into preallocated typed arrays:
 *
 *  - an **entity table** of `N_ENT` rows — the viewer's own entities, then
 *    allies, then the enemies in view, then the enemies remembered, then the
 *    mineral patches discovered — each a fixed row of features;
 *  - a **coarse grid**, `GRID` cells a side, of terrain, vision and who is
 *    where;
 *  - **scalars**: the HUD, and a little of what the bot itself just did.
 *
 * Everything is in the viewer's canonical frame (`frame.ts`), so a bot in the
 * second seat sees the map the first seat sees. Row order is decided by
 * integer arithmetic on canonical coordinates and per-owner serials, so the
 * observation of seat `p + n/2` in a mirrored world is byte-for-byte the
 * observation of seat `p` — `tests/observation.test.ts` asserts exactly that.
 *
 * Nothing here reads an enemy the side cannot see or does not remember. The
 * two things the HUD leaks — selecting an enemy through the shroud, and the
 * panel printing its hp and queue — are deliberately not reproduced.
 */

import {
  defOf,
  MAX_PRODUCTION_QUEUE,
  MINERALS_PER_TRIP,
  PATCH_AMOUNT,
} from '../../config/rules.js';
import { toInt } from '../../sim/fixed.js';
import { OCCUPIED_SOLID } from '../../sim/map.js';
import { UNEXPLORED, VISIBLE } from '../../vision/visibility.js';
import {
  BuildState,
  EntityType,
  ENTITY_TYPE_COUNT,
  MapLayout,
  NEUTRAL,
  NO_ENTITY,
  Order,
  type EntityId,
  type PlayerId,
} from '../../sim/types.js';
import type { World } from '../../sim/world.js';
import type { Visibility } from '../../vision/visibility.js';
import { canonFix, canonTileCoord, cellOf, RowKind, type Frame } from './frame.js';
import { gridIndexFor } from './grid.js';
import type { EntityMemory } from './memory.js';
import {
  ACTION_TYPE_COUNT,
  CELL_TILES,
  CRITIC_LEN,
  CRITIC_PER_PLAYER,
  CRITIC_PLAYERS,
  ENTITY_FEATURE_COUNT,
  GRID,
  GRID_CHANNEL_COUNT,
  N_ENT,
  SCALAR_COUNT,
  UNIT_MEMORY_TICKS,
} from './spec.js';

export interface Observation {
  readonly entities: Float32Array;
  readonly entityMask: Uint8Array;
  readonly grid: Float32Array;
  readonly scalars: Float32Array;
}

export function allocObservation(): Observation {
  return {
    entities: new Float32Array(N_ENT * ENTITY_FEATURE_COUNT),
    entityMask: new Uint8Array(N_ENT),
    grid: new Float32Array(GRID_CHANNEL_COUNT * GRID * GRID),
    scalars: new Float32Array(SCALAR_COUNT),
  };
}

/** What the bot itself did lately — a person remembers their last click. */
export interface RecentActions {
  /** ActionType of the previous decision. */
  prevType: number;
  /** Decisions since the last one that was not a Noop. */
  sinceNonNoop: number;
  /** Commands issued in the last ten seconds. */
  recentCommands: number;
  /** Entities named by the previous command. */
  lastUnits: ReadonlySet<EntityId>;
}

export const NO_RECENT: RecentActions = {
  prevType: 0,
  sinceNonNoop: 0,
  recentCommands: 0,
  lastUnits: new Set(),
};

// Column indices, from ENTITY_FEATURES.
const F_TYPE = 0;
const F_REL = F_TYPE + ENTITY_TYPE_COUNT;
const F_X = F_REL + 4;
const F_Y = F_X + 1;
const F_DX = F_Y + 1;
const F_DY = F_DX + 1;
const F_HP = F_DY + 1;
const F_BUILD = F_HP + 1;
const F_BUILD_PROGRESS = F_BUILD + 3;
const F_ORDER = F_BUILD_PROGRESS + 1;
const F_CARRYING = F_ORDER + 7;
const F_PROD_COUNT = F_CARRYING + 1;
const F_PROD_PROGRESS = F_PROD_COUNT + 1;
const F_HAS_RALLY = F_PROD_PROGRESS + 1;
const F_RALLY_DX = F_HAS_RALLY + 1;
const F_RALLY_DY = F_RALLY_DX + 1;
const F_COOLDOWN = F_RALLY_DY + 1;
const F_VISIBLE = F_COOLDOWN + 1;
const F_MEMORY_AGE = F_VISIBLE + 1;
const F_RESOURCE = F_MEMORY_AGE + 1;
const F_IN_LAST = F_RESOURCE + 1;
const F_SUPPLY = F_IN_LAST + 1;
const F_FLYING = F_SUPPLY + 1;
const F_CAN_HIT_AIR = F_FLYING + 1;
const F_DIST_POST = F_CAN_HIT_AIR + 1;

// Grid channels, from GRID_CHANNELS.
const G_WALKABLE = 0;
const G_BUILDABLE = 1;
const G_EXPLORED = 2;
const G_VISIBLE = 3;
const G_OWN_BUILDINGS = 4;
const G_OWN_UNITS = 5;
const G_ALLY = 6;
const G_ENEMY_BUILDINGS = 7;
const G_ENEMY_UNITS = 8;
const G_MINERALS = 9;
const G_FRIENDLY_STARTS = 10;
const G_ENEMY_STARTS = 11;
const G_EXPANSIONS = 12;

// Scalars, from SCALARS.
const S_MINERALS = 0;
const S_SUPPLY_USED = 1;
const S_SUPPLY_MAX = 2;
const S_SUPPLY_FREE = 3;
const S_TICK = 4;
const S_LAYOUT = 5;
const S_SEAT = 7;
const S_ALLIES = 8;
const S_OWN_COUNTS = 9;
const S_ENEMY_COUNTS = S_OWN_COUNTS + ENTITY_TYPE_COUNT;
const S_PREV = S_ENEMY_COUNTS + ENTITY_TYPE_COUNT;
const S_SINCE_NON_NOOP = S_PREV + ACTION_TYPE_COUNT;
const S_RECENT = S_SINCE_NON_NOOP + 1;

const CELLS = GRID * GRID;
const ORDER_COLUMNS = [
  Order.None,
  Order.Move,
  Order.AttackMove,
  Order.Attack,
  Order.Harvest,
  Order.Build,
  Order.Hold,
];

function clip01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

interface Candidate {
  index: number;
  kind: RowKind;
  /** Sort key: lower first. */
  key: number;
  key2: number;
}

export class ObservationEncoder {
  private readonly candidates: Candidate[] = [];
  private readonly cellSum = new Float32Array(4 * CELLS);

  constructor(
    private readonly world: World,
    readonly viewer: PlayerId,
  ) {}

  /**
   * Fill `out` and `frame` from the world as the viewer sees it now.
   *
   * `vis` and `mem` must already be updated for this tick.
   */
  encode(
    vis: Visibility,
    mem: EntityMemory,
    recent: RecentActions,
    out: Observation,
    frame: Frame,
  ): void {
    const world = this.world;
    const pool = world.pool;
    const map = world.map;
    const viewer = this.viewer;
    const flip = world.flipOf(viewer);
    const W = map.width;
    const H = map.height;
    const size = Math.max(W, H);

    frame.tick = world.tick;
    frame.viewer = viewer;
    frame.flip = flip;
    frame.width = W;
    frame.height = H;
    frame.rows.fill(NO_ENTITY);
    frame.rowKind.fill(RowKind.Empty);
    frame.rowOf.clear();
    out.entities.fill(0);
    out.entityMask.fill(0);
    out.grid.fill(0);
    out.scalars.fill(0);

    // --- where things are, in the canonical frame -------------------------
    // The floor is taken in the canonical frame (`tileOfPosFor`), which is
    // what makes a position exactly on a tile boundary — every building's
    // centre — land in the same canonical tile from both seats.
    const canonX = (x: number, y: number): number =>
      canonTileCoord(map.tileXOf(map.tileOfPosFor(x, y, flip)), W, flip);
    const canonY = (x: number, y: number): number =>
      canonTileCoord(map.tileYOf(map.tileOfPosFor(x, y, flip)), H, flip);
    const canonTileX = (i: number): number => canonX(pool.posX[i]!, pool.posY[i]!);
    const canonTileY = (i: number): number => canonY(pool.posX[i]!, pool.posY[i]!);

    // The viewer's own army centroid, in canonical tiles, for ordering enemies
    // by how close they are to it — integer arithmetic only.
    let ownCount = 0;
    let sumX = 0;
    let sumY = 0;
    const start = map.starts[viewer]!;
    const startX = canonTileCoord(start.tileX, W, flip);
    const startY = canonTileCoord(start.tileY, H, flip);
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || pool.owner[i] !== viewer) continue;
      ownCount++;
      sumX += canonTileX(i);
      sumY += canonTileY(i);
    }
    const centreX = ownCount > 0 ? Math.floor(sumX / ownCount) : startX;
    const centreY = ownCount > 0 ? Math.floor(sumY / ownCount) : startY;

    // --- choose the rows ---------------------------------------------------
    const candidates = this.candidates;
    candidates.length = 0;
    const rememberedRows: {
      entry: (typeof mem.entries)[number];
      key: number;
      key2: number;
      kind: RowKind;
    }[] = [];
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const owner = pool.owner[i]!;
      const type = pool.type[i]! as EntityType;
      if (owner === viewer) {
        candidates.push({
          index: i,
          kind: defOf(type).isBuilding ? RowKind.OwnBuilding : RowKind.OwnUnit,
          key: 0,
          key2: pool.serial[i]!,
        });
      } else if (owner !== NEUTRAL && world.areAllied(owner, viewer)) {
        candidates.push({
          index: i,
          kind: RowKind.Ally,
          key: 1,
          key2: world.ownerCanonical(owner) * 1048576 + pool.serial[i]!,
        });
      }
    }
    for (const entry of mem.entries) {
      const seenNow = entry.lastSeen === world.tick;
      if (entry.type === EntityType.MineralPatch) {
        // Ordered by canonical position, not by a rotated corner: a footprint's
        // top-left rotates onto its bottom-right, and only its centre onto itself.
        rememberedRows.push({
          entry,
          kind: RowKind.Patch,
          key: 4,
          key2: canonY(entry.posX, entry.posY) * W + canonX(entry.posX, entry.posY),
        });
      } else if (seenNow) {
        const dx = canonX(entry.posX, entry.posY) - centreX;
        const dy = canonY(entry.posX, entry.posY) - centreY;
        rememberedRows.push({
          entry,
          kind: RowKind.EnemyVisible,
          key: 2,
          key2:
            (dx * dx + dy * dy) * 4194304 +
            world.ownerCanonical(entry.owner) * 1048576 +
            entry.serial,
        });
      } else {
        rememberedRows.push({
          entry,
          kind: RowKind.EnemyRemembered,
          key: 3,
          key2:
            (world.tick - entry.lastSeen) * 4194304 +
            world.ownerCanonical(entry.owner) * 1048576 +
            entry.serial,
        });
      }
    }
    candidates.sort((a, b) => a.key - b.key || a.key2 - b.key2);
    rememberedRows.sort((a, b) => a.key - b.key || a.key2 - b.key2);

    // --- the entity table --------------------------------------------------
    const E = out.entities;
    const F = ENTITY_FEATURE_COUNT;
    let row = 0;

    // Nearest own Command Post, for the distance feature.
    const posts: number[] = [];
    for (let i = 0; i < pool.count; i++) {
      if (
        pool.alive[i] === 1 &&
        pool.owner[i] === viewer &&
        pool.type[i] === EntityType.CommandPost
      )
        posts.push(i);
    }
    const distToPost = (tx: number, ty: number): number => {
      let best = Number.POSITIVE_INFINITY;
      for (const p of posts) {
        const dx = canonTileX(p) - tx;
        const dy = canonTileY(p) - ty;
        best = Math.min(best, Math.sqrt(dx * dx + dy * dy));
      }
      return posts.length === 0 ? 1 : clip01(best / size);
    };

    const writeCommon = (
      r: number,
      type: EntityType,
      rel: number,
      tx: number,
      ty: number,
      hp: number,
    ): void => {
      const o = r * F;
      const def = defOf(type);
      E[o + F_TYPE + type] = 1;
      E[o + F_REL + rel] = 1;
      E[o + F_X] = tx / size;
      E[o + F_Y] = ty / size;
      E[o + F_DX] = (tx - startX) / size;
      E[o + F_DY] = (ty - startY) / size;
      E[o + F_HP] = def.maxHp > 0 ? clip01(hp / def.maxHp) : 0;
      E[o + F_SUPPLY] = def.supplyCost / 2;
      E[o + F_FLYING] = def.flying ? 1 : 0;
      E[o + F_CAN_HIT_AIR] = def.canHitAir ? 1 : 0;
      E[o + F_DIST_POST] = distToPost(tx, ty);
    };

    for (const c of candidates) {
      if (row >= N_ENT) break;
      const i = c.index;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      const tx = canonTileX(i);
      const ty = canonTileY(i);
      const o = row * F;
      const own = c.kind !== RowKind.Ally;
      writeCommon(row, type, own ? 0 : 1, tx, ty, pool.hp[i]!);
      if (def.isBuilding) {
        E[o + F_BUILD + pool.buildState[i]!] = 1;
        if (pool.buildState[i] !== BuildState.Complete && def.buildTicks > 0) {
          E[o + F_BUILD_PROGRESS] = clip01(pool.buildProgress[i]! / def.buildTicks);
        }
      }
      if (own) {
        const orderColumn = ORDER_COLUMNS.indexOf(pool.order[i]! as Order);
        if (orderColumn >= 0) E[o + F_ORDER + orderColumn] = 1;
        E[o + F_CARRYING] = clip01(pool.carrying[i]! / MINERALS_PER_TRIP);
        E[o + F_PROD_COUNT] = pool.prodCount[i]! / MAX_PRODUCTION_QUEUE;
        if (pool.prodCount[i]! > 0) {
          const head = defOf(pool.prodAt(i, 0));
          E[o + F_PROD_PROGRESS] =
            head.buildTicks > 0 ? clip01(pool.prodProgress[i]! / head.buildTicks) : 0;
        }
        E[o + F_HAS_RALLY] = pool.hasRally[i]!;
        if (pool.hasRally[i] === 1) {
          const rx = toInt(canonFix(pool.rallyX[i]!, W, flip));
          const ry = toInt(canonFix(pool.rallyY[i]!, H, flip));
          E[o + F_RALLY_DX] = (rx - tx) / size;
          E[o + F_RALLY_DY] = (ry - ty) / size;
        }
        E[o + F_COOLDOWN] =
          def.attackCooldown > 0 ? clip01(pool.attackCooldown[i]! / def.attackCooldown) : 0;
      }
      E[o + F_VISIBLE] = 1;
      const id = pool.idAt(i);
      E[o + F_IN_LAST] = recent.lastUnits.has(id) ? 1 : 0;
      frame.rows[row] = id;
      frame.rowKind[row] = c.kind;
      frame.rowOf.set(id, row);
      out.entityMask[row] = 1;
      row++;
    }

    for (const r of rememberedRows) {
      if (row >= N_ENT) break;
      const entry = r.entry;
      const o = row * F;
      const patch = entry.type === EntityType.MineralPatch;
      const tx = canonX(entry.posX, entry.posY);
      const ty = canonY(entry.posX, entry.posY);
      writeCommon(row, entry.type, patch ? 3 : 2, tx, ty, entry.hp);
      if (entry.isBuilding && !patch) E[o + F_BUILD + entry.buildState] = 1;
      E[o + F_VISIBLE] = r.kind === RowKind.EnemyRemembered ? 0 : 1;
      E[o + F_MEMORY_AGE] = clip01((world.tick - entry.lastSeen) / UNIT_MEMORY_TICKS);
      if (patch) E[o + F_RESOURCE] = clip01(entry.resourceAmount / PATCH_AMOUNT);
      E[o + F_IN_LAST] = recent.lastUnits.has(entry.id) ? 1 : 0;
      frame.rows[row] = entry.id;
      frame.rowKind[row] = r.kind;
      frame.rowOf.set(entry.id, row);
      out.entityMask[row] = 1;
      row++;
    }

    // --- the grid ------------------------------------------------------------
    const G = out.grid;
    const index = gridIndexFor(map, flip);
    const tileCell = index.tileCell;
    const cellTiles = index.cellTiles;
    const state = vis.state;
    const occupied = map.occupied;
    const sums = this.cellSum;
    sums.fill(0);
    for (let t = 0; t < W * H; t++) {
      const cell = tileCell[t]!;
      if (cell < 0) continue;
      const s = state[t]!;
      if (index.notCliff[t] === 1) sums[cell]!++;
      // Occupancy is only known where the ground has been seen; unexplored
      // ground reads as free, which is what a person would assume.
      if (index.ground[t] === 1 && (s === UNEXPLORED || occupied[t] !== OCCUPIED_SOLID)) {
        sums[CELLS + cell]!++;
      }
      if (s !== UNEXPLORED) sums[2 * CELLS + cell]!++;
      if (s === VISIBLE) sums[3 * CELLS + cell]!++;
    }
    for (let cell = 0; cell < CELLS; cell++) {
      const n = cellTiles[cell]!;
      if (n === 0) continue;
      G[G_WALKABLE * CELLS + cell] = sums[cell]! / n;
      G[G_BUILDABLE * CELLS + cell] = sums[CELLS + cell]! / n;
      G[G_EXPLORED * CELLS + cell] = sums[2 * CELLS + cell]! / n;
      G[G_VISIBLE * CELLS + cell] = sums[3 * CELLS + cell]! / n;
    }
    const bump = (channel: number, tx: number, ty: number, amount: number, cap: number): void => {
      const cell = cellOf(tx, ty);
      if (cell < 0) return;
      const k = channel * CELLS + cell;
      G[k] = clip01(G[k]! + amount / cap);
    };
    for (let r = 0; r < row; r++) {
      const o = r * F;
      const tx = Math.round(E[o + F_X]! * size);
      const ty = Math.round(E[o + F_Y]! * size);
      const kind = frame.rowKind[r]!;
      switch (kind) {
        case RowKind.OwnBuilding:
          bump(G_OWN_BUILDINGS, tx, ty, 1, 4);
          break;
        case RowKind.OwnUnit:
          bump(G_OWN_UNITS, tx, ty, 1, 8);
          break;
        case RowKind.Ally:
          bump(G_ALLY, tx, ty, 1, 8);
          break;
        case RowKind.EnemyVisible:
        case RowKind.EnemyRemembered: {
          const entry = mem.get(frame.rows[r]!)!;
          if (entry.isBuilding) bump(G_ENEMY_BUILDINGS, tx, ty, 1, 4);
          else if (kind === RowKind.EnemyVisible) bump(G_ENEMY_UNITS, tx, ty, 1, 8);
          break;
        }
        case RowKind.Patch:
          bump(G_MINERALS, tx, ty, E[o + F_RESOURCE]! * PATCH_AMOUNT, 4 * PATCH_AMOUNT);
          break;
        default:
          break;
      }
    }
    for (let p = 0; p < world.players.length; p++) {
      const s = map.starts[p]!;
      const sx = canonTileCoord(s.tileX, W, flip);
      const sy = canonTileCoord(s.tileY, H, flip);
      bump(world.areAllied(p, viewer) ? G_FRIENDLY_STARTS : G_ENEMY_STARTS, sx, sy, 1, 1);
    }
    for (const e of map.expansions) {
      bump(G_EXPANSIONS, canonTileCoord(e.tileX, W, flip), canonTileCoord(e.tileY, H, flip), 1, 1);
    }

    // --- the scalars ---------------------------------------------------------
    const S = out.scalars;
    const ps = world.player(viewer);
    S[S_MINERALS] = ps.minerals / 1000;
    S[S_SUPPLY_USED] = ps.supplyUsed / 200;
    S[S_SUPPLY_MAX] = ps.supplyMax / 200;
    S[S_SUPPLY_FREE] = clip01((ps.supplyMax - ps.supplyUsed) / 20);
    S[S_TICK] = world.tick / 24000;
    S[S_LAYOUT + (world.config.layout === MapLayout.Quarters ? 1 : 0)] = 1;
    S[S_SEAT] = world.ownerCanonical(viewer);
    S[S_ALLIES] = (world.playersOnTeam(world.teamOf(viewer)).length - 1) / 2;
    for (let r = 0; r < row; r++) {
      const kind = frame.rowKind[r]!;
      const o = r * F;
      let type = -1;
      for (let t = 0; t < ENTITY_TYPE_COUNT; t++) {
        if (E[o + F_TYPE + t] === 1) type = t;
      }
      if (type < 0) continue;
      if (kind === RowKind.OwnUnit || kind === RowKind.OwnBuilding)
        S[S_OWN_COUNTS + type] = clip01(S[S_OWN_COUNTS + type]! + 1 / 16);
      else if (kind === RowKind.EnemyVisible || kind === RowKind.EnemyRemembered)
        S[S_ENEMY_COUNTS + type] = clip01(S[S_ENEMY_COUNTS + type]! + 1 / 16);
    }
    if (recent.prevType >= 0 && recent.prevType < ACTION_TYPE_COUNT)
      S[S_PREV + recent.prevType] = 1;
    S[S_SINCE_NON_NOOP] = clip01(recent.sinceNonNoop / 50);
    S[S_RECENT] = clip01(recent.recentCommands / 20);
  }
}

/**
 * The whole truth, for the critic only.
 *
 * Per player in canonical order — the viewer, then its allies, then the
 * enemies — the economy and the army as they really are. Training-only: the
 * actor never sees this, and the exported model has no input for it.
 */
export function encodeCritic(world: World, viewer: PlayerId, out: Float32Array): void {
  if (out.length < CRITIC_LEN) throw new Error(`critic vector needs ${CRITIC_LEN} floats`);
  out.fill(0);
  const pool = world.pool;
  const map = world.map;
  const flip = world.flipOf(viewer);
  const size = Math.max(map.width, map.height);
  const order: PlayerId[] = [viewer];
  for (let p = 0; p < world.players.length; p++) {
    if (p !== viewer && world.areAllied(p, viewer)) order.push(p);
  }
  for (let p = 0; p < world.players.length; p++) {
    if (!world.areAllied(p, viewer)) order.push(p);
  }
  const per = CRITIC_PER_PLAYER.length;
  for (let k = 0; k < Math.min(order.length, CRITIC_PLAYERS); k++) {
    const p = order[k]!;
    const ps = world.player(p);
    const o = k * per;
    let workers = 0;
    let army = 0;
    let buildings = 0;
    let armyValue = 0;
    let buildingValue = 0;
    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || pool.owner[i] !== p) continue;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      if (def.isBuilding) {
        buildings++;
        buildingValue += def.mineralCost;
      } else if (type === EntityType.Worker) {
        workers++;
      } else {
        army++;
        armyValue += def.mineralCost;
        const tile = map.tileOfPosFor(pool.posX[i]!, pool.posY[i]!, flip);
        sumX += canonTileCoord(map.tileXOf(tile), map.width, flip);
        sumY += canonTileCoord(map.tileYOf(tile), map.height, flip);
      }
    }
    out[o] = ps.minerals / 1000;
    out[o + 1] = ps.supplyUsed / 200;
    out[o + 2] = ps.supplyMax / 200;
    out[o + 3] = workers / 32;
    out[o + 4] = army / 64;
    out[o + 5] = buildings / 16;
    out[o + 6] = armyValue / 4000;
    out[o + 7] = buildingValue / 4000;
    out[o + 8] = army > 0 ? Math.floor(sumX / army) / size : 0;
    out[o + 9] = army > 0 ? Math.floor(sumY / army) / size : 0;
    out[o + 10] = ps.defeated ? 1 : 0;
  }
  out[CRITIC_LEN - 2] = world.tick / 24000;
  out[CRITIC_LEN - 1] = world.config.layout === MapLayout.Quarters ? 1 : 0;
}

/** For tests: which cell a canonical tile falls in, exposed from the frame module. */
export { cellOf, CELL_TILES };
