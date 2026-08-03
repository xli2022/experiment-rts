/**
 * The complete simulation state.
 *
 * Everything that influences future ticks lives on this object and is covered by
 * `checksum()`. Anything not covered by the checksum must be either derived
 * (recomputable from checksummed state) or purely cosmetic — if it is neither,
 * it is a desync waiting to happen.
 *
 * The world knows nothing about rendering, input, or the network. It is stepped
 * by `tick.ts` with a set of commands and advances one fixed timestep.
 */

import { checksumInit, checksumU32 } from './checksum.js';
import { EntityPool } from './entities.js';
import { fromInt, type Fix } from './fixed.js';
import { GameMap, generateMap, MAP_SIZE } from './map.js';
import { Rng } from './rng.js';
import { SpatialGrid } from './spatial.js';
import {
  EntityType,
  MAX_PLAYERS,
  NEUTRAL,
  NO_ENTITY,
  type EntityId,
  type PlayerId,
} from './types.js';
import {
  defOf,
  PATCH_AMOUNT,
  PATCHES_PER_BASE,
  STARTING_MINERALS,
  STARTING_WORKERS,
  SUPPLY_MAX,
} from '../config/rules.js';

export interface PlayerState {
  minerals: number;
  /** Supply consumed by living units. */
  supplyUsed: number;
  /** Supply provided by completed buildings, capped at SUPPLY_MAX. */
  supplyMax: number;
  defeated: boolean;
}

/**
 * Transient per-tick events for the renderer and audio.
 *
 * Deliberately NOT part of the checksum: these are outputs of the tick, not
 * state carried into the next one. They are cleared at the start of every step,
 * so a peer that ignores them behaves identically to one that draws them.
 */
export interface SimEvents {
  /** (attacker, target) pairs that fired this tick, for tracers. */
  shots: number[];
  /** Entity slot indices that died this tick. */
  deaths: number[];
  /** Entity slot indices that finished construction this tick. */
  completed: number[];
}

export class World {
  readonly map: GameMap;
  readonly pool = new EntityPool();
  readonly grid: SpatialGrid;
  readonly rng: Rng;
  readonly players: PlayerState[] = [];

  /** Ticks elapsed since the match began. */
  tick = 0;

  /** -1 while the match is running; otherwise the winning player. */
  winner: PlayerId = NO_ENTITY;

  /**
   * True once the match has ended, win or draw.
   *
   * Distinct from `winner` because a draw is a real outcome here: both players
   * can be eliminated on the same tick, and "winner is nobody" must still stop
   * the match rather than leave it running forever.
   */
  matchOver = false;

  /**
   * Units waiting for an A* path, in request order.
   *
   * A plain array used as a FIFO. Pathfinding is budgeted per tick (see
   * `PATH_BUDGET_PER_TICK`), and this queue is what makes the overflow
   * deterministic: every peer defers exactly the same requests to the same
   * later tick, so a busy tick slows everyone identically rather than letting
   * one machine's spare CPU change the outcome.
   */
  readonly pathQueue: number[] = [];

  readonly events: SimEvents = { shots: [], deaths: [], completed: [] };

  constructor(seed: number, mapSize = MAP_SIZE) {
    this.map = generateMap(seed, mapSize);
    this.grid = new SpatialGrid(mapSize);
    this.rng = new Rng(seed);
    for (let p = 0; p < MAX_PLAYERS; p++) {
      this.players.push({
        minerals: STARTING_MINERALS,
        supplyUsed: 0,
        supplyMax: 0,
        defeated: false,
      });
    }
  }

  player(id: PlayerId): PlayerState {
    return this.players[id]!;
  }

  /** Centre position of an entity, in world units. */
  centreX(index: number): Fix {
    return this.pool.posX[index]!;
  }

  centreY(index: number): Fix {
    return this.pool.posY[index]!;
  }

  /**
   * Place a building and mark its tiles occupied.
   *
   * Buildings are positioned by their top-left tile and their stored position is
   * the footprint centre, which is what movement and combat measure against.
   */
  placeBuilding(
    type: EntityType,
    owner: PlayerId,
    tileX: number,
    tileY: number,
  ): EntityId {
    const def = defOf(type);
    const half = def.footprint / 2;
    const cx = fromInt(tileX) + Math.round(half * 65536);
    const cy = fromInt(tileY) + Math.round(half * 65536);
    const id = this.pool.spawn(type, owner, cx, cy);
    if (id === NO_ENTITY) return NO_ENTITY;
    const i = id & 0xffff;
    this.pool.tileX[i] = tileX;
    this.pool.tileY[i] = tileY;
    this.map.setOccupied(tileX, tileY, def.footprint, 1);
    return id;
  }

  /**
   * Fingerprint the entire simulation.
   *
   * Field order is fixed and must never change casually — the golden replay test
   * pins expected values, and reordering would invalidate them for no reason.
   */
  checksum(): number {
    let h = checksumInit();
    h = checksumU32(h, this.tick);
    h = checksumU32(h, this.rng.state);
    h = checksumU32(h, this.winner);
    h = checksumU32(h, this.matchOver ? 1 : 0);
    for (let p = 0; p < this.players.length; p++) {
      const ps = this.players[p]!;
      h = checksumU32(h, ps.minerals);
      h = checksumU32(h, ps.supplyUsed);
      h = checksumU32(h, ps.supplyMax);
      h = checksumU32(h, ps.defeated ? 1 : 0);
    }
    h = checksumU32(h, this.pathQueue.length);
    for (let i = 0; i < this.pathQueue.length; i++) {
      h = checksumU32(h, this.pathQueue[i]!);
    }
    h = this.pool.checksum(h);
    h = this.map.checksum(h);
    return h;
  }

  /** Recompute supply totals from scratch. Cheap, and immune to drift. */
  recomputeSupply(): void {
    for (let p = 0; p < this.players.length; p++) {
      this.players[p]!.supplyUsed = 0;
      this.players[p]!.supplyMax = 0;
    }
    const pool = this.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const owner = pool.owner[i]!;
      if (owner === NEUTRAL) continue;
      const ps = this.players[owner];
      if (!ps) continue;
      const def = defOf(pool.type[i]! as EntityType);
      if (def.isBuilding) {
        // Only finished buildings contribute supply.
        if (pool.buildState[i] === 2) ps.supplyMax += def.supplyProvided;
      } else {
        ps.supplyUsed += def.supplyCost;
      }
    }
    for (let p = 0; p < this.players.length; p++) {
      const ps = this.players[p]!;
      if (ps.supplyMax > SUPPLY_MAX) ps.supplyMax = SUPPLY_MAX;
    }
  }
}

/**
 * Offsets of each mineral patch from a base's centre tile, as top-left corners.
 *
 * Both players use the same relative layout, so neither gets a better opening —
 * which matters more in a mirror matchup than any map feature.
 */
const PATCH_OFFSETS: readonly (readonly [number, number])[] = [
  [-6, -2],
  [-6, 0],
  [-6, 2],
  [-5, -4],
  [-3, -5],
  [-1, -6],
  [1, -6],
  [3, -5],
];

/**
 * Build the opening position: a finished Command Post, starting workers, and a
 * mineral line for each player.
 *
 * Runs identically on every peer from the shared seed, so no starting state ever
 * crosses the network.
 */
export function setupMatch(world: World): void {
  const { map, pool } = world;

  for (let p = 0; p < MAX_PLAYERS; p++) {
    const start = map.starts[p]!;
    const hqDef = defOf(EntityType.CommandPost);
    const hqTileX = start.tileX - (hqDef.footprint >> 1);
    const hqTileY = start.tileY - (hqDef.footprint >> 1);
    const hq = world.placeBuilding(EntityType.CommandPost, p, hqTileX, hqTileY);
    if (hq !== NO_ENTITY) {
      const hi = hq & 0xffff;
      pool.buildState[hi] = 2; // starts complete
      pool.buildProgress[hi] = hqDef.buildTicks;
    }

    // Mineral line: a tight arc beside the base.
    //
    // Distance here is the single biggest lever on the pace of the whole game.
    // Patches scattered even a few tiles too far leave workers walking instead
    // of mining and the economy never gets going, so these offsets are chosen
    // to sit just clear of the Command Post footprint.
    const patchDef = defOf(EntityType.MineralPatch);
    let placed = 0;
    for (const [dx, dy] of PATCH_OFFSETS) {
      if (placed >= PATCHES_PER_BASE) break;
      const tx = start.tileX + dx;
      const ty = start.tileY + dy;
      if (!map.canPlace(tx, ty, patchDef.footprint)) continue;
      const patch = world.placeBuilding(EntityType.MineralPatch, NEUTRAL, tx, ty);
      if (patch === NO_ENTITY) continue;
      const pi = patch & 0xffff;
      pool.buildState[pi] = 2;
      pool.resourceAmount[pi] = PATCH_AMOUNT;
      placed++;
    }

    // Starting workers, fanned out below the Command Post.
    for (let wIdx = 0; wIdx < STARTING_WORKERS; wIdx++) {
      const ox = fromInt(start.tileX) + fromInt(wIdx - (STARTING_WORKERS >> 1));
      const oy = fromInt(start.tileY + 3);
      pool.spawn(EntityType.Worker, p, ox, oy);
    }
  }

  world.recomputeSupply();
  world.grid.rebuild(pool);
}
