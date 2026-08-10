/**
 * Entity storage: structure-of-arrays over pre-allocated typed arrays.
 *
 * ## Why not an ECS library
 *
 * The component set here is small and fixed, and every general-purpose ECS
 * stores its archetypes in hash maps somewhere. Iteration order over a hash map
 * is an implementation detail — exactly the kind of thing that is stable on one
 * engine and different on another, producing a desync that reproduces only
 * across browsers. Iterating a plain array by ascending index is trivially
 * identical everywhere, so that is what we do.
 *
 * ## Handles, not indices
 *
 * Slots are recycled, so a raw index would let a stale reference silently
 * resolve to whichever unit happened to reuse the slot. `EntityId` therefore
 * packs a 16-bit slot index with a 15-bit generation counter that increments on
 * every free. `isAlive` rejects handles whose generation no longer matches, so a
 * command issued against a unit that died two ticks ago resolves to nothing
 * instead of to an unrelated unit. The id stays positive so `NO_ENTITY` (-1)
 * remains unambiguous.
 */

import { checksumArray, checksumU32 } from './checksum.js';
import { defOf, MAX_PRODUCTION_QUEUE } from '../config/rules.js';
import type { Fix } from './fixed.js';
import {
  BuildState,
  EntityType,
  NEUTRAL,
  NO_ENTITY,
  Order,
  type EntityId,
  type PlayerId,
} from './types.js';

/** Maximum simultaneous entities. Two players at 200 supply plus terrain. */
export const ENTITY_CAPACITY = 2048;

/** Longest path we retain per unit. Longer routes are re-planned on arrival. */
export const MAX_PATH = 48;

/** `arriveBest` before a unit has measured its distance to a destination. */
export const ARRIVE_BEST_NONE = 0x7fffffff;

const INDEX_MASK = 0xffff;
const GEN_MASK = 0x7fff;

export function makeId(index: number, generation: number): EntityId {
  return (((generation & GEN_MASK) << 16) | (index & INDEX_MASK)) >>> 0;
}

export function idIndex(id: EntityId): number {
  return id & INDEX_MASK;
}

export function idGeneration(id: EntityId): number {
  return (id >>> 16) & GEN_MASK;
}

export class EntityPool {
  /** One past the highest slot ever used. Systems iterate `0..count`. */
  count = 0;

  // --- identity ---
  readonly alive = new Uint8Array(ENTITY_CAPACITY);
  readonly generation = new Uint16Array(ENTITY_CAPACITY);
  readonly type = new Uint8Array(ENTITY_CAPACITY);
  /** Player slot, or NEUTRAL for mineral patches. */
  readonly owner = new Int8Array(ENTITY_CAPACITY);

  // --- transform (Q16.16) ---
  readonly posX = new Int32Array(ENTITY_CAPACITY);
  readonly posY = new Int32Array(ENTITY_CAPACITY);
  /** Normalised facing vector. Never an angle — see fixed.ts. */
  readonly faceX = new Int32Array(ENTITY_CAPACITY);
  readonly faceY = new Int32Array(ENTITY_CAPACITY);

  // --- vitals ---
  readonly hp = new Int32Array(ENTITY_CAPACITY);

  // --- orders ---
  readonly order = new Uint8Array(ENTITY_CAPACITY);
  readonly orderTarget = new Int32Array(ENTITY_CAPACITY);
  readonly orderX = new Int32Array(ENTITY_CAPACITY);
  readonly orderY = new Int32Array(ENTITY_CAPACITY);

  // --- combat ---
  readonly attackCooldown = new Int32Array(ENTITY_CAPACITY);
  /** Entity currently being shot at, for rendering tracers and aggro stickiness. */
  readonly combatTarget = new Int32Array(ENTITY_CAPACITY);

  // --- buildings ---
  readonly buildState = new Uint8Array(ENTITY_CAPACITY);
  readonly buildProgress = new Int32Array(ENTITY_CAPACITY);
  /** Top-left tile of a building footprint. Meaningless for units. */
  readonly tileX = new Int32Array(ENTITY_CAPACITY);
  readonly tileY = new Int32Array(ENTITY_CAPACITY);

  /**
   * Current speed along `face`, in Q16.16 units per tick.
   *
   * Tracked per entity rather than read from the unit's definition because
   * units ramp up to their top speed and ease back down to a stop. Without it
   * every unit starts and stops instantly, which is most of what makes movement
   * look mechanical.
   */
  readonly speed = new Int32Array(ENTITY_CAPACITY);

  // --- economy ---
  readonly carrying = new Int32Array(ENTITY_CAPACITY);
  readonly harvestTimer = new Int32Array(ENTITY_CAPACITY);
  /** Patch a worker returns to after delivering its load. */
  readonly harvestPatch = new Int32Array(ENTITY_CAPACITY);
  /** Remaining minerals in a patch. */
  readonly resourceAmount = new Int32Array(ENTITY_CAPACITY);

  // --- production ---
  readonly prodQueue = new Uint8Array(ENTITY_CAPACITY * MAX_PRODUCTION_QUEUE);
  /**
   * Where units trained here walk to, or NO_RALLY when there is none.
   *
   * Kept per building rather than per player: a barracks near the front and one
   * at home want different answers, which is the entire point of a rally point.
   */
  readonly rallyX = new Int32Array(ENTITY_CAPACITY);
  readonly rallyY = new Int32Array(ENTITY_CAPACITY);
  readonly hasRally = new Uint8Array(ENTITY_CAPACITY);

  readonly prodCount = new Uint8Array(ENTITY_CAPACITY);
  readonly prodProgress = new Int32Array(ENTITY_CAPACITY);

  // --- pathing ---
  readonly pathNodes = new Int32Array(ENTITY_CAPACITY * MAX_PATH);
  readonly pathLen = new Uint8Array(ENTITY_CAPACITY);
  readonly pathCursor = new Uint8Array(ENTITY_CAPACITY);
  /** Set when a unit wants a path but the tick budget was already spent. */
  readonly pathPending = new Uint8Array(ENTITY_CAPACITY);
  /**
   * Destination tile when this unit is following a shared flow field, or -1 when
   * it is following its own A* path. Group moves use the field; individual
   * errands use A*.
   */
  readonly flowGoal = new Int32Array(ENTITY_CAPACITY);
  /**
   * The tile a move order navigates to, kept across a fight.
   *
   * `clearPath` wipes the active route — including `flowGoal` — which is right
   * when a unit stops to shoot but leaves nothing to resume from. This holds the
   * order's *intent* rather than its current route, so it survives combat. -1
   * means the order navigates by its own A* path instead of a shared field.
   */
  readonly navGoal = new Int32Array(ENTITY_CAPACITY).fill(-1);
  /**
   * Where a unit stood when it broke off to chase something, and whether it is
   * chasing at all.
   *
   * The leash has to be measured from here rather than from the unit's current
   * position. Recomputed each tick against where it now stands, the window
   * slides along with a retreating enemy and pursuit ratchets — measured, a
   * unit dragged sideways followed a fleeing Burstbot 14.6 tiles off its route.
   */
  readonly pursueX = new Int32Array(ENTITY_CAPACITY);
  readonly pursueY = new Int32Array(ENTITY_CAPACITY);
  readonly pursuing = new Uint8Array(ENTITY_CAPACITY);
  /**
   * The closest this unit has come to its destination, and how many ticks it
   * has gone without beating that.
   *
   * A unit ordered somewhere it cannot stand — a spot another unit has already
   * taken, a formation slot against a cliff — otherwise pushes at it forever.
   * It never counts as arrived, so it never goes idle, and a group move leaves
   * a handful of units shoving at the crowd for the rest of the match.
   */
  readonly arriveBest = new Int32Array(ENTITY_CAPACITY).fill(ARRIVE_BEST_NONE);
  readonly arriveStall = new Int32Array(ENTITY_CAPACITY);
  /**
   * Ticks to wait before retrying a path that failed.
   *
   * Without this, a unit that cannot reach its target re-runs a full-budget A*
   * search every single tick. Measured, that was 6.7 doomed searches per tick at
   * 3ms each — the dominant cost of the entire simulation.
   */
  readonly pathCooldown = new Int32Array(ENTITY_CAPACITY);

  /**
   * Free slots, most-recently-freed first.
   *
   * A stack rather than a scan keeps allocation O(1), and because frees happen
   * in a deterministic order the stack contents are identical on every peer —
   * which matters, since the allocation order decides entity ids and therefore
   * every subsequent iteration order.
   */
  private readonly freeList: number[] = [];

  /** Allocate an entity. Returns NO_ENTITY if the pool is exhausted. */
  spawn(type: EntityType, owner: PlayerId, x: Fix, y: Fix): EntityId {
    let index: number;
    const reused = this.freeList.pop();
    if (reused !== undefined) {
      index = reused;
    } else {
      if (this.count >= ENTITY_CAPACITY) return NO_ENTITY;
      index = this.count++;
    }

    const def = defOf(type);
    this.alive[index] = 1;
    this.type[index] = type;
    this.owner[index] = owner;
    this.posX[index] = x;
    this.posY[index] = y;
    this.faceX[index] = 0;
    this.faceY[index] = 65536; // facing +Y by default
    this.hp[index] = def.maxHp;
    this.order[index] = Order.None;
    this.orderTarget[index] = NO_ENTITY;
    this.orderX[index] = 0;
    this.orderY[index] = 0;
    this.attackCooldown[index] = 0;
    this.speed[index] = 0;
    this.combatTarget[index] = NO_ENTITY;
    this.buildState[index] = def.isBuilding ? BuildState.Site : BuildState.Complete;
    this.buildProgress[index] = 0;
    this.tileX[index] = 0;
    this.tileY[index] = 0;
    this.carrying[index] = 0;
    this.harvestTimer[index] = 0;
    this.harvestPatch[index] = NO_ENTITY;
    this.resourceAmount[index] = 0;
    this.prodCount[index] = 0;
    this.navGoal[index] = -1;
    this.pursuing[index] = 0;
    this.pursueX[index] = 0;
    this.pursueY[index] = 0;
    this.arriveBest[index] = ARRIVE_BEST_NONE;
    this.arriveStall[index] = 0;
    this.rallyX[index] = 0;
    this.rallyY[index] = 0;
    this.hasRally[index] = 0;
    this.prodProgress[index] = 0;
    this.pathLen[index] = 0;
    this.pathCursor[index] = 0;
    this.pathPending[index] = 0;
    this.flowGoal[index] = -1;
    this.pathCooldown[index] = 0;

    return makeId(index, this.generation[index]!);
  }

  /** Free an entity. Safe to call with a stale handle; it becomes a no-op. */
  destroy(id: EntityId): void {
    const index = idIndex(id);
    if (!this.isAlive(id)) return;
    this.alive[index] = 0;
    // Bump the generation so every outstanding handle to this slot goes stale.
    this.generation[index] = (this.generation[index]! + 1) & GEN_MASK;
    this.freeList.push(index);
  }

  /** True if the handle still refers to the entity it was issued for. */
  isAlive(id: EntityId): boolean {
    if (id < 0) return false;
    const index = idIndex(id);
    if (index >= this.count) return false;
    return this.alive[index] === 1 && this.generation[index] === idGeneration(id);
  }

  /** Handle for a slot index, or NO_ENTITY if the slot is empty. */
  idAt(index: number): EntityId {
    if (this.alive[index] !== 1) return NO_ENTITY;
    return makeId(index, this.generation[index]!);
  }

  /** Read a path waypoint (a tile index) for a unit. */
  pathNode(index: number, step: number): number {
    return this.pathNodes[index * MAX_PATH + step]!;
  }

  /** Overwrite a unit's path with `nodes`, truncating at MAX_PATH. */
  setPath(index: number, nodes: readonly number[]): void {
    const n = Math.min(nodes.length, MAX_PATH);
    const base = index * MAX_PATH;
    for (let i = 0; i < n; i++) this.pathNodes[base + i] = nodes[i]!;
    this.pathLen[index] = n;
    this.pathCursor[index] = 0;
    this.pathPending[index] = 0;
  }

  clearPath(index: number): void {
    this.pathLen[index] = 0;
    this.pathCursor[index] = 0;
    this.pathPending[index] = 0;
    this.flowGoal[index] = -1;
  }

  /** Read a queued production entry. */
  prodAt(index: number, slot: number): EntityType {
    return this.prodQueue[index * MAX_PRODUCTION_QUEUE + slot]! as EntityType;
  }

  /** Append to a production queue. Returns false when the queue is full. */
  prodPush(index: number, type: EntityType): boolean {
    const n = this.prodCount[index]!;
    if (n >= MAX_PRODUCTION_QUEUE) return false;
    this.prodQueue[index * MAX_PRODUCTION_QUEUE + n] = type;
    this.prodCount[index] = n + 1;
    return true;
  }

  /** Remove a queue entry, shifting the rest down. */
  prodRemove(index: number, slot: number): void {
    const n = this.prodCount[index]!;
    if (slot >= n) return;
    const base = index * MAX_PRODUCTION_QUEUE;
    for (let i = slot; i < n - 1; i++) {
      this.prodQueue[base + i] = this.prodQueue[base + i + 1]!;
    }
    this.prodCount[index] = n - 1;
    if (slot === 0) this.prodProgress[index] = 0;
  }

  /**
   * Fold entity state into a checksum.
   *
   * Only fields that influence future simulation are included. Purely cosmetic
   * state is excluded so that a rendering-only change cannot report a false
   * desync. `pathNodes` is deliberately omitted — it is derived from position
   * and destination, so including it would just add noise, but `pathLen` and
   * `pathCursor` are included because they gate movement.
   */
  checksum(h: number): number {
    let x = h;
    x = checksumU32(x, this.count);
    x = checksumArray(x, this.alive, this.count);
    x = checksumArray(x, this.generation, this.count);
    x = checksumArray(x, this.type, this.count);
    x = checksumArray(x, this.posX, this.count);
    x = checksumArray(x, this.posY, this.count);
    x = checksumArray(x, this.faceX, this.count);
    x = checksumArray(x, this.faceY, this.count);
    x = checksumArray(x, this.hp, this.count);
    x = checksumArray(x, this.order, this.count);
    x = checksumArray(x, this.orderTarget, this.count);
    x = checksumArray(x, this.orderX, this.count);
    x = checksumArray(x, this.orderY, this.count);
    x = checksumArray(x, this.attackCooldown, this.count);
    x = checksumArray(x, this.combatTarget, this.count);
    x = checksumArray(x, this.buildState, this.count);
    x = checksumArray(x, this.buildProgress, this.count);
    // Footprint tiles are what `GameMap.occupied` is derived from, so hashing
    // them here is what lets the map skip hashing 16k tiles every tick.
    x = checksumArray(x, this.tileX, this.count);
    x = checksumArray(x, this.tileY, this.count);
    x = checksumArray(x, this.speed, this.count);
    x = checksumArray(x, this.navGoal, this.count);
    x = checksumArray(x, this.pursuing, this.count);
    x = checksumArray(x, this.pursueX, this.count);
    x = checksumArray(x, this.pursueY, this.count);
    x = checksumArray(x, this.arriveBest, this.count);
    x = checksumArray(x, this.arriveStall, this.count);
    x = checksumArray(x, this.rallyX, this.count);
    x = checksumArray(x, this.rallyY, this.count);
    x = checksumArray(x, this.hasRally, this.count);
    x = checksumArray(x, this.carrying, this.count);
    x = checksumArray(x, this.harvestTimer, this.count);
    x = checksumArray(x, this.harvestPatch, this.count);
    x = checksumArray(x, this.resourceAmount, this.count);
    x = checksumArray(x, this.prodCount, this.count);
    x = checksumArray(x, this.prodProgress, this.count);
    x = checksumArray(x, this.prodQueue, this.count * MAX_PRODUCTION_QUEUE);
    x = checksumArray(x, this.pathLen, this.count);
    x = checksumArray(x, this.pathCursor, this.count);
    x = checksumArray(x, this.flowGoal, this.count);
    x = checksumArray(x, this.pathCooldown, this.count);
    // The free list decides future entity ids, so it is part of the state.
    x = checksumU32(x, this.freeList.length);
    for (let i = 0; i < this.freeList.length; i++) {
      x = checksumU32(x, this.freeList[i]!);
    }
    return x;
  }

  /** Owner test that treats neutral entities as hostile to nobody. */
  isHostile(index: number, toPlayer: PlayerId): boolean {
    const o = this.owner[index]!;
    return o !== NEUTRAL && o !== toPlayer;
  }
}
