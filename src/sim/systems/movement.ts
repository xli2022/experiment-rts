/**
 * Movement: path requests, path following, and local separation.
 *
 * Pathfinding is the one system that can realistically blow the tick budget, and
 * in lockstep a slow tick is not a local problem — every peer waits for the
 * slowest one. So A* requests are served from a bounded FIFO: at most
 * `PATH_BUDGET_PER_TICK` searches run per tick and the rest wait. Because the
 * queue is part of world state and drains in a fixed order, every peer defers
 * exactly the same requests to exactly the same later tick.
 */

import {
  PATH_BUDGET_PER_TICK,
  SEPARATION_STRENGTH,
  defOf,
  reachSlackFor,
} from '../../config/rules.js';
import type { FlowFieldCache } from '../pathing/flowfield.js';
import { idIndex, MAX_PATH } from '../entities.js';
import {
  FIX_HALF,
  fdiv,
  fmul,
  fromInt,
  sqRange,
  vecDist,
  vecLenSqRaw,
  vecNormalize,
  vecRotateToward,
  type Fix,
} from '../fixed.js';
import { AStar, nearestWalkable } from '../pathing/astar.js';
import type { EntityDef } from '../../config/rules.js';
import { EntityType, NO_ENTITY, Order } from '../types.js';
import type { World } from '../world.js';

/** How close counts as "standing on" a waypoint. */
const WAYPOINT_REACH = FIX_HALF; // 0.5 world units

/** How close to the ordered point counts as having arrived. */
const ARRIVAL_REACH = FIX_HALF;

/** Attack orders re-path this often while chasing a moving target. */
const CHASE_REPATH_INTERVAL = 10;

/** Ticks a unit waits before retrying a path search that found no route. */
const PATH_RETRY_COOLDOWN = 40;

export function movementSystem(world: World, astar: AStar, fields: FlowFieldCache): void {
  servePathRequests(world, astar);
  followFlowFields(world, fields);
  followPaths(world);
  separate(world);
}

/**
 * Advance every unit that is following a shared flow field.
 *
 * Each unit reads the field's next-best tile from where it stands and steers at
 * it. There is no stored route, so units re-evaluate every tick and naturally
 * flow around each other and around newly-placed buildings.
 */
function followFlowFields(world: World, fields: FlowFieldCache): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const goal = pool.flowGoal[i]!;
    if (goal < 0) continue;

    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding || def.speedPerTick === 0) {
      pool.flowGoal[i] = -1;
      continue;
    }

    const order = pool.order[i]!;
    if (order !== Order.Move && order !== Order.AttackMove) {
      pool.flowGoal[i] = -1;
      continue;
    }

    // Arrived? Measure against the ordered point, not the tile centre, so a
    // group converges on where the player actually clicked.
    const distToGoal = vecDist(pool.posX[i]!, pool.posY[i]!, pool.orderX[i]!, pool.orderY[i]!);
    if (distToGoal <= ARRIVAL_REACH) {
      pool.flowGoal[i] = -1;
      pool.order[i] = Order.None;
      continue;
    }

    const field = fields.get(world.map, goal);
    const here = world.map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
    if (here < 0 || field.isStranded(here)) {
      // No route from here — give up rather than jitter against a wall.
      pool.flowGoal[i] = -1;
      pool.order[i] = Order.None;
      continue;
    }

    const next = field.stepFrom(world.map, here);
    let tx: Fix;
    let ty: Fix;
    if (next < 0) {
      // Standing on the goal tile; close the last fraction of a tile directly.
      tx = pool.orderX[i]!;
      ty = pool.orderY[i]!;
    } else {
      tx = fromInt(world.map.tileXOf(next)) + FIX_HALF;
      ty = fromInt(world.map.tileYOf(next)) + FIX_HALF;
    }

    stepToward(world, i, tx, ty, def.speedPerTick, def.turnPerTick);
  }
}

/** Move `index` toward (tx, ty) by at most `speed`, turning to face the way. */
function stepToward(
  world: World,
  index: number,
  tx: Fix,
  ty: Fix,
  speed: Fix,
  turnRate: Fix,
): void {
  const pool = world.pool;
  const dx = tx - pool.posX[index]!;
  const dy = ty - pool.posY[index]!;
  const dist = vecDist(pool.posX[index]!, pool.posY[index]!, tx, ty);
  if (dist === 0) return;

  const step = dist < speed ? dist : speed;
  const dir = vecNormalize(dx, dy);
  const dirX = dir.x;
  const dirY = dir.y;
  pool.posX[index] = (pool.posX[index]! + fmul(dirX, step)) | 0;
  pool.posY[index] = (pool.posY[index]! + fmul(dirY, step)) | 0;

  const face = vecRotateToward(pool.faceX[index]!, pool.faceY[index]!, dirX, dirY, turnRate);
  pool.faceX[index] = face.x;
  pool.faceY[index] = face.y;
}

/**
 * Drain up to the per-tick budget of path requests.
 *
 * Entities that died or changed orders while queued are skipped without
 * consuming budget, so a burst of cancelled orders cannot starve live ones.
 */
function servePathRequests(world: World, astar: AStar): void {
  const pool = world.pool;
  const queue = world.pathQueue;
  let served = 0;
  let read = 0;

  while (read < queue.length && served < PATH_BUDGET_PER_TICK) {
    const i = queue[read++]!;
    if (pool.alive[i] !== 1 || pool.pathPending[i] !== 1) continue;

    const startTile = world.map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
    let goalTile = world.map.tileOfPos(pool.orderX[i]!, pool.orderY[i]!);

    // Right-clicking a cliff or a building should walk as close as possible
    // rather than being rejected outright.
    if (goalTile >= 0) {
      const gx = world.map.tileXOf(goalTile);
      const gy = world.map.tileYOf(goalTile);
      if (!world.map.isWalkable(gx, gy)) goalTile = nearestWalkable(world.map, gx, gy);
    }

    pool.pathPending[i] = 0;
    served++;

    if (startTile < 0 || goalTile < 0) {
      pool.clearPath(i);
      continue;
    }
    const path = astar.find(world.map, startTile, goalTile);
    if (path.length === 0) {
      // No route. Drop the order so the unit does not spin re-requesting, and
      // back off before trying again — a failed search costs the full expansion
      // budget, so retrying every tick is what made pathfinding dominate the
      // whole simulation.
      pool.clearPath(i);
      pool.pathCooldown[i] = PATH_RETRY_COOLDOWN;
      if (pool.order[i] === Order.Move || pool.order[i] === Order.AttackMove) {
        pool.order[i] = Order.None;
      }
    } else {
      pool.setPath(i, path);
    }
  }

  // Compact the queue in place, preserving order for anything still pending.
  if (read > 0) queue.splice(0, read);
}

function followPaths(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;
    const def = defOf(type);
    if (def.isBuilding || def.speedPerTick === 0) continue;

    const order = pool.order[i]!;
    if (order === Order.None || order === Order.Hold) continue;

    // Orders that chase an entity steer toward its live position; orders that
    // target ground follow the precomputed path.
    if (order === Order.Attack || order === Order.Build || order === Order.Harvest) {
      const targetId = pool.orderTarget[i]!;
      if (targetId !== NO_ENTITY && pool.isAlive(targetId)) {
        const ti = idIndex(targetId);
        maybeRepathToward(world, i, pool.posX[ti]!, pool.posY[ti]!);
      }
    }

    const len = pool.pathLen[i]!;
    if (len === 0) {
      // No path, but we may still have somewhere to be. When a unit is chasing
      // an entity and is only a tile or two short, A* is both unnecessary and
      // actively harmful: the nearest walkable tile to a building footprint is
      // often the one the unit is already standing on, so the search returns an
      // empty path, gets treated as a failure, and the unit parks just outside
      // build range forever. Close the last leg by steering straight at it.
      closeOnTarget(world, i, def);
      continue;
    }

    let cursor = pool.pathCursor[i]!;
    if (cursor >= len) {
      pool.clearPath(i);
      if (order === Order.Move || order === Order.AttackMove) pool.order[i] = Order.None;
      continue;
    }

    // Walk toward the current waypoint, consuming waypoints we have reached.
    let remaining = def.speedPerTick;
    while (remaining > 0 && cursor < len) {
      const tile = pool.pathNode(i, cursor);
      const wx = fromInt(world.map.tileXOf(tile)) + FIX_HALF;
      const wy = fromInt(world.map.tileYOf(tile)) + FIX_HALF;
      const dx = wx - pool.posX[i]!;
      const dy = wy - pool.posY[i]!;
      const dist = vecDist(pool.posX[i]!, pool.posY[i]!, wx, wy);

      if (dist <= WAYPOINT_REACH || dist === 0) {
        cursor++;
        continue;
      }

      const step = dist < remaining ? dist : remaining;
      const dir = vecNormalize(dx, dy);
      const dirX = dir.x;
      const dirY = dir.y;
      pool.posX[i] = (pool.posX[i]! + fmul(dirX, step)) | 0;
      pool.posY[i] = (pool.posY[i]! + fmul(dirY, step)) | 0;

      const face = vecRotateToward(
        pool.faceX[i]!,
        pool.faceY[i]!,
        dirX,
        dirY,
        def.turnPerTick,
      );
      pool.faceX[i] = face.x;
      pool.faceY[i] = face.y;

      remaining -= step;
      if (step === dist) cursor++;
    }

    pool.pathCursor[i] = cursor < MAX_PATH ? cursor : MAX_PATH;
    if (cursor >= len) {
      pool.clearPath(i);
      if (order === Order.Move || order === Order.AttackMove) pool.order[i] = Order.None;
    }
  }
}

/**
 * Walk the final short distance to an entity target without pathfinding.
 *
 * Only used when the unit already has no path and is close enough that
 * obstacles are unlikely to matter. `clampToMap` ejects anything that ends up
 * inside a footprint, so the worst case self-corrects.
 */
function closeOnTarget(world: World, index: number, def: EntityDef): void {
  const pool = world.pool;
  const order = pool.order[index]!;
  if (order !== Order.Build && order !== Order.Harvest && order !== Order.Attack) return;

  const targetId = pool.orderTarget[index]!;
  if (targetId === NO_ENTITY || !pool.isAlive(targetId)) return;

  const ti = idIndex(targetId);
  const dx = pool.posX[ti]! - pool.posX[index]!;
  const dy = pool.posY[ti]! - pool.posY[index]!;
  const reach =
    def.radius + defOf(pool.type[ti]! as EntityType).radius + reachSlackFor(order);
  const distSq = vecLenSqRaw(dx, dy);
  if (distSq <= sqRange(reach)) return; // already there

  // Only for the last leg; anything further away is a real navigation problem
  // and should wait for a path rather than walking into a wall.
  if (distSq > sqRange(reach + fromInt(6))) return;

  stepToward(
    world,
    index,
    pool.posX[ti]!,
    pool.posY[ti]!,
    def.speedPerTick,
    def.turnPerTick,
  );
}

/**
 * Re-path toward a moving target, but only occasionally.
 *
 * Chasing by re-running A* every tick would be both wasteful and jittery. Using
 * the tick counter as the phase means the interval is part of simulation state,
 * so peers re-path in lockstep rather than on wall-clock timers.
 */
function maybeRepathToward(world: World, index: number, tx: Fix, ty: Fix): void {
  const pool = world.pool;
  if (pool.pathPending[index] === 1) return;

  // Back off after a failed search instead of retrying immediately. A unit that
  // cannot reach its target would otherwise burn a full-budget A* every tick.
  if (pool.pathCooldown[index]! > 0) {
    pool.pathCooldown[index]! -= 1;
    return;
  }

  // Already standing at the target — harvesting, building, or in melee. There is
  // nothing to path to, and re-pathing here was the single largest source of
  // wasted searches.
  const targetId = pool.orderTarget[index]!;
  if (targetId !== NO_ENTITY && pool.isAlive(targetId)) {
    const ti = idIndex(targetId);
    const reach =
      defOf(pool.type[index]! as EntityType).radius +
      defOf(pool.type[ti]! as EntityType).radius +
      reachSlackFor(pool.order[index]! as Order);
    const dx = pool.posX[ti]! - pool.posX[index]!;
    const dy = pool.posY[ti]! - pool.posY[index]!;
    if (vecLenSqRaw(dx, dy) <= sqRange(reach)) {
      pool.clearPath(index);
      return;
    }
  }

  const stale =
    pool.pathLen[index] === 0 || (world.tick + index) % CHASE_REPATH_INTERVAL === 0;
  if (!stale) return;

  pool.orderX[index] = tx;
  pool.orderY[index] = ty;
  pool.pathPending[index] = 1;
  world.pathQueue.push(index);
}

/**
 * Push overlapping units apart.
 *
 * Without this, a group given one destination piles into a single tile and units
 * visually occupy the same space. This is a cheap positional relaxation rather
 * than true collision response — it resolves overlap over a few ticks, which is
 * what RTS movement wants anyway (units should squeeze past each other, not
 * bounce).
 *
 * Pairs are visited in ascending index order via the spatial grid, and each pair
 * is handled exactly once (`j > i`), so the result is order-independent in the
 * only sense that matters: it is the same on every peer.
 */
function separate(world: World): void {
  const pool = world.pool;
  const grid = world.grid;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const defI = defOf(pool.type[i]! as EntityType);
    if (defI.isBuilding) continue;

    const px = pool.posX[i]!;
    const py = pool.posY[i]!;
    const ri = defI.radius;

    grid.forEachNear(px, py, fromInt(2), (j) => {
      if (j <= i) return; // handle each pair once, from the lower index
      if (pool.alive[j] !== 1) return;
      const defJ = defOf(pool.type[j]! as EntityType);
      if (defJ.isBuilding) return;

      const dx = pool.posX[j]! - px;
      const dy = pool.posY[j]! - py;
      const minDist = ri + defJ.radius;
      const distSq = vecLenSqRaw(dx, dy);
      if (distSq >= sqRange(minDist)) return;

      if (distSq === 0) {
        // Exactly coincident: nudge apart along a fixed axis chosen by index
        // parity, so the resolution is deterministic rather than arbitrary.
        const nudge = (i & 1) === 0 ? SEPARATION_STRENGTH : -SEPARATION_STRENGTH;
        pool.posX[i] = (pool.posX[i]! - nudge) | 0;
        pool.posX[j] = (pool.posX[j]! + nudge) | 0;
        return;
      }

      const dist = Math.sqrt(distSq) | 0;
      const overlap = minDist - dist;
      const push = fmul(overlap, SEPARATION_STRENGTH);
      const inv = fdiv(push, dist);
      const ox = fmul(dx, inv);
      const oy = fmul(dy, inv);

      pool.posX[i] = (pool.posX[i]! - ox) | 0;
      pool.posY[i] = (pool.posY[i]! - oy) | 0;
      pool.posX[j] = (pool.posX[j]! + ox) | 0;
      pool.posY[j] = (pool.posY[j]! + oy) | 0;
    });
  }

  clampToMap(world);
}

/**
 * Keep everyone inside the playfield and out of solid tiles.
 *
 * Separation nudges can push a unit into a building footprint or a cliff. Beyond
 * looking wrong, it strands the unit: its start tile is unwalkable and possibly
 * enclosed, so every path request it makes explores the entire expansion budget
 * and fails. Ejecting it to the nearest walkable tile fixes both.
 */
function clampToMap(world: World): void {
  const pool = world.pool;
  const lo = 0;
  const hi = fromInt(world.map.width) - 1;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (defOf(pool.type[i]! as EntityType).isBuilding) continue;

    if (pool.posX[i]! < lo) pool.posX[i] = lo;
    if (pool.posY[i]! < lo) pool.posY[i] = lo;
    if (pool.posX[i]! > hi) pool.posX[i] = hi;
    if (pool.posY[i]! > hi) pool.posY[i] = hi;

    const tile = world.map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
    if (tile < 0) continue;
    const tx = world.map.tileXOf(tile);
    const ty = world.map.tileYOf(tile);
    if (world.map.isWalkable(tx, ty)) continue;

    const free = nearestWalkable(world.map, tx, ty, 6);
    if (free < 0) continue;
    pool.posX[i] = fromInt(world.map.tileXOf(free)) + FIX_HALF;
    pool.posY[i] = fromInt(world.map.tileYOf(free)) + FIX_HALF;
  }
}
