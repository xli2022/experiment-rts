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
  fsqrt,
  sqRange,
  vecDist,
  vecLenSqRaw,
  vecNormalize,
  vecRotateToward,
  type Fix,
} from '../fixed.js';
import { AStar, nearestWalkable } from '../pathing/astar.js';
import { approachPoint } from './economy.js';
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
  moveFlyers(world);
  servePathRequests(world, astar);
  followFlowFields(world, fields);
  followPaths(world);
  engageNearby(world);
  separate(world);
}

/**
 * Idle units step up to something they have already picked a fight with.
 *
 * Combat acquires a target within sight and then only shoots if it is already
 * in weapon range; nothing ever closed the gap. For a rifleman that is
 * invisible — its range covers everything it can see nearby — but a brawler
 * reaches 1.3 tiles and so stood still while an enemy two tiles away shot it,
 * which reads as melee units simply not fighting.
 *
 * Deliberately a short leash rather than a chase: a unit walks the last few
 * tiles onto a target and no further, so an idle army holds its ground instead
 * of being drawn across the map one straggler at a time. `Hold` never moves at
 * all — that is what it is for.
 */
function engageNearby(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding || def.speedPerTick === 0 || def.attackRange === 0) continue;
    if (pool.order[i] !== Order.None) continue;

    const targetId = pool.combatTarget[i]!;
    if (targetId === NO_ENTITY || !pool.isAlive(targetId)) continue;

    const ti = idIndex(targetId);
    const dx = pool.posX[ti]! - pool.posX[i]!;
    const dy = pool.posY[ti]! - pool.posY[i]!;
    const distSq = vecLenSqRaw(dx, dy);

    // Combat measures to the target's edge, so this has to agree with it or the
    // unit creeps forward for one more tick after it can already shoot.
    const reach = def.attackRange + defOf(pool.type[ti]! as EntityType).radius;
    if (distSq <= sqRange(reach)) continue;
    if (distSq > sqRange(reach + ENGAGE_LEASH)) continue;

    stepToward(world, i, pool.posX[ti]!, pool.posY[ti]!, def.speedPerTick, def.turnPerTick);
  }
}

/**
 * Air movement: steer straight at the destination, ignoring everything.
 *
 * Flyers need no pathfinding at all, which is both correct for the genre and a
 * useful property — a flying army costs nothing in the system that dominates
 * simulation time. They are handled entirely here and skipped by every
 * ground-movement pass below.
 */
function moveFlyers(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const def = defOf(pool.type[i]! as EntityType);
    if (!def.flying || def.speedPerTick === 0) continue;

    const order = pool.order[i]!;
    if (order === Order.None || order === Order.Hold) continue;

    // Chase orders track the target's live position; ground orders head for the
    // commanded point.
    let tx = pool.orderX[i]!;
    let ty = pool.orderY[i]!;
    let stopWithin = ARRIVAL_REACH;

    const targetId = pool.orderTarget[i]!;
    if (targetId !== NO_ENTITY && pool.isAlive(targetId)) {
      const ti = idIndex(targetId);
      tx = pool.posX[ti]!;
      ty = pool.posY[ti]!;
      // Stop at weapon range rather than flying into the target.
      stopWithin = def.attackRange + defOf(pool.type[ti]! as EntityType).radius;
    }

    const dist = vecDist(pool.posX[i]!, pool.posY[i]!, tx, ty);
    if (dist <= stopWithin) {
      if (order === Order.Move || order === Order.AttackMove) pool.order[i] = Order.None;
      continue;
    }

    stepToward(world, i, tx, ty, def.speedPerTick, def.turnPerTick);
  }
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
    if (def.isBuilding || def.speedPerTick === 0 || def.flying) {
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
  if (dist === 0) {
    pool.speed[index] = 0;
    return;
  }

  const step = accelerate(world, index, dist, speed);
  if (step > 0) {
    const dir = vecNormalize(dx, dy);
    pool.posX[index] = (pool.posX[index]! + fmul(dir.x, step)) | 0;
    pool.posY[index] = (pool.posY[index]! + fmul(dir.y, step)) | 0;
    // Facing is taken from the direction of travel, which is the same thing the
    // old code used — but read before the position update rather than after, so
    // a unit that arrives this tick still faces where it was going.
    const face = vecRotateToward(pool.faceX[index]!, pool.faceY[index]!, dir.x, dir.y, turnRate);
    pool.faceX[index] = face.x;
    pool.faceY[index] = face.y;
  }
}

/**
 * Roughly how far this unit still has to travel along its path.
 *
 * Straight-line to the last waypoint plus a tile per waypoint after the next,
 * which overestimates a winding route and underestimates nothing — and only
 * matters near the end, where the path is short and the estimate is tight. A
 * unit needs this to know when to start easing off, not to navigate by.
 */
function distanceLeft(world: World, index: number, len: number, cursor: number): Fix {
  const pool = world.pool;
  const last = pool.pathNode(index, len - 1);
  const lx = fromInt(world.map.tileXOf(last)) + FIX_HALF;
  const ly = fromInt(world.map.tileYOf(last)) + FIX_HALF;
  const direct = vecDist(pool.posX[index]!, pool.posY[index]!, lx, ly);
  const corners = len - cursor - 1;
  return corners > 0 ? direct + fromInt(corners) : direct;
}

/**
 * Advance this unit's speed one tick and return the distance to move.
 *
 * Units used to travel at their top speed on the tick they were ordered and
 * stop dead on the tick they arrived, which is most of what made movement look
 * mechanical rather than heavy. Now they ramp.
 *
 * The braking term is `v = sqrt(2 * a * d)`: the fastest a unit can be going
 * and still shed all of it before `d`. `Math.sqrt` is the one non-trivial
 * function allowed in here — IEEE-754 requires it to be correctly rounded, so
 * it agrees bit for bit across engines, which the transcendentals do not.
 */
function accelerate(world: World, index: number, dist: Fix, top: Fix): Fix {
  const pool = world.pool;
  const def = defOf(pool.type[index]! as EntityType);
  const accel = fmul(top, def.accelFraction);
  if (accel <= 0) {
    // A unit with no ramp behaves exactly as before, which keeps the door open
    // for something that genuinely should not ease in.
    pool.speed[index] = top;
    return dist < top ? dist : top;
  }

  // Fast enough to still stop in the distance left, and no faster than its legs.
  const brake = fsqrt(fmul(fromInt(2), fmul(accel, dist)));
  const want = brake < top ? brake : top;

  let v = pool.speed[index]!;
  if (v < want) v = v + accel > want ? want : v + accel;
  else if (v > want) v = v - accel < want ? want : v - accel;
  // A unit at rest with a target must get under way; without this floor, a
  // stationary unit whose braking distance rounds to zero never starts.
  if (v <= 0) v = accel < top ? accel : top;
  pool.speed[index] = v;

  return dist < v ? dist : v;
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

/**
 * How far past its weapon range a unit will walk to reach something.
 *
 * Short on purpose. Long enough that a melee unit engages anything that comes
 * to it, short enough that an idle line does not unravel into a chase.
 */
const ENGAGE_LEASH = fromInt(5);

/** Scratch for `approachPoint`; the simulation allocates nothing per tick. */
const approachOut = { x: 0 as Fix, y: 0 as Fix };

function followPaths(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;
    const def = defOf(type);
    if (def.isBuilding || def.speedPerTick === 0 || def.flying) continue;

    const order = pool.order[i]!;
    if (order === Order.None || order === Order.Hold) continue;

    // Orders that chase an entity steer toward its live position; orders that
    // target ground follow the precomputed path.
    if (order === Order.Attack || order === Order.Build || order === Order.Harvest) {
      const targetId = pool.orderTarget[i]!;
      if (targetId !== NO_ENTITY && pool.isAlive(targetId)) {
        const ti = idIndex(targetId);
        // Head for the near face of a building rather than its middle. The
        // centre of a footprint is not walkable, so A* substitutes the nearest
        // walkable tile to it — the same tile for everyone, whichever side they
        // came from, which is what made workers walk around a Command Post
        // instead of delivering where they stood.
        approachPoint(world, ti, pool.posX[i]!, pool.posY[i]!, approachOut);
        maybeRepathToward(world, i, approachOut.x, approachOut.y);
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
    //
    // The travel budget is the accelerated speed, and it is measured against
    // the distance still to run rather than to the next waypoint — braking for
    // every corner of an A* path would make a unit stutter its way across the
    // map instead of easing to a stop at the end of it.
    let remaining = accelerate(world, i, distanceLeft(world, i, len, cursor), def.speedPerTick);
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
    // Workers pass through everything: neither pushed nor pushing. Flyers do
    // collide, but only with each other — `collides` is false for them because
    // it also decides whether a thing occupies map tiles, and nothing in the
    // air should block the ground.
    if (!defI.collides && !defI.flying) continue;

    const px = pool.posX[i]!;
    const py = pool.posY[i]!;
    const ri = defI.radius;

    grid.forEachNear(px, py, fromInt(2), (j) => {
      if (j <= i) return; // handle each pair once, from the lower index
      if (pool.alive[j] !== 1) return;
      const defJ = defOf(pool.type[j]! as EntityType);
      if (defJ.isBuilding) return;
      // Same layer or no interaction. Air and ground share the map but not the
      // space, so a gunship never shoulders a brawler aside.
      if (defI.flying !== defJ.flying) return;
      if (!defJ.collides && !defJ.flying) return;

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
    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding) continue;

    if (pool.posX[i]! < lo) pool.posX[i] = lo;
    if (pool.posY[i]! < lo) pool.posY[i] = lo;
    if (pool.posX[i]! > hi) pool.posX[i] = hi;
    if (pool.posY[i]! > hi) pool.posY[i] = hi;

    // Flyers are over the terrain, not on it, so nothing ejects them.
    if (def.flying) continue;

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
