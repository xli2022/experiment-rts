/**
 * Movement: path requests, path following, and local separation.
 *
 * Pathfinding is the one system that can realistically blow the tick budget, and
 * in lockstep a slow tick is not a local problem — every peer waits for the
 * slowest one. So A* requests are served from a bounded queue: at most
 * `PATH_BUDGET_PER_TICK` searches run per tick and the rest wait. Because the
 * queue is part of world state and drains in a fixed order, every peer defers
 * exactly the same requests to exactly the same later tick.
 *
 * ## Every unit sees the tick as it started
 *
 * Units are processed in ascending slot order, and the first player's units
 * hold the low slots. Any pass that reads *another* unit's position while
 * writing its own therefore sees a half-updated world, in an order that is not
 * the mirrored order for the other side: a unit closing on an enemy stepped,
 * and its opposite number then measured against the moved position and stood
 * still one step short. So the movers read other units' positions from a
 * snapshot taken before any of them moves (`snapX`, `snapY`), and separation
 * accumulates every pair's push before applying any of them. Both halves then
 * compute from the same picture whatever order they are visited in.
 */

import {
  PATH_BUDGET_PER_TICK,
  SEPARATION_STRENGTH,
  defOf,
  reachSlackFor,
  BUILD_REACH,
} from '../../config/rules.js';
import type { FlowField, FlowFieldCache } from '../pathing/flowfield.js';
import { ARRIVE_BEST_NONE, ENTITY_CAPACITY, idIndex, MAX_PATH } from '../entities.js';
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
import type { ConstructionPaths } from '../pathing/construction.js';
import { lineOfSightClear, smoothPath, tileCentreX, tileCentreY } from '../pathing/los.js';
import { approachPoint, inReach } from './economy.js';
import { topSpeedOf } from './combat.js';
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

/**
 * Positions as they were when the movement pass began, for reading *other*
 * units' positions. Scratch, refilled every tick, never checksummed.
 */
const snapX = new Int32Array(ENTITY_CAPACITY);
const snapY = new Int32Array(ENTITY_CAPACITY);

export function movementSystem(
  world: World,
  astar: AStar,
  fields: FlowFieldCache,
  construction: ConstructionPaths,
): void {
  const pool = world.pool;
  snapX.set(pool.posX.subarray(0, pool.count));
  snapY.set(pool.posY.subarray(0, pool.count));
  moveFlyers(world);
  resumeAdvance(world);
  servePathRequests(world, astar, construction);
  followFlowFields(world, fields);
  followPaths(world);
  engageNearby(world);
  separate(world);
  settleArrivals(world);
}

/**
 * Stop a unit that is as close to its destination as it is ever going to get.
 *
 * Arrival is "within half a tile of the point you were given", and a unit whose
 * point is taken — by a unit that got there first, by a building, by rock —
 * can never satisfy it. Nothing else ever ends the order, so it pushes at the
 * spot forever: a group move left a fifth to a half of its units shoving at the
 * crowd for the rest of the match, still holding a Move order.
 *
 * Three things keep this from stopping units that are merely taking a while:
 *
 *   - It only runs near the destination. Further out, a unit walking around an
 *     obstacle genuinely fails to close the straight-line gap for a long
 *     stretch, and stopping it there would strand it mid-route.
 *   - Progress resets it, and progress means beating the closest approach so
 *     far by a real margin — not the jitter separation puts on every unit in a
 *     crowd every tick.
 *   - Fighting is not failing to arrive, so a unit with a live target holds.
 *
 * Runs after `separate`, which is what does the pushing, so the distance it
 * reads is where the unit actually ended the tick.
 */
function settleArrivals(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const order = pool.order[i]!;
    if (order !== Order.Move && order !== Order.AttackMove) continue;

    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding || def.speedPerTick === 0) continue;

    const dist = vecDist(pool.posX[i]!, pool.posY[i]!, pool.orderX[i]!, pool.orderY[i]!);
    if (dist > SETTLE_RANGE) {
      pool.arriveBest[i] = ARRIVE_BEST_NONE;
      pool.arriveStall[i] = 0;
      continue;
    }

    // Fighting is not failing to arrive — but a repairer's target is one of
    // ours, and standing next to something it is mending is not a reason to
    // keep shoving at a destination it has already reached.
    const target = pool.combatTarget[i]!;
    if (def.damage > 0 && target !== NO_ENTITY && pool.isAlive(target)) {
      pool.arriveStall[i] = 0;
      continue;
    }

    if (dist + SETTLE_MARGIN < pool.arriveBest[i]!) {
      pool.arriveBest[i] = dist;
      pool.arriveStall[i] = 0;
      continue;
    }

    pool.arriveStall[i]! += 1;
    if (pool.arriveStall[i]! < SETTLE_TICKS) continue;

    pool.clearPath(i);
    pool.order[i] = Order.None;
    pool.navGoal[i] = -1;
    pool.arriveBest[i] = ARRIVE_BEST_NONE;
    pool.arriveStall[i] = 0;
  }
}

/**
 * Put an attack-moving unit back on the road once its fight is over.
 *
 * Stopping to shoot calls `clearPath`, which wipes the route *and* the shared
 * flow-field goal. That is correct — a unit holding its ground should not also
 * be walking — but nothing ever put it back. So an army given one attack-move
 * across the map stopped at the first thing it killed and stood there for the
 * rest of the match, still holding an `AttackMove` order it would never
 * complete. It reads as a movement bug; it is a missing transition.
 *
 * Runs before the movers so a unit that resumes this tick walks this tick.
 */
function resumeAdvance(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.order[i] !== Order.AttackMove) continue;
    // Already routed, or waiting on a search we asked for.
    if (pool.pathLen[i]! > 0 || pool.flowGoal[i]! >= 0 || pool.pathPending[i] === 1) continue;

    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding || def.speedPerTick === 0) continue;
    // Flyers steer straight at the order point and never hold a route.
    if (def.flying) continue;

    // Something still worth walking at? Then this is a pause in the advance,
    // not the end of it, and `engageNearby` owns the unit until it is gone.
    if (shouldPursue(world, i, def)) continue;

    // Otherwise: arrived, or the fight is over and the advance continues.
    if (vecDist(pool.posX[i]!, pool.posY[i]!, pool.orderX[i]!, pool.orderY[i]!) <= ARRIVAL_REACH) {
      pool.order[i] = Order.None;
      pool.navGoal[i] = -1;
      continue;
    }

    if (pool.pathCooldown[i]! > 0) {
      pool.pathCooldown[i]! -= 1;
      continue;
    }

    if (pool.navGoal[i]! >= 0) {
      pool.flowGoal[i] = pool.navGoal[i]!;
    } else {
      pool.pathPending[i] = 1;
      world.pathQueue.push(i);
    }
  }
}

/**
 * Should this unit walk at its combat target rather than at its destination?
 *
 * The target has to be worth stepping to — inside the acquisition leash — and,
 * for an attack-mover, the chase has to stay near where it began. The anchor is
 * the whole point: measured from the unit's *current* position, the window
 * slides along with a retreating enemy and the chase ratchets indefinitely. A
 * unit dragged sideways followed a fleeing Burstbot 14.6 tiles off its route.
 *
 * Crossing the anchor leash commits the unit to resuming its objective until
 * it loses contact. Testing just the radius lets that resumed movement step
 * back inside the leash, where engagement immediately turns it around again.
 * It can still stop and shoot anything already in weapon range on its way.
 */
function shouldPursue(world: World, index: number, def: EntityDef): boolean {
  const pool = world.pool;
  if (def.attackRange === 0) return false;

  const targetId = pool.combatTarget[index]!;
  if (targetId === NO_ENTITY || !pool.isAlive(targetId)) {
    pool.pursuing[index] = 0;
    return false;
  }

  const ti = idIndex(targetId);
  const dx = snapX[ti]! - pool.posX[index]!;
  const dy = snapY[ti]! - pool.posY[index]!;
  const reach = def.attackRange + defOf(pool.type[ti]! as EntityType).radius;
  const targetDistSq = vecLenSqRaw(dx, dy);
  if (targetDistSq > sqRange(reach + ENGAGE_LEASH)) {
    pool.pursuing[index] = 0;
    return false;
  }

  if (pool.order[index] !== Order.AttackMove) return true;

  // Disengagement survives the return inside the anchor radius, including
  // another call later in this same movement pass. A target that closes into
  // weapon range may still be fought without starting another chase.
  if (pool.pursuing[index] === 2) return targetDistSq <= sqRange(reach);

  if (pool.pursuing[index] === 0) {
    pool.pursuing[index] = 1;
    pool.pursueX[index] = pool.posX[index]!;
    pool.pursueY[index] = pool.posY[index]!;
  }
  const strayed = vecDist(
    pool.posX[index]!,
    pool.posY[index]!,
    pool.pursueX[index]!,
    pool.pursueY[index]!,
  );
  if (strayed <= PURSUE_LEASH) return true;
  pool.pursuing[index] = 2;
  return targetDistSq <= sqRange(reach);
}

/**
 * Idle units step up to something they have already picked a fight with.
 *
 * Combat acquires a target within sight and then only shoots if it is already
 * in weapon range; nothing ever closed the gap. For a Burstbot that is
 * invisible — its range covers everything it can see nearby — but a Slicebot
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
    const order = pool.order[i]!;
    // Idle units defend themselves; attack-movers go and take what they saw. A
    // plain Move does neither, which is the whole reason attack-move exists.
    if (order !== Order.None && order !== Order.AttackMove) continue;
    if (!shouldPursue(world, i, def)) continue;

    const ti = idIndex(pool.combatTarget[i]!);
    const dx = snapX[ti]! - pool.posX[i]!;
    const dy = snapY[ti]! - pool.posY[i]!;
    // Combat measures to the target's edge, so this has to agree with it or the
    // unit creeps forward for one more tick after it can already shoot.
    const reach = def.attackRange + defOf(pool.type[ti]! as EntityType).radius;
    if (vecLenSqRaw(dx, dy) <= sqRange(reach)) continue;

    // Drop the route while closing, so path-following does not drag the unit
    // onward in the same tick. `resumeAdvance` restores it when the fight ends.
    if (order === Order.AttackMove) pool.clearPath(i);
    stepToward(world, i, snapX[ti]!, snapY[ti]!, topSpeedOf(world, i, def), def.turnPerTick);
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
    // A flying attack-mover has no route for combat to clear. Hand pursuit to
    // engageNearby explicitly, or it keeps flying past the target while firing
    // and can receive a second movement step from that pass on the same tick.
    if (order === Order.AttackMove && shouldPursue(world, i, def)) continue;

    // Chase orders track the target's live position; ground orders head for the
    // commanded point.
    let tx = pool.orderX[i]!;
    let ty = pool.orderY[i]!;
    let stopWithin = ARRIVAL_REACH;

    const targetId = pool.orderTarget[i]!;
    if (targetId !== NO_ENTITY && pool.isAlive(targetId)) {
      const ti = idIndex(targetId);
      tx = snapX[ti]!;
      ty = snapY[ti]!;
      // Stop at weapon range rather than flying into the target.
      stopWithin = def.attackRange + defOf(pool.type[ti]! as EntityType).radius;
    }

    const dist = vecDist(pool.posX[i]!, pool.posY[i]!, tx, ty);
    if (dist <= stopWithin) {
      if (order === Order.Move || order === Order.AttackMove) pool.order[i] = Order.None;
      continue;
    }

    stepToward(world, i, tx, ty, topSpeedOf(world, i, def), def.turnPerTick);
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

    // The last stretch is walked straight at this unit's own place in the
    // formation, not at the shared goal tile.
    //
    // The field steers the whole group at one tile, and near the destination
    // that tile is already full. Units that cannot get onto it are steered into
    // the scrum anyway, shoved back out by separation, and steered in again —
    // for the rest of the match, since only their own spread point counts as
    // arriving and they are being pushed away from it. Measured on open ground:
    // 20 of 24 units in a group move never came to rest.
    const flip = world.flipOf(pool.owner[i]!);
    // Large selections spread beyond the old three-tile handoff. Include the
    // slot's distance from the shared goal or far-side units keep steering back
    // into the centre before they ever get close enough to their own slot.
    const approachReach = Math.max(
      FORMATION_APPROACH,
      vecDist(
        pool.orderX[i]!,
        pool.orderY[i]!,
        tileCentreX(world.map, goal),
        tileCentreY(world.map, goal),
      ) + ARRIVAL_REACH,
    );
    if (
      distToGoal <= FORMATION_APPROACH ||
      (distToGoal <= approachReach &&
        lineOfSightClear(
          world.map,
          pool.posX[i]!,
          pool.posY[i]!,
          pool.orderX[i]!,
          pool.orderY[i]!,
          flip,
        ))
    ) {
      stepToward(
        world,
        i,
        pool.orderX[i]!,
        pool.orderY[i]!,
        topSpeedOf(world, i, def),
        def.turnPerTick,
      );
      continue;
    }

    const field = fields.get(world.map, goal);
    const here = world.map.tileOfPosFor(pool.posX[i]!, pool.posY[i]!, flip);
    if (here < 0 || field.isStranded(here)) {
      // No route from here — give up rather than jitter against a wall.
      pool.flowGoal[i] = -1;
      pool.order[i] = Order.None;
      continue;
    }

    const next = field.stepFrom(world.map, here, flip, pool.posX[i]!, pool.posY[i]!);
    let tx: Fix;
    let ty: Fix;
    if (next < 0) {
      // Standing on the goal tile; close the last fraction of a tile directly.
      tx = pool.orderX[i]!;
      ty = pool.orderY[i]!;
    } else {
      // Not at the next tile: at the furthest tile down the field this unit can
      // still walk to in a straight line. See `lookAhead`.
      lookAhead(world, field, i, next, flip, aimOut);
      tx = aimOut.x;
      ty = aimOut.y;
    }

    stepToward(world, i, tx, ty, topSpeedOf(world, i, def), def.turnPerTick);
  }
}

/**
 * How many tiles down the flow field to look for something to walk at.
 *
 * Far enough that a leg spans the longest straight a unit meets in a corridor,
 * short enough that the visibility tests stay cheap: this runs per unit per
 * tick, and the scan stops early the moment sight breaks, so an army in broken
 * ground pays for two or three tests rather than eight.
 */
const FLOW_LOOKAHEAD = 8;

/** Where `lookAhead` puts its answer. Scratch; never read across calls. */
const aimOut = { x: 0 as Fix, y: 0 as Fix };

/** The tiles `lookAhead` is considering. Scratch; never read across calls. */
const probe = new Int32Array(FLOW_LOOKAHEAD);

/**
 * The furthest point down the flow field a unit can walk to in a straight line.
 *
 * Steering at the *next* tile is what made flow-field movement zig-zag, and it
 * is worth being precise about why, because the obvious answer — that eight
 * neighbours can only express eight headings — is only half of it.
 *
 * A unit aims at the centre of the adjacent tile but never arrives there: it
 * crosses into that tile at the corner nearest it and the field immediately
 * names a new target, so every leg begins from a tile corner while every leg
 * ends at a tile centre. Walking a shallow diagonal, the heading came out 45
 * degrees, then 18, then 38, then 18 again, tile after tile, and `turnPerTick`
 * dutifully rotated the model into each one. That is the wobble.
 *
 * Aiming eight tiles out instead makes crossing a boundary a small correction
 * rather than a new direction, and on open ground the unit simply walks the
 * straight line it should always have walked. The field is still consulted
 * every tick from where the unit actually stands, so units still flow around
 * each other and around a building that went up a moment ago — nothing is
 * cached and nothing goes stale.
 *
 * The first step is taken on the field's word alone, exactly as before, so this
 * can only ever improve on the old behaviour: the flow field has already
 * applied the corner rule to that step. Every step after it has to survive
 * `lineOfSightClear`, which applies the same rule to the whole segment.
 */
function lookAhead(
  world: World,
  field: FlowField,
  index: number,
  next: number,
  flip: boolean,
  out: { x: Fix; y: Fix },
): void {
  const pool = world.pool;
  const px = pool.posX[index]!;
  const py = pool.posY[index]!;

  // Where the field leads over the next few tiles. Probing from each tile's
  // centre rather than from the unit, which is what `stepFrom` does when given
  // no position: past the first step the unit is nowhere near, and feeding it a
  // position a corridor away would break ties toward tiles behind it.
  probe[0] = next;
  let n = 1;
  let cur = next;
  while (n < FLOW_LOOKAHEAD) {
    const ahead = field.stepFromCentre(world.map, cur, flip);
    if (ahead < 0) break;
    probe[n++] = ahead;
    cur = ahead;
  }

  // Furthest first, so open ground — where the whole lookahead is visible and
  // this is most worth doing — costs one visibility test rather than one per
  // tile. A test that fails gives up at the blocking tile, so the walk back is
  // cheap too.
  for (let k = n - 1; k > 0; k--) {
    const cx = tileCentreX(world.map, probe[k]!);
    const cy = tileCentreY(world.map, probe[k]!);
    if (!lineOfSightClear(world.map, px, py, cx, cy, flip)) continue;
    out.x = cx;
    out.y = cy;
    return;
  }

  // Nothing further is visible: the next tile, exactly as before.
  out.x = tileCentreX(world.map, next);
  out.y = tileCentreY(world.map, next);
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
    // Through the same terrain clamp separation uses. Steering is not always
    // along a path: the final approach to a formation slot, closing on a build
    // site, and stepping up to a combat target all aim straight at a point, and
    // a straight line to a point near a cliff clips the cliff.
    nudgeBy(
      world,
      index,
      fmul(dir.x, step),
      fmul(dir.y, step),
      defOf(pool.type[index]! as EntityType),
    );
    // Facing is taken from the direction of travel, which is the same thing the
    // old code used — but read before the position update rather than after, so
    // a unit that arrives this tick still faces where it was going.
    const face = vecRotateToward(pool.faceX[index]!, pool.faceY[index]!, dir.x, dir.y, turnRate);
    pool.faceX[index] = face.x;
    pool.faceY[index] = face.y;
  }
}

/**
 * How far this unit still has to travel along its path.
 *
 * The true length of the remaining polyline: the unit to its next waypoint,
 * then waypoint to waypoint. A unit needs this to know when to start easing
 * off, not to navigate by.
 *
 * This used to be the straight line to the final waypoint plus a tile for each
 * waypoint in between, which was a fair estimate while a waypoint meant a tile
 * and consecutive ones were a tile apart. Now that paths are string-pulled the
 * waypoints are corners, two of them can be forty tiles apart, and that
 * estimate reads a route around a headland as very nearly the straight line
 * across it — so a unit braked to a crawl the moment it started one. Summing
 * the segments costs a handful of distances against a path that smoothing has
 * already made short.
 */
function distanceLeft(world: World, index: number, len: number, cursor: number): Fix {
  const pool = world.pool;
  let px = pool.posX[index]!;
  let py = pool.posY[index]!;
  let total = 0;

  for (let k = cursor; k < len; k++) {
    const tile = pool.pathNode(index, k);
    const wx = tileCentreX(world.map, tile);
    const wy = tileCentreY(world.map, tile);
    total += vecDist(px, py, wx, wy);
    px = wx;
    py = wy;
  }

  return total;
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

/** Per-owner views of the path queue. Scratch for `servePathRequests`. */
const ownerQueues: number[][] = [];
const ownerHeads = new Int32Array(8);

/**
 * Drain up to the per-tick budget of path requests, one round at a time.
 *
 * A round serves one request for every player with a request waiting, and
 * only whole rounds are served. The queue used to drain first come, first
 * served, and since commands execute in player order the first player's units
 * were always at the front of it: whenever more requests arrived than the
 * budget covered, the other side's routes were the ones deferred to the next
 * tick. Under a mirrored attack order both armies ask at once, so that is
 * exactly when it happened.
 *
 * Entities that died or changed orders while queued are skipped without
 * consuming budget, so a burst of cancelled orders cannot starve live ones.
 */
function servePathRequests(world: World, astar: AStar, construction: ConstructionPaths): void {
  const pool = world.pool;
  const queue = world.pathQueue;
  if (queue.length === 0) return;

  const owners = world.players.length;
  while (ownerQueues.length < owners) ownerQueues.push([]);
  for (let o = 0; o < owners; o++) {
    ownerQueues[o]!.length = 0;
    ownerHeads[o] = 0;
  }
  for (let k = 0; k < queue.length; k++) {
    const i = queue[k]!;
    if (pool.alive[i] !== 1 || pool.pathPending[i] !== 1) continue;
    const owner = pool.owner[i]!;
    if (owner >= 0 && owner < owners) ownerQueues[owner]!.push(i);
  }
  // Within a player's list, oldest unit first. Requests arrive in slot order,
  // which the two halves of a mirrored match do not share; when the budget
  // binds, the units that wait must be the same units on both sides.
  for (let o = 0; o < owners; o++) {
    ownerQueues[o]!.sort((a, b) => pool.serial[a]! - pool.serial[b]!);
  }

  let served = 0;
  let active = 0;
  for (let o = 0; o < owners; o++) if (ownerQueues[o]!.length > 0) active++;

  while (active > 0 && served + active <= PATH_BUDGET_PER_TICK) {
    for (let o = 0; o < owners; o++) {
      const list = ownerQueues[o]!;
      let head = ownerHeads[o]!;
      if (head >= list.length) continue;
      // The same unit can be queued twice; the second entry is stale once the
      // first has been served, and costs nothing.
      while (head < list.length) {
        const i = list[head++]!;
        if (pool.pathPending[i] !== 1) continue;
        servePathRequest(world, astar, construction, i);
        served++;
        break;
      }
      ownerHeads[o] = head;
      if (head >= list.length) active--;
    }
  }

  // What is left waits, in the order it arrived within each player's own list.
  queue.length = 0;
  for (let o = 0; o < owners; o++) {
    const list = ownerQueues[o]!;
    for (let k = ownerHeads[o]!; k < list.length; k++) queue.push(list[k]!);
  }
}

function servePathRequest(
  world: World,
  astar: AStar,
  construction: ConstructionPaths,
  i: number,
): void {
  const pool = world.pool;
  // Every tile decision on this unit's behalf is made in its owner's frame,
  // so the mirrored unit asks the mirrored question and gets the mirrored
  // route.
  const flip = world.flipOf(pool.owner[i]!);
  const startTile = world.map.tileOfPosFor(pool.posX[i]!, pool.posY[i]!, flip);
  const site = pool.orderTarget[i]!;
  const building = pool.order[i] === Order.Build && pool.isAlive(site);
  let goalTile = world.map.tileOfPosFor(pool.orderX[i]!, pool.orderY[i]!, flip);

  // Right-clicking a cliff or a building should walk as close as possible
  // rather than being rejected outright.
  if (!building && goalTile >= 0) {
    const gx = world.map.tileXOf(goalTile);
    const gy = world.map.tileYOf(goalTile);
    if (!world.map.isWalkable(gx, gy)) goalTile = nearestWalkable(world.map, gx, gy, 12, flip);
  }

  pool.pathPending[i] = 0;

  if (startTile < 0 || (!building && goalTile < 0)) {
    pool.clearPath(i);
    return;
  }
  if (building && inReach(world, i, idIndex(site), BUILD_REACH)) {
    pool.clearPath(i);
    return;
  }
  const path = building
    ? construction.find(world, i, site, pathScratch)
    : astar.find(world.map, startTile, goalTile, pathScratch, flip);
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
    // Store the corners, not the tiles. A* answers one node per tile, and a
    // unit walking that literally can only head in the eight directions the
    // grid offers, so it staircases across open ground. String-pulling drops
    // every node the unit can already see past: a straight run becomes one
    // leg, and what is left is the turns it genuinely has to make.
    //
    // It also buys back the route length that `MAX_PATH` used to cost. A path
    // longer than 48 tiles was truncated and the unit stopped partway to ask
    // again; 48 *corners* is further than any route on any map here.
    pool.setPath(i, smoothPath(world.map, path, pool.posX[i]!, pool.posY[i]!, flip, smoothScratch));
  }
}

/**
 * Reused buffers for the two stages of a path request. One search runs at a
 * time, and neither array outlives the call that fills it.
 */
const pathScratch: number[] = [];
const smoothScratch: number[] = [];

/**
 * How far past its weapon range a unit will walk to reach something.
 *
 * Short on purpose. Long enough that a melee unit engages anything that comes
 * to it, short enough that an idle line does not unravel into a chase.
 */
const ENGAGE_LEASH = fromInt(5);

/**
 * How close a unit walks straight at its own formation slot rather than
 * following the group's shared field. Short, because it cuts corners: the
 * field is what routes around terrain.
 */
const FORMATION_APPROACH = fromInt(3);

/**
 * How near its destination a unit has to be before it may give up on reaching
 * the exact point. Wide enough to cover a crowd around a busy destination,
 * narrow enough that a unit still walking a route is never in it.
 */
const SETTLE_RANGE = fromInt(5);

/** How much closer counts as progress, rather than as being shoved about. */
const SETTLE_MARGIN = FIX_HALF >> 2;

/** How long a unit tries for a spot it cannot reach. 1.5 seconds. */
const SETTLE_TICKS = 30;

/**
 * How far an attack-mover will stray from where it broke off to chase.
 *
 * Enough to close on anything it can acquire, short enough that a retreating
 * enemy cannot walk an army off its objective one tile at a time.
 */
const PURSUE_LEASH = fromInt(6);

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
        approachPoint(world, ti, pool.posX[i]!, pool.posY[i]!, approachOut, snapX[ti]!, snapY[ti]!);
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
    let remaining = accelerate(
      world,
      i,
      distanceLeft(world, i, len, cursor),
      topSpeedOf(world, i, def),
    );
    let routeBlocked = false;
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
      const nextX = (pool.posX[i]! + fmul(dirX, step)) | 0;
      const nextY = (pool.posY[i]! + fmul(dirY, step)) | 0;
      // A foundation can appear across an already-smoothed private route. The
      // destination may remain walkable, so check the actual short step, not
      // just its waypoint. Otherwise terrain clamping ejects the unit back to
      // the same edge forever. Checking only this tick's step bounds the work
      // independently of how far away the next smoothed waypoint is.
      if (
        !lineOfSightClear(
          world.map,
          pool.posX[i]!,
          pool.posY[i]!,
          nextX,
          nextY,
          world.flipOf(pool.owner[i]!),
        )
      ) {
        const pending = pool.pathPending[i] === 1;
        pool.clearPath(i);
        pool.pathPending[i] = 1;
        if (!pending) world.pathQueue.push(i);
        routeBlocked = true;
        break;
      }
      pool.posX[i] = nextX;
      pool.posY[i] = nextY;

      const face = vecRotateToward(pool.faceX[i]!, pool.faceY[i]!, dirX, dirY, def.turnPerTick);
      pool.faceX[i] = face.x;
      pool.faceY[i] = face.y;

      remaining -= step;
      if (step === dist) cursor++;
    }

    if (routeBlocked) continue;
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
  const dx = snapX[ti]! - pool.posX[index]!;
  const dy = snapY[ti]! - pool.posY[index]!;
  const reach = def.radius + defOf(pool.type[ti]! as EntityType).radius + reachSlackFor(order);
  const distSq = vecLenSqRaw(dx, dy);
  if (distSq <= sqRange(reach)) return; // already there

  // Only for the last leg; anything further away is a real navigation problem
  // and should wait for a path rather than walking into a wall.
  if (distSq > sqRange(reach + fromInt(6))) return;

  stepToward(world, index, snapX[ti]!, snapY[ti]!, topSpeedOf(world, index, def), def.turnPerTick);
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
    const dx = snapX[ti]! - pool.posX[index]!;
    const dy = snapY[ti]! - pool.posY[index]!;
    if (vecLenSqRaw(dx, dy) <= sqRange(reach)) {
      pool.clearPath(index);
      return;
    }
  }

  // Phased on the unit's serial rather than its slot: a mirrored pair share a
  // serial and so re-path on the same ticks, where slots put them up to nine
  // ticks apart, one of them chasing a stale position for longer every cycle.
  const stale =
    pool.pathLen[index] === 0 || (world.tick + pool.serial[index]!) % CHASE_REPATH_INTERVAL === 0;
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
 * Every pair is measured against where both units stood when the pass began,
 * and every push is summed before any is applied. Applied as it went, a push
 * moved the lower-slot unit before its neighbours were measured, and since
 * one side's units hold the lower slots the two halves of a mirrored crowd
 * relaxed in different orders and settled in different places. Summing first
 * makes the result the same whatever order the pairs are visited in — which is
 * also what lets `SEPARATION_STRENGTH` mean one thing for every unit in a jam.
 */
const sepX = new Int32Array(ENTITY_CAPACITY);
const sepY = new Int32Array(ENTITY_CAPACITY);
const pushX = new Int32Array(ENTITY_CAPACITY);
const pushY = new Int32Array(ENTITY_CAPACITY);

function separate(world: World): void {
  const pool = world.pool;
  const grid = world.grid;
  const count = pool.count;
  sepX.set(pool.posX.subarray(0, count));
  sepY.set(pool.posY.subarray(0, count));
  pushX.fill(0, 0, count);
  pushY.fill(0, 0, count);

  for (let i = 0; i < count; i++) {
    if (pool.alive[i] !== 1) continue;
    const defI = defOf(pool.type[i]! as EntityType);
    if (defI.isBuilding) continue;
    // Workers pass through everything: neither pushed nor pushing. Flyers do
    // collide, but only with each other — `collides` is false for them because
    // it also decides whether a thing occupies map tiles, and nothing in the
    // air should block the ground.
    if (!defI.collides && !defI.flying) continue;

    const px = sepX[i]!;
    const py = sepY[i]!;
    const ri = defI.radius;

    grid.forEachNear(px, py, fromInt(2), (j) => {
      if (j <= i) return; // handle each pair once
      if (pool.alive[j] !== 1) return;
      const defJ = defOf(pool.type[j]! as EntityType);
      if (defJ.isBuilding) return;
      // Same layer or no interaction. Air and ground share the map but not the
      // space, so a Beamdrone never shoulders a Slicebot aside.
      if (defI.flying !== defJ.flying) return;
      if (!defJ.collides && !defJ.flying) return;

      const dx = sepX[j]! - px;
      const dy = sepY[j]! - py;
      const minDist = ri + defJ.radius;
      const distSq = vecLenSqRaw(dx, dy);
      if (distSq >= sqRange(minDist)) return;

      if (distSq === 0) {
        // Exactly coincident: nudge apart along the x axis of the canonical
        // frame of whichever unit was created first, so the pair's mirror
        // image is pushed the mirrored way. Index parity chose the axis
        // before, and slot parity is unrelated between two halves of a match.
        const first = createdBefore(world, i, j) ? i : j;
        const sign = world.flipOf(pool.owner[first]!) ? -1 : 1;
        const nudge = first === i ? sign * SEPARATION_STRENGTH : -sign * SEPARATION_STRENGTH;
        pushX[i] = (pushX[i]! - nudge) | 0;
        pushX[j] = (pushX[j]! + nudge) | 0;
        return;
      }

      const dist = Math.sqrt(distSq) | 0;
      const overlap = minDist - dist;
      const push = fmul(overlap, SEPARATION_STRENGTH);
      const inv = fdiv(push, dist);
      const ox = fmul(dx, inv);
      const oy = fmul(dy, inv);

      pushX[i] = (pushX[i]! - ox) | 0;
      pushY[i] = (pushY[i]! - oy) | 0;
      pushX[j] = (pushX[j]! + ox) | 0;
      pushY[j] = (pushY[j]! + oy) | 0;
    });
  }

  for (let i = 0; i < count; i++) {
    if (pool.alive[i] !== 1) continue;
    const ox = pushX[i]!;
    const oy = pushY[i]!;
    if (ox === 0 && oy === 0) continue;
    nudgeBy(world, i, ox, oy, defOf(pool.type[i]! as EntityType));
  }

  clampToMap(world);
}

/**
 * Was `a` created before `b`, in the order both halves of a match share?
 *
 * Seat within the half, then creation ordinal, then the slot itself for two
 * units that agree on both — which can only be a unit and its own mirror
 * image, and any strict order will do for those.
 */
function createdBefore(world: World, a: number, b: number): boolean {
  const pool = world.pool;
  const sa = world.ownerCanonical(pool.owner[a]!);
  const sb = world.ownerCanonical(pool.owner[b]!);
  if (sa !== sb) return sa < sb;
  const ra = pool.serial[a]!;
  const rb = pool.serial[b]!;
  if (ra !== rb) return ra < rb;
  return pool.owner[a]! < pool.owner[b]!;
}

/**
 * Shove a unit, but never into terrain it cannot occupy.
 *
 * Separation used to write the offset straight onto the position, with no idea
 * that walls exist. A crowd fighting against a cliff pushed its outer members
 * into the rock, and `clampToMap` then teleported each of them to the middle of
 * the nearest open tile — a jump of up to a tile and a half, every tick, for as
 * long as the crowd lasted. That is the snapping-between-two-positions players
 * report, and it accounted for 22 of the 25 relocations in a bot match.
 *
 * The axes are tried separately so a unit shoved into a wall slides along it
 * rather than sticking, which is what makes a jam clear itself.
 */
function nudgeBy(world: World, index: number, ox: Fix, oy: Fix, def: EntityDef): void {
  const pool = world.pool;
  // Flyers are over the terrain, not on it.
  if (def.flying) {
    pool.posX[index] = (pool.posX[index]! + ox) | 0;
    pool.posY[index] = (pool.posY[index]! + oy) | 0;
    return;
  }

  const x = pool.posX[index]!;
  const y = pool.posY[index]!;
  const flip = world.flipOf(pool.owner[index]!);
  if (standable(world, (x + ox) | 0, (y + oy) | 0, flip)) {
    pool.posX[index] = (x + ox) | 0;
    pool.posY[index] = (y + oy) | 0;
    return;
  }
  if (ox !== 0 && standable(world, (x + ox) | 0, y, flip)) {
    pool.posX[index] = (x + ox) | 0;
    return;
  }
  if (oy !== 0 && standable(world, x, (y + oy) | 0, flip)) {
    pool.posY[index] = (y + oy) | 0;
  }
}

function standable(world: World, x: number, y: number, flip: boolean): boolean {
  const tile = world.map.tileOfPosFor(x, y, flip);
  if (tile < 0) return false;
  return world.map.isWalkable(world.map.tileXOf(tile), world.map.tileYOf(tile));
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
  // An interval that is its own rotation: `lo` maps to `hi` under
  // `x -> width - x`. `[0, width - 1]` was not, so a unit pinned against one
  // wall sat one unit further out than its mirror pinned against the other.
  const lo = 1;
  const hiX = fromInt(world.map.width) - 1;
  const hiY = fromInt(world.map.height) - 1;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const def = defOf(pool.type[i]! as EntityType);
    if (def.isBuilding) continue;

    if (pool.posX[i]! < lo) pool.posX[i] = lo;
    if (pool.posY[i]! < lo) pool.posY[i] = lo;
    if (pool.posX[i]! > hiX) pool.posX[i] = hiX;
    if (pool.posY[i]! > hiY) pool.posY[i] = hiY;

    // Flyers are over the terrain, not on it, so nothing ejects them.
    if (def.flying) continue;

    const flip = world.flipOf(pool.owner[i]!);
    const tile = world.map.tileOfPosFor(pool.posX[i]!, pool.posY[i]!, flip);
    if (tile < 0) continue;
    const tx = world.map.tileXOf(tile);
    const ty = world.map.tileYOf(tile);
    if (world.map.isWalkable(tx, ty)) continue;

    const free = nearestWalkable(world.map, tx, ty, 6, flip);
    if (free < 0) continue;
    pool.posX[i] = fromInt(world.map.tileXOf(free)) + FIX_HALF;
    pool.posY[i] = fromInt(world.map.tileYOf(free)) + FIX_HALF;
  }
}
