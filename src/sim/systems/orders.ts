/**
 * Command execution: turning player intent into entity orders.
 *
 * This is the trust boundary. Commands arrive from the local player, from the
 * AI, and from the network, and all three go through here. Every rule that
 * decides whether an action is legal — ownership, cost, placement, queue depth —
 * is enforced in this file and nowhere else.
 *
 * That is not defensive programming against cheaters (in peer-to-peer lockstep
 * there is no authority, and a modified client can already see the whole map).
 * It is about determinism: if the UI decided that a build was affordable and the
 * simulation disagreed, the two peers would apply different state changes. The
 * UI may grey a button out, but only this file decides what actually happens.
 */

import { defOf, GROUP_PATH_THRESHOLD, MAX_PRODUCTION_QUEUE } from '../../config/rules.js';
import { CommandType, type Command } from '../commands.js';
import { idIndex } from '../entities.js';
import { fromInt } from '../fixed.js';
import { BuildState, EntityType, NO_ENTITY, Order, type EntityId, type PlayerId } from '../types.js';
import type { World } from '../world.js';

/**
 * Apply one command. Illegal commands are silently dropped — identically on
 * every peer, which is what matters.
 */
export function executeCommand(world: World, cmd: Command): void {
  const player = cmd.player;
  if (player < 0 || player >= world.players.length) return;
  if (world.player(player).defeated) return;

  switch (cmd.type) {
    case CommandType.Move: {
      const grouped = cmd.units.length >= GROUP_PATH_THRESHOLD;
      let slot = 0;
      forEachOwned(world, cmd.units, player, (i) => {
        // The slot counts units the order actually applied to, in the order
        // `forEachOwned` walks them — identical on every peer.
        spreadDestination(world, cmd.units.length > 1 ? slot++ : 0, cmd.x, cmd.y, destOut);
        setMoveOrder(world, i, Order.Move, destOut.x, destOut.y, grouped, cmd.x, cmd.y);
      });
      break;
    }

    case CommandType.AttackMove: {
      const grouped = cmd.units.length >= GROUP_PATH_THRESHOLD;
      let slot = 0;
      forEachOwned(world, cmd.units, player, (i) => {
        spreadDestination(world, cmd.units.length > 1 ? slot++ : 0, cmd.x, cmd.y, destOut);
        setMoveOrder(world, i, Order.AttackMove, destOut.x, destOut.y, grouped, cmd.x, cmd.y);
      });
      break;
    }

    case CommandType.Attack: {
      const target = cmd.target;
      if (!world.pool.isAlive(target)) break;
      const ti = idIndex(target);
      const targetFlies = defOf(world.pool.type[ti]! as EntityType).flying;
      forEachOwned(world, cmd.units, player, (i) => {
        // Attacking your own units is not a thing; ignore rather than misfire.
        if (world.pool.owner[ti] === player) return;
        const attacker = defOf(world.pool.type[i]! as EntityType);
        if (attacker.attackRange === 0) return;
        // Dropped per unit rather than for the whole order, so a mixed
        // selection still sends its riflemen while the brawlers stay put
        // instead of jogging after something they cannot reach.
        if (targetFlies && !attacker.canHitAir) return;
        world.pool.order[i] = Order.Attack;
        world.pool.orderTarget[i] = target;
        world.pool.clearPath(i);
      });
      break;
    }

    case CommandType.Harvest: {
      const target = cmd.target;
      if (!world.pool.isAlive(target)) break;
      const ti = idIndex(target);
      if (world.pool.type[ti] !== EntityType.MineralPatch) break;
      forEachOwned(world, cmd.units, player, (i) => {
        if (world.pool.type[i] !== EntityType.Worker) return;
        world.pool.order[i] = Order.Harvest;
        world.pool.orderTarget[i] = target;
        world.pool.harvestPatch[i] = target;
        world.pool.harvestTimer[i] = 0;
        world.pool.clearPath(i);
      });
      break;
    }

    case CommandType.Build:
      executeBuild(world, cmd.worker, cmd.building, cmd.tileX, cmd.tileY, player);
      break;

    case CommandType.Stop:
      forEachOwned(world, cmd.units, player, (i) => {
        world.pool.order[i] = Order.None;
        world.pool.orderTarget[i] = NO_ENTITY;
        world.pool.combatTarget[i] = NO_ENTITY;
        world.pool.clearPath(i);
      });
      break;

    case CommandType.Hold:
      forEachOwned(world, cmd.units, player, (i) => {
        world.pool.order[i] = Order.Hold;
        world.pool.orderTarget[i] = NO_ENTITY;
        world.pool.clearPath(i);
      });
      break;

    case CommandType.Train:
      executeTrain(world, cmd.building, cmd.unit, player);
      break;

    case CommandType.CancelTrain:
      executeCancelTrain(world, cmd.building, cmd.slot, player);
      break;

    case CommandType.SetRally: {
      // Only your own finished production building, and only somewhere on the
      // map. Validated here rather than in the UI like every other command.
      const b = cmd.building;
      if (!world.pool.isAlive(b)) break;
      const bi = idIndex(b);
      if (world.pool.owner[bi] !== player) break;
      const bDef = defOf(world.pool.type[bi]! as EntityType);
      if (bDef.produces.length === 0) break;
      if (world.map.tileOfPos(cmd.x, cmd.y) < 0) break;
      world.pool.rallyX[bi] = cmd.x;
      world.pool.rallyY[bi] = cmd.y;
      world.pool.hasRally[bi] = 1;
      break;
    }

    case CommandType.Surrender:
      world.player(player).defeated = true;
      break;
  }
}

/**
 * Run `fn` for each live entity in `units` that the issuing player actually
 * owns, in the order given.
 *
 * The ownership check is essential: a malformed or malicious packet could name
 * someone else's units, and without this every peer would happily move them.
 */
function forEachOwned(
  world: World,
  units: readonly EntityId[],
  player: PlayerId,
  fn: (index: number) => void,
): void {
  const pool = world.pool;
  for (let k = 0; k < units.length; k++) {
    const id = units[k]!;
    if (!pool.isAlive(id)) continue;
    const i = idIndex(id);
    if (pool.owner[i] !== player) continue;
    // Buildings take production orders, not movement orders.
    if (defOf(pool.type[i]! as EntityType).isBuilding) continue;
    fn(i);
  }
}

/**
 * Where the `slot`-th unit of a group should actually stand.
 *
 * A group move used to send every unit to the same point, so they arrived in a
 * heap and then spent a second shoving each other back out of it — separation
 * doing the work a destination should have done. Spreading the destinations
 * means they arrive in a loose formation and stop.
 *
 * The layout walks a square spiral, which looks less natural than a ring and is
 * chosen anyway: a ring needs sine and cosine, and those are the functions that
 * genuinely differ between JavaScript engines. Integer offsets cannot desync.
 * Once units settle, separation rounds the corners off the square regardless.
 */
function spreadSlot(slot: number, out: { x: number; y: number }): void {
  let x = 0;
  let y = 0;
  let dx = 0;
  let dy = -1;
  for (let i = 0; i < slot; i++) {
    if (x === y || (x < 0 && x === -y) || (x > 0 && x === 1 - y)) {
      const swap = dx;
      dx = -dy;
      dy = swap;
    }
    x += dx;
    y += dy;
  }
  out.x = x;
  out.y = y;
}

/** Scratch for `spreadSlot`; the sim is single-threaded and allocates nothing. */
const slotOut = { x: 0, y: 0 };

/** Gap between neighbouring units in a formation. Wider than any unit's girth. */
const SPREAD_STEP = fromInt(1);

/**
 * Offset a group's destination so its members do not all target one tile.
 *
 * Falls back to the raw target whenever the spread tile is not somewhere a unit
 * could stand, which keeps a formation ordered against a cliff edge from
 * scattering its flank into the rock.
 */
function spreadDestination(
  world: World,
  slot: number,
  x: number,
  y: number,
  out: { x: number; y: number },
): void {
  out.x = x;
  out.y = y;
  if (slot <= 0) return;
  spreadSlot(slot, slotOut);
  const sx = (x + slotOut.x * SPREAD_STEP) | 0;
  const sy = (y + slotOut.y * SPREAD_STEP) | 0;
  if (world.map.tileOfPos(sx, sy) < 0) return;
  out.x = sx;
  out.y = sy;
}

const destOut = { x: 0, y: 0 };

/**
 * Point a unit at a destination.
 *
 * `grouped` selects the navigation strategy: a shared flow field when many units
 * were named in one order, or a private A* path when it is an individual errand.
 * See `GROUP_PATH_THRESHOLD` for why the split pays off.
 */
function setMoveOrder(
  world: World,
  index: number,
  order: Order,
  x: number,
  y: number,
  grouped: boolean,
  goalX: number = x,
  goalY: number = y,
): void {
  const pool = world.pool;
  pool.order[index] = order;
  pool.orderX[index] = x;
  pool.orderY[index] = y;
  pool.orderTarget[index] = NO_ENTITY;
  pool.combatTarget[index] = NO_ENTITY;
  pool.clearPath(index);

  // Flyers steer straight there; no path or field is ever built for them.
  if (defOf(pool.type[index]! as EntityType).flying) return;

  if (grouped) {
    // The *shared* destination, not this unit's spread arrival point. One
    // Dijkstra sweep serves the whole group, and that is the entire reason
    // grouped moves exist — giving each unit its own goal tile silently turns
    // one sweep into one per unit, which took the test match from 5s to over
    // 150s. Arrival is still measured against the spread point below, so the
    // formation survives.
    const goal = world.map.tileOfPos(goalX, goalY);
    pool.flowGoal[index] = goal >= 0 ? goal : -1;
    return;
  }

  // Request a path; movement system serves the queue within its tick budget.
  pool.pathPending[index] = 1;
  world.pathQueue.push(index);
}

function executeBuild(
  world: World,
  workerId: EntityId,
  building: EntityType,
  tileX: number,
  tileY: number,
  player: PlayerId,
): void {
  const pool = world.pool;
  if (!pool.isAlive(workerId)) return;
  const wi = idIndex(workerId);
  if (pool.owner[wi] !== player) return;
  if (pool.type[wi] !== EntityType.Worker) return;

  const def = defOf(building);
  if (!def.isBuilding || building === EntityType.MineralPatch) return;

  // Issuing a build order onto one of our own unfinished structures means
  // "finish that". Without it the order would be rejected by the placement
  // check below, which is what let abandoned sites sit half-built forever with
  // nobody assigned to them.
  const existing = findWorkableAt(world, player, tileX, tileY);
  if (existing !== NO_ENTITY) {
    assignBuilder(world, wi, existing);
    return;
  }

  const ps = world.player(player);
  if (ps.minerals < def.mineralCost) return;
  if (!world.map.canPlace(tileX, tileY, def.footprint)) return;

  // Charge immediately, like StarCraft: the cost is committed when the site is
  // placed, and refunded only if the site is cancelled or destroyed unfinished.
  ps.minerals -= def.mineralCost;

  const siteId = world.placeBuilding(building, player, tileX, tileY);
  if (siteId === NO_ENTITY) {
    ps.minerals += def.mineralCost; // pool exhausted; undo cleanly
    return;
  }
  const si = idIndex(siteId);
  pool.buildState[si] = 0; // Site
  pool.buildProgress[si] = 0;
  pool.hp[si] = Math.max(1, (def.maxHp / 10) | 0); // sites start fragile

  assignBuilder(world, wi, siteId);
}

/** Point a worker at a construction site and path it there. */
function assignBuilder(world: World, workerIndex: number, siteId: EntityId): void {
  const pool = world.pool;
  const si = idIndex(siteId);
  pool.order[workerIndex] = Order.Build;
  pool.orderTarget[workerIndex] = siteId;
  pool.orderX[workerIndex] = pool.posX[si]!;
  pool.orderY[workerIndex] = pool.posY[si]!;
  pool.clearPath(workerIndex);
  pool.pathPending[workerIndex] = 1;
  world.pathQueue.push(workerIndex);
}

/**
 * An unfinished building of ours at this tile, for a worker to finish.
 *
 * Damaged-but-complete structures are deliberately not workable: repair used to
 * live here and was removed for being too strong. A finished building is not a
 * job, so ordering a worker onto one does nothing rather than parking it there.
 *
 * Scans the pool in ascending index order, so every peer finds the same one.
 */
function findWorkableAt(
  world: World,
  player: PlayerId,
  tileX: number,
  tileY: number,
): EntityId {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.owner[i] !== player) continue;
    if (pool.tileX[i] !== tileX || pool.tileY[i] !== tileY) continue;
    const def = defOf(pool.type[i]! as EntityType);
    if (!def.isBuilding) continue;

    if (pool.buildState[i] !== BuildState.Complete) return pool.idAt(i);
  }
  return NO_ENTITY;
}

function executeTrain(
  world: World,
  buildingId: EntityId,
  unit: EntityType,
  player: PlayerId,
): void {
  const pool = world.pool;
  if (!pool.isAlive(buildingId)) return;
  const bi = idIndex(buildingId);
  if (pool.owner[bi] !== player) return;
  if (pool.buildState[bi] !== 2) return; // must be finished

  const bDef = defOf(pool.type[bi]! as EntityType);
  if (!bDef.produces.includes(unit)) return;
  if (pool.prodCount[bi]! >= MAX_PRODUCTION_QUEUE) return;

  const uDef = defOf(unit);
  const ps = world.player(player);
  if (ps.minerals < uDef.mineralCost) return;
  // Supply is checked at completion, not queueing, so a player can queue ahead
  // of a depot finishing — matching the genre's expectations.

  ps.minerals -= uDef.mineralCost;
  pool.prodPush(bi, unit);
}

function executeCancelTrain(
  world: World,
  buildingId: EntityId,
  slot: number,
  player: PlayerId,
): void {
  const pool = world.pool;
  if (!pool.isAlive(buildingId)) return;
  const bi = idIndex(buildingId);
  if (pool.owner[bi] !== player) return;
  if (slot < 0 || slot >= pool.prodCount[bi]!) return;

  const type = pool.prodAt(bi, slot);
  world.player(player).minerals += defOf(type).mineralCost; // full refund
  pool.prodRemove(bi, slot);
}
