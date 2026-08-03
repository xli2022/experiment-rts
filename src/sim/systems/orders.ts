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
import { EntityType, NO_ENTITY, Order, type EntityId, type PlayerId } from '../types.js';
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
      forEachOwned(world, cmd.units, player, (i) => {
        setMoveOrder(world, i, Order.Move, cmd.x, cmd.y, grouped);
      });
      break;
    }

    case CommandType.AttackMove: {
      const grouped = cmd.units.length >= GROUP_PATH_THRESHOLD;
      forEachOwned(world, cmd.units, player, (i) => {
        setMoveOrder(world, i, Order.AttackMove, cmd.x, cmd.y, grouped);
      });
      break;
    }

    case CommandType.Attack: {
      const target = cmd.target;
      if (!world.pool.isAlive(target)) break;
      const ti = idIndex(target);
      forEachOwned(world, cmd.units, player, (i) => {
        // Attacking your own units is not a thing; ignore rather than misfire.
        if (world.pool.owner[ti] === player) return;
        if (defOf(world.pool.type[i]! as EntityType).attackRange === 0) return;
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
): void {
  const pool = world.pool;
  pool.order[index] = order;
  pool.orderX[index] = x;
  pool.orderY[index] = y;
  pool.orderTarget[index] = NO_ENTITY;
  pool.combatTarget[index] = NO_ENTITY;
  pool.clearPath(index);

  if (grouped) {
    const goal = world.map.tileOfPos(x, y);
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

  // Issuing a build order onto an existing unfinished site of ours means "go
  // help with that", not "start a second one". Without this the order would be
  // rejected by the placement check below, which is what let abandoned sites sit
  // half-built forever with nobody assigned to them.
  const existing = findSiteAt(world, player, tileX, tileY);
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
 * An unfinished building of ours whose footprint starts at this tile.
 *
 * Scans the pool in ascending index order, so every peer finds the same one.
 */
function findSiteAt(
  world: World,
  player: PlayerId,
  tileX: number,
  tileY: number,
): EntityId {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.owner[i] !== player) continue;
    if (pool.buildState[i] === 2) continue; // already finished
    if (pool.tileX[i] !== tileX || pool.tileY[i] !== tileY) continue;
    if (!defOf(pool.type[i]! as EntityType).isBuilding) continue;
    return pool.idAt(i);
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
