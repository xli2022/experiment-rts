/**
 * Economy: harvesting, construction, and unit production.
 *
 * These three share a file because they are the same shape — a worker or
 * building accumulating ticks toward a state change — and because they all
 * mutate player mineral and supply totals, which is easier to reason about in
 * one place.
 */

import {
  BUILD_REACH,
  defOf,
  HARVEST_REACH,
  HARVEST_TICKS,
  MINERALS_PER_TRIP,
  REPAIR_HP_PER_TICK,
} from '../../config/rules.js';
import { idIndex } from '../entities.js';
import { fromInt, sqRange, vecLenSqRaw } from '../fixed.js';
import { BuildState, EntityType, NO_ENTITY, Order, type PlayerId } from '../types.js';
import type { World } from '../world.js';

export function economySystem(world: World): void {
  harvestSystem(world);
  constructionSystem(world);
  productionSystem(world);
}

// ---------------------------------------------------------------------------
// Harvesting
// ---------------------------------------------------------------------------

function harvestSystem(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.type[i] !== EntityType.Worker) continue;
    if (pool.order[i] !== Order.Harvest) continue;

    if (pool.carrying[i]! > 0) {
      deliverLoad(world, i);
    } else {
      gatherFromPatch(world, i);
    }
  }
}

function gatherFromPatch(world: World, i: number): void {
  const pool = world.pool;
  const patchId = pool.harvestPatch[i]!;

  if (patchId === NO_ENTITY || !pool.isAlive(patchId)) {
    // Patch exhausted while we walked to it — find another rather than idling,
    // which is what a player would do and keeps bots from stalling.
    const next = nearestPatch(world, i);
    if (next < 0) {
      pool.order[i] = Order.None;
      pool.orderTarget[i] = NO_ENTITY;
      return;
    }
    pool.harvestPatch[i] = pool.idAt(next);
    pool.orderTarget[i] = pool.idAt(next);
    return;
  }

  const pi = idIndex(patchId);
  pool.orderTarget[i] = patchId;

  if (!inReach(world, i, pi, HARVEST_REACH)) {
    pool.harvestTimer[i] = 0;
    return;
  }

  pool.harvestTimer[i]! += 1;
  if (pool.harvestTimer[i]! < HARVEST_TICKS) return;

  const available = pool.resourceAmount[pi]!;
  const taken = available < MINERALS_PER_TRIP ? available : MINERALS_PER_TRIP;
  pool.resourceAmount[pi] = available - taken;
  pool.carrying[i] = taken;
  pool.harvestTimer[i] = 0;

  if (pool.resourceAmount[pi]! <= 0) {
    // Exhausted patches vanish, freeing their tiles.
    world.events.deaths.push(pi);
  }
}

function deliverLoad(world: World, i: number): void {
  const pool = world.pool;
  const owner = pool.owner[i]! as PlayerId;
  const depot = nearestDropoff(world, i, owner);

  if (depot < 0) {
    // Nowhere to deliver; hold the load until a Command Post exists again.
    return;
  }

  pool.orderTarget[i] = pool.idAt(depot);

  if (!inReach(world, i, depot, HARVEST_REACH)) return;

  world.player(owner).minerals += pool.carrying[i]!;
  pool.carrying[i] = 0;
  // Head back to the remembered patch; `gatherFromPatch` re-targets if it is gone.
  pool.orderTarget[i] = pool.harvestPatch[i]!;
}

/** Nearest completed Command Post owned by `player`. Ties break by index. */
function nearestDropoff(world: World, i: number, player: PlayerId): number {
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let j = 0; j < pool.count; j++) {
    if (pool.alive[j] !== 1) continue;
    if (pool.owner[j] !== player) continue;
    if (pool.type[j] !== EntityType.CommandPost) continue;
    if (pool.buildState[j] !== BuildState.Complete) continue;
    const dx = pool.posX[j]! - pool.posX[i]!;
    const dy = pool.posY[j]! - pool.posY[i]!;
    const d = vecLenSqRaw(dx, dy);
    if (d < bestDist) {
      bestDist = d;
      best = j;
    }
  }
  return best;
}

/** Nearest mineral patch with minerals left. Ties break by index. */
function nearestPatch(world: World, i: number): number {
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let j = 0; j < pool.count; j++) {
    if (pool.alive[j] !== 1) continue;
    if (pool.type[j] !== EntityType.MineralPatch) continue;
    if (pool.resourceAmount[j]! <= 0) continue;
    const dx = pool.posX[j]! - pool.posX[i]!;
    const dy = pool.posY[j]! - pool.posY[i]!;
    const d = vecLenSqRaw(dx, dy);
    if (d < bestDist) {
      bestDist = d;
      best = j;
    }
  }
  return best;
}

function inReach(world: World, a: number, b: number, extra: number): boolean {
  const pool = world.pool;
  const defA = defOf(pool.type[a]! as EntityType);
  const defB = defOf(pool.type[b]! as EntityType);
  const dx = pool.posX[b]! - pool.posX[a]!;
  const dy = pool.posY[b]! - pool.posY[a]!;
  const reach = defA.radius + defB.radius + extra;
  return vecLenSqRaw(dx, dy) <= sqRange(reach);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

function constructionSystem(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.type[i] !== EntityType.Worker) continue;
    if (pool.order[i] !== Order.Build) continue;

    const siteId = pool.orderTarget[i]!;
    if (siteId === NO_ENTITY || !pool.isAlive(siteId)) {
      pool.order[i] = Order.None;
      pool.orderTarget[i] = NO_ENTITY;
      continue;
    }

    const si = idIndex(siteId);
    const def = defOf(pool.type[si]! as EntityType);
    const finished = pool.buildState[si] === BuildState.Complete;

    // A finished building at full health needs nothing; release the worker so it
    // does not stand there indefinitely.
    if (finished && pool.hp[si]! >= def.maxHp) {
      pool.order[i] = Order.None;
      pool.orderTarget[i] = NO_ENTITY;
      continue;
    }

    if (!inReach(world, i, si, BUILD_REACH)) continue;

    // The worker is on site; stop moving and work.
    pool.clearPath(i);

    // Repair: same order, same worker, different job depending on whether the
    // structure is unfinished or merely hurt.
    if (finished) {
      pool.hp[si] = Math.min(def.maxHp, pool.hp[si]! + REPAIR_HP_PER_TICK);
      if (pool.hp[si]! >= def.maxHp) {
        pool.order[i] = Order.None;
        pool.orderTarget[i] = NO_ENTITY;
      }
      continue;
    }

    pool.buildState[si] = BuildState.UnderConstruction;
    pool.buildProgress[si]! += 1;

    // Health climbs with progress, so a half-built structure is genuinely
    // fragile and worth defending.
    const frac = pool.buildProgress[si]! / def.buildTicks;
    const target = Math.max(1, Math.round(def.maxHp * (0.1 + 0.9 * Math.min(1, frac))));
    if (target > pool.hp[si]!) pool.hp[si] = target;

    if (pool.buildProgress[si]! >= def.buildTicks) {
      pool.buildState[si] = BuildState.Complete;
      pool.hp[si] = def.maxHp;
      pool.order[i] = Order.None;
      pool.orderTarget[i] = NO_ENTITY;
      world.events.completed.push(si);
      world.recomputeSupply();
    }
  }
}

// ---------------------------------------------------------------------------
// Production
// ---------------------------------------------------------------------------

function productionSystem(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.prodCount[i]! === 0) continue;
    if (pool.buildState[i] !== BuildState.Complete) continue;

    const owner = pool.owner[i]! as PlayerId;
    const unitType = pool.prodAt(i, 0);
    const unitDef = defOf(unitType);

    pool.prodProgress[i]! += 1;
    if (pool.prodProgress[i]! < unitDef.buildTicks) continue;

    // Finished, but supply may have run out while it was training. Hold the
    // completed unit in the queue rather than dropping it — the player gets it
    // the moment a depot finishes, which is the behaviour the genre expects.
    const ps = world.player(owner);
    if (ps.supplyUsed + unitDef.supplyCost > ps.supplyMax) {
      pool.prodProgress[i] = unitDef.buildTicks;
      continue;
    }

    const spawn = spawnPointFor(world, i);
    const id = pool.spawn(unitType, owner, spawn.x, spawn.y);
    if (id === NO_ENTITY) continue; // pool full; retry next tick
    // Facing out of the building it came from, so a unit does not have to turn
    // around before it can walk anywhere — and so it mirrors with the spawn.
    pool.faceY[id & 0xffff] = fromInt(spawn.faceY);

    pool.prodRemove(i, 0);
    pool.prodProgress[i] = 0;
    world.recomputeSupply();
  }
}

/**
 * Where a newly trained unit appears, and which way it faces.
 *
 * Clear of the building's footprint on the side facing the middle of the map,
 * nudged by the tick so a batch does not stack perfectly on one point. Uses only
 * integer state, no RNG, so it is reproducible.
 *
 * The side matters. Always spawning on the +Y side put one player's units four
 * tiles nearer the front every time they trained and the other player's four
 * tiles further — on a rotationally symmetric map that is a standing advantage
 * to whoever happens to be in the top-left corner. Choosing the side by which
 * half of the map the building sits in makes the rule rotate with everything
 * else.
 */
const spawnOut = { x: 0, y: 0, faceY: 0 };
function spawnPointFor(world: World, buildingIndex: number): typeof spawnOut {
  const pool = world.pool;
  const def = defOf(pool.type[buildingIndex]! as EntityType);
  const spread = (world.tick + buildingIndex) % 5;
  const towardCentre = pool.posY[buildingIndex]! < fromInt(world.map.height >> 1) ? 1 : -1;
  spawnOut.x = pool.posX[buildingIndex]! + fromInt((spread - 2) * towardCentre);
  spawnOut.y = pool.posY[buildingIndex]! + fromInt(def.footprint * towardCentre);
  spawnOut.faceY = towardCentre;
  return spawnOut;
}
