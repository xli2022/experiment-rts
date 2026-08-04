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
} from '../../config/rules.js';
import { idIndex } from '../entities.js';
import { FIX_HALF, fromInt, sqRange, vecLenSqRaw, type Fix } from '../fixed.js';
import { nearestWalkable } from '../pathing/astar.js';
import { standableTarget } from './orders.js';
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

/**
 * Squared distance from a point to the nearest part of entity `b`.
 *
 * Buildings are squares and were measured as the circle inscribed in their own
 * footprint, which is wrong in both directions: it reaches half a tile past the
 * middle of each face and falls short of every corner. On a Command Post, whose
 * footprint is four tiles, that left the diagonals out of range entirely, so a
 * worker arriving on one had to walk around to a face before it could deliver.
 *
 * Round things keep their radius; only footprints get the box treatment.
 */
export function distanceSqTo(world: World, b: number, px: Fix, py: Fix): number {
  const pool = world.pool;
  const def = defOf(pool.type[b]! as EntityType);
  let dx = px - pool.posX[b]!;
  let dy = py - pool.posY[b]!;

  if (def.footprint > 0) {
    // Outside the box on each axis independently; zero on an axis the point
    // already overlaps, which is what makes a face as reachable as a corner.
    const half = fromInt(def.footprint) >> 1;
    dx = dx < 0 ? Math.min(0, dx + half) : Math.max(0, dx - half);
    dy = dy < 0 ? Math.min(0, dy + half) : Math.max(0, dy - half);
    return vecLenSqRaw(dx, dy);
  }
  // Round: pull the radius off the straight-line distance later, as before.
  return vecLenSqRaw(dx, dy);
}

/**
 * The point on `b` nearest to (px, py) — where a unit should actually walk to.
 *
 * For a building this is a spot on the face it is approaching from, so every
 * side is an equally good way in. Pathing at the centre instead sent every unit
 * to whichever single tile `nearestWalkable` happened to pick for the middle of
 * the footprint, which is the long way round from three sides out of four.
 */
export function approachPoint(world: World, b: number, px: Fix, py: Fix, out: { x: Fix; y: Fix }): void {
  const pool = world.pool;
  const def = defOf(pool.type[b]! as EntityType);
  const bx = pool.posX[b]!;
  const by = pool.posY[b]!;
  if (def.footprint <= 0) {
    out.x = bx;
    out.y = by;
    return;
  }
  const half = fromInt(def.footprint) >> 1;
  out.x = px < bx - half ? bx - half : px > bx + half ? bx + half : px;
  out.y = py < by - half ? by - half : py > by + half ? by + half : py;
}

/**
 * Is `a` close enough to `b` to work on it, given that order's slack?
 *
 * Exported because the HUD has to ask the same question to tell a worker still
 * walking to a site apart from one already building it. Answering it a second
 * time in the UI would let the two drift, and a worker reported as building
 * while it is still walking is exactly the misinformation the status line
 * exists to prevent.
 */
export function inReach(world: World, a: number, b: number, extra: number): boolean {
  const pool = world.pool;
  const defA = defOf(pool.type[a]! as EntityType);
  const defB = defOf(pool.type[b]! as EntityType);
  const distSq = distanceSqTo(world, b, pool.posX[a]!, pool.posY[a]!);
  // A footprint has already had its own extent removed by `distanceSqTo`.
  const reach = defA.radius + extra + (defB.footprint > 0 ? 0 : defB.radius);
  return distSq <= sqRange(reach);
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

    // A finished building needs nothing further. Workers used to repair one
    // back to full for free, which made any attack that did not outright kill a
    // structure a waste of time — release the worker instead.
    if (pool.buildState[si] === BuildState.Complete) {
      pool.order[i] = Order.None;
      pool.orderTarget[i] = NO_ENTITY;
      continue;
    }

    if (!inReach(world, i, si, BUILD_REACH)) continue;

    // The worker is on site; stop moving and work.
    pool.clearPath(i);

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
    // Nowhere to put it. Hold the finished unit in the queue exactly as the
    // supply case above does, rather than dropping it on a tile it cannot
    // stand on and leaving the movers to sort it out.
    if (!spawn.ok) {
      pool.prodProgress[i] = unitDef.buildTicks;
      continue;
    }
    const id = pool.spawn(unitType, owner, spawn.x, spawn.y);
    if (id === NO_ENTITY) continue; // pool full; retry next tick
    // Facing out of the building it came from, so a unit does not have to turn
    // around before it can walk anywhere — and so it mirrors with the spawn.
    pool.faceY[id & 0xffff] = fromInt(spawn.faceY);

    // Walk to the rally point if this building has one. Issued as an ordinary
    // move order rather than anything special, so it paths, spreads and can be
    // overridden exactly like a player's own click.
    // Checked again here, not only where the rally was set: a building can be
    // raised on the rally spot afterwards, and then every unit this one trains
    // inherits a destination it can never reach.
    if (pool.hasRally[i] === 1 && standableTarget(world, pool.rallyX[i]!, pool.rallyY[i]!, rallyOut)) {
      const ui = id & 0xffff;
      pool.order[ui] = Order.Move;
      pool.orderX[ui] = rallyOut.x;
      pool.orderY[ui] = rallyOut.y;
      pool.orderTarget[ui] = NO_ENTITY;
      pool.clearPath(ui);
      pool.pathPending[ui] = 1;
      world.pathQueue.push(ui);
    }

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
 *
 * The point is computed geometrically, so on its own it is only a *preference*:
 * it can land in rock, inside a neighbouring building, or off the map edge, and
 * nothing used to check. A unit placed there is standing somewhere it cannot be,
 * and `clampToMap` teleports it to the nearest walkable tile on the next tick —
 * the snap players see. Measured over a bot match, 11 of 89 trained units
 * appeared inside a building footprint and 3 of those on solid tiles.
 *
 * So the preference is checked, and when it does not hold the nearest standable
 * tile is used instead. `ok` is false only when there is no room at all nearby,
 * in which case production waits rather than placing the unit anywhere.
 */
const spawnOut = { x: 0, y: 0, faceY: 0, ok: false };

/** Scratch for the rally check; the simulation allocates nothing per tick. */
const rallyOut = { x: 0, y: 0 };

/** How far from the preferred spot to look for somewhere the unit can stand. */
const SPAWN_SEARCH_RINGS = 6;

function spawnPointFor(world: World, buildingIndex: number): typeof spawnOut {
  const pool = world.pool;
  const map = world.map;
  const def = defOf(pool.type[buildingIndex]! as EntityType);
  const spread = (world.tick + buildingIndex) % 5;
  const towardCentre = pool.posY[buildingIndex]! < fromInt(map.height >> 1) ? 1 : -1;
  const px = pool.posX[buildingIndex]! + fromInt((spread - 2) * towardCentre);
  const py = pool.posY[buildingIndex]! + fromInt(def.footprint * towardCentre);
  spawnOut.faceY = towardCentre;

  const preferred = map.tileOfPos(px, py);
  if (preferred >= 0 && map.isWalkable(map.tileXOf(preferred), map.tileYOf(preferred))) {
    spawnOut.x = px;
    spawnOut.y = py;
    spawnOut.ok = true;
    return spawnOut;
  }

  // Search from the preferred tile when it is on the map at all, and from the
  // building itself when the preference fell off the edge.
  const from = preferred >= 0 ? preferred : map.tileOfPos(pool.posX[buildingIndex]!, pool.posY[buildingIndex]!);
  if (from < 0) {
    spawnOut.ok = false;
    return spawnOut;
  }

  const free = nearestWalkable(map, map.tileXOf(from), map.tileYOf(from), SPAWN_SEARCH_RINGS);
  if (free < 0) {
    spawnOut.ok = false;
    return spawnOut;
  }
  spawnOut.x = fromInt(map.tileXOf(free)) + FIX_HALF;
  spawnOut.y = fromInt(map.tileYOf(free)) + FIX_HALF;
  spawnOut.ok = true;
  return spawnOut;
}
