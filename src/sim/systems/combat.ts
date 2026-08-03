/**
 * Combat: target acquisition, firing, and death.
 *
 * Target selection is the classic determinism trap. "Attack the nearest enemy"
 * is ambiguous whenever two enemies are equidistant, and on a grid that happens
 * constantly. The comparator below is a strict total order — distance first,
 * then entity slot index — so there is never a tie to break arbitrarily.
 */

import { counterBonusPct, defOf } from '../../config/rules.js';
import { idIndex } from '../entities.js';
import { fromInt, sqRange, vecLenSqRaw, vecNormalize } from '../fixed.js';
import { BuildState, EntityType, NEUTRAL, NO_ENTITY, Order } from '../types.js';
import type { World } from '../world.js';

export function combatSystem(world: World): void {
  const pool = world.pool;

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;

    if (pool.attackCooldown[i]! > 0) pool.attackCooldown[i]! -= 1;

    const type = pool.type[i]! as EntityType;
    const def = defOf(type);
    if (def.attackRange === 0) continue;
    if (pool.owner[i] === NEUTRAL) continue;
    // Unfinished buildings cannot shoot.
    if (def.isBuilding && pool.buildState[i] !== BuildState.Complete) continue;

    const order = pool.order[i]!;

    // An explicit attack order pins the target; anything else acquires freely.
    let targetIndex = -1;
    if (order === Order.Attack) {
      const t = pool.orderTarget[i]!;
      if (pool.isAlive(t)) {
        targetIndex = idIndex(t);
      } else {
        // Target died: fall back to free acquisition rather than standing idle.
        pool.order[i] = Order.None;
        pool.orderTarget[i] = NO_ENTITY;
      }
    }

    if (targetIndex < 0) {
      // Units on a plain Move order do not stop to fight; that is what
      // attack-move is for.
      if (order === Order.Move || order === Order.Harvest || order === Order.Build) {
        pool.combatTarget[i] = NO_ENTITY;
        continue;
      }
      targetIndex = acquireTarget(world, i, def.sightRange);
    }

    if (targetIndex < 0) {
      pool.combatTarget[i] = NO_ENTITY;
      continue;
    }

    pool.combatTarget[i] = pool.idAt(targetIndex);

    const dx = pool.posX[targetIndex]! - pool.posX[i]!;
    const dy = pool.posY[targetIndex]! - pool.posY[i]!;
    const targetDef = defOf(pool.type[targetIndex]! as EntityType);
    // Measure to the target's edge, so large buildings are hittable from where
    // they visually begin rather than from their centre.
    const reach = def.attackRange + targetDef.radius;
    const inRange = vecLenSqRaw(dx, dy) <= sqRange(reach);

    if (!inRange) continue;

    // Face the target while firing. Turrets rotate; ground units snap since
    // they are already steering toward it.
    const dir = vecNormalize(dx, dy);
    pool.faceX[i] = dir.x;
    pool.faceY[i] = dir.y;

    // Standing in range with an attack-move order means stop and shoot.
    if (order === Order.AttackMove || order === Order.Attack) pool.clearPath(i);

    if (pool.attackCooldown[i]! > 0) continue;

    pool.attackCooldown[i] = def.attackCooldown;
    // Integer maths so the counter bonus stays exact across engines.
    const bonus = counterBonusPct(type, pool.type[targetIndex]! as EntityType);
    const damage = ((def.damage * bonus) / 100) | 0;
    applyDamage(world, targetIndex, damage);
    world.events.shots.push(i, targetIndex);
  }
}

/**
 * Find the best hostile target within `range`.
 *
 * Returns a slot index, or -1. The comparator is (distance, index): strictly
 * ordered, so two peers always choose the same unit even in a perfectly
 * symmetric engagement — which, on a mirror map, is exactly the situation that
 * arises constantly.
 */
function acquireTarget(world: World, index: number, range: number): number {
  const pool = world.pool;
  const owner = pool.owner[index]!;
  const px = pool.posX[index]!;
  const py = pool.posY[index]!;

  let bestIndex = -1;
  let bestDistSq = Number.POSITIVE_INFINITY;

  world.grid.forEachNear(px, py, range, (j) => {
    if (j === index) return;
    if (pool.alive[j] !== 1) return;
    if (!pool.isHostile(j, owner)) return;
    // Mineral patches are neutral, so `isHostile` already excludes them.

    const dx = pool.posX[j]! - px;
    const dy = pool.posY[j]! - py;
    const distSq = vecLenSqRaw(dx, dy);
    if (distSq > sqRange(range)) return;

    if (distSq < bestDistSq || (distSq === bestDistSq && j < bestIndex)) {
      bestDistSq = distSq;
      bestIndex = j;
    }
  });

  return bestIndex;
}

/** Apply damage and queue a death if it drops to zero. */
export function applyDamage(world: World, index: number, amount: number): void {
  const pool = world.pool;
  if (pool.alive[index] !== 1) return;
  pool.hp[index]! -= amount;
  if (pool.hp[index]! <= 0) {
    pool.hp[index] = 0;
    world.events.deaths.push(index);
  }
}

/**
 * Remove everything that died this tick.
 *
 * Deaths are collected during the tick and applied here, after every system has
 * run, so that no system observes a half-removed entity. The deaths list is
 * built in ascending index order by the combat loop, keeping the destroy order —
 * and therefore the free list, and therefore all future entity ids — identical
 * across peers.
 */
export function reapDead(world: World): void {
  const pool = world.pool;
  const deaths = world.events.deaths;
  if (deaths.length === 0) return;

  for (let k = 0; k < deaths.length; k++) {
    const i = deaths[k]!;
    if (pool.alive[i] !== 1) continue;

    const type = pool.type[i]! as EntityType;
    const def = defOf(type);

    // Free the tiles a building occupied so the ground becomes buildable again.
    if (def.isBuilding) {
      world.map.setOccupied(pool.tileX[i]!, pool.tileY[i]!, def.footprint, 0);
    }

    // Workers carrying minerals simply lose the load.
    pool.destroy(pool.idAt(i));
  }

  // Clear stale references so nothing chases a recycled slot. Generation
  // tagging already makes this safe, but zeroing keeps the checksum tidy and
  // avoids re-validating dead handles every tick.
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const t = pool.orderTarget[i]!;
    if (t !== NO_ENTITY && !pool.isAlive(t)) {
      pool.orderTarget[i] = NO_ENTITY;
      if (pool.order[i] === Order.Attack) pool.order[i] = Order.None;
    }
    const c = pool.combatTarget[i]!;
    if (c !== NO_ENTITY && !pool.isAlive(c)) pool.combatTarget[i] = NO_ENTITY;
    const hp = pool.harvestPatch[i]!;
    if (hp !== NO_ENTITY && !pool.isAlive(hp)) pool.harvestPatch[i] = NO_ENTITY;
  }

  world.recomputeSupply();
}

/** Distance helper shared with the AI, which reasons about threat ranges. */
export function withinRange(
  world: World,
  a: number,
  b: number,
  range: number,
): boolean {
  const pool = world.pool;
  const dx = pool.posX[b]! - pool.posX[a]!;
  const dy = pool.posY[b]! - pool.posY[a]!;
  return vecLenSqRaw(dx, dy) <= sqRange(range + fromInt(0));
}
