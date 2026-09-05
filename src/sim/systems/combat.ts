/**
 * Combat: target acquisition, firing, and death.
 *
 * Target selection is the classic determinism trap. "Attack the nearest enemy"
 * is ambiguous whenever two enemies are equidistant, and on a grid that happens
 * constantly. The comparator below is a strict total order — distance first,
 * then the target's seat and creation order — so there is never a tie to break
 * arbitrarily, and two mirrored units break it the same way.
 */

import { defOf } from '../../config/rules.js';
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

    // A started attack owns this unit until its authored impact tick. Locking a
    // generation-tagged target makes the result independent of later target
    // acquisition and prevents a recycled slot from receiving the blow.
    if (pool.attackWindup[i]! > 0) {
      pool.attackWindup[i]! -= 1;
      if (pool.attackWindup[i] === 0) {
        const target = pool.attackTarget[i]!;
        pool.attackTarget[i] = NO_ENTITY;
        resolveAttackImpact(world, i, target);
      }
      continue;
    }

    const order = pool.order[i]!;

    // An explicit attack order pins the target; anything else acquires freely.
    let targetIndex = -1;
    if (order === Order.Attack) {
      const t = pool.orderTarget[i]!;
      // A pinned target is still subject to what the weapon can reach. Orders
      // are refused at source, so this only catches a target that became
      // unreachable afterwards — but it is the difference between a unit that
      // stands there and one that lands impossible blows.
      if (
        pool.isAlive(t) &&
        (def.canHitAir || !defOf(pool.type[idIndex(t)]! as EntityType).flying)
      ) {
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

    // Cooldown begins with the wind-up rather than with impact. That preserves
    // the listed time between consecutive hits: foreswing delays the first hit,
    // but does not silently lower sustained damage by adding to every cycle.
    pool.attackCooldown[i] = def.attackCooldown;
    world.events.attackStarts.push(i, targetIndex);
    if (def.attackForeswing > 0) {
      pool.attackWindup[i] = def.attackForeswing;
      pool.attackTarget[i] = pool.idAt(targetIndex);
      continue;
    }

    resolveAttackImpact(world, i, pool.idAt(targetIndex));
  }
}

/**
 * Resolve the target locked at attack start on the exact impact tick.
 *
 * A target can die, be recycled, take off, or leave reach during foreswing. A
 * stale or now-impossible blow whiffs but still spends its cooldown, matching
 * the animation the renderer already began without granting damage at range.
 */
function resolveAttackImpact(world: World, attackerIndex: number, target: number): void {
  const pool = world.pool;
  // Presentation needs to distinguish a completed swing from an interruption.
  // Emit this before validation so a legitimate whiff still plays follow-through.
  world.events.attackImpacts.push(attackerIndex);
  if (!pool.isAlive(target)) {
    pool.combatTarget[attackerIndex] = NO_ENTITY;
    return;
  }

  const targetIndex = idIndex(target);
  const def = defOf(pool.type[attackerIndex]! as EntityType);
  const targetDef = defOf(pool.type[targetIndex]! as EntityType);
  if (!world.isHostile(targetIndex, pool.owner[attackerIndex]!)) return;
  if (targetDef.flying && !def.canHitAir) return;

  const dx = pool.posX[targetIndex]! - pool.posX[attackerIndex]!;
  const dy = pool.posY[targetIndex]! - pool.posY[attackerIndex]!;
  if (vecLenSqRaw(dx, dy) > sqRange(def.attackRange + targetDef.radius)) return;

  const dir = vecNormalize(dx, dy);
  pool.faceX[attackerIndex] = dir.x;
  pool.faceY[attackerIndex] = dir.y;

  // One number, whatever it is shooting. The HUD shows `def.damage` on the
  // info panel, and a per-matchup multiplier would have made that a lie.
  applyDamage(world, targetIndex, def.damage);
  world.events.shots.push(attackerIndex, targetIndex);
}

/**
 * Find the best hostile target within `range`.
 *
 * Returns a slot index, or -1. The comparator is (distance, seat, serial):
 * strictly ordered, so two peers always choose the same unit even in a
 * perfectly symmetric engagement — which, on a mirror map, is exactly the
 * situation that arises constantly. The second and third keys are the target
 * owner's seat within its half and the target's creation ordinal, both of
 * which a mirrored pair share; a slot index, which they do not, sent the two
 * halves' units at different members of an equidistant pair.
 */
function acquireTarget(world: World, index: number, range: number): number {
  const pool = world.pool;
  const owner = pool.owner[index]!;
  const canHitAir = defOf(pool.type[index]! as EntityType).canHitAir;
  const px = pool.posX[index]!;
  const py = pool.posY[index]!;

  let bestIndex = -1;
  let bestDistSq = Number.POSITIVE_INFINITY;
  let bestKey = 0;

  world.grid.forEachNear(px, py, range, (j) => {
    if (j === index) return;
    if (pool.alive[j] !== 1) return;
    if (!world.isHostile(j, owner)) return;
    // Mineral patches are neutral, so `isHostile` already excludes them, and so
    // is anything a partner owns — a co-op player's units walk through their
    // ally's army without either side taking a shot at it.
    // Something we cannot shoot is not a target, and must not be picked as one:
    // acquisition is what `engageNearby` walks toward, so a melee unit that
    // acquired a Beamdrone would trail after it without ever landing a blow.
    if (!canHitAir && defOf(pool.type[j]! as EntityType).flying) return;

    const dx = pool.posX[j]! - px;
    const dy = pool.posY[j]! - py;
    const distSq = vecLenSqRaw(dx, dy);
    if (distSq > sqRange(range)) return;

    const key = world.ownerCanonical(pool.owner[j]!) * 1048576 + pool.serial[j]!;
    if (distSq < bestDistSq || (distSq === bestDistSq && key < bestKey)) {
      bestDistSq = distSq;
      bestIndex = j;
      bestKey = key;
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
 *
 * `from` is where in the list to start, for the one caller that reaps a second
 * time in the same tick: `victorySystem` queues an eliminated player's estate
 * *after* this has already run, and the list is only cleared at the top of the
 * next tick. Skipping what has been reaped is not merely cheaper. An entry
 * resolves as `pool.destroy(pool.idAt(i))`, which reads the slot's *current*
 * generation — so a re-walk destroys whatever occupies that slot now, and the
 * only thing making that harmless today is that nothing happens to spawn
 * between the two calls. That is an invariant nobody declared and one inserted
 * system away from a silent entity delete.
 */
export function reapDead(world: World, from = 0): void {
  const pool = world.pool;
  const deaths = world.events.deaths;
  if (from >= deaths.length) return;

  for (let k = from; k < deaths.length; k++) {
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
    const a = pool.attackTarget[i]!;
    if (a !== NO_ENTITY && !pool.isAlive(a)) pool.cancelAttack(i);
    const hp = pool.harvestPatch[i]!;
    if (hp !== NO_ENTITY && !pool.isAlive(hp)) pool.harvestPatch[i] = NO_ENTITY;
  }

  world.recomputeSupply();
}

/** Distance helper shared with the AI, which reasons about threat ranges. */
export function withinRange(world: World, a: number, b: number, range: number): boolean {
  const pool = world.pool;
  const dx = pool.posX[b]! - pool.posX[a]!;
  const dy = pool.posY[b]! - pool.posY[a]!;
  return vecLenSqRaw(dx, dy) <= sqRange(range + fromInt(0));
}
