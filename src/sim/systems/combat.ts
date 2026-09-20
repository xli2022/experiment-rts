/**
 * Combat: target acquisition, firing, and death.
 *
 * Target selection is the classic determinism trap. "Attack the nearest enemy"
 * is ambiguous whenever two enemies are equidistant, and on a grid that happens
 * constantly. The comparator below is a strict total order — distance first,
 * then the target's seat and creation order — so there is never a tie to break
 * arbitrarily, and two mirrored units break it the same way.
 *
 * ## Attacks that land on more than one thing
 *
 * Splash, the Arclight's three coils, and the Piercebot's line all gather a
 * *set* of victims rather than one. Every one of them collects into `victims`
 * and sorts it on the same key `acquireTarget` breaks ties with — the target's
 * canonical seat and creation ordinal — before a single point of damage is
 * applied. Slot index would have been the obvious sort and is wrong: slots are
 * recycled from one shared free list, so two mirrored halves do not agree about
 * them, and the order damage lands in decides the order deaths are queued,
 * which decides the free list, which decides every entity id afterwards.
 */

import { CHILL_SPEED, defOf, MIN_DAMAGE, type EntityDef } from '../../config/rules.js';
import { idIndex } from '../entities.js';
import { fmul, fromInt, sqRange, vecLen, vecLenSqRaw, vecNormalize } from '../fixed.js';
import { BuildState, EntityType, NEUTRAL, NO_ENTITY, Order } from '../types.js';
import type { World } from '../world.js';

export function combatSystem(world: World): void {
  const pool = world.pool;

  // Chill wears off in a pass of its own, before anyone fires. Folded into the
  // loop below it would tick down *after* the attack that applied it whenever
  // the Ice Golem sat at a lower slot than what it shot, so the same chill
  // lasted one tick longer or shorter depending on which of the two was
  // spawned first — deterministic, but arbitrary, and visible on the panel.
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.chill[i]! > 0) pool.chill[i]! -= 1;
  }

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;

    if (pool.attackCooldown[i]! > 0) pool.attackCooldown[i]! -= 1;

    const type = pool.type[i]! as EntityType;
    const def = defOf(type);
    if (def.attackRange === 0) continue;
    // A repairer runs the same clock against the opposite list.
    if (def.repairAmount > 0) {
      serviceRepair(world, i, def);
      continue;
    }
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
    const distSq = vecLenSqRaw(dx, dy);
    if (distSq > sqRange(reach)) continue;
    // Too close to depress the barrel. The unit still faces what it cannot
    // shoot, so the player can see it is stuck rather than asleep, and
    // movement is left alone — backing artillery off is the player's job.
    if (def.minRange > 0 && distSq < sqRange(def.minRange)) continue;

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

  // One number, whatever it is shooting, and whatever else the shot reaches on
  // the way. The HUD shows `def.damage` on the info panel, and a per-matchup
  // multiplier would have made that a lie.
  victims.length = 0;
  addVictim(world, attackerIndex, targetIndex);
  if (def.splashRadius > 0) gatherSplash(world, attackerIndex, targetIndex, def);
  if (def.pierce) gatherLine(world, attackerIndex, targetIndex, def);
  if (def.maxTargets > 1) gatherExtraTargets(world, attackerIndex, def);
  sortVictims(world);

  for (let k = 0; k < victims.length; k++) {
    const v = victims[k]!;
    applyDamage(world, v, def.damage);
    if (def.chillTicks > 0) pool.chill[v] = def.chillTicks;
    world.events.shots.push(attackerIndex, v);
  }

  // The payload went off, so the thing carrying it is gone. Queued like any
  // other death so it is reaped with them, after every system has run.
  if (def.detonates) {
    pool.hp[attackerIndex] = 0;
    world.events.deaths.push(attackerIndex);
  }
}

/**
 * Victims of the attack being resolved, gathered before any of them is hurt.
 *
 * Module scope, reused every impact: this runs inside the tick loop, and a
 * fresh array per attack is garbage the simulation does not need to make. It is
 * only ever live between `victims.length = 0` and the loop that drains it.
 */
const victims: number[] = [];

/** Add a slot to `victims` if it is a legal target for this attacker and not already in. */
function addVictim(world: World, attackerIndex: number, index: number): void {
  if (!canVictimise(world, attackerIndex, index)) return;
  for (let k = 0; k < victims.length; k++) if (victims[k] === index) return;
  victims.push(index);
}

/** Whether this attacker's weapon is allowed to land on that slot at all. */
function canVictimise(world: World, attackerIndex: number, index: number): boolean {
  const pool = world.pool;
  if (index === attackerIndex) return false;
  if (pool.alive[index] !== 1) return false;
  if (!world.isHostile(index, pool.owner[attackerIndex]!)) return false;
  const def = defOf(pool.type[attackerIndex]! as EntityType);
  return def.canHitAir || !defOf(pool.type[index]! as EntityType).flying;
}

/**
 * Order the victims the way `acquireTarget` breaks ties.
 *
 * Insertion sort on a list that is almost always one to four long, and never
 * more than a crowd standing inside one blast. The key is the target's
 * canonical seat and creation ordinal, which two mirrored halves agree on;
 * slot index, which they do not, would make the two seats queue their deaths
 * in different orders and hand out different entity ids from then on.
 */
function sortVictims(world: World): void {
  for (let k = 1; k < victims.length; k++) {
    const v = victims[k]!;
    const key = tieKey(world, v);
    let j = k - 1;
    while (j >= 0 && tieKey(world, victims[j]!) > key) {
      victims[j + 1] = victims[j]!;
      j--;
    }
    victims[j + 1] = v;
  }
}

function tieKey(world: World, index: number): number {
  return world.ownerCanonical(world.pool.owner[index]!) * 1048576 + world.pool.serial[index]!;
}

/** Everything hostile standing within the blast, centred on what was hit. */
function gatherSplash(
  world: World,
  attackerIndex: number,
  targetIndex: number,
  def: EntityDef,
): void {
  const pool = world.pool;
  const cx = pool.posX[targetIndex]!;
  const cy = pool.posY[targetIndex]!;
  world.grid.forEachNear(cx, cy, def.splashRadius, (j) => {
    const other = defOf(pool.type[j]! as EntityType);
    // Measured to the edge, like every other range test in here, so a big
    // building caught by the rim of a blast takes it.
    const dx = pool.posX[j]! - cx;
    const dy = pool.posY[j]! - cy;
    if (vecLenSqRaw(dx, dy) > sqRange(def.splashRadius + other.radius)) return;
    addVictim(world, attackerIndex, j);
  });
}

/**
 * Everything hostile the bolt passes through on its way to the target.
 *
 * The test is distance from the segment, not from the line: something behind
 * the Piercebot is not on the shot, and neither is something past what it
 * aimed at — the bolt stops where it was aimed.
 */
function gatherLine(
  world: World,
  attackerIndex: number,
  targetIndex: number,
  def: EntityDef,
): void {
  const pool = world.pool;
  const ax = pool.posX[attackerIndex]!;
  const ay = pool.posY[attackerIndex]!;
  const sx = pool.posX[targetIndex]! - ax;
  const sy = pool.posY[targetIndex]! - ay;
  const lenSq = vecLenSqRaw(sx, sy);
  if (lenSq <= 0) return;
  const len = vecLen(sx, sy);

  // One sweep around the midpoint covers the whole segment: nothing on it is
  // further from the middle than half the shot's length.
  const midX = ax + ((sx / 2) | 0);
  const midY = ay + ((sy / 2) | 0);
  world.grid.forEachNear(midX, midY, def.attackRange, (j) => {
    if (j === targetIndex) return;
    const px = pool.posX[j]! - ax;
    const py = pool.posY[j]! - ay;

    // Along the shot: both products are exact integers well inside float64's
    // range (|dx| < 2^23 on the largest map, so each term is under 2^46), and
    // comparing them against `lenSq` needs no scaling — they share it.
    const along = sx * px + sy * py;
    if (along <= 0 || along >= lenSq) return;

    // Across it: the cross product over the length is the perpendicular
    // distance back in Q16.16. Division of exact integers is correctly
    // rounded, so every engine truncates to the same unit.
    const across = ((sx * py - sy * px) / len) | 0;
    const width = defOf(pool.type[j]! as EntityType).radius;
    if (across > width || across < -width) return;
    addVictim(world, attackerIndex, j);
  });
}

/**
 * The other enemies the remaining coils earth through.
 *
 * Chosen near the *shooter*, not near the primary target, which is what makes
 * this different from splash: an army that spreads out to beat a blast is
 * still three separate things inside an Arclight's reach. Taken nearest first,
 * one coil at a time, so the set does not depend on the order the spatial grid
 * happened to visit its cells in.
 */
function gatherExtraTargets(world: World, attackerIndex: number, def: EntityDef): void {
  const pool = world.pool;
  const px = pool.posX[attackerIndex]!;
  const py = pool.posY[attackerIndex]!;

  while (victims.length < def.maxTargets) {
    let bestIndex = -1;
    let bestDistSq = Number.POSITIVE_INFINITY;
    let bestKey = 0;

    world.grid.forEachNear(px, py, def.attackRange, (j) => {
      if (!canVictimise(world, attackerIndex, j)) return;
      for (let k = 0; k < victims.length; k++) if (victims[k] === j) return;

      const other = defOf(pool.type[j]! as EntityType);
      const distSq = distSqFrom(world, px, py, j);
      if (distSq > sqRange(def.attackRange + other.radius)) return;

      const key = tieKey(world, j);
      if (distSq < bestDistSq || (distSq === bestDistSq && key < bestKey)) {
        bestDistSq = distSq;
        bestIndex = j;
        bestKey = key;
      }
    });

    if (bestIndex < 0) return;
    victims.push(bestIndex);
  }
}

function distSqFrom(world: World, x: number, y: number, index: number): number {
  return vecLenSqRaw(world.pool.posX[index]! - x, world.pool.posY[index]! - y);
}

/**
 * A repairer's turn: mend the most broken friendly unit within reach.
 *
 * Deliberately not structures, and deliberately not itself. Free repair on
 * buildings was taken off the Worker for making any attack that did not
 * outright kill a structure a waste of time, and a pair of Fixomatics holding
 * each other up would be the same mistake at unit scale.
 */
function serviceRepair(world: World, index: number, def: EntityDef): void {
  const pool = world.pool;
  if (pool.owner[index] === NEUTRAL) return;

  const target = acquireRepairTarget(world, index, def.attackRange);
  pool.combatTarget[index] = target < 0 ? NO_ENTITY : pool.idAt(target);
  if (target < 0) return;

  const dx = pool.posX[target]! - pool.posX[index]!;
  const dy = pool.posY[target]! - pool.posY[index]!;
  const dir = vecNormalize(dx, dy);
  pool.faceX[index] = dir.x;
  pool.faceY[index] = dir.y;

  if (pool.attackCooldown[index]! > 0) return;
  pool.attackCooldown[index] = def.attackCooldown;

  const max = defOf(pool.type[target]! as EntityType).maxHp;
  const healed = pool.hp[target]! + def.repairAmount;
  pool.hp[target] = healed > max ? max : healed;

  // The same two events an attack raises. The renderer already knows how to
  // play a swing and draw a tracer from these, and a repair beam is one.
  world.events.attackStarts.push(index, target);
  world.events.attackImpacts.push(index);
  world.events.shots.push(index, target);
}

/**
 * The friendly unit most in need of mending, within `range`.
 *
 * Ordered by how much HP it is missing, then by the same canonical key every
 * other choice in this file breaks ties with — never by how close it is.
 * Nearest-first would have a repairer nurse a scratched Burstbot beside it
 * while the Dark Golem it is escorting dies two tiles away.
 */
function acquireRepairTarget(world: World, index: number, range: number): number {
  const pool = world.pool;
  const owner = pool.owner[index]!;
  const px = pool.posX[index]!;
  const py = pool.posY[index]!;

  let bestIndex = -1;
  let bestMissing = 0;
  let bestKey = 0;

  world.grid.forEachNear(px, py, range, (j) => {
    if (j === index) return;
    if (pool.alive[j] !== 1) return;
    if (pool.owner[j] !== owner) return;
    const other = defOf(pool.type[j]! as EntityType);
    if (other.isBuilding) return;
    const missing = other.maxHp - pool.hp[j]!;
    if (missing <= 0) return;

    const dx = pool.posX[j]! - px;
    const dy = pool.posY[j]! - py;
    if (vecLenSqRaw(dx, dy) > sqRange(range + other.radius)) return;

    const key = tieKey(world, j);
    if (missing > bestMissing || (missing === bestMissing && key < bestKey)) {
      bestMissing = missing;
      bestIndex = j;
      bestKey = key;
    }
  });

  return bestIndex;
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

/**
 * Apply damage and queue a death if it drops to zero.
 *
 * Armour comes off here rather than at the attacker, so it applies to every
 * source — a blast, a coil, a pierced bolt and a sword all meet the same
 * plate. Never below `MIN_DAMAGE`: armour is a bad matchup, not immunity.
 */
export function applyDamage(world: World, index: number, amount: number): void {
  const pool = world.pool;
  if (pool.alive[index] !== 1) return;
  const armor = defOf(pool.type[index]! as EntityType).armor;
  if (armor > 0 && amount > 0) {
    amount = amount - armor < MIN_DAMAGE ? MIN_DAMAGE : amount - armor;
  }
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

/**
 * This unit's top speed right now, after any chill.
 *
 * Movement asks for this instead of reading `def.speedPerTick` directly, so
 * every one of its passes — pursuit, flight, path-following, the final approach
 * — slows by the same amount rather than each remembering to.
 */
export function topSpeedOf(world: World, index: number, def: EntityDef): number {
  if (world.pool.chill[index]! <= 0) return def.speedPerTick;
  return fmul(def.speedPerTick, CHILL_SPEED);
}

/** Distance helper shared with the AI, which reasons about threat ranges. */
export function withinRange(world: World, a: number, b: number, range: number): boolean {
  const pool = world.pool;
  const dx = pool.posX[b]! - pool.posX[a]!;
  const dy = pool.posY[b]! - pool.posY[a]!;
  return vecLenSqRaw(dx, dy) <= sqRange(range + fromInt(0));
}
