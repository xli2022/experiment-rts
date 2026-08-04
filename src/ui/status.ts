/**
 * What a selected unit is currently doing, in words.
 *
 * A worker spends most of the match doing something invisible. Walking to a
 * construction site, standing on it building, standing on a patch mining,
 * walking back with a load — from the outside all look identical to a worker
 * that has been forgotten about, and the panel used to say nothing at all
 * unless it happened to be carrying minerals. So the one state a player must
 * act on, genuinely idle, was indistinguishable from four states they must not.
 *
 * This reads the simulation and never writes it. It lives apart from `hud.ts`
 * so it can be tested without a DOM.
 */

import { BUILD_REACH, defOf, HARVEST_REACH, HARVEST_TICKS } from '../config/rules.js';
import { idIndex } from '../sim/entities.js';
import { inReach } from '../sim/systems/economy.js';
import { BuildState, EntityType, NO_ENTITY, Order } from '../sim/types.js';
import type { World } from '../sim/world.js';

/**
 * A one-line account of `index`'s current job, or `null` if it has nothing
 * worth reporting beyond what the rest of the panel already shows.
 *
 * Phrased as what the unit is doing rather than which order it holds: a worker
 * on `Order.Build` may be walking or building, and those are different answers
 * to "why is it not mining?".
 */
export function activityOf(world: World, index: number): string | null {
  const pool = world.pool;
  if (pool.alive[index] !== 1) return null;
  const type = pool.type[index]! as EntityType;
  if (defOf(type).isBuilding) return null;

  const order = pool.order[index]!;

  if (type === EntityType.Worker) {
    if (order === Order.Build) return buildActivity(world, index);
    if (order === Order.Harvest) return harvestActivity(world, index);
  }

  switch (order) {
    case Order.Move:
      return 'moving';
    case Order.AttackMove:
      return 'attack-moving';
    case Order.Attack:
      return 'attacking';
    case Order.Hold:
      return 'holding position';
    default:
      // Every unit reports, not just workers: a soldier standing about is the
      // same question a player is asking, and answering it for one kind of unit
      // and not another reads as the panel being broken rather than terse. A
      // soldier with something in range is fighting even with no order at all.
      return pool.combatTarget[index] !== NO_ENTITY && pool.isAlive(pool.combatTarget[index]!)
        ? 'engaging'
        : 'idle';
  }
}

/** Building — and whether the worker has arrived yet. */
function buildActivity(world: World, index: number): string | null {
  const pool = world.pool;
  const siteId = pool.orderTarget[index]!;
  // The order outlives the site for one tick when a building is destroyed
  // mid-construction; the worker is about to be released.
  if (siteId === NO_ENTITY || !pool.isAlive(siteId)) return 'idle';

  const site = idIndex(siteId);
  const def = defOf(pool.type[site]! as EntityType);

  // The order outlives a finished building by a tick, while the worker is being
  // released. There is no repair to report.
  if (pool.buildState[site] === BuildState.Complete) return 'idle';

  if (!inReach(world, index, site, BUILD_REACH)) return `walking to ${def.name} site`;
  return `building ${def.name} ${pct100(pool.buildProgress[site]!, def.buildTicks)}%`;
}

/** The harvest cycle: out to the patch, mining, and back with a load. */
function harvestActivity(world: World, index: number): string | null {
  const pool = world.pool;

  // The load is the headline — it is what the existing panel showed, and it is
  // the half of the cycle a player can act on by adding a drop-off.
  if (pool.carrying[index]! > 0) return `returning ${pool.carrying[index]} minerals`;

  const patchId = pool.harvestPatch[index]!;
  if (patchId === NO_ENTITY || !pool.isAlive(patchId)) return 'looking for minerals';

  const patch = idIndex(patchId);
  if (!inReach(world, index, patch, HARVEST_REACH)) return 'walking to minerals';
  return `mining ${pct100(pool.harvestTimer[index]!, HARVEST_TICKS)}%`;
}

/** Whole percent, clamped — a progress counter reading 101% reads as a bug. */
function pct100(value: number, total: number): number {
  if (total <= 0) return 100;
  const pct = Math.floor((value / total) * 100);
  return pct < 0 ? 0 : pct > 100 ? 100 : pct;
}
