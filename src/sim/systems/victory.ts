/**
 * Win, loss, and draw conditions.
 *
 * A player is eliminated when every structure they own is destroyed — the rule
 * the genre has used since Dune II, and it stops matches dragging on while a
 * lone worker hides in a corner.
 *
 * There is a second elimination rule that is less traditional but necessary
 * here. A player can end up with buildings but *no units at all* and too few
 * minerals to train even a worker. Nothing can gather, nothing can be built,
 * nothing can attack: that position cannot change for the rest of time. Without
 * this rule a mined-out map where both armies wiped each other out simply never
 * ends, which is exactly what happened — two bots sat across a dead map for
 * fifteen simulated minutes with no possible action between them.
 */

import { defOf } from '../../config/rules.js';
import { EntityType, NEUTRAL, NO_ENTITY, type PlayerId } from '../types.js';
import type { World } from '../world.js';

export function victorySystem(world: World): void {
  if (world.matchOver) return;

  const pool = world.pool;
  const playerCount = world.players.length;
  const buildings = new Array<number>(playerCount).fill(0);
  const units = new Array<number>(playerCount).fill(0);
  /** Can this player still train something, given what it has banked? */
  const canProduce = new Array<boolean>(playerCount).fill(false);
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;

    if (type === EntityType.MineralPatch) continue;

    const owner = pool.owner[i]!;
    if (owner === NEUTRAL) continue;

    const def = defOf(type);
    if (def.isBuilding) {
      buildings[owner] = (buildings[owner] ?? 0) + 1;
      // Only a finished building can train anything.
      if (pool.buildState[i] === 2 && def.produces.length > 0) {
        const cheapest = cheapestOf(def.produces);
        if (world.player(owner as PlayerId).minerals >= cheapest) canProduce[owner] = true;
      }
    } else {
      units[owner] = (units[owner] ?? 0) + 1;
    }
  }

  for (let p = 0; p < playerCount; p++) {
    const ps = world.players[p]!;
    if (ps.defeated) continue;

    if (buildings[p] === 0) {
      ps.defeated = true;
      continue;
    }

    // Stalemate guard: no units, nothing affordable to train, and — since
    // gathering needs a worker — no prospect of ever affording one. Minerals
    // left in the ground are irrelevant with nobody alive to mine them.
    if (units[p] === 0 && !canProduce[p]) {
      ps.defeated = true;
    }
  }

  let survivors = 0;
  let lastAlive: PlayerId = NO_ENTITY;
  for (let p = 0; p < playerCount; p++) {
    if (world.players[p]!.defeated) continue;
    survivors++;
    lastAlive = p;
  }

  if (survivors <= 1) {
    world.matchOver = true;
    // Zero survivors means both sides were eliminated on the same tick — a
    // genuine draw, which needs to end the match rather than leave it running
    // with nobody able to win it.
    world.winner = lastAlive;
  }

}

/** Cheapest unit among a production list. */
function cheapestOf(types: readonly EntityType[]): number {
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < types.length; i++) {
    const cost = defOf(types[i]!).mineralCost;
    if (cost < best) best = cost;
  }
  return best;
}
