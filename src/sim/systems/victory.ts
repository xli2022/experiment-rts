/**
 * Win and loss conditions.
 *
 * A player is eliminated when every structure they own is destroyed. Surviving
 * units cannot win the game on their own — the same rule the genre has used
 * since Dune II, and it stops matches dragging on while a lone worker hides in a
 * corner.
 */

import { defOf } from '../../config/rules.js';
import { EntityType, NEUTRAL, NO_ENTITY, type PlayerId } from '../types.js';
import type { World } from '../world.js';

export function victorySystem(world: World): void {
  if (world.winner !== NO_ENTITY) return;

  const pool = world.pool;
  const buildingCount = new Array<number>(world.players.length).fill(0);

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const owner = pool.owner[i]!;
    if (owner === NEUTRAL) continue;
    const type = pool.type[i]! as EntityType;
    if (type === EntityType.MineralPatch) continue;
    if (!defOf(type).isBuilding) continue;
    buildingCount[owner] = (buildingCount[owner] ?? 0) + 1;
  }

  let survivors = 0;
  let lastAlive: PlayerId = NO_ENTITY;
  for (let p = 0; p < world.players.length; p++) {
    const ps = world.players[p]!;
    if (!ps.defeated && buildingCount[p] === 0) ps.defeated = true;
    if (!ps.defeated) {
      survivors++;
      lastAlive = p;
    }
  }

  if (survivors <= 1) {
    world.winner = lastAlive;
  }
}
