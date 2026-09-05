/**
 * Win, loss, and draw conditions.
 *
 * Elimination is per player; the *match* ends per team. A co-op player whose
 * base is razed is out of the game, but their partner plays on and can still
 * win it for both of them — which is the whole appeal of playing together, and
 * is why the surviving-side count below is over teams rather than slots.
 *
 * **Being out takes everything with it.** An eliminated player's remaining
 * units are destroyed on the tick they go out, which is what the genre does and
 * what closes an otherwise nasty seam: elimination is per player and the match
 * is per team, so without this a co-op player who lost their last building — or
 * who surrendered — left an army on the field that fought on but could take no
 * orders, because `executeCommand` drops commands from a defeated player. One
 * rule instead: out is out, and what you owned goes with you.
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
import { EntityType, NEUTRAL, NO_ENTITY, type PlayerId, type TeamId } from '../types.js';
import type { World } from '../world.js';
import { reapDead } from './combat.js';

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

  let eliminatedAny = false;
  for (let p = 0; p < playerCount; p++) {
    const ps = world.players[p]!;
    if (!ps.defeated) {
      // Stalemate guard, second clause: no units, nothing affordable to train,
      // and — since gathering needs a worker — no prospect of ever affording
      // one. Minerals left in the ground are irrelevant with nobody alive to
      // mine them.
      const out = buildings[p] === 0 || (units[p] === 0 && !canProduce[p]);
      if (!out) continue;
      ps.defeated = true;
    }

    // Defeated, however they got there. `Surrender` sets the flag itself, from
    // a command that has already executed by the time this runs, so keying the
    // sweep on "newly defeated here" would empty a razed player's base and
    // leave a conceding player's standing — the one case the button exists for.
    // Asking what they still own instead covers both, and does nothing on
    // every tick after the first because by then they own nothing.
    if (buildings[p]! > 0 || units[p]! > 0) {
      strip(world, p as PlayerId);
      eliminatedAny = true;
    }
  }

  // One reap for however many players went out this tick. The deaths are queued
  // rather than applied directly so the presentation layer sees them like any
  // other: a surrendering player's base blows up rather than blinking away.
  if (eliminatedAny) reapDead(world);

  // Count surviving *teams*, not players. A side with one partner left standing
  // has not lost, and in a 1v1 — where every team has exactly one member — this
  // is the same count it always was.
  //
  // Marked rather than rescanned: asking "did an earlier surviving slot already
  // claim this team" is a second encoding of the split `config.teams` already
  // holds, and it made the count quadratic on every tick of every match.
  const seen = new Uint8Array(playerCount);
  let survivingTeams = 0;
  let lastAlive: TeamId = NO_ENTITY;
  for (let p = 0; p < playerCount; p++) {
    if (world.players[p]!.defeated) continue;
    const team = world.teamOf(p);
    if (seen[team] === 1) continue;
    seen[team] = 1;
    survivingTeams++;
    lastAlive = team;
  }

  if (survivingTeams <= 1) {
    world.matchOver = true;
    // Zero survivors means both sides were eliminated on the same tick — a
    // genuine draw, which needs to end the match rather than leave it running
    // with nobody able to win it.
    world.winner = lastAlive;
  }
}

/**
 * Queue everything a player still owns for destruction.
 *
 * Ascending slot order, like every other pass over the pool, because the order
 * deaths are queued in decides the order slots return to the free list and
 * therefore every entity id issued afterwards.
 */
function strip(world: World, player: PlayerId): void {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.owner[i] !== player) continue;
    world.events.deaths.push(i);
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
