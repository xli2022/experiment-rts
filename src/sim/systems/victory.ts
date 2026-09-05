/**
 * Win, loss, and draw conditions.
 *
 * Elimination is per player; the *match* ends per team. A co-op player whose
 * base is razed is out of the game, but their partner plays on and can still
 * win it for both of them — which is the whole appeal of playing together, and
 * is why the surviving-side count below is over teams rather than slots.
 *
 * **Being out takes everything with it.** An eliminated player's remaining
 * units *and buildings* are destroyed on the tick they go out, which is what
 * the genre does and what closes an otherwise nasty seam: elimination is per
 * player and the match is per team, so without this a co-op player who lost
 * their last building — or who surrendered — left an army on the field that
 * fought on but could take no orders, because `executeCommand` drops commands
 * from a defeated player. One rule instead: out is out, and what you owned goes
 * with you. Buildings are the half with a visible side effect: `reapDead`
 * releases their footprint, so the ground a departing partner held becomes
 * buildable again.
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
 *
 * That rule now razes a base rather than only flagging a slot, so "cannot
 * change" has to be true rather than nearly true. A unit already *paid for* and
 * sitting in a production queue is one that will arrive without another
 * mineral being mined, so a queue that is not empty counts as prospects — see
 * `canProduce` below. Without that clause a player who spent their last
 * minerals on a worker and then lost their army was declared permanently stuck
 * one tick before the worker popped out, and had their Command Post demolished
 * with the worker still in it.
 */

import { defOf } from '../../config/rules.js';
import {
  EntityType,
  MAX_PLAYERS,
  NEUTRAL,
  NO_ENTITY,
  type PlayerId,
  type TeamId,
} from '../types.js';
import type { World } from '../world.js';
import { reapDead } from './combat.js';

/**
 * Per-tick tallies, reused rather than reallocated.
 *
 * This system runs on every tick of every match, and the sim allocates nothing
 * per tick — see `spawnOut` in `economy.ts` and `targetOut` in `orders.ts` for
 * the same pattern. Sized by `MAX_PLAYERS` and cleared to the roster length,
 * which `teamsFor` refuses to exceed. Pure scratch: never read before it is
 * written, never checksummed.
 */
const buildings = new Int32Array(MAX_PLAYERS);
const units = new Int32Array(MAX_PLAYERS);
/**
 * Can this player still put a unit on the field without mining another
 * mineral? True when something is affordable, and true when something has
 * already been bought and is sitting in a queue.
 */
const canProduce = new Uint8Array(MAX_PLAYERS);

export function victorySystem(world: World): void {
  if (world.matchOver) return;

  const pool = world.pool;
  const playerCount = world.players.length;
  buildings.fill(0, 0, playerCount);
  units.fill(0, 0, playerCount);
  canProduce.fill(0, 0, playerCount);
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;

    if (type === EntityType.MineralPatch) continue;

    const owner = pool.owner[i]!;
    if (owner === NEUTRAL) continue;

    const def = defOf(type);
    if (def.isBuilding) {
      buildings[owner]! += 1;
      // Only a finished building can train anything.
      if (pool.buildState[i] === 2 && def.produces.length > 0) {
        // Already paid for and under way: the minerals are spent, so the unit
        // arrives whatever the bank says.
        if (pool.prodCount[i]! > 0) canProduce[owner] = 1;
        const cheapest = cheapestOf(def.produces);
        if (world.player(owner as PlayerId).minerals >= cheapest) canProduce[owner] = 1;
      }
    } else {
      units[owner]! += 1;
    }
  }

  // Where this tick's deaths already reach. `reapDead` ran in `tick.ts` a
  // moment ago and does not clear the list, so anything `strip` appends starts
  // here — and the second reap below is told so, rather than re-walking slots
  // it has already freed.
  const alreadyReaped = world.events.deaths.length;

  let eliminatedAny = false;
  for (let p = 0; p < playerCount; p++) {
    const ps = world.players[p]!;
    if (!ps.defeated) {
      // Stalemate guard, second clause: no units, nothing affordable to train,
      // and — since gathering needs a worker — no prospect of ever affording
      // one. Minerals left in the ground are irrelevant with nobody alive to
      // mine them.
      const out = buildings[p] === 0 || (units[p] === 0 && canProduce[p] === 0);
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
  if (eliminatedAny) reapDead(world, alreadyReaped);

  // Count surviving *teams*, not players. A side with one partner left standing
  // has not lost, and in a 1v1 — where every team has exactly one member — this
  // is the same count it always was.
  //
  // Marked rather than rescanned: asking "did an earlier surviving slot already
  // claim this team" is a second encoding of the split `config.teams` already
  // holds, and it made the count quadratic on every tick of every match.
  //
  // A bit per team rather than an array, for two reasons. A typed array would
  // have to be sized in the *slot* dimension and indexed in the *team* one, and
  // an out-of-range typed-array write is silently discarded — so a roster whose
  // team ids reached the slot count would count one team twice and the match
  // would never end. And this runs every tick of every match, where the sim
  // allocates nothing — see the scratch tallies at the top of the file.
  // `MAX_PLAYERS` bounds the team ids too, comfortably inside 31 bits.
  let seenTeams = 0;
  let survivingTeams = 0;
  let lastAlive: TeamId = NO_ENTITY;
  for (let p = 0; p < playerCount; p++) {
    if (world.players[p]!.defeated) continue;
    const team = world.teamOf(p);
    const bit = 1 << team;
    if ((seenTeams & bit) !== 0) continue;
    seenTeams |= bit;
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
 *
 * Ascending *within one player*, not across the whole list: these are appended
 * after whatever combat already queued this tick, and one call per defeated
 * slot. What matters is that the order is the same on every peer, which it is —
 * `reapDead`'s note about ascending order describes the combat loop that
 * populates it, not this list once both have written to it.
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
