/**
 * Prints simulation checksums at fixed checkpoints.
 *
 * Run under two different JavaScript engines and compare the output: if the
 * simulation is genuinely portable, the numbers are identical. This is the
 * cheapest real proof available that the fixed-point layer works, because Node
 * runs on V8 and Bun runs on JavaScriptCore — two independent implementations of
 * `Math`, and the exact place where a simulation built on `Math.sin` or naive
 * float arithmetic would come apart.
 *
 *   npm run determinism:node
 *   npm run determinism:bun
 *   npm run determinism:cross    # runs both and diffs them
 *
 * A mismatch here means a Chrome player and a Safari player would desync — a bug
 * that is nearly impossible to reproduce once it reaches real users.
 */

import { checksumToHex } from '../src/sim/checksum.js';
import { CommandType } from '../src/sim/commands.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { recordMatch } from '../tests/helpers/scripted.js';

const SEED = 0x1234abcd;
// Long enough to reach a fight. The scripted match needs a Depot, then a
// Barracks, then units, then the walk to contact — first shot fired by a combat
// unit lands around tick 2800. Stopping at 1500, as this did, meant the
// strongest determinism check in the project covered combat with exactly one
// shot in the whole run, worker on worker.
const TICKS = 4000;
const CHECKPOINTS = [1, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000];

const lines: string[] = [];

lines.push(`scripted  seed=${SEED.toString(16)} ticks=${TICKS}`);
const { checksums } = recordMatch(SEED, TICKS);
for (const t of CHECKPOINTS) {
  const c = checksums[t - 1];
  if (c === undefined) continue;
  lines.push(`  tick ${String(t).padStart(5)}  ${checksumToHex(c)}`);
}

/**
 * A second leg driven by the bots.
 *
 * The scripted match is one fixed sequence, and a whole class of behaviour
 * simply never comes up in it: it does not set rally points, does not pack
 * buildings tightly enough for a trained unit to land in one, and does not jam
 * a crowd against a cliff. Fixes to all three landed without moving a single
 * checksum here, which means this check was blind to them. The bot builds,
 * expands and fights on its own, so it reaches states no fixed script will, and
 * it is deterministic — it is a simulation-side command source, so it costs
 * nothing to run under both engines.
 */
const BOT_SEED = 0x51ce7a11;
const BOT_TICKS = 6000;
lines.push('', `bot-vs-bot  seed=${BOT_SEED.toString(16)} ticks=${BOT_TICKS}`);
const bots = new Simulation(duelMatch(BOT_SEED, { botPlayers: [0, 1] }));
for (let t = 1; t <= BOT_TICKS; t++) {
  bots.step([]);
  if (t % 1000 === 0 || t === 1) {
    lines.push(`  tick ${String(t).padStart(5)}  ${checksumToHex(bots.checksum())}`);
  }
}

/**
 * A third leg on the four-player map.
 *
 * The other two legs are both a 1v1 on the duel map, so nothing in them ever
 * reaches four players, the larger grid, team hostility, or the bot's
 * team-level decisions — a whole map and half the roster's worth of arithmetic
 * that the strongest check in the project would otherwise never run under a
 * second engine. Bot-driven for the same reason the leg above is: it needs no
 * recorded script to reach states worth checksumming.
 */
const COOP_SEED = 0x51ce7a11;
const COOP_TICKS = 4000;
lines.push('', `co-op 2v2  seed=${COOP_SEED.toString(16)} ticks=${COOP_TICKS}`);
const coop = new Simulation(coopMatch(COOP_SEED, { botPlayers: [0, 1, 2, 3] }));
for (let t = 1; t <= COOP_TICKS; t++) {
  coop.step([]);
  if (t % 1000 === 0 || t === 1) {
    lines.push(`  tick ${String(t).padStart(5)}  ${checksumToHex(coop.checksum())}`);
  }
}

/**
 * A fourth leg that plays through an elimination.
 *
 * The three legs above never eliminate anybody — I checked: four seeds of
 * four-bot co-op run twelve thousand ticks without one — so the code that runs
 * when a player goes out has never been executed under a second engine. That
 * code is the most id-sensitive in the simulation: `strip` decides the order
 * slots return to the free list, and the free list decides every entity id
 * issued for the rest of the match. Two engines that disagreed there would
 * diverge on everything afterwards.
 *
 * Rather than wait for one to happen, make one happen: concede on a fixed tick
 * and keep going long enough for the ids to matter.
 */
const OUT_SEED = 0x51ce7a11;
const OUT_AT = 600;
const OUT_TICKS = 2400;
lines.push('', `elimination  seed=${OUT_SEED.toString(16)} ticks=${OUT_TICKS}`);
const out = new Simulation(coopMatch(OUT_SEED, { botPlayers: [0, 1, 2, 3] }));
for (let t = 1; t <= OUT_TICKS; t++) {
  out.step(t === OUT_AT ? [{ type: CommandType.Surrender, player: 0 }] : []);
  if (t === OUT_AT || t % 600 === 0) {
    lines.push(`  tick ${String(t).padStart(5)}  ${checksumToHex(out.checksum())}`);
  }
}

console.log(lines.join('\n'));
