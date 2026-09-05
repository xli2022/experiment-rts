/**
 * What a bot is.
 *
 * A bot is a player. It reads the world through this one interface and answers
 * with the same `Command`s a human's UI produces; a host — `AgentDriver` —
 * hands those to the lockstep runner, where they cross the wire and execute one
 * input delay later, exactly like a human's. The simulation never runs a bot:
 * `Simulation.step` applies what it is given, and `config.bots` is nothing to
 * it but a checksummed roster.
 *
 * ## Why bots are hosted rather than simulated
 *
 * The scripted bot used to run inside `Simulation.step` on every peer, which
 * costs no bandwidth and needs no host — and which only works for a bot that is
 * a pure function of the world. The neural bot samples its actions, so two
 * peers running it would disagree on its first decision and the match would
 * silently become two matches. Rather than keep two kinds of bot on two paths,
 * every bot is hosted: one peer owns each bot slot, derived from the agreed
 * config (`hostOf`), and sends for it. Bots are interchangeable — the roster
 * names a `BotKind`, `createAgent` turns it into one of these, and nothing
 * downstream can tell which it got.
 *
 * What that costs: a hosted bot's commands cross the wire (a few hundred bytes
 * a turn), they land one input delay after they are decided, and in online
 * co-op each bot reacts at its host's delay. The scripted bot is still a pure
 * function of the world — the determinism and mirror probes depend on that,
 * and `tests/sealed-sim.test.ts` scans it — but nothing structural does.
 */

import { MAX_COMMAND_UNITS, type Command } from '../sim/commands.js';
import type { PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';

export interface Agent {
  /**
   * Called once per simulation tick, after the tick. Returns the commands to
   * issue now, possibly none, each naming at most `MAX_COMMAND_UNITS` units.
   * Must never write the world.
   */
  act(world: World, player: PlayerId): Command[];
  /** Release anything held outside the world — a worker, a socket. */
  dispose?(): void;
}

/**
 * Split any command naming more than `MAX_COMMAND_UNITS` units into several
 * that each obey the cap, in the original order.
 *
 * The scripted bot orders its whole army in one command, which a human cannot
 * do: the UI caps a selection at the same number. Chunking keeps every bot
 * inside the human vocabulary and every packet inside the wire's budget.
 */
export function chunkCommands(commands: readonly Command[]): Command[] {
  const out: Command[] = [];
  for (const command of commands) {
    if (!('units' in command) || command.units.length <= MAX_COMMAND_UNITS) {
      out.push(command);
      continue;
    }
    for (let start = 0; start < command.units.length; start += MAX_COMMAND_UNITS) {
      out.push({ ...command, units: command.units.slice(start, start + MAX_COMMAND_UNITS) });
    }
  }
  return out;
}
