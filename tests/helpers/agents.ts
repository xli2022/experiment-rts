/**
 * Bots for headless matches.
 *
 * Every bot is hosted, so a test that wants a match played by bots builds the
 * agents itself and runs them through `HeadlessMatch`. One scripted bot exists;
 * where a test needs two *unequal* sides — because two identical bots on the
 * mirrored map can only draw — one of them thinks half as often.
 */

import type { Agent } from '../../src/ai/agent.js';
import { ScriptedAgent, type ScriptedOptions } from '../../src/ai/scripted.js';
import type { MatchConfig, PlayerId } from '../../src/sim/types.js';

/**
 * The think interval that makes a match against the real scripted bot resolve.
 *
 * Not a handicap, whatever it looks like: the half-speed thinker still lands
 * on every beat the bot's army logic fires on and re-issues its orders half as
 * often, and measured over eight seeds from both seats it *wins* 8–0. What the
 * tests need is only that the two sides differ, so that the match resolves and
 * swapping the seats swaps the winner. See `match.test.ts`.
 */
export const HALF_SPEED_THINK_INTERVAL = 20;

/** A scripted agent for every bot slot in the roster, with per-slot options. */
export function scriptedAgents(
  config: MatchConfig,
  options: Readonly<Record<number, ScriptedOptions>> = {},
): Map<PlayerId, Agent> {
  const agents = new Map<PlayerId, Agent>();
  for (const bot of config.bots) {
    agents.set(bot.player, new ScriptedAgent(options[bot.player] ?? {}));
  }
  return agents;
}

/** The full-speed bot in `fullSpeedSeat`, the half-speed one in the other, for a two-bot duel. */
export function unequalAgents(config: MatchConfig, fullSpeedSeat: PlayerId): Map<PlayerId, Agent> {
  const options: Record<number, ScriptedOptions> = {};
  for (const bot of config.bots) {
    options[bot.player] =
      bot.player === fullSpeedSeat ? {} : { thinkInterval: HALF_SPEED_THINK_INTERVAL };
  }
  return scriptedAgents(config, options);
}
