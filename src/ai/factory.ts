/**
 * From a roster to agents.
 *
 * The scripted bot needs nothing; the neural bot needs a runtime the browser
 * loads before the match starts, so it arrives as a callback. A headless caller
 * that has no runtime simply never hosts a neural slot.
 */

import { hostedBy } from '../sim/match.js';
import { BotKind, type MatchConfig, type PlayerId } from '../sim/types.js';
import type { Agent } from './agent.js';
import { ScriptedAgent } from './scripted.js';

export interface AgentDeps {
  /** Makes a neural agent for a slot. Required only when the roster has one. */
  readonly neural?: (player: PlayerId) => Agent;
}

export function createAgent(kind: BotKind, player: PlayerId, deps: AgentDeps = {}): Agent {
  switch (kind) {
    case BotKind.Scripted:
      return new ScriptedAgent();
    case BotKind.Neural: {
      if (!deps.neural) {
        throw new Error(`slot ${player} is a neural bot and no neural runtime was provided`);
      }
      return deps.neural(player);
    }
  }
}

/** One agent per bot slot `localPlayer` hosts, keyed by slot. */
export function createHostedAgents(
  config: MatchConfig,
  localPlayer: PlayerId,
  deps: AgentDeps = {},
): Map<PlayerId, Agent> {
  const agents = new Map<PlayerId, Agent>();
  for (const player of hostedBy(config, localPlayer)) {
    const kind = config.bots.find((bot) => bot.player === player)!.kind;
    agents.set(player, createAgent(kind, player, deps));
  }
  return agents;
}
