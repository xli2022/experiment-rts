/**
 * The scripted bot as an `Agent`.
 *
 * `bot.ts` decides *what* to order; this decides *when* to ask it, and cuts
 * its orders down to the human vocabulary.
 */

import type { Command } from '../sim/commands.js';
import type { PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { chunkCommands, type Agent } from './agent.js';
import { botThink, THINK_INTERVAL } from './bot.js';

export interface ScriptedOptions {
  /**
   * Ticks between thinks. `THINK_INTERVAL` is the real bot.
   *
   * A different interval makes a *different* bot — one that a match against
   * the real one resolves instead of mirroring to a draw, which is what the
   * seat-fairness tests need. It is not a weaker bot: thinking every 20 ticks
   * beats the real bot 8–0 over eight seeds, because the beat gates in
   * `bot.ts` still fall on its thinks and it re-issues orders half as often.
   * The lobby never sets it — one scripted bot, one strength.
   */
  readonly thinkInterval?: number;
}

export class ScriptedAgent implements Agent {
  readonly thinkInterval: number;

  constructor(options: ScriptedOptions = {}) {
    const interval = options.thinkInterval ?? THINK_INTERVAL;
    if (!Number.isInteger(interval) || interval < 1) {
      throw new Error(`thinkInterval must be a positive integer, got ${interval}`);
    }
    this.thinkInterval = interval;
  }

  act(world: World, player: PlayerId): Command[] {
    // Every bot thinks on the same tick. See THINK_INTERVAL.
    if (world.tick % this.thinkInterval !== 0) return [];
    return chunkCommands(botThink(world, player));
  }
}
