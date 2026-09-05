/**
 * Any agent at a human's pace.
 *
 * The neural bot decides every `DECISION_TICKS` and issues at most one command
 * per decision, which is inside the rate measured for people in
 * `tests/wire.test.ts`. Wrapping the scripted bot the same way turns it into
 * the *teacher* for imitation: the student can only ever emit one command per
 * decision, so its labels must come from something that did too.
 */

import type { Command } from '../sim/commands.js';
import type { PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { chunkCommands, type Agent } from './agent.js';

/** Ticks between the neural bot's decisions: 200 ms, five a second. */
export const DECISION_TICKS = 4;

export interface CadenceOptions {
  readonly decisionTicks?: number;
  /**
   * Orders waiting their turn, at most. A think that wants more than this is
   * trimmed from the front — the oldest order is the one most likely to be
   * stale by the time it would go out.
   */
  readonly queueCap?: number;
}

export function humanCadence(inner: Agent, options: CadenceOptions = {}): Agent {
  const decisionTicks = options.decisionTicks ?? DECISION_TICKS;
  const queueCap = options.queueCap ?? 8;
  const queue: Command[] = [];
  return {
    act(world: World, player: PlayerId): Command[] {
      for (const command of chunkCommands(inner.act(world, player))) queue.push(command);
      while (queue.length > queueCap) queue.shift();
      if (world.tick % decisionTicks !== 0 || queue.length === 0) return [];
      return [queue.shift()!];
    },
    dispose(): void {
      inner.dispose?.();
    },
  };
}
