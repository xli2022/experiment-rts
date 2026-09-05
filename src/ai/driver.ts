/**
 * The one host for bots.
 *
 * Runs every hosted agent once a tick and routes its commands into a sink: the
 * lockstep runner's `issue` in the browser, a latency queue in a headless
 * match. Nothing else ever calls `Agent.act`, so a bot behaves the same in a
 * test, in training and in play.
 *
 * The driver is also where the wire's budget is kept. A hosted slot may send
 * at most `HOSTED_COMMANDS_PER_TURN` commands a turn; what an agent produces
 * beyond that waits in a queue, in order, and goes out on the following turns.
 * Nothing is dropped — the runner's own cap on hosted slots is a last line the
 * driver never reaches.
 */

import { HOSTED_COMMANDS_PER_TURN, TICKS_PER_TURN } from '../net/lockstep.js';
import type { Command } from '../sim/commands.js';
import type { PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { chunkCommands, type Agent } from './agent.js';

/** Where a hosted slot's commands go. Returns false when one was refused. */
export type CommandSink = (command: Command, player: PlayerId) => boolean;

export interface SlotStats {
  /** Commands handed to the sink and accepted. */
  issued: number;
  /** Commands the sink refused. */
  rejected: number;
  /** Commands waiting for a turn with budget. */
  queued: number;
}

interface HostedSlot {
  readonly player: PlayerId;
  readonly agent: Agent;
  readonly queue: Command[];
  readonly stats: SlotStats;
  /** The turn the release count belongs to. */
  turn: number;
  releasedThisTurn: number;
}

export class AgentDriver {
  private readonly slots: HostedSlot[] = [];
  private disposed = false;

  constructor(
    agents: Iterable<readonly [PlayerId, Agent]>,
    private readonly sink: CommandSink,
  ) {
    for (const [player, agent] of agents) {
      this.slots.push({
        player,
        agent,
        queue: [],
        stats: { issued: 0, rejected: 0, queued: 0 },
        turn: -1,
        releasedThisTurn: 0,
      });
    }
    // Ascending slot order: the order bots think in is visible to the
    // simulation, so it must not depend on how a caller built the map.
    this.slots.sort((a, b) => a.player - b.player);
  }

  /** The slots this driver hosts, ascending. */
  get players(): PlayerId[] {
    return this.slots.map((slot) => slot.player);
  }

  agentFor(player: PlayerId): Agent | undefined {
    return this.slots.find((slot) => slot.player === player)?.agent;
  }

  statsFor(player: PlayerId): SlotStats | undefined {
    return this.slots.find((slot) => slot.player === player)?.stats;
  }

  /**
   * Call after every simulation tick.
   *
   * A command issued after tick t is drained by the runner at the next turn
   * boundary at or after t, so both ticks that drain together count against the
   * same turn: `ceil(t / TICKS_PER_TURN)`.
   */
  tick(world: World): void {
    const turn = Math.ceil(world.tick / TICKS_PER_TURN);
    for (const slot of this.slots) {
      for (const command of chunkCommands(slot.agent.act(world, slot.player))) {
        slot.queue.push(command);
      }
      if (slot.turn !== turn) {
        slot.turn = turn;
        slot.releasedThisTurn = 0;
      }
      while (slot.queue.length > 0 && slot.releasedThisTurn < HOSTED_COMMANDS_PER_TURN) {
        const command = slot.queue.shift()!;
        // The slot is the driver's to assert, not the agent's — the runner
        // re-stamps it again on the way in, for the same reason.
        command.player = slot.player;
        if (this.sink(command, slot.player)) {
          slot.stats.issued++;
          slot.releasedThisTurn++;
        } else {
          slot.stats.rejected++;
        }
      }
      slot.stats.queued = slot.queue.length;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots) slot.agent.dispose?.();
  }
}
