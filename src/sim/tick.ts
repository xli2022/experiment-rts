/**
 * The simulation step.
 *
 * One call to `step()` advances the world by exactly one fixed tick. Nothing in
 * here consults wall-clock time, frame duration, or anything else that varies
 * between machines: given the same world and the same commands, it produces the
 * same world, always.
 *
 * ## System order is part of the contract
 *
 * The sequence below is fixed and must not be reordered casually. Combat reads
 * positions that movement just wrote; the spatial grid is rebuilt before anyone
 * queries it; the dead are reaped only after every system has had its turn, so
 * no system ever observes a half-removed entity. Changing the order changes
 * results — legitimately, but it invalidates the golden replay fixture, so it
 * should be a deliberate decision rather than a drive-by edit.
 */

import { generateBotCommands } from '../ai/bot.js';
import { sortCommands, type Command } from './commands.js';
import { AStar } from './pathing/astar.js';
import { FlowFieldCache } from './pathing/flowfield.js';
import { combatSystem, reapDead } from './systems/combat.js';
import { economySystem } from './systems/economy.js';
import { movementSystem } from './systems/movement.js';
import { executeCommand } from './systems/orders.js';
import { victorySystem } from './systems/victory.js';
import type { MatchConfig } from './types.js';
import { setupMatch, World } from './world.js';

/**
 * Owns a world plus the scratch structures used to advance it.
 *
 * The A* instance lives here rather than on `World` because it is pure scratch —
 * nothing it holds between calls influences results, so it is deliberately
 * excluded from the checksum.
 */
export class Simulation {
  readonly world: World;
  private readonly astar: AStar;
  /**
   * Shared flow fields for group movement. Pure scratch, derived from the map
   * and the requested destination, so it is excluded from the checksum — two
   * peers with differently-warmed caches still compute identical fields.
   */
  private readonly fields: FlowFieldCache;

  /**
   * Difficulty per player slot, or -1 for a slot no bot plays.
   *
   * The bot is deterministic, so it runs here — inside the simulation, on every
   * peer — rather than on one machine that broadcasts its orders. That costs no
   * bandwidth, needs no host, and makes single-player and multiplayer the same
   * code path. Which slots it plays therefore has to be agreed before the match
   * rather than set afterwards, which is why it arrives in the config and this
   * array is built once and never written again.
   */
  private readonly botOf: Int8Array;

  constructor(config: MatchConfig | number, mapSize?: number) {
    this.world = new World(config, mapSize);
    setupMatch(this.world);
    this.astar = new AStar(this.world.map);
    this.fields = new FlowFieldCache(this.world.map.width * this.world.map.height);

    this.botOf = new Int8Array(this.world.players.length).fill(-1);
    for (const bot of this.world.config.bots) {
      if (bot.player >= 0 && bot.player < this.botOf.length) {
        this.botOf[bot.player] = bot.difficulty;
      }
    }
  }

  /** Does the AI play this slot? */
  isBot(player: number): boolean {
    return (this.botOf[player] ?? -1) >= 0;
  }

  /** Advance one tick, applying `commands` at the start of it. */
  step(commands: Command[]): void {
    const world = this.world;

    world.events.attackStarts.length = 0;
    world.events.attackImpacts.length = 0;
    world.events.shots.length = 0;
    world.events.deaths.length = 0;
    world.events.completed.length = 0;

    // Bot commands are generated locally rather than received, but are
    // otherwise indistinguishable from a human's — same type, same validation,
    // same ordering.
    let all = commands;
    if (world.config.bots.length > 0) {
      all = commands.slice();
      // Ascending slot order, from a dense array rather than the config list —
      // the order bots think in is part of the simulation, so it must not
      // depend on how a caller happened to sort the roster.
      for (let p = 0; p < this.botOf.length; p++) {
        const difficulty = this.botOf[p]!;
        if (difficulty < 0) continue;
        const botCmds = generateBotCommands(world, p, difficulty);
        for (let i = 0; i < botCmds.length; i++) all.push(botCmds[i]!);
      }
    }

    // Commands arrive from several sources and in arbitrary packet order, so
    // impose a canonical order before applying any of them.
    if (all.length > 0) {
      const ordered = sortCommands(all.slice());
      for (let i = 0; i < ordered.length; i++) {
        executeCommand(world, ordered[i]!);
      }
    }

    world.grid.rebuild(world.pool);

    movementSystem(world, this.astar, this.fields);
    combatSystem(world);
    economySystem(world);

    reapDead(world);
    victorySystem(world);

    world.tick++;
  }

  checksum(): number {
    return this.world.checksum();
  }
}
