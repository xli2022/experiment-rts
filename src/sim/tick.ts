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

  constructor(config: MatchConfig | number) {
    this.world = new World(config);
    setupMatch(this.world);
    this.astar = new AStar(this.world.map);
    this.fields = new FlowFieldCache(this.world.map.width * this.world.map.height);
  }

  /**
   * Advance one tick, applying `commands` at the start of it.
   *
   * Every command arrives from outside — a human's UI or a hosted bot, by way
   * of the lockstep runner or a headless driver. The simulation used to run the
   * scripted bot here itself; it runs nothing now, so the only thing that can
   * make two peers diverge is a command one of them did not receive, and that
   * is exactly what lockstep waits for.
   */
  step(commands: Command[]): void {
    const world = this.world;

    world.events.attackStarts.length = 0;
    world.events.attackImpacts.length = 0;
    world.events.shots.length = 0;
    world.events.deaths.length = 0;
    world.events.completed.length = 0;

    // Commands arrive from several sources and in arbitrary packet order, so
    // impose a canonical order before applying any of them.
    if (commands.length > 0) {
      const ordered = sortCommands(commands.slice());
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
