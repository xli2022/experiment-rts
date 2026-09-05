/**
 * A match with no transport: the lockstep loop, minus the lockstep.
 *
 * Tests, the determinism and mirror probes, the arena and the training
 * environment all run bots through this. It hosts every agent it is given
 * through the same `AgentDriver` the browser uses, and delays each command by
 * the same rule the runner applies — issued after tick t, executed at the start
 * of turn `ceil(t / TICKS_PER_TURN) + INPUT_DELAY_TURNS` — so a bot sees the
 * world here exactly as it would see it in play. `tests/headless.test.ts` pins
 * that a headless match and a solo lockstep match agree tick for tick.
 *
 * Given agents that are pure functions of the world, a headless match is one
 * too, which is what lets the probes compare it across engines and against its
 * own rotation.
 */

import { INPUT_DELAY_TURNS, TICKS_PER_TURN } from '../net/lockstep.js';
import type { Command } from '../sim/commands.js';
import { Simulation } from '../sim/tick.js';
import type { MatchConfig, PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import type { Agent } from './agent.js';
import { AgentDriver } from './driver.js';

export interface HeadlessOptions {
  /** Turns between issuing a command and executing it. The runner's starting value by default. */
  readonly inputDelayTurns?: number;
}

/** The tick a command issued after `tick` executes on, under the runner's schedule. */
export function executeTickFor(tick: number, inputDelayTurns = INPUT_DELAY_TURNS): number {
  return TICKS_PER_TURN * (Math.ceil(tick / TICKS_PER_TURN) + inputDelayTurns);
}

export class HeadlessMatch {
  readonly sim: Simulation;
  readonly driver: AgentDriver;
  private readonly delay: number;
  /** Commands by the tick they execute on. */
  private readonly scheduled = new Map<number, Command[]>();

  constructor(
    config: MatchConfig | number,
    agents: Iterable<readonly [PlayerId, Agent]>,
    options: HeadlessOptions = {},
  ) {
    this.sim = new Simulation(config);
    this.delay = options.inputDelayTurns ?? INPUT_DELAY_TURNS;
    this.driver = new AgentDriver(agents, (command) => this.schedule(command));
  }

  get world(): World {
    return this.sim.world;
  }

  /** Apply what is due, advance one tick, then let the hosted agents act. */
  step(): void {
    const tick = this.world.tick;
    const due = this.scheduled.get(tick);
    if (due) this.scheduled.delete(tick);
    this.sim.step(due ?? []);
    this.driver.tick(this.world);
  }

  /** Step until `ticks` have passed or the match is over. Returns ticks stepped. */
  run(ticks: number): number {
    let stepped = 0;
    while (stepped < ticks && !this.world.matchOver) {
      this.step();
      stepped++;
    }
    return stepped;
  }

  /**
   * Issue a command from outside the hosted agents — a test standing in for a
   * human — on the same schedule a hosted command gets.
   */
  issue(command: Command, player: PlayerId): boolean {
    command.player = player;
    return this.schedule(command);
  }

  /** Commands scheduled and not yet executed, for tests. */
  get pending(): number {
    let n = 0;
    for (const list of this.scheduled.values()) n += list.length;
    return n;
  }

  private schedule(command: Command): boolean {
    const at = executeTickFor(this.world.tick, this.delay);
    let list = this.scheduled.get(at);
    if (!list) {
      list = [];
      this.scheduled.set(at, list);
    }
    list.push(command);
    return true;
  }

  dispose(): void {
    this.driver.dispose();
  }
}
