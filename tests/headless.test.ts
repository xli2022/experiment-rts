/**
 * A headless match is the lockstep loop with the lockstep taken out.
 *
 * Every bot-driven probe in the project runs through it, so it has to see the
 * world exactly as the browser's runner would show it: the same driver, and a
 * command landing on the same tick. The first test pins that directly, by
 * running the same bots through both and comparing the entity pool after every
 * tick.
 */

import { describe, expect, it } from 'vitest';
import type { Agent } from '../src/ai/agent.js';
import { AgentDriver } from '../src/ai/driver.js';
import { createHostedAgents } from '../src/ai/factory.js';
import { executeTickFor, HeadlessMatch } from '../src/ai/headless.js';
import { INPUT_DELAY_TURNS, LockstepRunner, TICKS_PER_TURN } from '../src/net/lockstep.js';
import { SoloTransport } from '../src/net/localTransport.js';
import { checksumInit } from '../src/sim/checksum.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import {
  EntityType,
  MS_PER_TICK,
  Order,
  type MatchConfig,
  type PlayerId,
} from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { scriptedAgents } from './helpers/agents.js';

const SEED = 0x51ce7a11;

function poolChecksum(world: World): number {
  return world.pool.checksum(checksumInit()) >>> 0;
}

/** The first live worker `player` owns, as a slot index. */
function firstWorker(world: World, player: PlayerId): number {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === player && pool.type[i] === EntityType.Worker) {
      return i;
    }
  }
  throw new Error(`player ${player} has no worker`);
}

/** Orders `player`'s first worker across the map, once, on `onTick`. */
function oneMove(onTick: number, player: PlayerId): Agent {
  return {
    act(world: World): Command[] {
      if (world.tick !== onTick) return [];
      const worker = world.pool.idAt(firstWorker(world, player));
      return [{ type: CommandType.Move, player, units: [worker], x: fromInt(60), y: fromInt(60) }];
    },
  };
}

/** Run a single-player lockstep match with the bots hosted, one tick per update. */
function lockstepHosted(config: MatchConfig): { sim: Simulation; tick(): void } {
  const sim = new Simulation(config);
  let driver: AgentDriver;
  const runner = new LockstepRunner(sim, new SoloTransport(), {
    onStep: () => driver.tick(sim.world),
  });
  driver = new AgentDriver(createHostedAgents(config, 0), (command, player) =>
    runner.issue(command, player),
  );
  return {
    sim,
    tick: () => {
      runner.update(MS_PER_TICK);
      if (runner.currentTick !== sim.world.tick) throw new Error('runner did not step');
    },
  };
}

describe('headless match', () => {
  for (const [name, config] of [
    ['a duel against one bot', duelMatch(SEED, { botPlayers: [1] })],
    ['a solo co-op hosting three bots', coopMatch(SEED, { botPlayers: [1, 2, 3] })],
  ] as const) {
    it(`agrees with a solo lockstep match tick for tick, on ${name}`, () => {
      const headless = new HeadlessMatch(config, scriptedAgents(config));
      const lockstep = lockstepHosted(config);
      for (let t = 0; t < 3000; t++) {
        headless.step();
        lockstep.tick();
        expect(lockstep.sim.world.tick).toBe(headless.world.tick);
        if (poolChecksum(lockstep.sim.world) !== poolChecksum(headless.world)) {
          throw new Error(`headless and lockstep diverged after tick ${t}`);
        }
      }
      // And the bots played, or the agreement is about nothing.
      expect(headless.world.player(1).supplyUsed).toBeGreaterThan(0);
    });
  }

  it('executes a command on the turn the runner would', () => {
    // Issued after tick t, a command executes at the start of turn
    // ceil(t / TICKS_PER_TURN) + INPUT_DELAY_TURNS — the runner drains what is
    // pending at the next turn boundary and schedules it that many turns out.
    for (const issuedAt of [5, 6, 7]) {
      const config = duelMatch(SEED, { botPlayers: [1] });
      const match = new HeadlessMatch(config, [[1, oneMove(issuedAt, 1)]]);
      const executesAt = executeTickFor(issuedAt);
      expect(executesAt).toBe(
        TICKS_PER_TURN * (Math.ceil(issuedAt / TICKS_PER_TURN) + INPUT_DELAY_TURNS),
      );
      while (match.world.tick < executesAt) match.step();
      const worker = firstWorker(match.world, 1);
      expect(match.world.pool.order[worker]).toBe(Order.None);
      expect(match.pending).toBe(1);
      match.step();
      expect(match.world.pool.order[worker]).toBe(Order.Move);
      expect(match.pending).toBe(0);
    }
  });

  it('puts an outside command on the same schedule', () => {
    const config = duelMatch(SEED, { botPlayers: [1] });
    const match = new HeadlessMatch(config, []);
    while (match.world.tick < 5) match.step();
    const worker = firstWorker(match.world, 0);
    match.issue(
      {
        type: CommandType.Move,
        player: 0,
        units: [match.world.pool.idAt(worker)],
        x: fromInt(60),
        y: fromInt(60),
      },
      0,
    );
    while (match.world.tick < executeTickFor(5)) match.step();
    expect(match.world.pool.order[worker]).toBe(Order.None);
    match.step();
    expect(match.world.pool.order[worker]).toBe(Order.Move);
  });

  it('is reproducible run to run', () => {
    const config = duelMatch(SEED, { botPlayers: [0, 1] });
    const a = new HeadlessMatch(config, scriptedAgents(config));
    const b = new HeadlessMatch(config, scriptedAgents(config));
    for (let t = 0; t < 2000; t++) {
      a.step();
      b.step();
      if (a.sim.checksum() !== b.sim.checksum()) throw new Error(`runs diverged at tick ${t}`);
    }
    expect(a.world.player(0).supplyUsed).toBeGreaterThan(0);
  });
});
