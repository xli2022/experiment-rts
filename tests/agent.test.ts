/**
 * Bots are players, and there is one scripted bot.
 *
 * Easy, Normal and Hard used to be three tunings of the scripted bot; Hard is
 * the one that survives, and the merge has to be provably behaviour-preserving.
 * The fixtures under `tests/fixtures/` were recorded from the pre-merge build:
 * every command the Hard bot decided on every think of two whole matches, plus
 * the entity pool's checksum after every tick. Replaying them here and asking
 * the merged bot the same questions on the same worlds pins the logic exactly.
 *
 * The rest covers the interface every bot shares — the driver, the cadence
 * wrapper, the unit cap — and the fact that the simulation runs no bot itself.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { chunkCommands, type Agent } from '../src/ai/agent.js';
import { botThink, THINK_INTERVAL } from '../src/ai/bot.js';
import { DECISION_TICKS, humanCadence } from '../src/ai/cadence.js';
import { AgentDriver } from '../src/ai/driver.js';
import { createAgent } from '../src/ai/factory.js';
import { ScriptedAgent } from '../src/ai/scripted.js';
import { checksumInit } from '../src/sim/checksum.js';
import { CommandType, MAX_COMMAND_UNITS, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { duelMatch, matchConfig } from '../src/sim/match.js';
import { HOSTED_COMMANDS_PER_TURN, TICKS_PER_TURN } from '../src/net/lockstep.js';
import { Simulation } from '../src/sim/tick.js';
import { BotKind, type MapLayout, type PlayerId } from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';
import { cloneCommands } from './helpers/scripted.js';

interface BotFixture {
  name: string;
  layout: MapLayout;
  seed: number;
  players: number;
  ticks: number;
  thinks: { tick: number; player: number; commands: Command[] }[];
  poolChecksums: number[];
}

function loadFixture(name: string): BotFixture {
  const url = new URL(`./fixtures/${name}.json`, import.meta.url);
  return JSON.parse(readFileSync(url, 'utf8')) as BotFixture;
}

/** An agent that emits a fixed script of commands on chosen ticks. */
function scripted(script: Record<number, Command[]>): Agent {
  return {
    act(world: World): Command[] {
      return cloneCommands(script[world.tick] ?? []);
    },
  };
}

function move(units: number[], player = 0): Command {
  return { type: CommandType.Move, player, units, x: fromInt(10), y: fromInt(10) };
}

describe('the merged scripted bot', () => {
  for (const name of ['bot-hard-duel', 'bot-hard-quarters']) {
    it(`decides exactly what Hard decided, tick for tick, on ${name}`, () => {
      const fixture = loadFixture(name);
      const players = Array.from({ length: fixture.players }, (_, p) => p);
      const sim = new Simulation(
        matchConfig(fixture.layout, fixture.seed, { botPlayers: players }),
      );
      const world = sim.world;

      const thinks = new Map<number, Command[]>();
      for (const think of fixture.thinks) thinks.set(think.tick * 8 + think.player, think.commands);

      for (let t = 0; t < fixture.ticks; t++) {
        expect(world.tick).toBe(t);
        const due: Command[] = [];
        if (t % THINK_INTERVAL === 0) {
          for (const p of players) {
            const want = thinks.get(t * 8 + p);
            expect(want, `no recorded think for player ${p} at tick ${t}`).toBeDefined();
            // The same world, the same question: the answer must be the same
            // command list, before any chunking.
            expect(botThink(world, p)).toEqual(want);
            for (const c of cloneCommands(want!)) due.push(c);
          }
        }
        // Replay what Hard decided, so the world keeps following the recording
        // rather than whatever the merged bot would have done next.
        sim.step(due);
        const checksum = world.pool.checksum(checksumInit()) >>> 0;
        if (checksum !== fixture.poolChecksums[t]) {
          throw new Error(`entity pool diverged from the recording after tick ${t}`);
        }
      }
    });
  }

  it('thinks on the same tick as every other scripted bot, and only then', () => {
    const config = duelMatch(0x51ce7a11, { botPlayers: [0, 1] });
    const sim = new Simulation(config);
    const agent = new ScriptedAgent();
    expect(agent.thinkInterval).toBe(THINK_INTERVAL);
    let thinks = 0;
    for (let t = 0; t < 40; t++) {
      const out = agent.act(sim.world, 0);
      if (sim.world.tick % THINK_INTERVAL !== 0) expect(out).toEqual([]);
      else thinks++;
      sim.step([]);
    }
    expect(thinks).toBe(4);
  });

  it('accepts a slower think interval as a handicap and refuses nonsense', () => {
    const slow = new ScriptedAgent({ thinkInterval: 20 });
    expect(slow.thinkInterval).toBe(20);
    expect(() => new ScriptedAgent({ thinkInterval: 0 })).toThrow();
    expect(() => new ScriptedAgent({ thinkInterval: 2.5 })).toThrow();
  });
});

describe('the human vocabulary every bot obeys', () => {
  it('chunks an oversize order into commands that each obey the unit cap', () => {
    const units = Array.from({ length: 60 }, (_, i) => i + 1);
    const out = chunkCommands([move(units), { type: CommandType.Stop, player: 0, units: [1, 2] }]);
    expect(out.map((c) => ('units' in c ? c.units.length : -1))).toEqual([24, 24, 12, 2]);
    expect(out.every((c) => !('units' in c) || c.units.length <= MAX_COMMAND_UNITS)).toBe(true);
    // Order preserved, nothing lost.
    expect(out.flatMap((c) => ('units' in c ? c.units : []))).toEqual([...units, 1, 2]);
  });

  it('runs a bot at a human pace: one command per decision, in order', () => {
    const sim = new Simulation(duelMatch(1, { botPlayers: [] }));
    const inner = scripted({ 0: [move(Array.from({ length: 60 }, (_, i) => i + 1))] });
    const paced = humanCadence(inner);
    const released: number[] = [];
    for (let t = 0; t <= 4 * DECISION_TICKS; t++) {
      const out = paced.act(sim.world, 0);
      expect(out.length).toBeLessThanOrEqual(1);
      if (out.length === 1) released.push(sim.world.tick);
      sim.step([]);
    }
    // Sixty units are three commands, one per decision tick, starting on the
    // first decision boundary after they were queued.
    expect(released).toEqual([0, DECISION_TICKS, 2 * DECISION_TICKS]);
  });

  it('caps the paced queue by dropping the oldest, and stays reproducible', () => {
    const many = Array.from({ length: 12 }, (_, i) => move([i + 1]));
    const run = (): number[] => {
      const sim = new Simulation(duelMatch(1, { botPlayers: [] }));
      const paced = humanCadence(scripted({ 0: many }), { queueCap: 8 });
      const seen: number[] = [];
      for (let t = 0; t < 12 * DECISION_TICKS; t++) {
        for (const c of paced.act(sim.world, 0)) seen.push('units' in c ? c.units[0]! : -1);
        sim.step([]);
      }
      return seen;
    };
    const a = run();
    // The four oldest were trimmed; the eight newest went out in order.
    expect(a).toEqual([5, 6, 7, 8, 9, 10, 11, 12]);
    expect(run()).toEqual(a);
  });
});

describe('the driver', () => {
  it('visits hosted slots in ascending order whatever order it was given', () => {
    const sim = new Simulation(duelMatch(1, { botPlayers: [] }));
    const visits: number[] = [];
    const noting = (slot: number): Agent => ({
      act(): Command[] {
        visits.push(slot);
        return [];
      },
    });
    const driver = new AgentDriver(
      [
        [3, noting(3)],
        [1, noting(1)],
        [2, noting(2)],
      ],
      () => true,
    );
    driver.tick(sim.world);
    expect(visits).toEqual([1, 2, 3]);
    expect(driver.players).toEqual([1, 2, 3]);
  });

  it('paces a burst inside the per-turn budget without dropping anything', () => {
    const sim = new Simulation(duelMatch(1, { botPlayers: [] }));
    const burst = Array.from({ length: 6 }, (_, i) => move([i + 1]));
    const issued: { tick: number; unit: number }[] = [];
    const driver = new AgentDriver([[0, scripted({ 0: burst })]], (command) => {
      issued.push({ tick: sim.world.tick, unit: 'units' in command ? command.units[0]! : -1 });
      return true;
    });
    for (let t = 0; t < 3 * TICKS_PER_TURN; t++) {
      driver.tick(sim.world);
      sim.step([]);
    }
    expect(issued.map((i) => i.unit)).toEqual([1, 2, 3, 4, 5, 6]);
    // The first budget's worth on the turn the burst arrived, the rest on the
    // following one — and never more than the budget in any one turn.
    const perTurn = new Map<number, number>();
    for (const i of issued) {
      const turn = Math.ceil(i.tick / TICKS_PER_TURN);
      perTurn.set(turn, (perTurn.get(turn) ?? 0) + 1);
    }
    for (const n of perTurn.values()) expect(n).toBeLessThanOrEqual(HOSTED_COMMANDS_PER_TURN);
    expect([...perTurn.values()]).toEqual([HOSTED_COMMANDS_PER_TURN, 6 - HOSTED_COMMANDS_PER_TURN]);
    expect(driver.statsFor(0)).toEqual({ issued: 6, rejected: 0, queued: 0 });
  });

  it('stamps the slot on every command and counts what the sink refuses', () => {
    const sim = new Simulation(duelMatch(1, { botPlayers: [] }));
    const lying: Agent = {
      act(): Command[] {
        return [move([1], 7)]; // claims to be player 7
      },
    };
    const players: PlayerId[] = [];
    let refuse = true;
    const driver = new AgentDriver([[1, lying]], (command) => {
      players.push(command.player);
      return !refuse;
    });
    driver.tick(sim.world);
    refuse = false;
    driver.tick(sim.world);
    expect(players).toEqual([1, 1]);
    expect(driver.statsFor(1)).toEqual({ issued: 1, rejected: 1, queued: 0 });
  });

  it('makes an agent for every kind, and needs a runtime only for the neural one', () => {
    expect(createAgent(BotKind.Scripted, 1)).toBeInstanceOf(ScriptedAgent);
    expect(() => createAgent(BotKind.Neural, 1)).toThrow(/neural/);
    const stub: Agent = { act: () => [] };
    expect(createAgent(BotKind.Neural, 1, { neural: () => stub })).toBe(stub);
  });
});

describe('the simulation', () => {
  it('runs no bot of any kind', () => {
    // Whatever the roster says, a bare simulation applies only what it is
    // given. A bot is a player, hosted outside; a simulation that ran one would
    // be applying commands a peer never received.
    for (const kind of [BotKind.Scripted, BotKind.Neural]) {
      const sim = new Simulation(duelMatch(0x51ce7a11, { botPlayers: [0, 1], kind }));
      const before = sim.world.player(1).supplyUsed;
      const count = sim.world.pool.count;
      for (let t = 0; t < 300; t++) sim.step([]);
      expect(sim.world.player(1).supplyUsed).toBe(before);
      expect(sim.world.pool.count).toBe(count);
      expect(sim.world.player(1).minerals).toBe(sim.world.player(0).minerals);
    }
  });
});
