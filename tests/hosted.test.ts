/**
 * Bots are players hosted by a peer, and any bot is interchangeable with any
 * other.
 *
 * The scripted bot and the neural bot take the same path: a slot in the
 * roster, a host derived from the config, a driver that issues through the
 * runner, commands on the wire. These tests run that path with both — the
 * neural slot played by a stand-in, since the real one needs a model — and
 * check that swapping the kind changes nothing but the agent.
 */

import { describe, expect, it } from 'vitest';
import type { Agent } from '../src/ai/agent.js';
import { AgentDriver } from '../src/ai/driver.js';
import { createHostedAgents, type AgentDeps } from '../src/ai/factory.js';
import { RandomAgent } from '../src/ai/neural/random.js';
import { ScriptedAgent } from '../src/ai/scripted.js';
import {
  HOSTED_COMMANDS_PER_TURN,
  INPUT_DELAY_TURNS,
  LockstepRunner,
  TICKS_PER_TURN,
} from '../src/net/lockstep.js';
import { LocalNetwork, SoloTransport } from '../src/net/localTransport.js';
import type { Packet } from '../src/net/transport.js';
import { CommandType, MAX_COMMAND_UNITS, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { coopMatch, duelMatch, hostOf, isBotSlot } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import {
  BotKind,
  CHECKSUM_INTERVAL,
  EntityType,
  MS_PER_TICK,
  Order,
  type MatchConfig,
  type PlayerId,
} from '../src/sim/types.js';
import type { World } from '../src/sim/world.js';

const SEED = 0x51ce7a11;

/** The first live worker `player` owns, as a slot index, or -1. */
function firstWorker(world: World, player: PlayerId): number {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === player && pool.type[i] === EntityType.Worker) {
      return i;
    }
  }
  return -1;
}

/**
 * A stand-in for the neural bot: every fifty ticks it walks its first worker
 * to a fixed spot. Nothing like a real player, but it issues real commands
 * through the real path, which is what these tests are about.
 */
class PulseAgent implements Agent {
  act(world: World, player: PlayerId): Command[] {
    if (world.tick % 50 !== 25) return [];
    const worker = firstWorker(world, player);
    if (worker < 0) return [];
    return [
      {
        type: CommandType.Move,
        player,
        units: [world.pool.idAt(worker)],
        x: fromInt(60 + (world.tick % 100 === 25 ? 0 : 10)),
        y: fromInt(60),
      },
    ];
  }
}

/** A stand-in that never does anything. */
class IdleAgent implements Agent {
  act(): Command[] {
    return [];
  }
}

const DEPS: AgentDeps = { neural: () => new PulseAgent() };

interface Peer {
  sim: Simulation;
  runner: LockstepRunner;
  driver: AgentDriver;
  stalls: PlayerId[][];
  desyncs: number;
}

/** Two peers on one simulated network, each hosting the bots dealt to it. */
function makePeers(net: LocalNetwork, config: MatchConfig, deps: AgentDeps = DEPS): Peer[] {
  const peers: Peer[] = [];
  for (let p = 0; p < net.playerCount; p++) {
    const sim = new Simulation(config);
    const peer: Peer = {
      sim,
      runner: null as never,
      driver: null as never,
      stalls: [],
      desyncs: 0,
    };
    peer.runner = new LockstepRunner(
      sim,
      net.createTransport(p),
      {
        onStep: () => peer.driver.tick(sim.world),
        onStall: (waiting) => peer.stalls.push(waiting),
        onDesync: () => {
          peer.desyncs++;
        },
      },
      () => net.nowMs,
    );
    peer.driver = new AgentDriver(createHostedAgents(config, p, deps), (command, player) =>
      peer.runner.issue(command, player),
    );
    peers.push(peer);
  }
  return peers;
}

function run(net: LocalNetwork, peers: Peer[], frames: number): void {
  for (let f = 0; f < frames; f++) {
    for (const peer of peers) peer.runner.update(MS_PER_TICK);
    net.advance(MS_PER_TICK);
  }
}

function expectAgreementAtCommonTick(peers: Peer[]): void {
  const minTick = Math.min(...peers.map((p) => p.runner.currentTick));
  for (let t = minTick - (minTick % CHECKSUM_INTERVAL); t > 0; t -= CHECKSUM_INTERVAL) {
    const values = peers.map((p) => p.runner.checksumAt(t));
    if (values.some((v) => v === undefined)) continue;
    expect(values[0]).toBe(values[1]);
    return;
  }
  throw new Error(`no common checksummed tick below ${minTick}`);
}

/** Which slots' commands crossed the wire, and from which peer each came. */
function watchWire(net: LocalNetwork): { onWire: Set<number>; sentBy: Map<number, Set<number>> } {
  const onWire = new Set<number>();
  const sentBy = new Map<number, Set<number>>();
  const originalSubmit = net.submit.bind(net);
  net.submit = (from, packet): void => {
    for (const turn of packet.turns) {
      for (const command of turn.commands) {
        onWire.add(command.player);
        let senders = sentBy.get(command.player);
        if (!senders) sentBy.set(command.player, (senders = new Set()));
        senders.add(from);
      }
    }
    originalSubmit(from, packet);
  };
  return { onWire, sentBy };
}

describe('a hosted bot in single-player', () => {
  for (const kind of [BotKind.Scripted, BotKind.Neural]) {
    it(`plays its slot through the runner without ever stalling (${BotKind[kind]})`, () => {
      const config = duelMatch(SEED, { botPlayers: [1], kind });
      expect(isBotSlot(config, 1)).toBe(true);
      expect(hostOf(config, 1)).toBe(0);

      const sim = new Simulation(config);
      let driver: AgentDriver;
      const stalls: PlayerId[][] = [];
      const runner = new LockstepRunner(sim, new SoloTransport(), {
        onStep: () => driver.tick(sim.world),
        onStall: (waiting) => stalls.push(waiting),
      });
      driver = new AgentDriver(createHostedAgents(config, 0, DEPS), (command, player) =>
        runner.issue(command, player),
      );
      expect(runner.ownedPlayers).toEqual([0, 1]);

      for (let t = 0; t < 600; t++) runner.update(MS_PER_TICK);
      expect(runner.currentTick).toBe(600);
      expect(runner.state).toBe('running');
      expect(stalls).toEqual([]);

      // The bot's commands executed: the scripted one trained, the stand-in
      // walked a worker away from the mineral line.
      const stats = driver.statsFor(1)!;
      expect(stats.issued).toBeGreaterThan(0);
      expect(stats.rejected).toBe(0);
      if (kind === BotKind.Scripted) {
        expect(sim.world.player(1).supplyUsed).toBeGreaterThan(sim.world.player(0).supplyUsed);
      } else {
        expect(sim.world.pool.order[firstWorker(sim.world, 1)]).toBe(Order.Move);
      }
    });
  }
});

describe('hosted bots on the wire', () => {
  const rosters: [string, MatchConfig][] = [
    ['two scripted bots', coopMatch(SEED, { kind: BotKind.Scripted })],
    ['two neural bots', coopMatch(SEED, { kind: BotKind.Neural })],
    [
      'a scripted and a neural bot',
      coopMatch(SEED, {
        botSlots: [
          { player: 2, kind: BotKind.Scripted },
          { player: 3, kind: BotKind.Neural },
        ],
      }),
    ],
  ];

  for (const [name, config] of rosters) {
    it(`deals ${name} one to a peer and carries their commands, agreeing throughout`, () => {
      const net = new LocalNetwork(2, 5);
      net.latencyMs = 60;
      net.jitterMs = 20;
      const peers = makePeers(net, config);
      const { onWire, sentBy } = watchWire(net);

      // Whatever the kinds, the dealing and the ownership are the same.
      expect(peers[0]!.runner.ownedPlayers).toEqual([0, 2]);
      expect(peers[1]!.runner.ownedPlayers).toEqual([1, 3]);

      run(net, peers, 700);

      expectAgreementAtCommonTick(peers);
      expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
      expect([...onWire].sort()).toEqual([2, 3]);
      expect([...sentBy.get(2)!]).toEqual([0]);
      expect([...sentBy.get(3)!]).toEqual([1]);
      // And both bots actually did something.
      for (const slot of [2, 3]) {
        expect(peers[hostOf(config, slot)]!.driver.statsFor(slot)!.issued).toBeGreaterThan(0);
      }
    });
  }

  it('keeps agreeing through heavy packet loss', () => {
    const config = coopMatch(SEED);
    const net = new LocalNetwork(2, 1234);
    net.latencyMs = 50;
    net.dropRate = 0.3;
    const peers = makePeers(net, config);

    run(net, peers, 900);

    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(200);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(200);
    expect(peers[0]!.sim.world.player(2).supplyUsed).toBeGreaterThan(0);
  });

  it('reports a silent peer, never the bot that peer hosts', () => {
    const config = coopMatch(SEED);
    const net = new LocalNetwork(2);
    const peers = makePeers(net, config);
    run(net, peers, 40);

    // Peer 1 goes silent, and with it the bot in slot 3.
    for (let f = 0; f < 200; f++) {
      peers[0]!.runner.update(MS_PER_TICK);
      net.advance(MS_PER_TICK);
    }

    expect(peers[0]!.runner.state).toBe('stalled');
    expect(peers[0]!.stalls.length).toBeGreaterThan(0);
    for (const waiting of peers[0]!.stalls) expect(waiting).toEqual([1]);
  });

  it('ignores a packet claiming a slot its sender does not host', () => {
    const config = coopMatch(SEED, { kind: BotKind.Neural });
    const net = new LocalNetwork(2);
    // Idle bots, so the only order the victim could receive is the forged one.
    const peers = makePeers(net, config, { neural: () => new IdleAgent() });
    run(net, peers, 40);

    const world = peers[1]!.sim.world;
    const victim = firstWorker(world, 2); // slot 2 is hosted by peer 0
    expect(victim).toBeGreaterThanOrEqual(0);
    const turn = Math.floor(peers[1]!.runner.currentTick / TICKS_PER_TURN) + INPUT_DELAY_TURNS + 2;
    const forged: Packet = {
      player: 1,
      turns: [
        {
          turn,
          player: 2,
          commands: [
            {
              type: CommandType.Move,
              player: 2,
              units: [world.pool.idAt(victim)],
              x: fromInt(60),
              y: fromInt(60),
            },
          ],
        },
      ],
    };
    net.submit(1, forged);

    run(net, peers, 200);

    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
    for (const peer of peers) {
      expect(peer.sim.world.pool.order[victim]).toBe(Order.None);
    }
  });

  it('keeps every owned slot on a contiguous schedule, whatever the delay does', () => {
    const config = coopMatch(SEED);
    const net = new LocalNetwork(2, 0xc0de);
    net.latencyMs = 20;

    const scheduled = new Map<PlayerId, Set<number>>();
    const sim = new Simulation(config);
    const inner = net.createTransport(0);
    const spy = {
      get localPlayer() {
        return inner.localPlayer;
      },
      get playerCount() {
        return inner.playerCount;
      },
      get ready() {
        return inner.ready;
      },
      send(packet: Packet) {
        for (const t of packet.turns) {
          let turns = scheduled.get(t.player);
          if (!turns) scheduled.set(t.player, (turns = new Set()));
          turns.add(t.turn);
        }
        inner.send(packet);
      },
      onPacket: inner.onPacket.bind(inner),
      onPeerLost: inner.onPeerLost.bind(inner),
      close: inner.close.bind(inner),
    };
    let driver: AgentDriver;
    const a = new LockstepRunner(
      sim,
      spy,
      { onStep: () => driver.tick(sim.world) },
      () => net.nowMs,
    );
    driver = new AgentDriver(createHostedAgents(config, 0), (c, p) => a.issue(c, p));
    const other = new Simulation(config);
    let driverB: AgentDriver;
    const b = new LockstepRunner(
      other,
      net.createTransport(1),
      { onStep: () => driverB.tick(other.world) },
      () => net.nowMs,
    );
    driverB = new AgentDriver(createHostedAgents(config, 1), (c, p) => b.issue(c, p));

    for (let round = 0; round < 24; round++) {
      net.latencyMs = round % 2 === 0 ? 10 : 300;
      net.jitterMs = round % 2 === 0 ? 0 : 40;
      for (let f = 0; f < 60; f++) {
        a.update(MS_PER_TICK);
        b.update(MS_PER_TICK);
        net.advance(MS_PER_TICK);
      }
    }

    expect([...scheduled.keys()].sort()).toEqual([0, 2]);
    for (const slot of [0, 2]) {
      const turns = [...scheduled.get(slot)!].sort((x, y) => x - y);
      expect(turns.length).toBeGreaterThan(50);
      expect(turns[0]).toBe(INPUT_DELAY_TURNS);
      const holes = turns.filter((t, i) => i > 0 && t !== turns[i - 1]! + 1);
      expect(`slot ${slot} holes: ${holes.join(',')}`).toBe(`slot ${slot} holes: `);
    }
    expect(a.inputDelayTurns !== INPUT_DELAY_TURNS || b.inputDelayTurns !== INPUT_DELAY_TURNS).toBe(
      true,
    );
  });
});

describe('a random legal bot in a neural slot', () => {
  it('plays through the whole hosted path, and the simulation takes what it says', () => {
    // The stand-in for the real neural bot before a model exists: it sees
    // through the codec, answers with uniformly random legal decisions, and
    // every one of them goes through the driver and the runner like a human's.
    const config = duelMatch(SEED, { botPlayers: [1], kind: BotKind.Neural });
    const sim = new Simulation(config);
    let driver: AgentDriver;
    const agent = new RandomAgent(7);
    const runner = new LockstepRunner(sim, new SoloTransport(), {
      onStep: () => driver.tick(sim.world),
    });
    driver = new AgentDriver(createHostedAgents(config, 0, { neural: () => agent }), (c, p) =>
      runner.issue(c, p),
    );
    for (let t = 0; t < 1200; t++) runner.update(MS_PER_TICK);
    expect(runner.state).toBe('running');
    expect(agent.decisions).toBeGreaterThan(50);
    expect(driver.statsFor(1)!.issued).toBe(agent.decisions);
    expect(driver.statsFor(1)!.rejected).toBe(0);
    expect(runner.droppedByBudget).toBe(0);
  });
});

describe('the roster a runner accepts', () => {
  it('refuses humans seated above the bots, the stall a count check lets through', () => {
    // Two humans and two bots pass a count check against a two-peer
    // transport; seated in slots 2 and 3 they would each wait forever on a
    // bot slot nobody sends for. `Game` used to be the only place that knew.
    const net = new LocalNetwork(2);
    const config = coopMatch(SEED, { botPlayers: [0, 1] });
    expect(() => new LockstepRunner(new Simulation(config), net.createTransport(0))).toThrow(
      /must be the low slots/,
    );
    expect(
      () => new LockstepRunner(new Simulation(coopMatch(SEED)), net.createTransport(1)),
    ).not.toThrow();
  });
});

describe('the budget on a hosted slot', () => {
  it('caps commands per turn and units per command, and never the human', () => {
    const config = duelMatch(SEED, { botPlayers: [1] });
    const sim = new Simulation(config);
    const runner = new LockstepRunner(sim, new SoloTransport());
    const worker = sim.world.pool.idAt(firstWorker(sim.world, 1));
    const small = (player: PlayerId): Command => ({
      type: CommandType.Move,
      player,
      units: [worker],
      x: fromInt(30),
      y: fromInt(30),
    });

    const accepted: boolean[] = [];
    for (let i = 0; i < HOSTED_COMMANDS_PER_TURN + 1; i++) accepted.push(runner.issue(small(1), 1));
    expect(accepted).toEqual([...Array(HOSTED_COMMANDS_PER_TURN).fill(true), false]);
    expect(runner.droppedByBudget).toBe(1);

    const oversize: Command = {
      type: CommandType.Move,
      player: 1,
      units: Array.from({ length: MAX_COMMAND_UNITS + 1 }, () => worker),
      x: fromInt(30),
      y: fromInt(30),
    };
    // Even with room in the turn, a command past the unit cap is refused.
    runner.update(MS_PER_TICK * TICKS_PER_TURN);
    expect(runner.issue(oversize, 1)).toBe(false);
    expect(runner.droppedByBudget).toBe(2);

    // The human's own slot has no budget at all.
    for (let i = 0; i < 10; i++) expect(runner.issue(small(0))).toBe(true);
    expect(runner.issue(oversize, 0)).toBe(true);

    // A slot this peer does not send for is a programming error, not a drop.
    expect(() => runner.issue(small(1), 5)).toThrow(/does not send for slot 5/);
  });

  it('is the runner refusing what the driver would never send', () => {
    // The driver paces a bot inside the budget; the runner's check is the last
    // line. Run a bot that bursts and confirm the line is never crossed.
    const config = duelMatch(SEED, { botPlayers: [1] });
    const sim = new Simulation(config);
    const bursting: Agent = {
      act(world: World, player: PlayerId): Command[] {
        const worker = firstWorker(world, player);
        if (worker < 0 || world.tick % 10 !== 0) return [];
        return Array.from({ length: 9 }, () => ({
          type: CommandType.Move,
          player,
          units: [world.pool.idAt(worker)],
          x: fromInt(30),
          y: fromInt(30),
        }));
      },
    };
    let driver: AgentDriver;
    const runner = new LockstepRunner(sim, new SoloTransport(), {
      onStep: () => driver.tick(sim.world),
    });
    driver = new AgentDriver([[1, bursting]], (c, p) => runner.issue(c, p));
    for (let t = 0; t < 200; t++) runner.update(MS_PER_TICK);
    expect(runner.droppedByBudget).toBe(0);
    expect(driver.statsFor(1)!.rejected).toBe(0);
    expect(driver.statsFor(1)!.issued).toBeGreaterThan(100);
    void ScriptedAgent;
  });
});
