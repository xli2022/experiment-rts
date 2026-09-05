/**
 * The online transport, end to end, against a fake room.
 *
 * `joinOnlineRoom` is the one piece of the multiplayer path that never ran in
 * a test: it needs a browser, WebRTC and a relay. Behind the `RoomProvider`
 * seam it needs none of them, so everything above the network can be checked
 * here — the handshake and its refusals, the slot dealing, hosted bots of
 * either kind riding the wire between two peers, a stranger in the room, a
 * peer leaving. What the fake cannot model is the network itself, which is
 * why a real link still gets a manual soak.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import type { Agent } from '../src/ai/agent.js';
import { AgentDriver } from '../src/ai/driver.js';
import { createHostedAgents, type AgentDeps } from '../src/ai/factory.js';
import { RandomAgent } from '../src/ai/neural/random.js';
import { INPUT_DELAY_TURNS, LockstepRunner, TICKS_PER_TURN } from '../src/net/lockstep.js';
import type { Packet } from '../src/net/transport.js';
import {
  JOIN_ABANDONED,
  joinOnlineRoom,
  PROTOCOL_VERSION,
  slotFromPeerIds,
  TRANSPORT_CHUNK_BYTES,
  type JoinResult,
} from '../src/net/trysteroTransport.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { fromInt } from '../src/sim/fixed.js';
import { coopMatch, hostOf } from '../src/sim/match.js';
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
import { FakeRoomNetwork, type Delivery } from './helpers/fakeRoom.js';

const SEED = 0x51ce7a11;
const ROOM = 'abcde';

/** The mode string the lobby sends: the config with the seed zeroed. */
function modeFor(kind: BotKind): string {
  return JSON.stringify(
    coopMatch(0, {
      botSlots: [
        { player: 2, kind },
        { player: 3, kind },
      ],
    }),
  );
}

interface JoinOptions {
  modeA?: string;
  modeB?: string;
  seed?: number;
  timeoutMs?: number;
  ids?: [string, string];
}

function joinPair(net: FakeRoomNetwork, options: JoinOptions = {}): Promise<JoinResult>[] {
  const [a, b] = options.ids ?? ['peer-a', 'peer-b'];
  const seed = options.seed ?? SEED;
  const mode = modeFor(BotKind.Scripted);
  return [
    joinOnlineRoom(
      { roomCode: ROOM, seed, mode: options.modeA ?? mode },
      options.timeoutMs ?? 8000,
      net.provider(a),
    ),
    joinOnlineRoom(
      { roomCode: ROOM, seed, mode: options.modeB ?? mode },
      options.timeoutMs ?? 8000,
      net.provider(b),
    ),
  ];
}

/** Run the clock until the joins have had every chance to settle. */
async function settle<T>(net: FakeRoomNetwork, promises: Promise<T>[], frames = 40): Promise<T[]> {
  for (let f = 0; f < frames; f++) {
    net.advance(MS_PER_TICK);
    await Promise.resolve();
  }
  return Promise.all(promises);
}

/** Which peer id a settled transport talks to. */
function pairedId(result: JoinResult): string | null {
  return (result.transport as unknown as { pairedPeerId: string | null }).pairedPeerId;
}

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

class IdleAgent implements Agent {
  act(): Command[] {
    return [];
  }
}

interface Peer {
  sim: Simulation;
  runner: LockstepRunner;
  driver: AgentDriver;
  result: JoinResult;
  stalls: PlayerId[][];
  timeouts: PlayerId[];
  desyncs: number;
}

/** Two settled peers, each hosting the bots dealt to it, as `Game` would build them. */
function makeOnlinePeers(
  net: FakeRoomNetwork,
  config: MatchConfig,
  results: JoinResult[],
  deps: AgentDeps,
): Peer[] {
  return results.map((result) => {
    const sim = new Simulation(config);
    const peer: Peer = {
      sim,
      runner: null as never,
      driver: null as never,
      result,
      stalls: [],
      timeouts: [],
      desyncs: 0,
    };
    peer.runner = new LockstepRunner(
      sim,
      result.transport,
      {
        onStep: () => peer.driver.tick(sim.world),
        onStall: (waiting) => peer.stalls.push(waiting),
        onPeerTimeout: (p) => peer.timeouts.push(p),
        onDesync: () => {
          peer.desyncs++;
        },
      },
      () => net.nowMs,
    );
    peer.driver = new AgentDriver(
      createHostedAgents(config, result.localPlayer, deps),
      (command, player) => peer.runner.issue(command, player),
    );
    return peer;
  });
}

function run(net: FakeRoomNetwork, peers: Peer[], frames: number): void {
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

/** Every `cmd` delivery: which slots' commands came from which peer id, and to whom. */
function watchWire(net: FakeRoomNetwork): {
  sentBy: Map<number, Set<string>>;
  targets: Set<string>;
  packets: number;
} {
  const sentBy = new Map<number, Set<string>>();
  const targets = new Set<string>();
  const counter = { packets: 0 };
  net.onDeliver = (d: Delivery) => {
    if (d.action !== 'cmd') return;
    counter.packets++;
    targets.add(d.to);
    for (const turn of (d.data as Packet).turns) {
      if (turn.commands.length === 0) continue;
      let senders = sentBy.get(turn.player);
      if (!senders) sentBy.set(turn.player, (senders = new Set()));
      senders.add(d.from);
    }
  };
  return {
    sentBy,
    targets,
    get packets() {
      return counter.packets;
    },
  };
}

describe('the handshake over a room', () => {
  it('settles both peers on complementary slots, however discovery is ordered', async () => {
    for (let seed = 1; seed <= 8; seed++) {
      const net = new FakeRoomNetwork({ seed, latencyMs: 30, jitterMs: 40 });
      const [a, b] = await settle(net, joinPair(net, { seed: 77 }));
      expect([a!.localPlayer, b!.localPlayer].sort()).toEqual([0, 1]);
      expect(a!.localPlayer).toBe(slotFromPeerIds('peer-a', 'peer-b'));
      expect(b!.localPlayer).toBe(slotFromPeerIds('peer-b', 'peer-a'));
      expect(a!.transport.ready && b!.transport.ready).toBe(true);
      expect(a!.seed).toBe(77);
      expect(pairedId(a!)).toBe('peer-b');
      expect(pairedId(b!)).toBe('peer-a');
    }
  });

  it('refuses a peer on another protocol version, and leaves the room', async () => {
    const net = new FakeRoomNetwork({ latencyMs: 10 });
    const joining = joinOnlineRoom(
      { roomCode: ROOM, seed: SEED, mode: modeFor(BotKind.Scripted) },
      30000,
      net.provider('peer-a'),
    );
    const old = net.join('peer-old', ROOM);
    net.advance(20);
    void old
      .makeAction('hello')
      .send(
        { protocol: PROTOCOL_VERSION - 1, seed: SEED, mode: modeFor(BotKind.Scripted) },
        { target: 'peer-a' },
      );
    net.advance(20);
    await expect(joining).rejects.toThrow(
      `protocol ${PROTOCOL_VERSION - 1}, expected ${PROTOCOL_VERSION}`,
    );
    expect(net.members(ROOM)).not.toContain('peer-a');
  });

  it('refuses two peers whose bot chips differ, on both sides', async () => {
    const net = new FakeRoomNetwork({ latencyMs: 10 });
    const [a, b] = joinPair(net, {
      modeA: modeFor(BotKind.Scripted),
      modeB: modeFor(BotKind.Neural),
    });
    const outcomes = await settle(net, [
      a!.then(
        () => 'ok',
        (e: Error) => e.message,
      ),
      b!.then(
        () => 'ok',
        (e: Error) => e.message,
      ),
    ]);
    for (const outcome of outcomes) expect(outcome).toMatch(/different modes/);
    expect(net.members(ROOM)).toEqual([]);
  });

  it('can be abandoned, before or after joining', async () => {
    const net = new FakeRoomNetwork({ latencyMs: 10 });
    const early = new AbortController();
    early.abort();
    await expect(
      joinOnlineRoom(
        { roomCode: ROOM, seed: SEED, mode: 'm', signal: early.signal },
        30000,
        net.provider('peer-a'),
      ),
    ).rejects.toThrow(JOIN_ABANDONED);
    expect(net.members(ROOM)).toEqual([]);

    const late = new AbortController();
    const joining = joinOnlineRoom(
      { roomCode: ROOM, seed: SEED, mode: 'm', signal: late.signal },
      30000,
      net.provider('peer-b'),
    );
    expect(net.members(ROOM)).toEqual(['peer-b']);
    late.abort();
    await expect(joining).rejects.toThrow(JOIN_ABANDONED);
    expect(net.members(ROOM)).toEqual([]);
  });

  it('gives up on an empty room at the timeout', async () => {
    vi.useFakeTimers();
    try {
      const net = new FakeRoomNetwork();
      const joining = joinOnlineRoom(
        { roomCode: ROOM, seed: SEED, mode: 'm' },
        1000,
        net.provider('peer-a'),
      );
      vi.advanceTimersByTime(1000);
      await expect(joining).rejects.toThrow(/Could not connect/);
      expect(net.members(ROOM)).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('is the same protocol version on both transports', () => {
    // The two-tab transport imports the constant rather than keeping its own,
    // so a bump for the wire cannot leave the two disagreeing.
    const source = readFileSync(
      new URL('../src/net/broadcastChannelTransport.ts', import.meta.url),
      'utf8',
    );
    expect(source).toMatch(
      /import \{[^}]*PROTOCOL_VERSION[^}]*\} from '\.\/trysteroTransport\.js'/,
    );
    expect(source).not.toMatch(/PROTOCOL_VERSION\s*=/);
  });
});

describe('hosted bots over a room', () => {
  const rosters: [string, MatchConfig, AgentDeps][] = [
    ['two scripted bots', coopMatch(SEED, { kind: BotKind.Scripted }), {}],
    [
      'two neural bots',
      coopMatch(SEED, { kind: BotKind.Neural }),
      { neural: (player) => new RandomAgent(7 + player) },
    ],
    [
      'a scripted and a neural bot',
      coopMatch(SEED, {
        botSlots: [
          { player: 2, kind: BotKind.Scripted },
          { player: 3, kind: BotKind.Neural },
        ],
      }),
      { neural: () => new RandomAgent(11) },
    ],
  ];

  for (const [name, config, deps] of rosters) {
    it(`carries ${name}, one per peer, through jitter and loss, agreeing throughout`, async () => {
      const net = new FakeRoomNetwork({ seed: 5, latencyMs: 60, jitterMs: 20 });
      const results = await settle(net, joinPair(net));
      const peers = makeOnlinePeers(net, config, results, deps);
      const wire = watchWire(net);
      // Loss only once the one-shot greetings are through: the real channel
      // retransmits those, the fake does not.
      net.dropRate = 0.3;

      expect(peers.map((p) => p.runner.ownedPlayers)).toEqual([
        [0, 2],
        [1, 3],
      ]);
      run(net, peers, 900);

      expectAgreementAtCommonTick(peers);
      expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
      for (const peer of peers) {
        expect(peer.runner.currentTick).toBeGreaterThan(200);
        expect(peer.runner.droppedByBudget).toBe(0);
        for (const waiting of peer.stalls) expect(waiting.every((p) => p < 2)).toBe(true);
      }
      // Each bot's commands came from its host and nobody else, addressed to
      // the paired peer and nobody else.
      const idOf = (slot: PlayerId): string =>
        results[0]!.localPlayer === slot ? 'peer-a' : 'peer-b';
      expect([...wire.sentBy.get(2)!]).toEqual([idOf(hostOf(config, 2))]);
      expect([...wire.sentBy.get(3)!]).toEqual([idOf(hostOf(config, 3))]);
      expect([...wire.targets].sort()).toEqual(['peer-a', 'peer-b']);
      expect(wire.packets).toBeGreaterThan(100);
      for (const slot of [2, 3] as PlayerId[]) {
        const host = peers.find((p) => p.result.localPlayer === hostOf(config, slot))!;
        expect(host.driver.statsFor(slot)!.issued).toBeGreaterThan(0);
      }
    });
  }

  it('drops packets from a stranger in the room, whatever slot they claim', async () => {
    const net = new FakeRoomNetwork({ seed: 9, latencyMs: 10 });
    const results = await settle(net, joinPair(net));
    const config = coopMatch(SEED, { kind: BotKind.Neural });
    const peers = makeOnlinePeers(net, config, results, { neural: () => new IdleAgent() });
    run(net, peers, 40);

    // A third member forges what a host would send: entries for the bot slot
    // the other peer hosts, on turns still to come, moving that bot's worker.
    const forger = net.join('peer-c', ROOM);
    run(net, peers, 10);
    const cmd = forger.makeAction('cmd');
    const forged: { to: string; victim: Peer; slot: PlayerId; unit: number }[] = [];
    for (const victim of peers) {
      const claimant = (1 - victim.result.localPlayer) as PlayerId;
      const slot = (claimant + 2) as PlayerId;
      const index = firstWorker(victim.sim.world, slot);
      expect(index).toBeGreaterThanOrEqual(0);
      const unit = victim.sim.world.pool.idAt(index);
      const turn = Math.floor(victim.runner.currentTick / TICKS_PER_TURN) + INPUT_DELAY_TURNS + 2;
      const packet: Packet = {
        player: claimant,
        turns: [
          {
            turn,
            player: slot,
            commands: [
              {
                type: CommandType.Move,
                player: slot,
                units: [unit],
                x: fromInt(40),
                y: fromInt(40),
              },
            ],
          },
        ],
      };
      const to = victim.result.localPlayer === results[0]!.localPlayer ? 'peer-a' : 'peer-b';
      forged.push({ to, victim, slot, unit });
      void cmd.send(packet, { target: to });
      void cmd.send(packet);
    }
    let reached = 0;
    net.onDeliver = (d) => {
      if (d.from === 'peer-c') reached++;
    };
    run(net, peers, 200);

    // The fake delivered the forgeries; the transports refused them.
    expect(reached).toBeGreaterThan(0);
    for (const { victim, slot, unit } of forged) {
      const world = victim.sim.world;
      expect(world.pool.isAlive(unit)).toBe(true);
      const index = idIndex(unit);
      expect(world.pool.owner[index]).toBe(slot);
      expect(world.pool.order[index]).toBe(Order.None);
    }
    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
  });

  it('leaves a settled pair alone when a third peer tries to join, and times that peer out', async () => {
    vi.useFakeTimers();
    try {
      const net = new FakeRoomNetwork({ seed: 3, latencyMs: 10 });
      const results = await settle(net, joinPair(net));
      const config = coopMatch(SEED);
      const peers = makeOnlinePeers(net, config, results, {});
      run(net, peers, 40);

      const third = joinOnlineRoom(
        { roomCode: ROOM, seed: SEED, mode: modeFor(BotKind.Scripted) },
        500,
        net.provider('peer-c'),
      );
      let outcome: string | null = null;
      void third.then(
        () => {
          outcome = 'settled';
        },
        (e: Error) => {
          outcome = e.message;
        },
      );
      run(net, peers, 200);
      await Promise.resolve();

      // Nobody greeted the newcomer, so it settled against nobody.
      expect(outcome).toBeNull();
      expect(pairedId(results[0]!)).toBe('peer-b');
      expect(pairedId(results[1]!)).toBe('peer-a');
      expectAgreementAtCommonTick(peers);

      vi.advanceTimersByTime(500);
      await Promise.resolve();
      expect(outcome).toMatch(/Could not connect/);
      expect(net.members(ROOM).sort()).toEqual(['peer-a', 'peer-b']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('reports the peer that left, as the runner would to the HUD', async () => {
    const net = new FakeRoomNetwork({ seed: 4, latencyMs: 10 });
    const results = await settle(net, joinPair(net));
    const peers = makeOnlinePeers(net, coopMatch(SEED), results, {});
    run(net, peers, 40);
    const leaver = peers.find((p) => p.result.localPlayer === 1)!;
    const stayer = peers.find((p) => p.result.localPlayer === 0)!;
    leaver.result.transport.close();
    run(net, [stayer], 10);
    expect(stayer.timeouts).toEqual([1]);
    expect(leaver.timeouts).toEqual([]);
  });
});

describe('chunking in the fake room', () => {
  function pair(net: FakeRoomNetwork): { received: unknown[]; send: (data: unknown) => void } {
    const x = net.join('x', ROOM);
    const y = net.join('y', ROOM);
    net.advance(100);
    const received: unknown[] = [];
    y.makeAction('blob').onMessage = (data) => {
      received.push(data);
    };
    const action = x.makeAction('blob');
    return { received, send: (data) => void action.send(data, { target: 'y' }) };
  }

  it('scrambles payloads over the chunk limit under jitter, as Trystero does', () => {
    const net = new FakeRoomNetwork({ seed: 2, latencyMs: 10, jitterMs: 20, chunking: true });
    const { received, send } = pair(net);
    const big = { values: Array.from({ length: 12000 }, (_, i) => i) };
    expect(JSON.stringify(big).length).toBeGreaterThan(2 * TRANSPORT_CHUNK_BYTES);
    for (let k = 0; k < 10; k++) send(big);
    net.advance(500);
    expect(net.scrambled).toBeGreaterThan(0);
    expect(received.length + net.scrambled).toBe(10);
  });

  it('delivers payloads under the limit intact', () => {
    const net = new FakeRoomNetwork({ seed: 2, latencyMs: 10, jitterMs: 20, chunking: true });
    const { received, send } = pair(net);
    const small = { values: Array.from({ length: 1000 }, (_, i) => i) };
    expect(JSON.stringify(small).length).toBeLessThan(TRANSPORT_CHUNK_BYTES);
    for (let k = 0; k < 10; k++) send(small);
    net.advance(500);
    expect(net.scrambled).toBe(0);
    expect(received).toHaveLength(10);
    for (const data of received) expect(data).toEqual(small);
  });
});
