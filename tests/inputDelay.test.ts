/**
 * Input delay that follows the connection.
 *
 * Lockstep buys determinism with latency: a command issued now executes a fixed
 * number of turns later, and that number used to be 2 for everyone forever —
 * 200ms of lag on a LAN game, and not nearly enough on a bad link, where the
 * same constant produced stalls instead.
 *
 * It can move because the wire carries the absolute turn each command belongs
 * to. A peer files what it receives at the stated turn and never infers a
 * schedule, so the delay is a local choice that needs no agreement and cannot
 * desync. What it *can* do, if done carelessly, is leave a hole in the sequence
 * of turns a peer sends for — and a hole is a permanent deadlock, because every
 * peer waits for every player's commands before executing a turn. Hence the
 * contiguous-prefix rule these tests exist to defend.
 */

import { describe, expect, it } from 'vitest';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { CHECKSUM_INTERVAL, MS_PER_TICK, type PlayerId } from '../src/sim/types.js';
import { LocalNetwork } from '../src/net/localTransport.js';
import {
  INPUT_DELAY_TURNS,
  LockstepRunner,
  MAX_INPUT_DELAY_TURNS,
  MIN_INPUT_DELAY_TURNS,
  TICKS_PER_TURN,
} from '../src/net/lockstep.js';
import type { Packet, Transport } from '../src/net/transport.js';

const SEED = 0xbeef01;

interface Peer {
  sim: Simulation;
  runner: LockstepRunner;
  desyncs: number;
}

function makeMatch(net: LocalNetwork): Peer[] {
  const peers: Peer[] = [];
  for (let p = 0; p < net.playerCount; p++) {
    const sim = new Simulation(SEED);
    const peer: Peer = { sim, runner: null as never, desyncs: 0 };
    peer.runner = new LockstepRunner(
      sim,
      net.createTransport(p),
      {
        onDesync: () => {
          peer.desyncs++;
        },
      },
      () => net.nowMs,
    );
    peers.push(peer);
  }
  return peers;
}

/**
 * Holds inbound packets for a few frames, so one direction of the link is
 * slower than the other. `LocalNetwork` applies one latency to everything.
 */
class OneWayDelay implements Transport {
  private handler: ((p: Packet) => void) | undefined;
  private readonly queue: { packet: Packet; releaseIn: number }[] = [];

  constructor(
    private readonly inner: Transport,
    private readonly frames: number,
  ) {
    inner.onPacket((packet) => this.queue.push({ packet, releaseIn: this.frames }));
  }

  /** Advance the hold-back by one frame; call alongside `net.advance`. */
  pump(): void {
    for (let i = this.queue.length - 1; i >= 0; i--) {
      const entry = this.queue[i]!;
      if (--entry.releaseIn > 0) continue;
      this.queue.splice(i, 1);
      this.handler?.(entry.packet);
    }
  }

  get localPlayer(): PlayerId {
    return this.inner.localPlayer;
  }
  get playerCount(): number {
    return this.inner.playerCount;
  }
  get ready(): boolean {
    return this.inner.ready;
  }
  send(packet: Packet): void {
    this.inner.send(packet);
  }
  onPacket(handler: (p: Packet) => void): void {
    this.handler = handler;
  }
  onPeerLost(handler: (p: PlayerId) => void): void {
    this.inner.onPeerLost(handler);
  }
  close(): void {
    this.inner.close();
  }
}

function makeRunner(
  sim: Simulation,
  transport: Transport,
  peer: Peer,
  net: LocalNetwork,
): LockstepRunner {
  return new LockstepRunner(
    sim,
    transport,
    {
      onDesync: () => {
        peer.desyncs++;
      },
    },
    () => net.nowMs,
  );
}

function run(net: LocalNetwork, peers: Peer[], frames: number): void {
  for (let f = 0; f < frames; f++) {
    for (const peer of peers) peer.runner.update(MS_PER_TICK);
    net.advance(MS_PER_TICK);
  }
}

function expectAgreement(peers: Peer[]): void {
  const minTick = Math.min(...peers.map((p) => p.runner.currentTick));
  for (let t = minTick - (minTick % CHECKSUM_INTERVAL); t > 0; t -= CHECKSUM_INTERVAL) {
    const values = peers.map((p) => p.runner.checksumAt(t));
    if (values.some((v) => v === undefined)) continue;
    expect(values[0]).toBe(values[1]);
    return;
  }
  throw new Error(`no common checksummed tick below ${minTick}`);
}

/** Play a while on a given link and report where the delay settled. */
function settle(
  latencyMs: number,
  jitterMs = 0,
  dropRate = 0,
  frames = 900,
): { delays: number[]; peers: Peer[] } {
  const net = new LocalNetwork(2, 0x5a17);
  net.latencyMs = latencyMs;
  net.jitterMs = jitterMs;
  net.dropRate = dropRate;
  const peers = makeMatch(net);
  run(net, peers, frames);
  return { delays: peers.map((p) => p.runner.inputDelayTurns), peers };
}

describe('adapting the input delay', () => {
  it('speeds up on a fast link', () => {
    // The whole point: a LAN game should not pay 200ms of input lag.
    const { delays, peers } = settle(5);
    expect(`delays ${delays.join(',')} < start ${INPUT_DELAY_TURNS}`).toBe(
      `delays ${MIN_INPUT_DELAY_TURNS},${MIN_INPUT_DELAY_TURNS} < start ${INPUT_DELAY_TURNS}`,
    );
    expectAgreement(peers);
    expect(peers[0]!.desyncs).toBe(0);
  });

  it('backs off on a slow one', () => {
    const fast = settle(5).delays[0]!;
    const slow = settle(260, 60).delays[0]!;
    expect(`fast ${fast} < slow ${slow}`).toBe(`fast ${fast} < slow ${slow}`);
    expect(slow).toBeGreaterThan(fast);
    expect(slow).toBeGreaterThan(INPUT_DELAY_TURNS);
  });

  it('stays inside its bounds however bad the link gets', () => {
    // Unbounded growth would be worse than the stalls it avoids: half a second
    // of input lag is a broken-feeling game rather than an honest overlay.
    const { delays } = settle(600, 200, 0.3, 1500);
    for (const d of delays) {
      expect(d).toBeLessThanOrEqual(MAX_INPUT_DELAY_TURNS);
      expect(d).toBeGreaterThanOrEqual(MIN_INPUT_DELAY_TURNS);
    }
  });

  it('keeps peers in sync while the delay is moving on both sides', () => {
    // The delays change independently and at different moments; nothing about
    // the simulation may depend on that.
    const net = new LocalNetwork(2, 0x31d);
    net.latencyMs = 20;
    const peers = makeMatch(net);

    const changes = [0, 0];
    let last = peers.map((p) => p.runner.inputDelayTurns);
    for (let round = 0; round < 30; round++) {
      // Alternate between a good and a bad link so the delay is forced to move.
      net.latencyMs = round % 2 === 0 ? 10 : 280;
      net.jitterMs = round % 2 === 0 ? 0 : 50;
      run(net, peers, 60);
      const now = peers.map((p) => p.runner.inputDelayTurns);
      for (let p = 0; p < 2; p++) if (now[p] !== last[p]) changes[p]!++;
      last = now;

      for (const p of [0, 1] as PlayerId[]) {
        const pool = peers[p]!.sim.world.pool;
        const units: number[] = [];
        for (let i = 0; i < pool.count && units.length < 3; i++) {
          if (pool.alive[i] === 1 && pool.owner[i] === p) units.push(pool.idAt(i));
        }
        const cmd: Command = {
          type: CommandType.Move,
          player: p,
          units,
          x: fromInt(35 + (round % 7)),
          y: fromInt(45 + (round % 5)),
        };
        peers[p]!.runner.issue(cmd);
      }
    }

    // Non-vacuous: the delay really did move, repeatedly.
    expect(changes[0]).toBeGreaterThan(2);
    expectAgreement(peers);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
  });

  it('backs off the peer that is actually late, not the one that noticed', () => {
    // An asymmetric link is the case that broke the first design. When peer 1's
    // packets arrive late, it is *peer 1* that must schedule further ahead —
    // peer 0's own delay governs peer 0's sends and cannot help. Driving it from
    // peer 0's stalls instead parks peer 0 at the ceiling, stalling forever,
    // while peer 1 sees a roomy link and tightens further.
    const net = new LocalNetwork(2, 0x9f1);
    net.latencyMs = 15;

    const peers: Peer[] = [];
    let holdback: OneWayDelay | null = null;
    for (let p = 0; p < 2; p++) {
      const sim = new Simulation(SEED);
      const peer: Peer = { sim, runner: null as never, desyncs: 0 };
      let transport: Transport = net.createTransport(p);
      // Only what peer 0 receives is held up, so only peer 1 is genuinely late.
      if (p === 0) {
        holdback = new OneWayDelay(transport, 4);
        transport = holdback;
      }
      peer.runner = makeRunner(sim, transport, peer, net);
      peers.push(peer);
    }

    for (let f = 0; f < 1800; f++) {
      for (const peer of peers) peer.runner.update(MS_PER_TICK);
      holdback!.pump();
      net.advance(MS_PER_TICK);
    }

    const slow = peers[1]!.runner.inputDelayTurns;
    const fast = peers[0]!.runner.inputDelayTurns;
    expect(`late peer ${slow} > prompt peer ${fast}`).toBe(`late peer ${slow} > prompt peer ${fast}`);
    expect(slow).toBeGreaterThan(fast);

    expectAgreement(peers);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(600);
  });

  it('keeps both peers advancing across repeated delay changes', () => {
    // A hole in the schedule is a permanent deadlock, so sustained progress
    // through many delay changes is the coarse end-to-end version of the
    // contiguity check below.
    const net = new LocalNetwork(2, 0x77a);
    net.latencyMs = 20;
    const peers = makeMatch(net);

    for (let round = 0; round < 24; round++) {
      net.latencyMs = round % 2 === 0 ? 10 : 300;
      run(net, peers, 60);
    }

    expect(peers[0]!.runner.currentTick).toBeGreaterThan(800);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(800);
    expectAgreement(peers);
  });
});

describe('the schedule a peer puts on the wire', () => {
  it('is a contiguous run of turns, whatever the delay does', () => {
    // Read the packets directly rather than inferring from "it did not
    // deadlock": a hole only deadlocks if the run happens to reach it.
    const net = new LocalNetwork(2, 0xc0de);
    net.latencyMs = 20;

    const scheduled = new Set<number>();
    const sim = new Simulation(SEED);
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
      send(packet: Parameters<typeof inner.send>[0]) {
        for (const t of packet.turns) {
          if (t.player === inner.localPlayer) scheduled.add(t.turn);
        }
        inner.send(packet);
      },
      onPacket: inner.onPacket.bind(inner),
      onPeerLost: inner.onPeerLost.bind(inner),
      close: inner.close.bind(inner),
    };

    const a = new LockstepRunner(sim, spy, {}, () => net.nowMs);
    const other = new Simulation(SEED);
    const b = new LockstepRunner(other, net.createTransport(1), {}, () => net.nowMs);

    for (let round = 0; round < 24; round++) {
      net.latencyMs = round % 2 === 0 ? 10 : 300;
      net.jitterMs = round % 2 === 0 ? 0 : 40;
      for (let f = 0; f < 60; f++) {
        a.update(MS_PER_TICK);
        b.update(MS_PER_TICK);
        net.advance(MS_PER_TICK);
      }
    }

    const turns = [...scheduled].sort((x, y) => x - y);
    expect(turns.length).toBeGreaterThan(50);
    // Turns 0..INPUT_DELAY_TURNS-1 are seeded locally rather than sent, so the
    // sequence starts there and must then run unbroken.
    expect(turns[0]).toBe(INPUT_DELAY_TURNS);
    const holes = turns.filter((t, i) => i > 0 && t !== turns[i - 1]! + 1);
    expect(`holes after ${turns[0]}: ${holes.join(',')}`).toBe(`holes after ${turns[0]}: `);
    // And the delay really moved during the run, or this proves nothing.
    expect(a.inputDelayTurns !== INPUT_DELAY_TURNS || b.inputDelayTurns !== INPUT_DELAY_TURNS).toBe(
      true,
    );
    void TICKS_PER_TURN;
  });
});
