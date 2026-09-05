/**
 * The two properties that let the data channel run unordered.
 *
 * A WebRTC data channel defaults to ordered delivery, which is head-of-line
 * blocking by another name: one lost packet holds back every packet behind it.
 * That is precisely wrong for lockstep, because each packet already repeats the
 * previous turns' commands — the packet being held back is usually the one
 * carrying what the receiver is waiting for. So the channel is opened with
 * `ordered: false`.
 *
 * Two things have to hold for that to be safe, and neither is obvious from
 * reading the transport:
 *
 * 1. **The receiver must not care what order packets arrive in.** It does not —
 *    turns are keyed absolutely and first-write-wins — but that is a property of
 *    the lockstep layer that nothing else was pinning.
 * 2. **A packet must fit in one Trystero chunk.** Trystero splits larger
 *    payloads and reassembles them by *arrival order*, with no sequence number,
 *    so a multi-chunk message would be scrambled by the very reordering we just
 *    asked for.
 */

import { describe, expect, it } from 'vitest';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { CHECKSUM_INTERVAL, MS_PER_TICK, type PlayerId } from '../src/sim/types.js';
import { LocalNetwork } from '../src/net/localTransport.js';
import { LockstepRunner } from '../src/net/lockstep.js';
import { MAX_SELECTION } from '../src/input/selection.js';
import { TRANSPORT_CHUNK_BYTES } from '../src/net/trysteroTransport.js';
import type { Packet, Transport } from '../src/net/transport.js';

const SEED = 0xbeef01;

/** One packet in four arrives late. */
const LATE_EVERY = 4;

/**
 * How many packets overtake a late one.
 *
 * Chosen to exceed the redundancy window, though it turns out not to matter —
 * see the note on `out-of-order delivery` below for what was measured.
 */
const DISPLACE_BY = 4;

/**
 * Delivers most packets immediately and a few of them late.
 *
 * This is what unordered SCTP actually does, and the distinction matters: it
 * does *not* hold packets back to reorder them. Batching arrivals and releasing
 * them in reverse — the obvious model — adds a whole window of latency to every
 * packet, which stalls lockstep for reasons that have nothing to do with order.
 * Here a late packet is overtaken by its successors while everything else
 * arrives on time, so mean added delay stays at one packet.
 */
class ReorderingTransport implements Transport {
  private handler: ((p: Packet) => void) | undefined;
  private readonly held: { packet: Packet; releaseAfter: number }[] = [];
  private received = 0;
  private delivered = 0;

  constructor(
    private readonly inner: Transport,
    private readonly displaceBy: number,
  ) {
    inner.onPacket((packet) => {
      if (++this.received % LATE_EVERY === 0) {
        this.held.push({ packet, releaseAfter: this.delivered + this.displaceBy });
        return;
      }
      this.handler?.(packet);
      this.delivered++;
      for (let i = this.held.length - 1; i >= 0; i--) {
        const entry = this.held[i]!;
        if (entry.releaseAfter > this.delivered) continue;
        this.held.splice(i, 1);
        this.handler?.(entry.packet);
        this.delivered++;
      }
    });
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

interface Peer {
  sim: Simulation;
  runner: LockstepRunner;
  desyncs: number;
}

function makeMatch(net: LocalNetwork, wrap: (t: Transport) => Transport): Peer[] {
  const peers: Peer[] = [];
  for (let p = 0; p < net.playerCount; p++) {
    const sim = new Simulation(SEED);
    const peer: Peer = { sim, runner: null as never, desyncs: 0 };
    peer.runner = new LockstepRunner(
      sim,
      wrap(net.createTransport(p)),
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

function run(net: LocalNetwork, peers: Peer[], frames: number): void {
  for (let f = 0; f < frames; f++) {
    for (const peer of peers) peer.runner.update(MS_PER_TICK);
    net.advance(MS_PER_TICK);
  }
}

/** Newest checksummed tick both peers have reached, and their values there. */
function commonChecksum(peers: Peer[]): number[] {
  const minTick = Math.min(...peers.map((p) => p.runner.currentTick));
  for (let t = minTick - (minTick % CHECKSUM_INTERVAL); t > 0; t -= CHECKSUM_INTERVAL) {
    const values = peers.map((p) => p.runner.checksumAt(t));
    if (values.some((v) => v === undefined)) continue;
    return values as number[];
  }
  throw new Error(`no common checksummed tick below ${minTick}`);
}

/** Some of a player's own units, as ids. */
function unitsOf(sim: Simulation, player: PlayerId, count: number): number[] {
  const pool = sim.world.pool;
  const ids: number[] = [];
  for (let i = 0; i < pool.count && ids.length < count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === player) ids.push(pool.idAt(i));
  }
  return ids;
}

/** Play a match, issuing real orders throughout, and report where it ended up. */
function playMatch(wrap: (t: Transport) => Transport): {
  checksums: number[];
  tick: number;
  desyncs: number;
} {
  const net = new LocalNetwork(2, 4242);
  net.latencyMs = 60;
  const peers = makeMatch(net, wrap);

  run(net, peers, 40);
  for (let round = 0; round < 8; round++) {
    for (const p of [0, 1] as PlayerId[]) {
      const units = unitsOf(peers[p]!.sim, p, 3);
      if (units.length === 0) continue;
      const cmd: Command = {
        type: CommandType.Move,
        player: p,
        units,
        x: fromInt(30 + round * 3),
        y: fromInt(40 + round * 2),
      };
      peers[p]!.runner.issue(cmd);
    }
    run(net, peers, 60);
  }

  return {
    checksums: commonChecksum(peers),
    tick: Math.min(...peers.map((p) => p.runner.currentTick)),
    desyncs: peers[0]!.desyncs + peers[1]!.desyncs,
  };
}

/**
 * These pin a property that, when measured, turns out to be over-determined.
 *
 * An "assume packets arrive in order" bug was injected into the scheduler to
 * check these tests were not vacuous. They did not catch it — and the reason is
 * worth knowing rather than tuning around. A turn's commands ride in three
 * consecutive packets, so a late packet is always covered by an on-time
 * successor; discarding it costs nothing. Reordering only does real work once
 * loss has removed the covering copies, and at 25% loss the injected bug shows
 * up merely as slower progress (1168 ticks against 1188), which is too soft a
 * signal to assert on.
 *
 * So the safety of unordered delivery does not rest on these tests. It rests on
 * the receive path being structurally order-free — turns keyed absolutely,
 * first write wins — with the redundancy as a second, independent reason.
 * These pin the end-to-end consequence of both; `packets fit in one chunk`
 * below is the test that guards something genuinely fragile.
 */
describe('out-of-order delivery', () => {
  it('leaves two peers in agreement', () => {
    const swapped = playMatch((t) => new ReorderingTransport(t, DISPLACE_BY));
    expect(swapped.checksums[0]).toBe(swapped.checksums[1]);
    expect(swapped.desyncs).toBe(0);
    expect(swapped.tick).toBeGreaterThan(300);
  });

  it('produces the identical match an in-order network does', () => {
    // Stronger than "both peers agree": reordering must not change the game at
    // all. Agreement alone would still hold if both peers dropped the same
    // command.
    const ordered = playMatch((t) => t);
    const swapped = playMatch((t) => new ReorderingTransport(t, DISPLACE_BY));
    expect(`${swapped.checksums[0]} @ tick ${swapped.tick}`).toBe(
      `${ordered.checksums[0]} @ tick ${ordered.tick}`,
    );
  });

  it('still agrees when reordering is piled on top of loss and jitter', () => {
    const net = new LocalNetwork(2, 31337);
    net.latencyMs = 70;
    net.jitterMs = 40;
    net.dropRate = 0.25;
    const peers = makeMatch(net, (t) => new ReorderingTransport(t, DISPLACE_BY));

    run(net, peers, 1200);

    const values = commonChecksum(peers);
    expect(values[0]).toBe(values[1]);
    expect(peers[0]!.desyncs + peers[1]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(200);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(200);
  });
});

describe('packets fit in one chunk', () => {
  /** What Trystero actually puts on the wire for a packet. */
  function wireBytes(packet: Packet): number {
    return new TextEncoder().encode(JSON.stringify(packet)).byteLength;
  }

  /** The largest command the UI can produce: a full selection, ordered to move. */
  function fullSelectionMove(): Command {
    const units: number[] = [];
    // Worst case ids: high slot, high generation, so every one serialises long.
    for (let i = 0; i < MAX_SELECTION; i++) units.push((((2047 - i) << 16) | 0xffff) >>> 0);
    return { type: CommandType.Move, player: 1, units, x: -2147483648, y: -2147483648 };
  }

  /** A packet carrying `perTurn` worst-case commands in each of its three turns. */
  function worstPacket(perTurn: number): Packet {
    return {
      player: 1,
      turns: [0, 1, 2].map((t) => ({
        turn: 99999 + t,
        player: 1 as PlayerId,
        commands: Array.from({ length: perTurn }, fullSelectionMove),
      })),
      checksum: { tick: 999999, value: 0xffffffff },
    };
  }

  it('fits a hard burst of full-selection orders with room to spare', () => {
    // Four commands per 100ms turn is already beyond human clicking. If this
    // ever fails, unordered delivery is no longer safe — a packet split across
    // chunks is reassembled by arrival order and would be scrambled.
    const bytes = wireBytes(worstPacket(4));
    expect(
      `${bytes} bytes vs ${TRANSPORT_CHUNK_BYTES} limit: ${bytes < TRANSPORT_CHUNK_BYTES}`,
    ).toBe(`${bytes} bytes vs ${TRANSPORT_CHUNK_BYTES} limit: true`);
    // And not merely fitting: an order of magnitude of headroom.
    expect(bytes * 4).toBeLessThan(TRANSPORT_CHUNK_BYTES);
  });

  it('reports the burst size that would actually overflow', () => {
    // Documents the real ceiling rather than asserting a number nobody checked.
    // Human input cannot approach it; the guard is against a future change to
    // MAX_SELECTION, the packet shape, or how many turns each packet repeats.
    let perTurn = 1;
    while (wireBytes(worstPacket(perTurn)) <= TRANSPORT_CHUNK_BYTES) perTurn++;
    expect(perTurn).toBeGreaterThan(16);
  });

  it('is bounded by a selection cap the UI enforces', () => {
    // The only producer of wire commands is local human input: bot commands are
    // generated inside the simulation on every peer and never sent. So the cap
    // on selection size is the cap on packet size.
    expect(MAX_SELECTION).toBeLessThanOrEqual(64);
  });
});
