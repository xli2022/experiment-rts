/**
 * In-process transport.
 *
 * Connects several `LockstepRunner`s inside one JavaScript context. It backs
 * two things:
 *
 *  - **Single-player.** A one-peer network where the only other participant is
 *    the AI. Because the AI is a deterministic command source, single-player is
 *    literally the multiplayer code path with a different transport, so it stays
 *    exercised rather than rotting.
 *  - **Netcode tests.** With `latencyMs`, `jitterMs` and `dropRate`, a test can
 *    reproduce the conditions that break lockstep — late packets, lost packets,
 *    reordered packets — deterministically and in milliseconds.
 *
 * Time is explicit (`advance`) rather than read from a clock, so tests are
 * repeatable and do not sleep.
 */

import { Rng } from '../sim/rng.js';
import type { PlayerId } from '../sim/types.js';
import type { Packet, Transport } from './transport.js';

interface InFlight {
  deliverAt: number;
  to: PlayerId;
  packet: Packet;
  /** Tiebreaker so equal-time deliveries keep a stable order. */
  seq: number;
}

export class LocalNetwork {
  /** One-way delay applied to every packet. */
  latencyMs = 0;
  /** Random extra delay in [0, jitterMs], which also reorders packets. */
  jitterMs = 0;
  /** Fraction of packets discarded outright, in [0, 1]. */
  dropRate = 0;

  private readonly peers: (LocalTransport | undefined)[] = [];
  private readonly inFlight: InFlight[] = [];
  private clock = 0;
  private seq = 0;
  private readonly rng: Rng;

  constructor(
    readonly playerCount: number,
    seed = 1,
  ) {
    this.rng = new Rng(seed);
  }

  /** Current virtual time, in milliseconds. */
  get nowMs(): number {
    return this.clock;
  }

  createTransport(player: PlayerId): LocalTransport {
    const t = new LocalTransport(this, player, this.playerCount);
    this.peers[player] = t;
    return t;
  }

  /** Called by a transport when its runner sends. */
  submit(from: PlayerId, packet: Packet): void {
    for (let to = 0; to < this.playerCount; to++) {
      if (to === from) continue; // never loop a packet back to its sender
      if (this.dropRate > 0 && this.rng.nextInt(10000) < this.dropRate * 10000) {
        continue; // dropped in transit
      }
      const jitter = this.jitterMs > 0 ? this.rng.nextInt(this.jitterMs + 1) : 0;
      this.inFlight.push({
        deliverAt: this.clock + this.latencyMs + jitter,
        to,
        packet,
        seq: this.seq++,
      });
    }
  }

  /**
   * Advance virtual time and deliver everything that has come due.
   *
   * Delivery is sorted by (time, sequence) so that a given network
   * configuration always produces the same arrival order — otherwise a test
   * that passes once could fail on a rerun for reasons unrelated to the code.
   */
  advance(ms: number): void {
    this.clock += ms;
    const due = this.inFlight.filter((p) => p.deliverAt <= this.clock);
    if (due.length === 0) return;

    for (let i = this.inFlight.length - 1; i >= 0; i--) {
      if (this.inFlight[i]!.deliverAt <= this.clock) this.inFlight.splice(i, 1);
    }

    due.sort((a, b) => (a.deliverAt !== b.deliverAt ? a.deliverAt - b.deliverAt : a.seq - b.seq));
    for (const item of due) {
      this.peers[item.to]?.deliver(item.packet);
    }
  }

  /** Packets still in transit. Useful for asserting a test drained cleanly. */
  get pendingCount(): number {
    return this.inFlight.length;
  }
}

export class LocalTransport implements Transport {
  ready = true;
  private packetHandler: ((p: Packet) => void) | undefined;
  private lostHandler: ((p: PlayerId) => void) | undefined;
  private closed = false;

  constructor(
    private readonly network: LocalNetwork,
    readonly localPlayer: PlayerId,
    readonly playerCount: number,
  ) {}

  send(packet: Packet): void {
    if (this.closed) return;
    this.network.submit(this.localPlayer, packet);
  }

  onPacket(handler: (p: Packet) => void): void {
    this.packetHandler = handler;
  }

  onPeerLost(handler: (p: PlayerId) => void): void {
    this.lostHandler = handler;
  }

  /** Called by the network when a packet arrives. */
  deliver(packet: Packet): void {
    if (this.closed) return;
    this.packetHandler?.(packet);
  }

  /** Simulate a peer vanishing, for testing disconnect handling. */
  reportLost(player: PlayerId): void {
    this.lostHandler?.(player);
  }

  close(): void {
    this.closed = true;
  }
}

/**
 * Single-player transport: no peers, nothing sent anywhere.
 *
 * Declares a player count of 1 so the scheduler never waits on anyone. The AI
 * opponent does not need a slot here — it runs inside the simulation on every
 * peer, so it costs no bandwidth and needs no transport.
 */
export class SoloTransport implements Transport {
  readonly localPlayer: PlayerId = 0;
  readonly playerCount = 1;
  readonly ready = true;

  send(): void {
    /* nowhere to send */
  }
  onPacket(): void {
    /* nothing ever arrives */
  }
  onPeerLost(): void {
    /* no peers to lose */
  }
  close(): void {
    /* nothing to close */
  }
}
