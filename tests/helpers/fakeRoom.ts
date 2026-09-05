/**
 * A Trystero room with no network under it.
 *
 * `joinOnlineRoom` talks to a room through the `RoomProvider` seam; this is a
 * switchboard that stands in for Trystero, so the online transport — the
 * handshake, the slot dealing, hosted bots, a stranger in the room — runs in
 * `npm test` the way the lockstep layer runs over `LocalNetwork`. It models
 * the same things `LocalNetwork` does, on the same virtual clock: latency,
 * jitter, loss, delivery in `(time, sequence)` order so a seed reproduces a
 * run exactly. And three things Trystero has that `LocalNetwork` does not:
 *
 * - **Symmetric discovery.** Every member hears of every other, both ways,
 *   each report on its own jittered path — the race `slotFromPeerIds` exists
 *   to make harmless.
 * - **Targeted sends.** A message goes to one peer or to everyone else in the
 *   room; one to a peer that is not a member is silently dropped, as Trystero
 *   drops it.
 * - **Chunking by arrival order.** With `chunking` on, a payload over
 *   `TRANSPORT_CHUNK_BYTES` is split, each chunk jittered separately, and
 *   reassembled in the order the pieces arrive — Trystero's behaviour, and the
 *   reason `tests/wire.test.ts` keeps packets under the limit. A scramble is
 *   counted and delivers nothing.
 *
 * Loss here *drops*; the real channel is reliable and retransmits, so the
 * fake is harsher than the wire. Payloads are cloned through JSON on
 * delivery, as Trystero serialises them, so a receiver that mutates what it
 * got — `setTurn` restamps `command.player` — never touches the sender's copy.
 *
 * What it cannot model is the network itself: ICE, NAT, TURN, DTLS, SCTP.
 */

import {
  TRANSPORT_CHUNK_BYTES,
  type ActionLike,
  type MessageContextLike,
  type RoomJoinOptions,
  type RoomLike,
  type RoomProvider,
} from '../../src/net/trysteroTransport.js';
import { Rng } from '../../src/sim/rng.js';

export interface FakeRoomOptions {
  seed?: number;
  latencyMs?: number;
  jitterMs?: number;
  dropRate?: number;
  chunking?: boolean;
}

/** One message as it reaches a member, after loss and reassembly. */
export interface Delivery {
  room: string;
  from: string;
  to: string;
  action: string;
  data: unknown;
}

/** A discovery report on its way to a member. */
interface DiscoveryItem {
  kind: 'join' | 'leave';
  deliverAt: number;
  seq: number;
  room: string;
  at: string;
  about: string;
}

/** A message, or one chunk of one, on its way to a member. */
interface MessageItem {
  kind: 'message';
  deliverAt: number;
  seq: number;
  room: string;
  from: string;
  to: string;
  action: string;
  text: string;
  /** Set on a chunk of a larger payload. */
  chunk?: { id: number; total: number };
}

type Item = DiscoveryItem | MessageItem;
type Unscheduled<T> = Omit<T, 'deliverAt' | 'seq'>;

class FakeAction implements ActionLike {
  onMessage: ((data: unknown, context: MessageContextLike) => void) | null = null;

  constructor(
    private readonly net: FakeRoomNetwork,
    private readonly member: FakeRoom,
    readonly name: string,
  ) {}

  send(data: unknown, options?: { target?: string }): Promise<void> {
    this.net.submit(this.member, this.name, data, options?.target);
    return Promise.resolve();
  }
}

class FakeRoom implements RoomLike {
  onPeerJoin: ((peerId: string) => void) | null = null;
  onPeerLeave: ((peerId: string) => void) | null = null;
  readonly actions = new Map<string, FakeAction>();
  left = false;

  constructor(
    private readonly net: FakeRoomNetwork,
    readonly id: string,
    readonly room: string,
  ) {}

  makeAction(name: string): ActionLike {
    let action = this.actions.get(name);
    if (!action) {
      action = new FakeAction(this.net, this, name);
      this.actions.set(name, action);
    }
    return action;
  }

  leave(): Promise<void> {
    this.net.depart(this);
    return Promise.resolve();
  }
}

export class FakeRoomNetwork {
  /** One-way delay applied to every message and every discovery report. */
  latencyMs: number;
  /** Random extra delay in [0, jitterMs], which also reorders. */
  jitterMs: number;
  /** Fraction of messages discarded outright, in [0, 1]. Discovery is never lost. */
  dropRate: number;
  /** Split payloads over `TRANSPORT_CHUNK_BYTES` and reassemble by arrival. */
  chunking: boolean;
  /** Every message as it is delivered, after loss; for watching the wire. */
  onDeliver: ((delivery: Delivery) => void) | null = null;
  /** Multi-chunk payloads that came back together in the wrong order. */
  scrambled = 0;

  private readonly rooms = new Map<string, FakeRoom[]>();
  private readonly inFlight: Item[] = [];
  private readonly partial = new Map<string, string[]>();
  private clock = 0;
  private seq = 0;
  private nextChunkId = 1;
  private readonly rng: Rng;

  constructor(options: FakeRoomOptions = {}) {
    this.latencyMs = options.latencyMs ?? 0;
    this.jitterMs = options.jitterMs ?? 0;
    this.dropRate = options.dropRate ?? 0;
    this.chunking = options.chunking ?? false;
    this.rng = new Rng(options.seed ?? 1);
  }

  /** Current virtual time, in milliseconds. */
  get nowMs(): number {
    return this.clock;
  }

  /** What `joinOnlineRoom` takes: this participant, as Trystero would present it. */
  provider(selfId: string): RoomProvider {
    return {
      selfId,
      join: (roomId: string, _options: RoomJoinOptions) => this.join(selfId, roomId),
    };
  }

  /** Enrol a member by hand — a forger, a stray tab, a peer on an old build. */
  join(selfId: string, roomId: string): FakeRoom {
    const members = this.rooms.get(roomId) ?? [];
    if (members.some((m) => m.id === selfId)) throw new Error(`${selfId} is already in ${roomId}`);
    const member = new FakeRoom(this, selfId, roomId);
    // Both halves of discovery, each on its own path, so which side hears
    // first is a matter of jitter rather than of who joined.
    for (const other of members) {
      this.schedule({ kind: 'join', room: roomId, at: other.id, about: selfId });
      this.schedule({ kind: 'join', room: roomId, at: selfId, about: other.id });
    }
    members.push(member);
    this.rooms.set(roomId, members);
    return member;
  }

  members(roomId: string): string[] {
    return (this.rooms.get(roomId) ?? []).map((m) => m.id);
  }

  /** Messages, chunks and discovery reports still in transit. */
  get pendingCount(): number {
    return this.inFlight.length;
  }

  /** Called by a member's action when it sends. */
  submit(from: FakeRoom, action: string, data: unknown, target: string | undefined): void {
    if (from.left) return;
    const members = this.rooms.get(from.room) ?? [];
    const recipients =
      target === undefined
        ? members.filter((m) => m.id !== from.id).map((m) => m.id)
        : members.some((m) => m.id === target)
          ? [target]
          : [];
    const text = JSON.stringify(data);
    for (const to of recipients) {
      if (this.chunking && text.length > TRANSPORT_CHUNK_BYTES) {
        const id = this.nextChunkId++;
        const total = Math.ceil(text.length / TRANSPORT_CHUNK_BYTES);
        for (let k = 0; k < total; k++) {
          const piece = text.slice(k * TRANSPORT_CHUNK_BYTES, (k + 1) * TRANSPORT_CHUNK_BYTES);
          this.post(from.room, from.id, to, action, piece, { id, total });
        }
      } else {
        this.post(from.room, from.id, to, action, text);
      }
    }
  }

  private post(
    room: string,
    from: string,
    to: string,
    action: string,
    text: string,
    chunk?: { id: number; total: number },
  ): void {
    if (this.dropRate > 0 && this.rng.nextInt(10000) < this.dropRate * 10000) return;
    this.schedule({ kind: 'message', room, from, to, action, text, ...(chunk ? { chunk } : {}) });
  }

  private schedule(item: Unscheduled<DiscoveryItem> | Unscheduled<MessageItem>): void {
    const jitter = this.jitterMs > 0 ? this.rng.nextInt(this.jitterMs + 1) : 0;
    const scheduled: Item = {
      ...item,
      deliverAt: this.clock + this.latencyMs + jitter,
      seq: this.seq++,
    };
    this.inFlight.push(scheduled);
  }

  /** Called by a member when it leaves. */
  depart(member: FakeRoom): void {
    if (member.left) return;
    member.left = true;
    member.onPeerJoin = null;
    member.onPeerLeave = null;
    const members = (this.rooms.get(member.room) ?? []).filter((m) => m !== member);
    this.rooms.set(member.room, members);
    for (const other of members) {
      this.schedule({ kind: 'leave', room: member.room, at: other.id, about: member.id });
    }
  }

  /**
   * Advance virtual time and deliver everything that has come due, in
   * `(time, sequence)` order, so a seed always produces the same run.
   */
  advance(ms: number): void {
    this.clock += ms;
    const due = this.inFlight.filter((item) => item.deliverAt <= this.clock);
    if (due.length === 0) return;
    for (let i = this.inFlight.length - 1; i >= 0; i--) {
      if (this.inFlight[i]!.deliverAt <= this.clock) this.inFlight.splice(i, 1);
    }
    due.sort((a, b) => (a.deliverAt !== b.deliverAt ? a.deliverAt - b.deliverAt : a.seq - b.seq));
    for (const item of due) this.deliver(item);
  }

  private memberOf(room: string, id: string): FakeRoom | undefined {
    return (this.rooms.get(room) ?? []).find((m) => m.id === id);
  }

  private deliver(item: Item): void {
    if (item.kind !== 'message') {
      const at = this.memberOf(item.room, item.at);
      if (!at) return;
      if (item.kind === 'join') {
        if (this.memberOf(item.room, item.about)) at.onPeerJoin?.(item.about);
      } else {
        at.onPeerLeave?.(item.about);
      }
      return;
    }
    // A message already on the wire arrives even if its sender has since
    // left: the channel drains before it closes. Only the recipient must
    // still be there.
    const to = this.memberOf(item.room, item.to);
    if (!to) return;
    let text = item.text;
    if (item.chunk) {
      const key = `${item.from}>${item.to}#${item.chunk.id}`;
      const pieces = this.partial.get(key) ?? [];
      pieces.push(item.text);
      if (pieces.length < item.chunk.total) {
        this.partial.set(key, pieces);
        return;
      }
      this.partial.delete(key);
      text = pieces.join('');
    }
    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      this.scrambled++;
      return;
    }
    this.onDeliver?.({ room: item.room, from: item.from, to: item.to, action: item.action, data });
    to.actions.get(item.action)?.onMessage?.(data, { peerId: item.from });
  }
}
