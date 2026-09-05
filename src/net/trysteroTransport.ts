/**
 * Peer-to-peer multiplayer over WebRTC, with no server of our own.
 *
 * ## How this works without infrastructure
 *
 * WebRTC cannot bootstrap itself: two browsers must exchange session
 * descriptions before they can talk directly. Trystero handles that by
 * piggybacking on public infrastructure that already exists — by default the
 * Nostr relay network, which has many independent relays and needs no account,
 * no key, and nothing deployed. Once the handshake completes, the relays are out
 * of the picture entirely and game traffic flows directly between the two
 * browsers, end-to-end encrypted.
 *
 * ## The limitation, stated honestly
 *
 * Some networks — symmetric NAT on both ends, roughly one connection in ten —
 * cannot establish a direct peer connection at all. Fixing that requires a TURN
 * relay to forward the traffic, which is the one piece that genuinely cannot be
 * borrowed for free at scale. `turnConfig` below accepts credentials if the
 * player has them; otherwise the lobby says plainly that the connection failed
 * rather than spinning forever.
 *
 * ## Why lockstep suits this
 *
 * Only commands cross the wire, so a match uses a trickle of bandwidth no matter
 * how large the armies get. That matters far more on a peer connection between
 * two home broadband lines than it would against a datacentre.
 *
 * ## The channel is unordered on purpose
 *
 * A WebRTC data channel runs SCTP over DTLS over UDP, and defaults to *ordered*
 * delivery — which means head-of-line blocking, exactly like TCP. A lost packet
 * holds back every packet behind it until it has been retransmitted.
 *
 * That default fights the lockstep layer rather than helping it. Each packet
 * already repeats the previous two turns' commands, so the packet queued behind
 * a lost one usually carries the very commands the receiver is waiting for.
 * Ordered delivery holds that packet in the receive buffer, turning a loss the
 * protocol was built to absorb into a full round-trip stall for *both* players
 * — a peer only advances when everyone's commands have arrived.
 *
 * So the channel is opened unordered. Reliability is deliberately kept: see
 * `unorderedPeerConnection` for why dropping that too would be a step backwards.
 */

import { joinRoom, selfId, type DataPayload, type MessageAction, type Room } from 'trystero/nostr';
import type { PlayerId } from '../sim/types.js';
import type { Packet, Transport } from './transport.js';

/** Namespaces our rooms so unrelated Trystero apps never collide with ours. */
const APP_ID = 'experiment-rts-v1';

/**
 * Payload bytes Trystero fits into one data-channel message.
 *
 * 16 KiB less its 36-byte framing header. This is not a tuning knob — it is the
 * threshold above which Trystero splits a message into chunks, and chunking is
 * what makes unordered delivery unsafe. See `TRANSPORT_CHUNK_BYTES` usage in
 * `tests/wire.test.ts`.
 */
export const TRANSPORT_CHUNK_BYTES = 16 * 1024 - 36;

/**
 * Open the data channel unordered, leaving retransmission on.
 *
 * Trystero creates the channel itself and takes no options for it, but it does
 * accept a replacement `RTCPeerConnection` class — so the options are injected
 * by overriding the one method that opens it. The channel's reliability is
 * negotiated by whichever peer creates it and then applies to both, and both
 * peers run this code, so it does not matter which of them is the initiator.
 *
 * **Unordered, but still reliable.** Going further to `maxRetransmits: 0` is
 * tempting and would be worse on both counts that matter:
 *
 * - Trystero's version handshake rides this same channel and is sent exactly
 *   once. Losing it would let two peers on incompatible builds start a match
 *   and desync, which is the failure this transport works hardest to prevent.
 * - When all three redundant copies of a turn are lost, recovery falls to the
 *   lockstep layer's history resend, which is throttled to 120ms. SCTP
 *   retransmits in one round trip and without stalling anything else. Keeping
 *   reliability makes the rare case faster, not slower.
 *
 * The bandwidth this costs is nil: packets are under a kilobyte, ten a second.
 *
 * Created lazily because `RTCPeerConnection` does not exist outside a browser
 * and this module is imported by tests that only want the pure helpers.
 */
function unorderedPeerConnection(): typeof RTCPeerConnection | undefined {
  if (typeof RTCPeerConnection === 'undefined') return undefined;
  return class extends RTCPeerConnection {
    override createDataChannel(label: string, options?: RTCDataChannelInit): RTCDataChannel {
      return super.createDataChannel(label, { ...options, ordered: false });
    }
  };
}

/**
 * Bumped whenever the simulation or wire format changes incompatibly.
 *
 * Two peers running different builds would desync within seconds, and the cause
 * is invisible from inside the game. Refusing to start is far kinder than
 * letting them play two diverging matches.
 *
 * 2: packets carry `peerHeadroom`, the feedback that lets each peer size its own
 *    input delay. An older peer never sends it, so a newer one would sit at the
 *    starting delay forever while the older one adapted against silence.
 * 3: attack wind-up is checksummed simulation state. A v2 peer would still
 *    apply Slicebot damage immediately and diverge on the first melee attack.
 * 4: the handshake carries the match mode, and is now *waited for* rather than
 *    raced. Before this a peer resolved the moment it saw the other join, so
 *    the version check it is here to perform could not actually stop a match
 *    starting — it only reported afterwards.
 * 5: elimination destroys what a defeated player still owns. `pool.alive`, the
 *    generation counters and the free list are all checksummed, so a v4 peer
 *    would leave a conceded army standing while a v5 peer razes it, and the two
 *    diverge on the first player to go out.
 * 6: bots are players hosted by a peer, and their commands travel on the wire.
 *    A v5 peer runs its bots inside the simulation and never sends for their
 *    slots, so a v6 peer would wait forever on a turn that is never coming —
 *    and the roster field the mode string compares was renamed besides.
 */
export const PROTOCOL_VERSION = 6;

export interface HostConfig {
  roomCode: string;
  seed: number;
  /**
   * Opaque identifier for what this peer wants to play.
   *
   * The transport never interprets it; it only checks that both sides sent the
   * same string. Which map, how many players and how hard the AI plays all
   * change what the simulation computes, so two peers that disagree would
   * desync on the first tick — and a lobby is the one place that can still be
   * honest about it.
   */
  mode: string;
  /** Optional TURN servers, for players behind symmetric NAT. */
  turnConfig?: RTCIceServer[];
  onStatus?: (message: string) => void;
  /**
   * Abandon the attempt and leave the room.
   *
   * A lobby that only navigates away leaves this room joined: the promise stays
   * live for the whole timeout, and a peer arriving inside that window still
   * completes a handshake and starts a match nobody is waiting for.
   */
  signal?: AbortSignal;
}

export interface JoinResult {
  transport: Transport;
  seed: number;
  localPlayer: PlayerId;
}

/**
 * The slice of a Trystero room this module uses, as a seam.
 *
 * Trystero needs a browser, WebRTC and a relay to reach anyone, none of which a
 * test has. `joinOnlineRoom` therefore takes a `RoomProvider`, and the default
 * one wraps Trystero; `tests/helpers/fakeRoom.ts` provides a switchboard on a
 * virtual clock, so the handshake, the slot dealing, hosted bots and a room
 * with a stranger in it can all be run in `npm test`. What the fake cannot
 * model is the network itself — ICE, NAT, TURN, SCTP retransmission — which is
 * why a real link still gets a manual soak before a release.
 */
export interface MessageContextLike {
  readonly peerId: string;
}

export interface ActionLike {
  send(data: unknown, options?: { target?: string }): Promise<void> | void;
  onMessage: ((data: unknown, context: MessageContextLike) => void) | null;
}

export interface RoomLike {
  makeAction(name: string): ActionLike;
  onPeerJoin: ((peerId: string) => void) | null;
  onPeerLeave: ((peerId: string) => void) | null;
  leave(): Promise<void> | void;
}

export interface RoomJoinOptions {
  turnConfig?: RTCIceServer[];
  rtcPolyfill?: typeof RTCPeerConnection;
}

export interface RoomProvider {
  /**
   * This participant's id. Trystero's is one constant per page, which is all
   * a browser needs; a fake supplies one per peer, since two peers sharing an
   * id would both derive the same slot.
   */
  readonly selfId: string;
  join(roomId: string, options: RoomJoinOptions): RoomLike;
}

/** Trystero itself, behind the seam. */
export function trysteroRoomProvider(): RoomProvider {
  return {
    selfId,
    join(roomId, options) {
      const room: Room = joinRoom(
        {
          appId: APP_ID,
          ...(options.turnConfig ? { turnConfig: options.turnConfig } : {}),
          ...(options.rtcPolyfill ? { rtcPolyfill: options.rtcPolyfill } : {}),
        },
        roomId,
      );
      return adaptRoom(room);
    },
  };
}

/**
 * Trystero types payloads as an index-signature JSON object. Our command and
 * handshake types are perfectly good JSON but are declared as interfaces, and
 * TypeScript will not structurally match those against an index signature —
 * nor a typed handler property against the seam's `unknown` one — so the
 * casts live here, at the one boundary, and nowhere else. Nothing is lost:
 * Trystero serialises the value as JSON either way.
 */
function adaptRoom(room: Room): RoomLike {
  return {
    makeAction(name: string): ActionLike {
      const action = room.makeAction(name) as MessageAction;
      let handler: ActionLike['onMessage'] = null;
      return {
        send: (data, options) =>
          action.send(
            data as unknown as DataPayload,
            options?.target === undefined ? undefined : { target: options.target },
          ),
        get onMessage() {
          return handler;
        },
        set onMessage(next) {
          handler = next;
          action.onMessage =
            next === null ? null : (data, context) => next(data, { peerId: context.peerId });
        },
      };
    },
    get onPeerJoin() {
      return room.onPeerJoin;
    },
    set onPeerJoin(next) {
      room.onPeerJoin = next;
    },
    get onPeerLeave() {
      return room.onPeerLeave;
    },
    set onPeerLeave(next) {
      room.onPeerLeave = next;
    },
    leave: () => room.leave(),
  };
}

/**
 * A send or a leave that fails has nothing to tell us: the peer is gone, and
 * `onPeerLeave` is the report of that. Leaving the rejection unhandled would
 * only surface as an unhandled-rejection error in the console.
 */
function swallow(result: Promise<void> | void): void {
  void Promise.resolve(result).catch(() => undefined);
}

/**
 * Rejection message for a join the caller abandoned.
 *
 * A sentinel rather than a subclass so both transports can share it and a
 * caller can tell "you cancelled this" from "this failed", which is the
 * difference between saying nothing and showing an error nobody caused.
 */
export const JOIN_ABANDONED = 'join abandoned';

interface Handshake {
  protocol: number;
  seed: number;
  mode: string;
}

/**
 * Decide which player slot we take, from the two peer ids alone.
 *
 * This has to be derived rather than negotiated. The obvious "whoever sees the
 * other arrive is the host" rule looks fine and is broken: peer discovery is
 * symmetric, so *both* sides observe the other joining and both claim slot 0.
 * Each then sits waiting forever for commands from a player 1 that does not
 * exist — which is exactly how it failed, with both screens showing
 * "waiting for player 1".
 *
 * Comparing the two ids gives both peers the same answer with no messages, no
 * ordering assumptions, and no race.
 */
export function slotFromPeerIds(localId: string, remoteId: string): PlayerId {
  return localId < remoteId ? 0 : 1;
}

/**
 * Create or join a room and wait for the other player.
 *
 * Both sides call this. Whoever is already in the room when the other arrives
 * acts as host and owns the seed, so neither side needs to know in advance which
 * role it will play — the same code path serves "host" and "join", and the room
 * code is the only thing a player has to share.
 */
export function joinOnlineRoom(
  config: HostConfig,
  timeoutMs = 90000,
  provider: RoomProvider = trysteroRoomProvider(),
): Promise<JoinResult> {
  const { roomCode, seed, mode, turnConfig, onStatus, signal } = config;

  if (signal?.aborted) return Promise.reject(new Error(JOIN_ABANDONED));

  const rtcPolyfill = unorderedPeerConnection();
  const room = provider.join(roomCode.trim().toLowerCase(), {
    ...(turnConfig ? { turnConfig } : {}),
    ...(rtcPolyfill ? { rtcPolyfill } : {}),
  });

  // Two actions: the one-shot greeting, and the packets. Both are addressed
  // to a peer, never broadcast to the room.
  const handshakeAction = room.makeAction('hello');
  const packetAction = room.makeAction('cmd');

  onStatus?.('Waiting for the other player…');

  return new Promise((resolve, reject) => {
    let settled = false;
    let localPlayer: PlayerId | null = null;
    const agreedSeed = seed;
    let peerId: string | null = null;

    // Packets go to the settled peer and are taken only from it. A room can
    // hold more than two (see the greeting below), and the runner's own guard
    // trusts the `player` a packet claims — it can only check that the claimed
    // player hosts the slots the packet fills — so a stranger in the room
    // could otherwise fill a slot the other peer hosts, and the lockstep
    // would apply it first-write-wins. The transport pins the sender instead:
    // once the handshake has named the peer, that id is the wire.
    const transport = new TrysteroTransport(
      room,
      (packet, target) => swallow(packetAction.send(packet, { target })),
      () => localPlayer ?? 0,
    );
    packetAction.onMessage = (data, context) => transport.receive(data as Packet, context.peerId);

    /**
     * Settle on a slot once the other peer's handshake has arrived.
     *
     * Deliberately *not* called from the join callback. Resolving there was a
     * turn faster and made both the version and mode checks decorative: the
     * match had already begun by the time the mismatch was discovered, so the
     * only thing either check could do was report a desync it was meant to
     * prevent. One extra message is a cheap price for the guarantee.
     */
    const resolveWith = (remoteId: string): void => {
      if (settled) return;
      peerId = remoteId;
      const slot = slotFromPeerIds(provider.selfId, remoteId);
      settled = true;
      clearTimeout(timer);
      localPlayer = slot;
      transport.markReady(remoteId);
      onStatus?.('Connected.');
      resolve({ transport, seed: agreedSeed, localPlayer: slot });
    };

    const refuse = (message: string): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      swallow(room.leave());
      reject(new Error(message));
    };

    signal?.addEventListener('abort', () => refuse(JOIN_ABANDONED), { once: true });

    // Sent once *per peer*. Peer discovery is symmetric but its two halves are
    // not ordered, so the greeting has to go out from whichever side of that
    // race we happen to be on: if their hello reaches us before our own join
    // callback fires, and we only ever greeted from the callback, neither peer
    // would ever hear from the other.
    //
    // Per peer, not once overall. Now that resolution waits for a handshake,
    // a peer that never receives one waits out the whole timeout — so greeting
    // only the first id we hear about strands the second. A room can hold more
    // than two: a player with the page open in a spare tab, a reload whose old
    // peer has not been reaped yet, or two pairs that landed on the same code.
    // Greet whoever we meet; the extra message is a few dozen bytes and the
    // first handshake back is still what settles the slot.
    //
    // Until settled, that is. A greeting from a peer that already has its
    // match is a promise it cannot keep: the newcomer would settle against
    // it, start a match, and wait forever for packets that go only to the
    // settled pair. Left ungreeted, it times out with an honest message.
    const greeted = new Set<string>();
    const greet = (id: string): void => {
      if (settled || greeted.has(id)) return;
      greeted.add(id);
      const hello: Handshake = { protocol: PROTOCOL_VERSION, seed, mode };
      swallow(handshakeAction.send(hello, { target: id }));
    };

    room.onPeerJoin = (id: string) => {
      onStatus?.('Peer found, agreeing on the map…');
      greet(id);
    };

    handshakeAction.onMessage = (data, context) => {
      greet(context.peerId);
      const msg = data as Handshake;
      if (msg.protocol !== PROTOCOL_VERSION) {
        refuse(
          `The other player is running a different version of the game ` +
            `(protocol ${msg.protocol}, expected ${PROTOCOL_VERSION}).`,
        );
        return;
      }
      if (msg.mode !== mode) {
        refuse(
          `You and the other player chose different modes. Both of you need to ` +
            `pick the same one and enter the same code.`,
        );
        return;
      }
      resolveWith(context.peerId);
    };

    room.onPeerLeave = (id: string) => {
      if (id === peerId) transport.reportLost();
    };

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      swallow(room.leave());
      reject(
        new Error(
          'Could not connect. Either nobody joined with that code, or both ' +
            'networks block direct peer connections — the latter needs a TURN ' +
            'server to work around.',
        ),
      );
    }, timeoutMs);
  });
}

class TrysteroTransport implements Transport {
  readonly playerCount = 2;
  ready = false;

  private packetHandler: ((p: Packet) => void) | undefined;
  private lostHandler: ((p: PlayerId) => void) | undefined;
  private closed = false;
  /** The peer this transport is paired with, once the handshake has named it. */
  private peerId: string | null = null;

  constructor(
    private readonly room: RoomLike,
    private readonly sendTo: (packet: Packet, target: string) => void,
    private readonly slot: () => PlayerId,
  ) {}

  get localPlayer(): PlayerId {
    return this.slot();
  }

  /** The paired peer's id, for tests that check who a packet went to. */
  get pairedPeerId(): string | null {
    return this.peerId;
  }

  markReady(peerId: string): void {
    this.peerId = peerId;
    this.ready = true;
  }

  send(packet: Packet): void {
    if (this.closed || this.peerId === null) return;
    this.sendTo(packet, this.peerId);
  }

  /** A packet from anyone but the paired peer is dropped, whatever it claims. */
  receive(packet: Packet, from: string): void {
    if (this.closed || from !== this.peerId) return;
    this.packetHandler?.(packet);
  }

  onPacket(handler: (p: Packet) => void): void {
    this.packetHandler = handler;
  }

  onPeerLost(handler: (p: PlayerId) => void): void {
    this.lostHandler = handler;
  }

  reportLost(): void {
    this.lostHandler?.(this.localPlayer === 0 ? 1 : 0);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      swallow(this.room.leave());
    } catch {
      // Leaving a room that is already torn down is not an error worth raising.
    }
  }
}

/** Short, unambiguous room codes: no vowels (so no accidental words), no 0/O/1/I. */
const CODE_ALPHABET = '23456789BCDFGHJKMNPQRSTVWXZ';

export function generateRoomCode(random: () => number = Math.random): string {
  let out = '';
  for (let i = 0; i < 5; i++) {
    out += CODE_ALPHABET[Math.floor(random() * CODE_ALPHABET.length)];
  }
  return out;
}

/** Derive a match seed from a room code so both peers agree even if packets race. */
export function seedFromRoomCode(code: string): number {
  let h = 0x811c9dc5;
  const norm = code.trim().toLowerCase();
  for (let i = 0; i < norm.length; i++) {
    h ^= norm.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
