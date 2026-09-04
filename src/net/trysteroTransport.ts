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
    override createDataChannel(
      label: string,
      options?: RTCDataChannelInit,
    ): RTCDataChannel {
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
 */
export const PROTOCOL_VERSION = 4;

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
}

export interface JoinResult {
  transport: Transport;
  seed: number;
  localPlayer: PlayerId;
}

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
export function joinOnlineRoom(config: HostConfig, timeoutMs = 90000): Promise<JoinResult> {
  const { roomCode, seed, mode, turnConfig, onStatus } = config;

  const rtcPolyfill = unorderedPeerConnection();
  const room: Room = joinRoom(
    {
      appId: APP_ID,
      ...(turnConfig ? { turnConfig } : {}),
      ...(rtcPolyfill ? { rtcPolyfill } : {}),
    },
    roomCode.trim().toLowerCase(),
  );

  // `makeAction` returns an object whose `onMessage` is assigned, and whose
  // `send` takes an options object for targeting a specific peer.
  //
  // Trystero types payloads as an index-signature JSON object. Our command and
  // handshake types are perfectly good JSON but are declared as interfaces, and
  // TypeScript will not structurally match those against an index signature, so
  // the payload is cast at this boundary. Nothing is lost: Trystero serialises
  // the value as JSON either way.
  const handshakeAction = room.makeAction('hello') as MessageAction;
  const packetAction = room.makeAction('cmd') as MessageAction;

  onStatus?.('Waiting for the other player…');

  return new Promise((resolve, reject) => {
    let settled = false;
    let localPlayer: PlayerId | null = null;
    const agreedSeed = seed;
    let peerId: string | null = null;

    const transport = new TrysteroTransport(
      room,
      (packet) => void packetAction.send(packet as unknown as DataPayload),
      () => localPlayer ?? 0,
    );
    packetAction.onMessage = (data) => transport.receive(data as unknown as Packet);

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
      const slot = slotFromPeerIds(selfId, remoteId);
      settled = true;
      clearTimeout(timer);
      localPlayer = slot;
      transport.markReady();
      onStatus?.('Connected.');
      resolve({ transport, seed: agreedSeed, localPlayer: slot });
    };

    const refuse = (message: string): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      room.leave();
      reject(new Error(message));
    };

    // Sent once, to whichever peer we learn about first. Peer discovery is
    // symmetric but its two halves are not ordered, so the greeting has to go
    // out from whichever side of that race we happen to be on: if their hello
    // reaches us before our own join callback fires, and we only ever greeted
    // from the callback, neither peer would ever hear from the other.
    let greeted = false;
    const greet = (id: string): void => {
      if (greeted) return;
      greeted = true;
      const hello: Handshake = { protocol: PROTOCOL_VERSION, seed, mode };
      void handshakeAction.send(hello as unknown as DataPayload, { target: id });
    };

    room.onPeerJoin = (id: string) => {
      onStatus?.('Peer found, agreeing on the map…');
      greet(id);
    };

    handshakeAction.onMessage = (data, context) => {
      greet(context.peerId);
      const msg = data as unknown as Handshake;
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
      room.leave();
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

  constructor(
    private readonly room: Room,
    private readonly sendAction: (packet: Packet) => void,
    private readonly slot: () => PlayerId,
  ) {}

  get localPlayer(): PlayerId {
    return this.slot();
  }

  markReady(): void {
    this.ready = true;
  }

  send(packet: Packet): void {
    if (this.closed || !this.ready) return;
    this.sendAction(packet);
  }

  receive(packet: Packet): void {
    if (this.closed) return;
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
      this.room.leave();
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
