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
 */

import { joinRoom, type DataPayload, type MessageAction, type Room } from 'trystero/nostr';
import type { PlayerId } from '../sim/types.js';
import type { Packet, Transport } from './transport.js';

/** Namespaces our rooms so unrelated Trystero apps never collide with ours. */
const APP_ID = 'experiment-rts-v1';

/**
 * Bumped whenever the simulation or wire format changes incompatibly.
 *
 * Two peers running different builds would desync within seconds, and the cause
 * is invisible from inside the game. Refusing to start is far kinder than
 * letting them play two diverging matches.
 */
export const PROTOCOL_VERSION = 1;

export interface HostConfig {
  roomCode: string;
  seed: number;
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
  /** The host assigns slots; whoever created the room takes 0. */
  slot: PlayerId;
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
  const { roomCode, seed, turnConfig, onStatus } = config;

  const room: Room = joinRoom(
    turnConfig ? { appId: APP_ID, turnConfig } : { appId: APP_ID },
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
    let agreedSeed = seed;
    let peerId: string | null = null;

    const transport = new TrysteroTransport(
      room,
      (packet) => void packetAction.send(packet as unknown as DataPayload),
      () => localPlayer ?? 0,
    );
    packetAction.onMessage = (data) => transport.receive(data as unknown as Packet);

    const finish = (slot: PlayerId, s: number): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      localPlayer = slot;
      agreedSeed = s;
      transport.markReady();
      onStatus?.('Connected.');
      resolve({ transport, seed: agreedSeed, localPlayer: slot });
    };

    // A peer arrived. We were here first, so we host: we take slot 0 and tell
    // them the seed and their slot.
    room.onPeerJoin = (id: string) => {
      peerId = id;
      onStatus?.('Peer found, agreeing on the map…');
      if (localPlayer === null) {
        const hello: Handshake = { protocol: PROTOCOL_VERSION, seed, slot: 1 };
        void handshakeAction.send(hello as unknown as DataPayload, { target: id });
        finish(0, seed);
      }
    };

    // We arrived second: adopt whatever the host tells us.
    handshakeAction.onMessage = (data, context) => {
      const msg = data as unknown as Handshake;
      peerId = context.peerId;
      if (msg.protocol !== PROTOCOL_VERSION) {
        settled = true;
        clearTimeout(timer);
        room.leave();
        reject(
          new Error(
            `The other player is running a different version of the game ` +
              `(protocol ${msg.protocol}, expected ${PROTOCOL_VERSION}).`,
          ),
        );
        return;
      }
      if (localPlayer === null) finish(msg.slot, msg.seed);
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
