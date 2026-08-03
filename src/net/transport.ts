/**
 * The transport seam.
 *
 * Everything above this interface — the lockstep scheduler, the simulation, the
 * UI — is identical whether the opponent is a bot in this tab, a second browser
 * tab, or a stranger across a WebRTC connection. That is the whole point: there
 * is one code path, exercised constantly in tests, rather than a well-tested
 * single-player path and a fragile multiplayer one.
 *
 * Three implementations:
 *
 *  - `LocalTransport`         in-process, for single-player and unit tests
 *  - `BroadcastChannelTransport`  two tabs on one machine, fully offline
 *  - `TrysteroTransport`      real peer-to-peer over WebRTC
 *
 * The first two exist so that lockstep bugs can be found without the network in
 * the picture. Debugging a desync and a connection problem at the same time is
 * how projects lose weeks.
 */

import type { Command } from '../sim/commands.js';
import type { PlayerId } from '../sim/types.js';

/** One player's commands for one turn. */
export interface TurnCommands {
  turn: number;
  player: PlayerId;
  commands: Command[];
}

/**
 * What actually goes over the wire.
 *
 * `turns` carries the newest turn plus a couple of older ones. Commands are
 * tiny, so repeating them is far cheaper than detecting loss and asking for a
 * resend — and it means a dropped packet usually costs nothing at all instead of
 * stalling every peer for a round trip.
 */
export interface Packet {
  player: PlayerId;
  turns: TurnCommands[];
  /** Periodic state fingerprint, for desync detection. */
  checksum?: { tick: number; value: number };
}

export interface Transport {
  /** Which player slot this peer controls. */
  readonly localPlayer: PlayerId;
  /** Total players in the match, including this one. */
  readonly playerCount: number;
  /** True once every peer is connected and the match can start. */
  readonly ready: boolean;

  send(packet: Packet): void;
  onPacket(handler: (packet: Packet) => void): void;
  /** Notified when a peer drops. */
  onPeerLost(handler: (player: PlayerId) => void): void;
  close(): void;
}
