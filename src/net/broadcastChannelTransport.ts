/**
 * Two-tab multiplayer over `BroadcastChannel`.
 *
 * Same machine, no network, no signalling, no NAT. This exists because it makes
 * the entire lockstep path testable in a real browser without any of WebRTC's
 * failure modes in the picture — when something desyncs here, it is a
 * simulation bug, full stop.
 *
 * It is also a genuinely useful way to play: two people at one keyboard, or a
 * developer checking a change against a second tab in seconds.
 *
 * The join protocol is deliberately tiny: each tab announces itself, and once
 * both ids are known, both derive the same slot assignment independently. There
 * is no matchmaking because there is nothing to match — both tabs are already on
 * the same machine.
 */

import type { PlayerId } from '../sim/types.js';
import { slotFromPeerIds } from './trysteroTransport.js';
import type { Packet, Transport } from './transport.js';

const CHANNEL_PREFIX = 'experiment-rts:';

type Envelope =
  | { kind: 'hello'; from: string; seed: number; mode: string; reply: boolean }
  | { kind: 'packet'; from: string; packet: Packet }
  | { kind: 'bye'; from: string };

export interface LobbyResult {
  transport: Transport;
  seed: number;
  localPlayer: PlayerId;
}

/**
 * Join (or create) a two-player room on this machine.
 *
 * Resolves once both tabs are present. Slots and the seed are both *derived*
 * from the two tab ids rather than negotiated: whoever greets whom first is a
 * race, and resolving it by "the receiver hosts" makes both tabs claim slot 0
 * whenever they open close together. See `slotFromPeerIds`.
 */
export function joinLocalRoom(
  room: string,
  seedIfHost: number,
  /**
   * Opaque identifier for what this tab wants to play, compared for equality
   * with the other tab's. Two tabs that picked different modes would generate
   * different maps from the same seed, which is a desync on tick zero — so the
   * mismatch is refused with something a person can act on instead.
   */
  mode: string,
  timeoutMs = 60000,
): Promise<LobbyResult> {
  const channel = new BroadcastChannel(CHANNEL_PREFIX + room);
  const selfId = `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;

  return new Promise((resolve, reject) => {
    let settled = false;

    const finish = (seed: number, slot: PlayerId, peer: string): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      channel.removeEventListener('message', onMessage);
      resolve({
        transport: new BroadcastChannelTransport(channel, selfId, peer, slot),
        seed,
        localPlayer: slot,
      });
    };

    const onMessage = (event: MessageEvent<Envelope>): void => {
      const msg = event.data;
      if (!msg || msg.from === selfId) return;
      if (msg.kind !== 'hello') return;

      // A tab that joined before us never saw our greeting, so answer once so
      // both sides end up knowing both ids and both seeds.
      if (msg.reply) {
        channel.postMessage({
          kind: 'hello',
          from: selfId,
          seed: seedIfHost,
          mode,
          reply: false,
        } satisfies Envelope);
      }

      if (msg.mode !== mode) {
        settled = true;
        clearTimeout(timer);
        channel.removeEventListener('message', onMessage);
        channel.close();
        reject(new Error('The other tab chose a different mode. Pick the same one in both.'));
        return;
      }

      const slot = slotFromPeerIds(selfId, msg.from);
      // Both tabs now hold both seeds and agree on who is slot 0, so both pick
      // the same one without anyone having to be "the host".
      const seed = slot === 0 ? seedIfHost : msg.seed;
      finish(seed, slot, msg.from);
    };

    channel.addEventListener('message', onMessage);
    channel.postMessage({
      kind: 'hello',
      from: selfId,
      seed: seedIfHost,
      mode,
      reply: true,
    } satisfies Envelope);

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      channel.removeEventListener('message', onMessage);
      channel.close();
      reject(new Error('no second player joined'));
    }, timeoutMs);
  });
}

class BroadcastChannelTransport implements Transport {
  readonly playerCount = 2;
  ready = true;
  private packetHandler: ((p: Packet) => void) | undefined;
  private lostHandler: ((p: PlayerId) => void) | undefined;
  private closed = false;

  constructor(
    private readonly channel: BroadcastChannel,
    private readonly selfId: string,
    private readonly peerId: string,
    readonly localPlayer: PlayerId,
  ) {
    channel.addEventListener('message', (event: MessageEvent<Envelope>) => {
      const msg = event.data;
      if (!msg || msg.from === this.selfId) return;
      if (msg.kind === 'packet') {
        this.packetHandler?.(msg.packet);
      } else if (msg.kind === 'bye' && msg.from === this.peerId) {
        this.lostHandler?.(this.localPlayer === 0 ? 1 : 0);
      }
    });

    // Best-effort goodbye so the other tab shows "player left" rather than
    // sitting on a stall until the timeout.
    window.addEventListener('pagehide', () => this.close());
  }

  send(packet: Packet): void {
    if (this.closed) return;
    this.channel.postMessage({ kind: 'packet', from: this.selfId, packet } satisfies Envelope);
  }

  onPacket(handler: (p: Packet) => void): void {
    this.packetHandler = handler;
  }

  onPeerLost(handler: (p: PlayerId) => void): void {
    this.lostHandler = handler;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.channel.postMessage({ kind: 'bye', from: this.selfId } satisfies Envelope);
      this.channel.close();
    } catch {
      // The channel may already be torn down during page unload.
    }
  }
}
