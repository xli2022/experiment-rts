import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { joinLocalRoom, type LobbyResult } from '../src/net/broadcastChannelTransport.js';
import { JOIN_ABANDONED } from '../src/net/trysteroTransport.js';

/** BroadcastChannel messages are queued and copied for every live recipient. */
class FakeChannel extends EventTarget {
  static readonly channels = new Set<FakeChannel>();
  closed = false;

  constructor(readonly name: string) {
    super();
    FakeChannel.channels.add(this);
  }

  postMessage(data: unknown): void {
    if (this.closed) throw new Error('closed channel');
    for (const recipient of FakeChannel.channels) {
      if (recipient === this || recipient.name !== this.name) continue;
      const copy = structuredClone(data);
      void Promise.resolve().then(() => {
        if (!recipient.closed) recipient.dispatchEvent(new MessageEvent('message', { data: copy }));
      });
    }
  }

  close(): void {
    this.closed = true;
    FakeChannel.channels.delete(this);
  }
}

async function flush(): Promise<void> {
  for (let pass = 0; pass < 20; pass++) await Promise.resolve();
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('BroadcastChannel', FakeChannel);
  vi.stubGlobal('window', new EventTarget());
  let nextId = 0;
  vi.spyOn(Math, 'random').mockImplementation(() => ++nextId / 1000);
});

afterEach(() => {
  for (const channel of FakeChannel.channels) channel.close();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('two-tab handshake', () => {
  it('chooses complementary slots and one seed, and delivers packets only to the pair', async () => {
    const joins = [joinLocalRoom('r', 100, 'm'), joinLocalRoom('r', 200, 'm')];
    await flush();
    const [a, b] = await Promise.all(joins);
    expect([a.localPlayer, b.localPlayer].sort()).toEqual([0, 1]);
    expect(a.seed).toBe(b.seed);
    expect(a.seed).toBe(a.localPlayer === 0 ? 100 : 200);
    const packets = vi.fn();
    const lost = vi.fn();
    b.transport.onPacket(packets);
    b.transport.onPeerLost(lost);
    const packet = { player: a.localPlayer, turns: [] };
    a.transport.send(packet);
    await flush();
    expect(packets).toHaveBeenCalledExactlyOnceWith(packet);
    a.transport.close();
    await flush();
    expect(lost).toHaveBeenCalledExactlyOnceWith(a.localPlayer);
  });

  it('never resolves mismatched pairings when three tabs greet at once', async () => {
    const results: LobbyResult[] = [];
    const joins = [100, 200, 300].map((seed) =>
      joinLocalRoom('r', seed, 'm', undefined, 1000).then(
        (result) => {
          results.push(result);
        },
        () => undefined,
      ),
    );
    await flush();
    vi.advanceTimersByTime(1000);
    await Promise.all(joins);
    // A reservation cycle can time everyone out, but nobody may start a match
    // against a tab that reserved someone else.
    expect([0, 2]).toContain(results.length);
    const paired = results.map(
      (result) => result.transport as unknown as { selfId: string; peerId: string },
    );
    for (const peer of paired) {
      expect(paired.find((other) => other.selfId === peer.peerId)?.peerId).toBe(peer.selfId);
    }
  });

  it('closes an abandoned channel so another attempt cannot pair with it', async () => {
    const controller = new AbortController();
    const abandoned = joinLocalRoom('r', 100, 'm', controller.signal);
    controller.abort();
    await expect(abandoned).rejects.toThrow(JOIN_ABANDONED);
    expect(FakeChannel.channels.size).toBe(0);
    const next = joinLocalRoom('r', 100, 'm', undefined, 1000);
    await flush();
    vi.advanceTimersByTime(1000);
    await expect(next).rejects.toThrow('no second player joined');
  });

  it('refuses different modes on both tabs', async () => {
    const joins = [joinLocalRoom('r', 100, 'a'), joinLocalRoom('r', 200, 'b')].map((join) =>
      join.then(
        () => 'resolved',
        (error: Error) => error.message,
      ),
    );
    await flush();
    expect(await Promise.all(joins)).toEqual([
      expect.stringContaining('different mode'),
      expect.stringContaining('different mode'),
    ]);
    expect(FakeChannel.channels.size).toBe(0);
  });
});
