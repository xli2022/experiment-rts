/**
 * Player-slot assignment.
 *
 * This guards a bug that made every online match unplayable. Slots used to be
 * decided by "whoever sees the other peer arrive is the host, and takes slot 0".
 * Peer discovery is symmetric, so both sides observe the other joining and both
 * took slot 0. Each then blocked forever waiting on commands from a player 1
 * that did not exist — both screens showing "waiting for player 1…".
 *
 * The fix is to derive the slot from the two peer ids instead of negotiating it,
 * so the assignment cannot depend on message ordering. These tests pin that
 * property rather than the specific comparison.
 */

import { describe, expect, it } from 'vitest';
import { slotFromPeerIds } from '../src/net/trysteroTransport.js';

describe('slot assignment', () => {
  it('gives the two peers different slots', () => {
    const a = 'aaa-peer';
    const b = 'zzz-peer';
    expect(slotFromPeerIds(a, b)).not.toBe(slotFromPeerIds(b, a));
  });

  it('covers exactly slots 0 and 1', () => {
    const slots = [slotFromPeerIds('alpha', 'beta'), slotFromPeerIds('beta', 'alpha')];
    expect(slots.slice().sort()).toEqual([0, 1]);
  });

  it('does not depend on who observes whom first', () => {
    // The original bug in one assertion: both peers running the same logic, each
    // believing itself to be the one that "received" the greeting.
    const ids = ['peer-one', 'peer-two'];
    const asSeenByFirst = slotFromPeerIds(ids[0]!, ids[1]!);
    const asSeenBySecond = slotFromPeerIds(ids[1]!, ids[0]!);
    expect(asSeenByFirst + asSeenBySecond).toBe(1);
  });

  it('is stable across many random id pairs', () => {
    // Deterministic pseudo-ids, so a failure reproduces.
    let seed = 987654321;
    const nextId = (): string => {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      seed |= 0;
      return `peer-${(seed >>> 0).toString(36)}`;
    };

    for (let i = 0; i < 500; i++) {
      const a = nextId();
      const b = nextId();
      if (a === b) continue;
      const slotA = slotFromPeerIds(a, b);
      const slotB = slotFromPeerIds(b, a);
      expect(slotA).not.toBe(slotB);
      expect(slotA + slotB).toBe(1);
    }
  });

  it('is self-consistent when called repeatedly', () => {
    const a = 'stable-a';
    const b = 'stable-b';
    const first = slotFromPeerIds(a, b);
    for (let i = 0; i < 20; i++) expect(slotFromPeerIds(a, b)).toBe(first);
  });
});
