import { describe, expect, it } from 'vitest';
import { Rng } from '../src/sim/rng.js';

describe('seeded rng', () => {
  it('produces the same stream for the same seed', () => {
    const a = new Rng(12345);
    const b = new Rng(12345);
    for (let i = 0; i < 1000; i++) {
      expect(a.nextU32()).toBe(b.nextU32());
    }
  });

  it('produces different streams for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    let same = 0;
    for (let i = 0; i < 100; i++) {
      if (a.nextU32() === b.nextU32()) same++;
    }
    expect(same).toBeLessThan(5);
  });

  it('terminates for power-of-two bounds', () => {
    // Regression: the rejection-sampling limit was coerced with `>>> 0`, and
    // when `bound` divides 2^32 evenly the limit is exactly 2^32 — which
    // `>>> 0` turns into 0, making the rejection test always true and hanging
    // the generator forever. Map generation calls nextInt(128) immediately, so
    // this froze the whole simulation before it produced a single tick.
    for (const bound of [2, 4, 8, 16, 64, 128, 256, 1024, 65536]) {
      const rng = new Rng(99);
      for (let i = 0; i < 200; i++) {
        const v = rng.nextInt(bound);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(bound);
      }
    }
  });

  it('handles non-power-of-two bounds', () => {
    for (const bound of [3, 5, 7, 100, 1000]) {
      const rng = new Rng(7);
      for (let i = 0; i < 200; i++) {
        const v = rng.nextInt(bound);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(bound);
      }
    }
  });

  it('stays roughly uniform', () => {
    const rng = new Rng(4242);
    const buckets = new Array<number>(8).fill(0);
    const n = 80000;
    for (let i = 0; i < n; i++) buckets[rng.nextInt(8)]! += 1;
    const expected = n / 8;
    for (const b of buckets) {
      expect(Math.abs(b - expected) / expected).toBeLessThan(0.05);
    }
  });

  it('respects inclusive range bounds', () => {
    const rng = new Rng(11);
    for (let i = 0; i < 500; i++) {
      const v = rng.nextRange(3, 7);
      expect(v).toBeGreaterThanOrEqual(3);
      expect(v).toBeLessThanOrEqual(7);
    }
  });

  it('clones without disturbing the source stream', () => {
    const rng = new Rng(555);
    rng.nextU32();
    const clone = rng.clone();
    const fromClone = [clone.nextU32(), clone.nextU32()];
    const fromSource = [rng.nextU32(), rng.nextU32()];
    expect(fromClone).toEqual(fromSource);
  });
});
