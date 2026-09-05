/**
 * The noise the model samples with is Gumbel(0, 1), made from 32-bit words:
 * if it were anything else, "argmax of logits plus noise" would not be a draw
 * from the softmax and the browser's bot would play a different policy from
 * the one that was trained.
 */

import { describe, expect, it } from 'vitest';
import { fillGumbel, gumbelFromWord } from '../src/ai/neural/noise.js';

describe('gumbel noise', () => {
  it('is finite at both ends of the word range', () => {
    expect(Number.isFinite(gumbelFromWord(0))).toBe(true);
    expect(Number.isFinite(gumbelFromWord(0xffffffff))).toBe(true);
    expect(gumbelFromWord(0)).toBeLessThan(gumbelFromWord(0xffffffff));
  });

  it('has the Gumbel mean and variance', () => {
    const n = 200000;
    const words = new Uint32Array(n);
    let x = 12345;
    for (let i = 0; i < n; i++) {
      x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
      words[i] = x;
    }
    const out = new Float32Array(n);
    fillGumbel(out, words);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += out[i]!;
    const mean = sum / n;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += (out[i]! - mean) ** 2;
    const variance = sq / n;
    expect(Math.abs(mean - 0.5772)).toBeLessThan(0.02);
    expect(Math.abs(variance - (Math.PI * Math.PI) / 6)).toBeLessThan(0.05);
  });

  it('refuses too few words', () => {
    expect(() => fillGumbel(new Float32Array(4), new Uint32Array(3))).toThrow();
  });
});
