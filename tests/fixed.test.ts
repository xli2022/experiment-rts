import { describe, expect, it } from 'vitest';
import {
  FIX_ONE,
  fdiv,
  fmul,
  fromFloat,
  fromInt,
  fsqrt,
  toFloat,
  toInt,
  vecLen,
  vecNormalize,
  vecRotateToward,
} from '../src/sim/fixed.js';

/** Exact reference multiply via BigInt, used to prove `fmul` is not approximate. */
function refMul(a: number, b: number): number {
  return Number(BigInt.asIntN(32, (BigInt(a) * BigInt(b)) >> 16n));
}

describe('fixed-point multiply', () => {
  it('matches exact BigInt arithmetic across the int32 range', () => {
    // Deterministic sample set including the extremes where the naive
    // `(a * b) >> 16` implementation silently loses precision past 2^53.
    const values: number[] = [
      0,
      1,
      -1,
      FIX_ONE,
      -FIX_ONE,
      2,
      -2,
      65535,
      -65535,
      0x7fffffff,
      -0x80000000,
      100000,
      -99999,
      1 << 20,
      -(1 << 20),
    ];
    let seed = 12345;
    for (let i = 0; i < 120; i++) {
      // xorshift so the sample set is fixed rather than Math.random-dependent.
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      seed |= 0;
      values.push(seed);
    }

    for (const a of values) {
      for (const b of values) {
        expect(fmul(a, b)).toBe(refMul(a, b));
      }
    }
  });

  it('treats FIX_ONE as the multiplicative identity', () => {
    for (const v of [0, 1, -1, 12345, -98765, 1 << 20]) {
      expect(fmul(v, FIX_ONE)).toBe(v);
    }
  });
});

describe('fixed-point divide', () => {
  it('round-trips against multiply within one ulp', () => {
    const pairs: [number, number][] = [
      [fromInt(10), fromInt(4)],
      [fromInt(-7), fromInt(3)],
      [fromFloat(1.5), fromFloat(0.25)],
      [fromFloat(-0.125), fromFloat(2)],
    ];
    for (const [a, b] of pairs) {
      const q = fdiv(a, b);
      expect(Math.abs(fmul(q, b) - a)).toBeLessThanOrEqual(2);
    }
  });

  it('saturates instead of returning NaN on divide by zero', () => {
    // NaN would poison the checksum and desync every peer simultaneously.
    expect(Number.isFinite(fdiv(fromInt(5), 0))).toBe(true);
    expect(Number.isFinite(fdiv(fromInt(-5), 0))).toBe(true);
  });
});

describe('fixed-point sqrt', () => {
  it('computes exact roots for perfect squares', () => {
    expect(fsqrt(fromInt(4))).toBe(fromInt(2));
    expect(fsqrt(fromInt(9))).toBe(fromInt(3));
    expect(fsqrt(fromInt(144))).toBe(fromInt(12));
  });

  it('returns zero for non-positive input rather than NaN', () => {
    expect(fsqrt(0)).toBe(0);
    expect(fsqrt(fromInt(-5))).toBe(0);
  });
});

describe('conversions', () => {
  it('round-trips integers', () => {
    for (const n of [0, 1, -1, 127, -128, 1000, -1000]) {
      expect(toInt(fromInt(n))).toBe(n);
    }
  });

  it('round-trips floats within fixed-point resolution', () => {
    for (const f of [0.5, -0.25, 3.75, -12.125]) {
      expect(Math.abs(toFloat(fromFloat(f)) - f)).toBeLessThan(1 / FIX_ONE);
    }
  });
});

describe('vectors', () => {
  it('measures length of a 3-4-5 triangle exactly', () => {
    expect(vecLen(fromInt(3), fromInt(4))).toBe(fromInt(5));
  });

  it('normalises to approximately unit length', () => {
    const v = vecNormalize(fromInt(3), fromInt(4));
    const len = vecLen(v.x, v.y);
    expect(Math.abs(len - FIX_ONE)).toBeLessThan(64); // ~0.001
  });

  it('leaves a zero vector at zero', () => {
    const v = vecNormalize(0, 0);
    expect(v.x).toBe(0);
    expect(v.y).toBe(0);
  });

  it('rotates toward a target without using trigonometry', () => {
    // Facing +X, turning toward +Y: after enough steps it should arrive, and
    // every intermediate value should stay near unit length.
    let fx = FIX_ONE;
    let fy = 0;
    for (let i = 0; i < 40; i++) {
      const r = vecRotateToward(fx, fy, 0, FIX_ONE, fromFloat(0.2));
      fx = r.x;
      fy = r.y;
      const len = vecLen(fx, fy);
      expect(Math.abs(len - FIX_ONE)).toBeLessThan(256);
    }
    expect(Math.abs(fy - FIX_ONE)).toBeLessThan(1024);
    expect(Math.abs(fx)).toBeLessThan(1024);
  });
});
