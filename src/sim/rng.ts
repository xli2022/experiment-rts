/**
 * Seeded pseudo-random number generator (mulberry32).
 *
 * This is the *only* source of randomness in the simulation. `Math.random` is
 * banned inside `src/sim/**` and enforced by `tests/sealed-sim.test.ts`, because
 * two peers calling it would immediately diverge.
 *
 * The generator state is a single uint32, which means it is trivially part of
 * the world checksum: if two peers' RNG streams ever drift apart, the desync
 * detector catches it on the next checksum exchange rather than three minutes
 * later when the results become visible.
 *
 * Every operation is `Math.imul` / shift / xor on 32-bit integers, all of which
 * are exactly specified by ECMAScript, so the stream is identical everywhere.
 */

export class Rng {
  /** Current generator state. Part of the world checksum. */
  state: number;

  constructor(seed: number) {
    // Force to uint32. A zero seed is fine for mulberry32.
    this.state = seed >>> 0;
  }

  /** Next raw uint32 in the stream. */
  nextU32(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  /**
   * Uniform integer in [0, bound). Uses rejection sampling so the distribution
   * is exactly uniform — a plain modulo would bias low values, which matters
   * for things like scatter patterns and AI decisions.
   */
  nextInt(bound: number): number {
    if (bound <= 1) return 0;
    // Kept as a plain number, deliberately NOT coerced with `>>> 0`: when
    // `bound` divides 2^32 evenly the limit *is* 2^32, and `2**32 >>> 0` is 0,
    // which would make the rejection test below always true and loop forever.
    const limit = 0x100000000 - (0x100000000 % bound);
    let v = this.nextU32();
    // Every peer rejects the same values in the same order, so the retry count
    // varies without affecting determinism.
    while (v >= limit) v = this.nextU32();
    return v % bound;
  }

  /** Uniform integer in [lo, hi], inclusive. */
  nextRange(lo: number, hi: number): number {
    return lo + this.nextInt(hi - lo + 1);
  }

  /** Deterministic copy, for speculative evaluation that must not disturb the stream. */
  clone(): Rng {
    return new Rng(this.state);
  }
}
