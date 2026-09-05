/**
 * Q16.16 fixed-point arithmetic.
 *
 * ## Why this file exists
 *
 * The simulation must produce bit-identical results on every peer, on every JS
 * engine. IEEE-754 guarantees that `+ - * /` and `Math.sqrt` are correctly
 * rounded, so those are safe. But `Math.sin`, `Math.cos`, `Math.atan2`,
 * `Math.pow` and `Math.exp` are *implementation-defined* — V8, SpiderMonkey and
 * JavaScriptCore legitimately return different last bits. A simulation built on
 * those desyncs between a Chrome player and a Firefox player, and only then.
 *
 * So all simulation state is stored as Q16.16: a 32-bit signed integer where
 * `1.0` is represented by `65536`. Integer arithmetic has no such ambiguity.
 *
 * ## Where float64 is used, and why it is still exact
 *
 * A few helpers below compute intermediates as plain JS numbers. That is safe
 * *only* because the values are provably exact integers below 2^53, where
 * float64 represents every integer exactly and `+`, `-`, `*` are error-free.
 * Each such site is commented with its bound. `Math.sqrt` is additionally
 * correctly-rounded by IEEE-754, so it is deterministic too.
 *
 * Nothing here may use a transcendental function. The simulation stores facing
 * as a normalised vector rather than an angle precisely so that no trig is ever
 * needed — see `vecRotateToward`.
 */

/** A Q16.16 fixed-point value. Alias for documentation; it is an int32. */
export type Fix = number;

export const FIX_SHIFT = 16;
/** Q16.16 representation of 1.0. */
export const FIX_ONE = 1 << FIX_SHIFT; // 65536
export const FIX_HALF = FIX_ONE >> 1;
export const FIX_ZERO = 0;

/** Largest/smallest representable values, useful as sentinels. */
export const FIX_MAX = 0x7fffffff;
export const FIX_MIN = -0x80000000;

/** Exact integer -> Q16.16. `n` must satisfy |n| < 32768. */
export function fromInt(n: number): Fix {
  return (n << FIX_SHIFT) | 0;
}

/** Q16.16 -> integer, truncating toward negative infinity. */
export function toInt(a: Fix): number {
  return a >> FIX_SHIFT;
}

/**
 * Float -> Q16.16. Rounds to nearest.
 *
 * Only for authoring constants (tuning tables, map data) at load time. Never
 * call this with a value derived from simulation state.
 */
export function fromFloat(x: number): Fix {
  return Math.round(x * FIX_ONE) | 0;
}

/**
 * Q16.16 -> float. For rendering and UI only — the result must never flow back
 * into simulation state.
 */
export function toFloat(a: Fix): number {
  return a / FIX_ONE;
}

/**
 * Fixed-point multiply, truncating toward zero.
 *
 * The obvious `(a * b) >> 16` is wrong: `int32 * int32` reaches 2^62, far past
 * the 2^53 where float64 stops representing integers exactly, so it silently
 * returns the wrong answer for roughly 2.5% of operand pairs. Instead both
 * magnitudes are split into 16-bit halves whose partial products are all exact
 * integers below 2^47, and the sign is applied afterwards.
 *
 * Applied afterwards, deliberately. This used to floor, and a floor is not
 * symmetric under negation: `fmul(-a, b)` came out one below `-fmul(a, b)`
 * for every operand pair whose product was not a whole number, which is nearly
 * all of them. Every unit's per-tick step goes through here with a signed
 * direction, so the two halves of a mirrored match drifted apart by one unit
 * per axis per tick from the first step. Truncation toward zero makes
 * `fmul(-a, b) === -fmul(a, b)` exactly, and changes nothing for non-negative
 * products. Overflow still wraps to int32, as before.
 */
export function fmul(a: Fix, b: Fix): Fix {
  const negative = a < 0 !== b < 0;
  const x = a < 0 ? -a : a;
  const y = b < 0 ? -b : b;
  const xh = (x / FIX_ONE) | 0;
  const xl = x - xh * FIX_ONE;
  const yh = (y / FIX_ONE) | 0;
  const yl = y - yh * FIX_ONE;
  // xh, yh <= 2^15 and xl, yl < 2^16, so every term is exact and the sum stays
  // under 2^47.
  const magnitude = xh * yh * FIX_ONE + xh * yl + xl * yh + (((xl * yl) / FIX_ONE) | 0);
  return (negative ? -magnitude : magnitude) | 0;
}

/**
 * Fixed-point divide, truncating toward zero.
 *
 * `a * FIX_ONE` peaks at 2^31 * 2^16 = 2^47, comfortably under 2^53, so the
 * intermediate is an exact float64 integer and the division is correctly
 * rounded. Division by zero yields a saturated value rather than NaN, because
 * NaN would poison the checksum and desync every peer at once.
 */
export function fdiv(a: Fix, b: Fix): Fix {
  if (b === 0) return a < 0 ? FIX_MIN : FIX_MAX;
  return ((a * FIX_ONE) / b) | 0;
}

/**
 * Fixed-point square root.
 *
 * `a / FIX_ONE` and the final `* FIX_ONE` are exact (powers of two), and
 * `Math.sqrt` is correctly rounded by IEEE-754, so this is deterministic.
 * Negative inputs return 0 rather than NaN, for the reason given in `fdiv`.
 */
export function fsqrt(a: Fix): Fix {
  if (a <= 0) return 0;
  return (Math.sqrt(a / FIX_ONE) * FIX_ONE) | 0;
}

export function fabs(a: Fix): Fix {
  return a < 0 ? -a : a;
}

export function fmin(a: Fix, b: Fix): Fix {
  return a < b ? a : b;
}

export function fmax(a: Fix, b: Fix): Fix {
  return a > b ? a : b;
}

export function fclamp(a: Fix, lo: Fix, hi: Fix): Fix {
  return a < lo ? lo : a > hi ? hi : a;
}

export function fsign(a: Fix): number {
  return a > 0 ? 1 : a < 0 ? -1 : 0;
}

/** Linear interpolation. `t` is Q16.16 in [0, FIX_ONE]. */
export function flerp(a: Fix, b: Fix, t: Fix): Fix {
  return (a + fmul(b - a, t)) | 0;
}

// ---------------------------------------------------------------------------
// Vector helpers
//
// Vectors are passed as loose (x, y) pairs rather than objects so the hot paths
// allocate nothing. The simulation is 2D on the ground plane; terrain height is
// a rendering concern only and never affects gameplay.
// ---------------------------------------------------------------------------

/**
 * Squared length as an exact float64 integer.
 *
 * Returns a plain number, NOT a Fix — the true Q16.16 square of a map-scale
 * distance overflows int32 (a 128x128 map has a 181-unit diagonal, whose square
 * in Q16.16 is 2^31). Callers compare these raw values against each other, which
 * is exact and deterministic as long as both sides come from this function or
 * from `sqRange`.
 *
 * Bound: |dx|,|dy| < 2^23 for a 128-unit map, so dx*dx + dy*dy < 2^47 < 2^53.
 */
export function vecLenSqRaw(dx: Fix, dy: Fix): number {
  return dx * dx + dy * dy;
}

/**
 * Convert a Q16.16 range into the same raw squared space as `vecLenSqRaw`, so
 * the two can be compared without a square root.
 */
export function sqRange(r: Fix): number {
  return r * r;
}

/**
 * Vector length in Q16.16.
 *
 * `dx*dx + dy*dy` is an exact float64 integer (see `vecLenSqRaw`), and
 * `Math.sqrt` of an exact value is correctly rounded, so truncating the result
 * is deterministic across engines.
 */
export function vecLen(dx: Fix, dy: Fix): Fix {
  if (dx === 0 && dy === 0) return 0;
  return Math.sqrt(dx * dx + dy * dy) | 0;
}

/** Distance between two points, in Q16.16. */
export function vecDist(ax: Fix, ay: Fix, bx: Fix, by: Fix): Fix {
  return vecLen(bx - ax, by - ay);
}

/**
 * Scratch space for the two-component returns below. Reused to keep the hot
 * paths allocation-free; copy the values out immediately.
 *
 * This is safe for determinism — it is scratch, never simulation state, and is
 * fully written before every read.
 */
export const vecOut = { x: 0 as Fix, y: 0 as Fix };

/** Normalise a vector into `vecOut`. A zero vector stays zero. */
export function vecNormalize(dx: Fix, dy: Fix): typeof vecOut {
  const len = vecLen(dx, dy);
  if (len === 0) {
    vecOut.x = 0;
    vecOut.y = 0;
  } else {
    vecOut.x = fdiv(dx, len);
    vecOut.y = fdiv(dy, len);
  }
  return vecOut;
}

/**
 * Rotate the unit vector (fx, fy) toward the unit vector (tx, ty) by at most
 * `maxStep` (Q16.16, roughly the chord length of the turn), writing the
 * renormalised result to `vecOut`.
 *
 * This is how units turn. Doing it as a clamped lerp plus renormalise — rather
 * than by comparing angles — is what lets the whole simulation avoid trig, and
 * therefore avoid the one part of `Math` that is not portable.
 */
export function vecRotateToward(fx: Fix, fy: Fix, tx: Fix, ty: Fix, maxStep: Fix): typeof vecOut {
  const dx = tx - fx;
  const dy = ty - fy;
  const d = vecLen(dx, dy);
  if (d === 0) {
    vecOut.x = fx;
    vecOut.y = fy;
    return vecOut;
  }
  if (d <= maxStep) {
    // Close enough to snap; still normalise so callers always get a unit vector.
    return vecNormalize(tx, ty);
  }
  const t = fdiv(maxStep, d);
  return vecNormalize(fx + fmul(dx, t), fy + fmul(dy, t));
}
