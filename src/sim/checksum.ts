/**
 * FNV-1a 32-bit hashing, used to fingerprint the whole simulation state.
 *
 * Peers exchange a checksum every `CHECKSUM_INTERVAL` ticks. If two peers ever
 * disagree, the game has desynced and every frame after that point is fiction —
 * so we want to know within a second, not when a unit visibly teleports.
 *
 * FNV-1a is chosen because it is order-sensitive (so it also catches iteration
 * order bugs, the most common cause of desync), needs no allocation, and is a
 * handful of int32 ops per value.
 */

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/** Start a new hash accumulator. */
export function checksumInit(): number {
  return FNV_OFFSET_BASIS >>> 0;
}

/** Fold one 32-bit value into the hash. */
export function checksumU32(h: number, v: number): number {
  let x = h ^ (v & 0xff);
  x = Math.imul(x, FNV_PRIME);
  x ^= (v >>> 8) & 0xff;
  x = Math.imul(x, FNV_PRIME);
  x ^= (v >>> 16) & 0xff;
  x = Math.imul(x, FNV_PRIME);
  x ^= (v >>> 24) & 0xff;
  x = Math.imul(x, FNV_PRIME);
  return x >>> 0;
}

/**
 * Fold a slice of a typed array into the hash.
 *
 * Only the first `count` entries are read; the arrays are capacity-sized pools
 * and the tail contains stale data from freed entities that must not contribute.
 */
export function checksumArray(
  h: number,
  arr: Int32Array | Uint8Array | Uint16Array | Uint32Array,
  count: number,
): number {
  let x = h;
  for (let i = 0; i < count; i++) {
    x = checksumU32(x, arr[i]!);
  }
  return x;
}

/** Format a checksum for logs and the desync banner. */
export function checksumToHex(h: number): string {
  return (h >>> 0).toString(16).padStart(8, '0');
}
