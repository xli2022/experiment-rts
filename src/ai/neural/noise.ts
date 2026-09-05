/**
 * The randomness the model samples with, made outside the model.
 *
 * Sampling lives inside the exported graph — every head is an argmax over its
 * logits plus Gumbel(0, 1) noise, which is an exact draw from its softmax —
 * and the noise is an *input*, so the graph carries no random number
 * generator and torch and onnxruntime can be compared exactly. The browser
 * fills it from `crypto.getRandomValues`, one 32-bit word per value; the
 * `+ 0.5` keeps the uniform strictly inside (0, 1), where both logarithms are
 * finite.
 */

const TWO_32 = 4294967296;

/** Gumbel(0, 1) from one uniform 32-bit word. */
export function gumbelFromWord(word: number): number {
  const u = (word + 0.5) / TWO_32;
  return -Math.log(-Math.log(u));
}

/** Fill `out` from `words`, which must be at least as long. */
export function fillGumbel(out: Float32Array, words: Uint32Array): void {
  if (words.length < out.length) throw new Error('not enough random words for the noise');
  for (let i = 0; i < out.length; i++) out[i] = gumbelFromWord(words[i]!);
}
