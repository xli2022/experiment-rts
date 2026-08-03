/**
 * Prints simulation checksums at fixed checkpoints.
 *
 * Run under two different JavaScript engines and compare the output: if the
 * simulation is genuinely portable, the numbers are identical. This is the
 * cheapest real proof available that the fixed-point layer works, because Node
 * runs on V8 and Bun runs on JavaScriptCore — two independent implementations of
 * `Math`, and the exact place where a simulation built on `Math.sin` or naive
 * float arithmetic would come apart.
 *
 *   npm run determinism:node
 *   npm run determinism:bun
 *   npm run determinism:cross    # runs both and diffs them
 *
 * A mismatch here means a Chrome player and a Safari player would desync — a bug
 * that is nearly impossible to reproduce once it reaches real users.
 */

import { checksumToHex } from '../src/sim/checksum.js';
import { recordMatch } from '../tests/helpers/scripted.js';

const SEED = 0x1234abcd;
const TICKS = 1500;
const CHECKPOINTS = [1, 100, 250, 500, 750, 1000, 1250, 1500];

const { checksums } = recordMatch(SEED, TICKS);

const lines: string[] = [`seed=${SEED.toString(16)} ticks=${TICKS}`];
for (const t of CHECKPOINTS) {
  const c = checksums[t - 1];
  if (c === undefined) continue;
  lines.push(`tick ${String(t).padStart(5)}  ${checksumToHex(c)}`);
}

console.log(lines.join('\n'));
