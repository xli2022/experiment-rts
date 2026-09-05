/**
 * How fast the environment steps, and whether two engines see the same bytes.
 *
 *   bun run tools/ml/bench.ts --envs 4 --ticks 4000 --hash
 *   npx vite-node tools/ml/bench.ts --envs 1 --ticks 400 --hash
 *
 * Decisions per second is the number training is bounded by. The hash folds
 * every observation the policy slot receives, so the same run under Bun and
 * Node proves the codec produces identical bytes on both engines — the
 * browser is a third, and `Math.fround` on every store is what keeps them
 * together.
 */

import { checksumInit, checksumToHex, checksumU32 } from '../../src/sim/checksum.js';
import { MapLayout } from '../../src/sim/types.js';
import { MatchEnv } from './env.js';

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2)
  args.set(process.argv[i]!.replace(/^--/, ''), process.argv[i + 1] ?? '1');
const envCount = Number(args.get('envs') ?? 1);
const ticks = Number(args.get('ticks') ?? 4000);
const hashing = args.has('hash');

function foldBytes(h: number, view: ArrayBufferView): number {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  for (let i = 0; i < bytes.length; i++) h = checksumU32(h, bytes[i]!);
  return h;
}

const envs: MatchEnv[] = [];
for (let e = 0; e < envCount; e++) {
  envs.push(
    new MatchEnv({
      seed: 0x51ce7a11 + e,
      layout: MapLayout.Lanes,
      slots: [{ kind: 'policy' }, { kind: 'scripted' }],
      maxTicks: ticks,
    }),
  );
}
let hash = checksumInit();
let decisions = 0;
const noop = new Int32Array(29).fill(-1);
noop[0] = 0;
const t0 = performance.now();
for (const env of envs) {
  while (!env.done) {
    const slot = env.observe(0);
    if (hashing) {
      hash = foldBytes(hash, slot.observation.entities);
      hash = foldBytes(hash, slot.observation.grid);
      hash = foldBytes(hash, slot.observation.scalars);
      hash = foldBytes(hash, slot.masks.type);
      hash = foldBytes(hash, slot.masks.selection);
      hash = foldBytes(hash, slot.masks.buildCell);
    }
    env.step(new Map([[0, noop]]));
    decisions++;
  }
}
const seconds = (performance.now() - t0) / 1000;
console.log(
  `${decisions} decisions in ${seconds.toFixed(2)}s: ${(decisions / seconds).toFixed(0)} decisions/s`,
);
if (hashing) console.log(`observation hash ${checksumToHex(hash >>> 0)}`);
