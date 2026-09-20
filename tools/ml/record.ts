/** Record fixed validation matches: bun run tools/ml/record.ts --layout lanes --matches 4 --out ml/data/validation. */
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { MapLayout } from '../../src/sim/types.js';
import { recordTeacherMatch } from './recording.js';
import type { SlotSpec } from './env.js';

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2)
  args.set(process.argv[i]!.replace(/^--/, ''), process.argv[i + 1] ?? '');
const matches = Number(args.get('matches') ?? 4);
const out = args.get('out') ?? 'ml/data/validation';
const maxTicks = Number(args.get('maxTicks') ?? 12000);
const seed0 = Number(args.get('seed0') ?? 0x7ea1);
const mode = args.get('layout') ?? 'lanes';
if (
  !Number.isInteger(matches) ||
  matches < 1 ||
  !Number.isInteger(maxTicks) ||
  maxTicks < 4 ||
  !Number.isInteger(seed0)
)
  throw new Error(
    'matches and maxTicks must be positive integers (maxTicks >= 4), seed0 must be an integer',
  );
if (!['lanes', 'quarters', 'mix'].includes(mode))
  throw new Error('layout must be lanes, quarters, or mix');
mkdirSync(out, { recursive: true });
for (let m = 0; m < matches; m++) {
  const seed = (seed0 + m * 104729) >>> 0;
  const layout =
    mode === 'quarters' || (mode === 'mix' && m % 2 === 1) ? MapLayout.Quarters : MapLayout.Lanes;
  const count = layout === MapLayout.Quarters ? 4 : 2;
  const layoutOrdinal = mode === 'mix' ? Math.floor(m / 2) : m;
  const player = layoutOrdinal % count;
  const slots: SlotSpec[] = Array.from({ length: count }, (_, p) => ({
    kind: p === player ? 'teacher' : 'scripted',
  }));
  const name = `shard-${String(m).padStart(3, '0')}`;
  const summary = recordTeacherMatch({ seed, layout, slots, maxTicks }, join(out, name), player);
  console.log(
    `${name}: ${summary.decisions} decisions, ${summary.labels} labelled, ${summary.dropped} dropped, ${(summary.bytes / 1e6).toFixed(1)} MB`,
  );
}
