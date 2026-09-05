/**
 * A small fixed set of teacher decisions, for validation.
 *
 *   npm run ml:record -- --matches 4 --out ml/data/validation
 *
 * Imitation streams its training data straight from the environment; this
 * writes a few matches' worth of (observation, masks, label) to disk so that
 * a validation score means the same thing from one run to the next. One
 * shard per match: raw arrays back to back, described by a JSON index.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { MapLayout } from '../../src/sim/types.js';
import { MatchEnv } from './env.js';
import { encodeFrame, Kind } from './protocol.js';
import {
  ACTION_INTS,
  ACTION_TYPE_COUNT,
  CRITIC_LEN,
  ENTITY_FEATURE_COUNT,
  GRID,
  GRID_CHANNEL_COUNT,
  N_ENT,
  SCALAR_COUNT,
  SPEC,
} from '../../src/ai/neural/spec.js';
import { ENTITY_TYPE_COUNT } from '../../src/sim/types.js';

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2)
  args.set(process.argv[i]!.replace(/^--/, ''), process.argv[i + 1] ?? '');
const matches = Number(args.get('matches') ?? 4);
const out = args.get('out') ?? 'ml/data/validation';
const maxTicks = Number(args.get('maxTicks') ?? 12000);
mkdirSync(out, { recursive: true });

for (let m = 0; m < matches; m++) {
  const seed = (0x7ea1 + m * 104729) >>> 0;
  const layout = m % 2 === 0 ? MapLayout.Lanes : MapLayout.Quarters;
  const slots =
    layout === MapLayout.Lanes
      ? [{ kind: 'teacher' as const }, { kind: 'scripted' as const }]
      : [
          { kind: 'teacher' as const },
          { kind: 'scripted' as const },
          { kind: 'scripted' as const },
          { kind: 'scripted' as const },
        ];
  const env = new MatchEnv({ seed, layout, slots, maxTicks });
  const frames: Uint8Array[] = [];
  let labels = 0;
  while (!env.done) {
    const slot = env.observe(0);
    if (slot.label[0] !== 0) labels++;
    frames.push(
      encodeFrame(Kind.Obs, { tick: env.tick }, [
        { name: 'entities', view: slot.observation.entities, shape: [N_ENT, ENTITY_FEATURE_COUNT] },
        { name: 'entity_mask', view: slot.observation.entityMask, shape: [N_ENT] },
        { name: 'grid', view: slot.observation.grid, shape: [GRID_CHANNEL_COUNT, GRID, GRID] },
        { name: 'scalars', view: slot.observation.scalars, shape: [SCALAR_COUNT] },
        { name: 'mask_type', view: slot.masks.type, shape: [ACTION_TYPE_COUNT] },
        { name: 'mask_selection', view: slot.masks.selection, shape: [ACTION_TYPE_COUNT, N_ENT] },
        { name: 'mask_target', view: slot.masks.target, shape: [ACTION_TYPE_COUNT, N_ENT] },
        { name: 'mask_cell', view: slot.masks.cell, shape: [ACTION_TYPE_COUNT, GRID * GRID] },
        {
          name: 'mask_build_cell',
          view: slot.masks.buildCell,
          shape: [ENTITY_TYPE_COUNT, GRID * GRID],
        },
        {
          name: 'mask_row_entity_type',
          view: slot.masks.rowEntityType,
          shape: [N_ENT, ENTITY_TYPE_COUNT],
        },
        { name: 'mask_build_type', view: slot.masks.buildType, shape: [ENTITY_TYPE_COUNT] },
        { name: 'critic', view: slot.critic, shape: [CRITIC_LEN] },
        { name: 'label', view: slot.label, shape: [ACTION_INTS] },
      ]),
    );
    env.step(new Map());
  }
  const total = frames.reduce((n, f) => n + f.length, 0);
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const f of frames) {
    bytes.set(f, at);
    at += f.length;
  }
  const name = `shard-${String(m).padStart(3, '0')}`;
  writeFileSync(join(out, `${name}.bin`), bytes);
  writeFileSync(
    join(out, `${name}.json`),
    JSON.stringify(
      {
        specVersion: SPEC.version,
        seed,
        layout,
        decisions: frames.length,
        labels,
        ticks: env.tick,
      },
      null,
      2,
    ),
  );
  console.log(
    `${name}: ${frames.length} decisions, ${labels} labelled, ${(total / 1e6).toFixed(1)} MB`,
  );
  env.dispose();
}
