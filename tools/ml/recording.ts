/** Stream one teacher match to disk without retaining every observation in RAM. */
import { closeSync, openSync, writeFileSync, writeSync } from 'node:fs';
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
import { MatchEnv, type EnvConfig } from './env.js';
import { encodeFrame, Kind } from './protocol.js';

/** `stem` names the .bin shard and its .json index; its directory must exist. */
export function recordTeacherMatch(config: EnvConfig, stem: string, player = 0) {
  if (config.slots[player]?.kind !== 'teacher') throw new Error('recorded slot must be a teacher');
  const env = new MatchEnv(config);
  let fd: number | undefined;
  let decisions = 0;
  let labels = 0;
  let dropped = 0;
  let bytes = 0;
  let previousTick = -1;
  try {
    fd = openSync(`${stem}.bin`, 'w');
    while (!env.done) {
      env.step(new Map());
      // The last decision cannot issue after a terminal/reset boundary. Match
      // the live server, which resets before returning terminal observations.
      if (env.done) break;
      const slot = env.observe(player);
      if (slot.frame.tick <= previousTick) continue;
      previousTick = slot.frame.tick;
      decisions++;
      if (slot.label[0]! > 0) labels++;
      if (slot.label[0] === -1) dropped++;
      const frame = encodeFrame(
        Kind.Obs,
        { specVersion: SPEC.version, tick: slot.frame.tick, player },
        [
          {
            name: 'entities',
            view: slot.observation.entities,
            shape: [N_ENT, ENTITY_FEATURE_COUNT],
          },
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
        ],
      );
      let offset = 0;
      while (offset < frame.length) {
        const written = writeSync(fd, frame, offset, frame.length - offset);
        if (written <= 0) throw new Error('teacher shard write made no progress');
        offset += written;
      }
      bytes += frame.length;
    }
    const summary = {
      specVersion: SPEC.version,
      seed: config.seed,
      layout: config.layout,
      player,
      decisions,
      labels,
      dropped,
      noop: decisions - labels - dropped,
      ticks: env.tick,
      bytes,
    };
    writeFileSync(`${stem}.json`, JSON.stringify(summary, null, 2) + '\n');
    return summary;
  } finally {
    if (fd !== undefined) closeSync(fd);
    env.dispose();
  }
}
