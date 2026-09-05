/**
 * Environments for Python, over stdio.
 *
 *   bun run tools/ml/serve.ts
 *
 * Python sends `Reset` with a list of environment configs and then `Step`
 * frames carrying one decision per policy slot per environment; each is
 * answered with an `Obs` frame carrying every observed slot's observation,
 * masks, critic view, label and reward, plus per-environment status in the
 * header. An environment that finishes is reset on the next seed before its
 * observation is sent, so the stream never stalls on a finished match.
 *
 * The frame layout is `protocol.ts`; the Python side is `rtsml/env.py`.
 */

import {
  ACTION_INTS,
  N_ENT,
  CRITIC_LEN,
  GRID,
  GRID_CHANNEL_COUNT,
  SCALAR_COUNT,
  ENTITY_FEATURE_COUNT,
  ACTION_TYPE_COUNT,
  SPEC,
} from '../../src/ai/neural/spec.js';
import { ENTITY_TYPE_COUNT } from '../../src/sim/types.js';
import { MatchEnv, type EnvConfig } from './env.js';
import { decodeFrame, encodeFrame, FrameParser, Kind } from './protocol.js';

interface ResetHeader {
  envs: EnvConfig[];
}

interface StepHeader {
  /** Actions: i32 [envs × maxObserved × ACTION_INTS]; unused entries ignored. */
  maxObserved: number;
}

const envs: MatchEnv[] = [];

function observationFrame(rewards: Float32Array[], status: Record<string, unknown>[]): Uint8Array {
  const arrays: { name: string; view: ArrayBufferView; shape: number[] }[] = [];
  const perSlot: Record<string, unknown>[] = [];
  envs.forEach((env, e) => {
    env.observed.forEach((player) => {
      const slot = env.observe(player);
      perSlot.push({ env: e, player });
      const o = slot.observation;
      const m = slot.masks;
      arrays.push(
        { name: 'entities', view: o.entities, shape: [N_ENT, ENTITY_FEATURE_COUNT] },
        { name: 'entity_mask', view: o.entityMask, shape: [N_ENT] },
        { name: 'grid', view: o.grid, shape: [GRID_CHANNEL_COUNT, GRID, GRID] },
        { name: 'scalars', view: o.scalars, shape: [SCALAR_COUNT] },
        { name: 'mask_type', view: m.type, shape: [ACTION_TYPE_COUNT] },
        { name: 'mask_selection', view: m.selection, shape: [ACTION_TYPE_COUNT, N_ENT] },
        { name: 'mask_target', view: m.target, shape: [ACTION_TYPE_COUNT, N_ENT] },
        { name: 'mask_cell', view: m.cell, shape: [ACTION_TYPE_COUNT, GRID * GRID] },
        { name: 'mask_build_cell', view: m.buildCell, shape: [ENTITY_TYPE_COUNT, GRID * GRID] },
        { name: 'mask_row_entity_type', view: m.rowEntityType, shape: [N_ENT, ENTITY_TYPE_COUNT] },
        { name: 'mask_build_type', view: m.buildType, shape: [ENTITY_TYPE_COUNT] },
        { name: 'critic', view: slot.critic, shape: [CRITIC_LEN] },
        { name: 'label', view: slot.label, shape: [ACTION_INTS] },
      );
    });
    arrays.push({ name: 'reward', view: rewards[e]!, shape: [rewards[e]!.length] });
  });
  return encodeFrame(Kind.Obs, { specVersion: SPEC.version, slots: perSlot, envs: status }, arrays);
}

function write(bytes: Uint8Array): void {
  // A single write per frame; Bun's stdout is a stream and will not interleave.
  process.stdout.write(bytes);
}

function fail(message: string): void {
  write(encodeFrame(Kind.Error, { message }));
}

function handle(bytes: Uint8Array): boolean {
  const frame = decodeFrame<Record<string, unknown>>(bytes);
  switch (frame.kind) {
    case Kind.Hello:
      write(encodeFrame(Kind.Hello, { specVersion: SPEC.version, spec: SPEC }));
      return true;
    case Kind.Reset: {
      const header = frame.header as unknown as ResetHeader;
      for (const env of envs) env.dispose();
      envs.length = 0;
      for (const config of header.envs) envs.push(new MatchEnv(config));
      const rewards = envs.map((env) => new Float32Array(env.observed.length));
      const status = envs.map((env) => ({
        tick: env.tick,
        done: false,
        truncated: false,
        winner: -1,
        reset: true,
      }));
      write(observationFrame(rewards, status));
      return true;
    }
    case Kind.Step: {
      const header = frame.header as unknown as StepHeader;
      const actions = frame.arrays[0] as Int32Array | undefined;
      if (!actions || actions.length !== envs.length * header.maxObserved * ACTION_INTS) {
        fail(
          `expected ${envs.length * header.maxObserved * ACTION_INTS} action ints, got ${actions?.length ?? 0}`,
        );
        return true;
      }
      const rewards: Float32Array[] = [];
      const status: Record<string, unknown>[] = [];
      envs.forEach((env, e) => {
        const map = new Map<number, ArrayLike<number>>();
        env.observed.forEach((player, k) => {
          const at = (e * header.maxObserved + k) * ACTION_INTS;
          map.set(player, actions.subarray(at, at + ACTION_INTS));
        });
        const result = env.step(map);
        rewards.push(result.rewards);
        status.push({
          tick: result.tick,
          done: result.done,
          truncated: result.truncated,
          winner: result.winner,
          issued: [...result.issued],
          reset: false,
        });
        if (result.done) {
          env.reset();
          status[e]!.reset = true;
        }
      });
      write(observationFrame(rewards, status));
      return true;
    }
    case Kind.Close:
      for (const env of envs) env.dispose();
      return false;
    default:
      fail(`unexpected frame kind ${frame.kind}`);
      return true;
  }
}

async function main(): Promise<void> {
  const parser = new FrameParser();
  for await (const chunk of process.stdin) {
    const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk as ArrayBuffer);
    for (const frame of parser.push(bytes)) {
      let more = true;
      try {
        more = handle(frame);
      } catch (error) {
        fail(error instanceof Error ? error.message : String(error));
      }
      if (!more) return;
    }
  }
}

void main();
