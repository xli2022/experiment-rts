/**
 * A bot that plays legal nonsense.
 *
 * It sees through the same codec the neural bot does and answers with a
 * uniformly random legal decision, so it exercises the whole hosted path —
 * observation, masks, decoding, the driver, the wire — before any model
 * exists, and in tests that need a neural-kind slot to actually do things.
 * Seeded, so a test that uses it is reproducible; not a pure function of the
 * world, so it is never a scripted stand-in.
 */

import type { Command } from '../../sim/commands.js';
import { Rng } from '../../sim/rng.js';
import { ENTITY_TYPE_COUNT, type PlayerId } from '../../sim/types.js';
import type { World } from '../../sim/world.js';
import { Visibility } from '../../vision/visibility.js';
import type { Agent } from '../agent.js';
import {
  allocAction,
  allocMasks,
  computeMasks,
  decode,
  selectsMany,
  selectsOne,
  usesEntityType,
  usesLocation,
  usesTarget,
  type Action,
  type Masks,
} from './actions.js';
import { allocFrame, type Frame } from './frame.js';
import { EntityMemory } from './memory.js';
import {
  allocObservation,
  NO_RECENT,
  ObservationEncoder,
  type Observation,
} from './observation.js';
import {
  ACTION_TYPE_COUNT,
  ActionType,
  DECISION_TICKS,
  GRID,
  N_ENT,
  SELECTION_MAX,
  SUB,
} from './spec.js';

const CELLS = GRID * GRID;

export class RandomAgent implements Agent {
  private readonly rng: Rng;
  private vis: Visibility | null = null;
  private mem: EntityMemory | null = null;
  private encoder: ObservationEncoder | null = null;
  private readonly obs: Observation = allocObservation();
  private readonly masks: Masks = allocMasks();
  private readonly frame: Frame = allocFrame();
  private readonly action: Action = allocAction();
  /** Decisions that were not a Noop, for tests. */
  decisions = 0;

  constructor(seed = 1) {
    this.rng = new Rng(seed);
  }

  act(world: World, player: PlayerId): Command[] {
    if (!this.vis || !this.encoder || !this.mem) {
      this.vis = new Visibility(world.map);
      this.mem = new EntityMemory(player);
      this.encoder = new ObservationEncoder(world, player);
    }
    this.vis.update(world, player);
    this.mem.update(world, this.vis);
    if (world.tick % DECISION_TICKS !== 0) return [];

    this.encoder.encode(this.vis, this.mem, NO_RECENT, this.obs, this.frame);
    computeMasks(world, this.frame, this.vis, this.mem, this.masks);
    sampleUniform(this.masks, this.rng, this.action);
    const command = decode(this.action, world, this.frame);
    if (command === null) return [];
    this.decisions++;
    return [command];
  }
}

function pick(rng: Rng, mask: Uint8Array, offset: number, length: number): number {
  let n = 0;
  for (let k = 0; k < length; k++) if (mask[offset + k] === 1) n++;
  if (n === 0) return -1;
  let choice = rng.nextInt(n);
  for (let k = 0; k < length; k++) {
    if (mask[offset + k] !== 1) continue;
    if (choice === 0) return k;
    choice--;
  }
  return -1;
}

/** A uniformly random legal decision: each head uniform over what the ones before it left legal. */
export function sampleUniform(masks: Masks, rng: Rng, out: Action): void {
  out.selection.fill(-1);
  out.entityType = -1;
  out.target = -1;
  out.cell = -1;
  out.sub = -1;
  out.type = pick(rng, masks.type, 0, ACTION_TYPE_COUNT);
  const type = out.type;
  if (type <= ActionType.Noop) {
    out.type = ActionType.Noop;
    return;
  }
  if (selectsMany(type)) {
    // Every legal row with probability one half, at least one, at most the cap.
    const legal: number[] = [];
    for (let r = 0; r < N_ENT; r++) if (masks.selection[type * N_ENT + r] === 1) legal.push(r);
    const chosen: number[] = [];
    for (const r of legal) if (rng.nextInt(2) === 0) chosen.push(r);
    if (chosen.length === 0) chosen.push(legal[rng.nextInt(legal.length)]!);
    for (let k = 0; k < Math.min(chosen.length, SELECTION_MAX); k++) out.selection[k] = chosen[k]!;
  } else if (selectsOne(type)) {
    out.selection[0] = pick(rng, masks.selection, type * N_ENT, N_ENT);
  }
  if (usesEntityType(type)) {
    if (type === ActionType.Build) {
      out.entityType = pick(rng, masks.buildType, 0, ENTITY_TYPE_COUNT);
    } else {
      const row = out.selection[0]!;
      out.entityType = pick(rng, masks.rowEntityType, row * ENTITY_TYPE_COUNT, ENTITY_TYPE_COUNT);
    }
  }
  if (usesTarget(type)) out.target = pick(rng, masks.target, type * N_ENT, N_ENT);
  if (usesLocation(type)) {
    out.cell =
      type === ActionType.Build
        ? pick(rng, masks.buildCell, out.entityType * CELLS, CELLS)
        : pick(rng, masks.cell, type * CELLS, CELLS);
    out.sub = rng.nextInt(SUB);
  }
}
