/**
 * The codec spec is shared with Python by a committed JSON file, and a stale
 * file is the one way the two sides can silently disagree.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  ACTION_INTS,
  ACTION_TYPES,
  ENTITY_FEATURE_COUNT,
  ENTITY_FEATURES,
  GRID,
  GRID_CHANNELS,
  N_ENT,
  NOISE_LEN,
  NOISE_SEGMENTS,
  SCALARS,
  SELECTION_MAX,
  SPEC,
  SUB,
} from '../src/ai/neural/spec.js';
import { ENTITY_TYPE_COUNT } from '../src/sim/types.js';

describe('the codec spec', () => {
  it('matches the committed spec.json that Python reads', () => {
    const committed = readFileSync(new URL('../ml/rtsml/spec.json', import.meta.url), 'utf8');
    expect(JSON.parse(committed)).toEqual(JSON.parse(JSON.stringify(SPEC)));
  });

  it('names every feature once', () => {
    for (const list of [ENTITY_FEATURES, GRID_CHANNELS, SCALARS, ACTION_TYPES]) {
      expect(new Set(list).size).toBe(list.length);
    }
    expect(ENTITY_FEATURE_COUNT).toBe(ENTITY_FEATURES.length);
  });

  it('sizes the noise vector from the heads', () => {
    const sum = NOISE_SEGMENTS.reduce((n, s) => n + s.size, 0);
    expect(NOISE_LEN).toBe(sum);
    expect(NOISE_SEGMENTS.map((s) => s.name)).toEqual(SPEC.actions.heads);
    // Selection draws two Gumbels per row (one per class) so the same noise
    // serves the multi-select threshold and the single-select argmax.
    expect(NOISE_LEN).toBe(
      ACTION_TYPES.length + 2 * N_ENT + ENTITY_TYPE_COUNT + N_ENT + GRID * GRID + SUB,
    );
  });

  it('lays a flat action out as five choices plus the selection', () => {
    expect(ACTION_INTS).toBe(5 + SELECTION_MAX);
    expect(SPEC.actions.ints).toBe(ACTION_INTS);
  });
});
