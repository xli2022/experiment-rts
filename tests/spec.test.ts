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
  QUEUED_UNIT_TYPES,
  SCALARS,
  SELECTION_MAX,
  SPEC,
  SUB,
} from '../src/ai/neural/spec.js';
import { EntityType, ENTITY_TYPE_COUNT } from '../src/sim/types.js';
import { DEFS } from '../src/config/rules.js';

describe('the codec spec', () => {
  it('appends own order and queue columns without moving codec-3 inputs', () => {
    const previous = [
      'type:Worker',
      'type:Burstbot',
      'type:Slicebot',
      'type:CommandPost',
      'type:Depot',
      'type:Barracks',
      'type:Turret',
      'type:MineralPatch',
      'type:Beamdrone',
      'type:Boomwalker',
      'type:Fixomatic',
      'type:Factory',
      'type:Firespout',
      'type:Arclight',
      'type:Piercebot',
      'type:Sentry',
      'type:DarkGolem',
      'type:IceGolem',
      'type:Plasmodrone',
      'type:Airport',
      'rel:own',
      'rel:ally',
      'rel:enemy',
      'rel:neutral',
      'x',
      'y',
      'dxFromStart',
      'dyFromStart',
      'hp',
      'build:Site',
      'build:UnderConstruction',
      'build:Complete',
      'buildProgress',
      'order:None',
      'order:Move',
      'order:AttackMove',
      'order:Attack',
      'order:Harvest',
      'order:Build',
      'order:Hold',
      'carrying',
      'prodCount',
      'prodProgress',
      'hasRally',
      'rallyDx',
      'rallyDy',
      'cooldown',
      'buildingLevel',
      'upgrading',
      'upgradeProgress',
      'visibleNow',
      'memoryAge',
      'resourceAmount',
      'inLastCommand',
      'supplyCost',
      'flying',
      'canHitAir',
      'distToOwnPost',
    ];
    expect(previous).toHaveLength(58);
    expect(ENTITY_FEATURES.slice(0, previous.length)).toEqual(previous);
    expect(ENTITY_FEATURES.slice(previous.length, 73)).toEqual([
      'orderDx',
      'orderDy',
      ...QUEUED_UNIT_TYPES.map((type) => `queued:${EntityType[type]}`),
    ]);
    expect(QUEUED_UNIT_TYPES).toEqual(
      [...new Set(DEFS.flatMap((def) => def.produces))].sort((a, b) => a - b),
    );
  });

  it('appends construction staffing without changing any codec-4 contract', () => {
    const previous = JSON.parse(
      readFileSync(new URL('../ml/rtsml/spec-v4.json', import.meta.url), 'utf8'),
    );
    expect(previous.version).toBe(4);
    expect(previous.entities.features).toHaveLength(73);
    expect(ENTITY_FEATURES.slice(0, 73)).toEqual(previous.entities.features);
    expect(ENTITY_FEATURES.slice(73)).toEqual(['hasAssignedBuilder']);
    const codec5 = JSON.parse(
      readFileSync(new URL('../ml/rtsml/spec-v5.json', import.meta.url), 'utf8'),
    );
    expect(codec5).toEqual({
      ...previous,
      version: 5,
      entities: {
        ...previous.entities,
        features: [...previous.entities.features, 'hasAssignedBuilder'],
      },
    });
  });

  it('versions the observation allocation change without changing tensor or action shapes', () => {
    const previous = JSON.parse(
      readFileSync(new URL('../ml/rtsml/spec-v5.json', import.meta.url), 'utf8'),
    );
    expect(previous.version).toBe(5);
    expect(SPEC).toEqual({ ...previous, version: 6 });
  });

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
