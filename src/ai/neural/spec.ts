/**
 * The shape of what the neural bot sees and says.
 *
 * Every number here is shared with the training code in Python: `npm run
 * ml:spec` dumps `SPEC` to `ml/rtsml/spec.json`, and `tests/spec.test.ts`
 * fails the moment the two disagree. Change a feature here and the model has
 * to be retrained; `version` says so to a model file that was not.
 *
 * The bot sees what a human sees: entities inside its side's vision or in its
 * own memory, a coarse map, and the numbers on the HUD — all in the canonical
 * frame, so the seat it sits in is invisible to it. It answers with the human
 * vocabulary: one command, built from a handful of choices made in order.
 */

import { MAX_COMMAND_UNITS } from '../../sim/commands.js';
import { ENTITY_TYPE_COUNT } from '../../sim/types.js';
import { DECISION_TICKS } from '../cadence.js';

export { DECISION_TICKS };

/** Rows in the entity table. Own entities first, then everything else in priority order. */
export const N_ENT = 160;

/**
 * Per-row features, in column order. Own-only fields are zero on other rows —
 * a human sees an enemy's health bar, not its orders.
 */
export const ENTITY_FEATURES = [
  // type one-hot, in EntityType order
  'type:Worker',
  'type:Burstbot',
  'type:Slicebot',
  'type:CommandPost',
  'type:Depot',
  'type:Barracks',
  'type:Turret',
  'type:MineralPatch',
  'type:Beamdrone',
  // relation to the viewer
  'rel:own',
  'rel:ally',
  'rel:enemy',
  'rel:neutral',
  // where, in the canonical frame, as fractions of the map
  'x',
  'y',
  'dxFromStart',
  'dyFromStart',
  'hp',
  'build:Site',
  'build:UnderConstruction',
  'build:Complete',
  'buildProgress',
  // own-only: what it is doing
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
  // how it is known
  'visibleNow',
  'memoryAge',
  'resourceAmount',
  'inLastCommand',
  'supplyCost',
  'flying',
  'canHitAir',
  'distToOwnPost',
] as const;
export const ENTITY_FEATURE_COUNT = ENTITY_FEATURES.length;

/** Cells per side of the coarse map. Both layouts fit: 128 tiles is 32 cells, 152 is 38. */
export const GRID = 40;
/** Tiles per cell side. */
export const CELL_TILES = 4;

/** Per-cell channels. Fractions of the cell's tiles, or counts scaled and clipped to one. */
export const GRID_CHANNELS = [
  'walkable',
  'buildable',
  'explored',
  'visible',
  'ownBuildings',
  'ownUnits',
  'allyEntities',
  'enemyBuildings',
  'enemyUnitsVisible',
  'minerals',
  'friendlyStarts',
  'enemyStarts',
  'expansions',
] as const;
export const GRID_CHANNEL_COUNT = GRID_CHANNELS.length;

/** The numbers on the HUD, and a little of what the bot itself just did. */
export const SCALARS = [
  'minerals',
  'supplyUsed',
  'supplyMax',
  'supplyFree',
  'tick',
  'layout:Lanes',
  'layout:Quarters',
  'seatInHalf',
  'allies',
  // own counts by type, in EntityType order
  'own:Worker',
  'own:Burstbot',
  'own:Slicebot',
  'own:CommandPost',
  'own:Depot',
  'own:Barracks',
  'own:Turret',
  'own:MineralPatch',
  'own:Beamdrone',
  // known enemy counts by type — visible or remembered
  'enemy:Worker',
  'enemy:Burstbot',
  'enemy:Slicebot',
  'enemy:CommandPost',
  'enemy:Depot',
  'enemy:Barracks',
  'enemy:Turret',
  'enemy:MineralPatch',
  'enemy:Beamdrone',
  // the previous decision's type, one-hot in ActionType order
  'prev:Noop',
  'prev:Move',
  'prev:AttackMove',
  'prev:Attack',
  'prev:Harvest',
  'prev:Build',
  'prev:Stop',
  'prev:Hold',
  'prev:Train',
  'prev:CancelTrain',
  'prev:SetRally',
  'decisionsSinceNonNoop',
  'recentCommands',
] as const;
export const SCALAR_COUNT = SCALARS.length;

/** What one decision can be. Every human command but Surrender, plus doing nothing. */
export enum ActionType {
  Noop = 0,
  Move = 1,
  AttackMove = 2,
  Attack = 3,
  Harvest = 4,
  Build = 5,
  Stop = 6,
  Hold = 7,
  Train = 8,
  CancelTrain = 9,
  SetRally = 10,
}
export const ACTION_TYPES = [
  'Noop',
  'Move',
  'AttackMove',
  'Attack',
  'Harvest',
  'Build',
  'Stop',
  'Hold',
  'Train',
  'CancelTrain',
  'SetRally',
] as const;
export const ACTION_TYPE_COUNT = ACTION_TYPES.length;

/** Tiles inside a cell a location head can point at: CELL_TILES squared. */
export const SUB = CELL_TILES * CELL_TILES;

/** Units named at most, per command; the UI's selection cap. */
export const SELECTION_MAX = MAX_COMMAND_UNITS;

/** A flat action: type, entityType, target, cell, sub, then SELECTION_MAX rows (-1 padded). */
export const ACTION_INTS = 5 + SELECTION_MAX;

/** A remembered enemy unit is forgotten this long after it was last seen. */
export const UNIT_MEMORY_TICKS = 300;

/**
 * Noise the model samples with, supplied as an input so the graph carries no
 * random number generator. One segment per head, in head order, every value
 * Gumbel(0, 1): a categorical head adds its segment and takes the argmax,
 * which is an exact draw from its softmax. The selection segment holds two
 * values per row: a type that names one row draws it by argmax over the
 * first; a type that names many keeps every row whose logit plus the
 * difference of the two — a Logistic(0, 1) — is above zero, which is an exact
 * Bernoulli draw per row.
 */
export const NOISE_SEGMENTS = [
  { name: 'type', size: ACTION_TYPE_COUNT, distribution: 'gumbel' },
  { name: 'selection', size: 2 * N_ENT, distribution: 'gumbel' },
  { name: 'entityType', size: ENTITY_TYPE_COUNT, distribution: 'gumbel' },
  { name: 'target', size: N_ENT, distribution: 'gumbel' },
  { name: 'cell', size: GRID * GRID, distribution: 'gumbel' },
  { name: 'sub', size: SUB, distribution: 'gumbel' },
] as const;
export const NOISE_LEN = NOISE_SEGMENTS.reduce((n, s) => n + s.size, 0);

/**
 * The critic's extra view, training only: per player in canonical order (own,
 * allies, enemies), the whole truth about their economy and army.
 */
export const CRITIC_PER_PLAYER = [
  'minerals',
  'supplyUsed',
  'supplyMax',
  'workers',
  'army',
  'buildings',
  'armyValue',
  'buildingValue',
  'armyX',
  'armyY',
  'defeated',
] as const;
export const CRITIC_PLAYERS = 4;
export const CRITIC_LEN = CRITIC_PER_PLAYER.length * CRITIC_PLAYERS + 2;

/** Everything Python needs to build and export a model that fits this codec. */
export const SPEC = {
  version: 1,
  decisionTicks: DECISION_TICKS,
  unitMemoryTicks: UNIT_MEMORY_TICKS,
  entities: { rows: N_ENT, features: ENTITY_FEATURES },
  grid: { size: GRID, cellTiles: CELL_TILES, channels: GRID_CHANNELS },
  scalars: SCALARS,
  critic: { perPlayer: CRITIC_PER_PLAYER, players: CRITIC_PLAYERS, length: CRITIC_LEN },
  actions: {
    types: ACTION_TYPES,
    selectionMax: SELECTION_MAX,
    entityTypes: ENTITY_TYPE_COUNT,
    sub: SUB,
    ints: ACTION_INTS,
    /** Head order; each head's mask is gathered by the choices before it. */
    heads: ['type', 'selection', 'entityType', 'target', 'cell', 'sub'],
  },
  masks: {
    type: [ACTION_TYPE_COUNT],
    selection: [ACTION_TYPE_COUNT, N_ENT],
    target: [ACTION_TYPE_COUNT, N_ENT],
    cell: [ACTION_TYPE_COUNT, GRID * GRID],
    buildCell: [ENTITY_TYPE_COUNT, GRID * GRID],
    rowEntityType: [N_ENT, ENTITY_TYPE_COUNT],
    buildType: [ENTITY_TYPE_COUNT],
  },
  noise: { segments: NOISE_SEGMENTS, length: NOISE_LEN },
} as const;
