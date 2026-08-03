/**
 * All balance data in one file.
 *
 * Every tunable number the simulation reads lives here, so balancing is editing
 * a table rather than hunting through systems. Values are converted to
 * fixed-point / ticks at module load, which is deterministic — `fromFloat` is
 * only ever applied to literals here, never to simulation state.
 *
 * Distances are in world units (1 unit = 1 map tile). Speeds are authored in
 * units per second and stored per tick.
 */

import { fromFloat, type Fix } from '../sim/fixed.js';
import { EntityType, Order, seconds, TICKS_PER_SECOND } from '../sim/types.js';

/** Author a speed in units/second, store it as movement per tick. */
function speed(unitsPerSecond: number): Fix {
  return fromFloat(unitsPerSecond / TICKS_PER_SECOND);
}

export interface EntityDef {
  readonly type: EntityType;
  readonly name: string;
  readonly isBuilding: boolean;
  /**
   * Whether this entity pushes other units out of its way.
   *
   * Workers are deliberately false. A mineral line packs a dozen of them into a
   * few tiles, all converging on the same patch from the same direction, and
   * mutual push-apart turns that into a permanent traffic jam that costs far
   * more income than the overlap costs realism. Letting them pass through each
   * other is the standard fix and is what the genre does. It only affects
   * unit-to-unit contact: buildings and terrain still block everyone, so nothing
   * can walk through a wall.
   */
  readonly collides: boolean;
  /**
   * Flying units ignore terrain entirely: no pathfinding, no cliffs, no
   * buildings in the way. They steer straight at wherever they are going, which
   * is both how the genre treats air and far cheaper than pathing them.
   */
  readonly flying: boolean;
  readonly maxHp: number;
  /** Collision radius in world units. */
  readonly radius: Fix;
  /** Buildings occupy footprint x footprint tiles. Zero for units. */
  readonly footprint: number;
  readonly speedPerTick: Fix;
  /** Max chord step when turning toward a new facing, per tick. */
  readonly turnPerTick: Fix;
  readonly sightRange: Fix;
  /** Zero means this entity cannot attack. */
  readonly attackRange: Fix;
  readonly damage: number;
  readonly attackCooldown: number;
  readonly mineralCost: number;
  readonly buildTicks: number;
  readonly supplyCost: number;
  readonly supplyProvided: number;
  /** Unit types this building can train, in menu order. */
  readonly produces: readonly EntityType[];
}

const NONE: readonly EntityType[] = [];

/**
 * Indexed by `EntityType`. Order must match the enum exactly — the simulation
 * looks defs up by numeric type, and `tests/rules.test.ts` asserts alignment.
 */
export const DEFS: readonly EntityDef[] = [
  {
    type: EntityType.Worker,
    name: 'Worker',
    isBuilding: false,
    collides: false,
    flying: false,
    maxHp: 40,
    radius: fromFloat(0.32),
    footprint: 0,
    speedPerTick: speed(3.2),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(7),
    attackRange: fromFloat(0.6),
    damage: 5,
    attackCooldown: seconds(1.0),
    mineralCost: 50,
    buildTicks: seconds(12),
    supplyCost: 1,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Rifleman,
    name: 'Rifleman',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 45,
    radius: fromFloat(0.32),
    footprint: 0,
    speedPerTick: speed(2.9),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(8),
    attackRange: fromFloat(5),
    damage: 6,
    attackCooldown: seconds(0.8),
    mineralCost: 50,
    buildTicks: seconds(17),
    supplyCost: 1,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Brawler,
    name: 'Brawler',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 90,
    radius: fromFloat(0.42),
    footprint: 0,
    speedPerTick: speed(3.6),
    turnPerTick: fromFloat(0.6),
    sightRange: fromFloat(7),
    attackRange: fromFloat(0.9),
    damage: 13,
    attackCooldown: seconds(1.2),
    mineralCost: 75,
    buildTicks: seconds(20),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.CommandPost,
    name: 'Command Post',
    isBuilding: true,
    collides: true,
    flying: false,
    maxHp: 1500,
    radius: fromFloat(2.0),
    footprint: 4,
    speedPerTick: 0,
    turnPerTick: 0,
    sightRange: fromFloat(9),
    attackRange: 0,
    damage: 0,
    attackCooldown: 0,
    mineralCost: 400,
    buildTicks: seconds(55),
    supplyCost: 0,
    supplyProvided: 10,
    produces: [EntityType.Worker],
  },
  {
    type: EntityType.Depot,
    name: 'Supply Depot',
    isBuilding: true,
    collides: true,
    flying: false,
    maxHp: 500,
    radius: fromFloat(1.0),
    footprint: 2,
    speedPerTick: 0,
    turnPerTick: 0,
    sightRange: fromFloat(6),
    attackRange: 0,
    damage: 0,
    attackCooldown: 0,
    mineralCost: 100,
    buildTicks: seconds(25),
    supplyCost: 0,
    // Generous on purpose: at 8 supply each, reaching the 200 cap needs roughly
    // two dozen depots, and that many structures ring a base so densely that
    // its own army cannot get out.
    supplyProvided: 15,
    produces: NONE,
  },
  {
    type: EntityType.Barracks,
    name: 'Barracks',
    isBuilding: true,
    collides: true,
    flying: false,
    maxHp: 1000,
    radius: fromFloat(1.5),
    footprint: 3,
    speedPerTick: 0,
    turnPerTick: 0,
    sightRange: fromFloat(7),
    attackRange: 0,
    damage: 0,
    attackCooldown: 0,
    mineralCost: 150,
    buildTicks: seconds(45),
    supplyCost: 0,
    supplyProvided: 0,
    produces: [EntityType.Rifleman, EntityType.Brawler, EntityType.Gunship],
  },
  {
    type: EntityType.Turret,
    name: 'Turret',
    isBuilding: true,
    collides: true,
    flying: false,
    maxHp: 600,
    radius: fromFloat(1.0),
    footprint: 2,
    speedPerTick: 0,
    turnPerTick: fromFloat(0.9),
    sightRange: fromFloat(8),
    attackRange: fromFloat(6.5),
    damage: 11,
    attackCooldown: seconds(0.8),
    mineralCost: 100,
    buildTicks: seconds(28),
    supplyCost: 0,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.MineralPatch,
    name: 'Mineral Patch',
    isBuilding: true, // static and blocks placement, though not player-owned
    // Walked over, not walked around. A mineral line is eight patches in a tight
    // arc right beside the Command Post, and as solid ground it was a wall
    // across the busiest few tiles on the map: workers queued round the ends of
    // it, and a unit told to move through the base took the long way.
    collides: false,
    flying: false,
    maxHp: 1,
    radius: fromFloat(0.8),
    footprint: 2,
    speedPerTick: 0,
    turnPerTick: 0,
    sightRange: 0,
    attackRange: 0,
    damage: 0,
    attackCooldown: 0,
    mineralCost: 0,
    buildTicks: 0,
    supplyCost: 0,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Gunship,
    name: 'Gunship',
    isBuilding: false,
    // Flies over everything, including its own army.
    collides: false,
    flying: true,
    maxHp: 70,
    radius: fromFloat(0.45),
    footprint: 0,
    speedPerTick: speed(4.4),
    turnPerTick: fromFloat(0.7),
    sightRange: fromFloat(9),
    attackRange: fromFloat(3.5),
    damage: 10,
    attackCooldown: seconds(1.0),
    mineralCost: 100,
    buildTicks: seconds(24),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
];

export function defOf(type: EntityType): EntityDef {
  return DEFS[type]!;
}

/**
 * Counter bonus, as a percentage of base damage.
 *
 * A rock-paper-scissors triangle between the three combat units:
 *
 *     Rifleman (ranged)  ->  Gunship (air)
 *     Gunship  (air)     ->  Brawler (melee)
 *     Brawler  (melee)   ->  Rifleman (ranged)
 *
 * Expressed as a damage multiplier rather than "cannot target", so a player who
 * has committed to one unit type is disadvantaged but never completely helpless
 * — a hard counter turns the match into a coin flip decided before contact.
 *
 * Integer percent, applied with integer arithmetic, so it stays exact and
 * deterministic.
 */
export function counterBonusPct(attacker: EntityType, target: EntityType): number {
  if (attacker === EntityType.Rifleman && target === EntityType.Gunship) return COUNTER_PCT;
  if (attacker === EntityType.Gunship && target === EntityType.Brawler) return COUNTER_PCT;
  if (attacker === EntityType.Brawler && target === EntityType.Rifleman) return COUNTER_PCT;
  return 100;
}

/** How hard a counter hits. 200 = double damage. */
export const COUNTER_PCT = 200;

// ---------------------------------------------------------------------------
// Global economy and match rules
// ---------------------------------------------------------------------------

export const STARTING_MINERALS = 50;
export const STARTING_WORKERS = 6;

/** Hard ceiling on supply regardless of how many depots are built. */
export const SUPPLY_MAX = 200;

/**
 * Minerals a worker carries per trip.
 *
 * Together with HARVEST_TICKS and the mineral-line distance this sets the pace
 * of the entire game: too low and neither side can afford an army before the
 * patches run dry.
 */
export const MINERALS_PER_TRIP = 8;
/** Ticks spent standing in a patch before the load is full. */
export const HARVEST_TICKS = seconds(2.0);
/** Total minerals in a patch before it is exhausted and removed. */
export const PATCH_AMOUNT = 1500;
/** Patches per starting base. */
export const PATCHES_PER_BASE = 8;
/**
 * Patches at an expansion site.
 *
 * Fewer than a main, so a second base is a meaningful boost rather than a
 * doubling — the reason to take one is that the mains run dry, and a game where
 * expanding is strictly better than not is a game with one opening.
 */
export const PATCHES_PER_EXPANSION = 6;

/** Build orders can be queued this deep per production building. */
export const MAX_PRODUCTION_QUEUE = 5;

/**
 * Slack added to the two radii when deciding whether a worker can reach a
 * *construction site*.
 *
 * Radii are circles but building footprints are squares, so a worker standing
 * against the corner of a 3x3 barracks is further from its centre than the
 * radius suggests. Too tight a value leaves workers hovering one step outside
 * build range, unable to work and unable to path closer.
 */
export const BUILD_REACH = fromFloat(1.7);

/**
 * Slack for mining a patch and for delivering to a drop-off.
 *
 * Much tighter than BUILD_REACH, and deliberately so: sharing the construction
 * value meant a worker could mine from nearly two units clear of the crystals,
 * so it never visibly walked to the patch at all and the whole gather cycle read
 * as broken. Workers should have to travel — that trip is the economy.
 */
export const HARVEST_REACH = fromFloat(0.35);

/**
 * Hit points a worker restores per tick when repairing.
 *
 * Repair is free in minerals but not in time: the worker has to stand there and
 * is not mining while it does. At this rate a badly damaged Command Post takes
 * most of a minute with one worker, which is worth defending rather than an
 * instant undo of an attack.
 */
export const REPAIR_HP_PER_TICK = 6;

/**
 * How much slack an order gets when deciding "am I close enough yet".
 *
 * Movement and the economy systems must agree on this exactly. If movement
 * thought a worker had arrived while harvesting still considered it too far, the
 * worker would stand still forever next to a patch it refused to mine.
 */
export function reachSlackFor(order: Order): Fix {
  return order === Order.Build ? BUILD_REACH : HARVEST_REACH;
}

/**
 * A* requests served per tick. Excess requests wait in a deterministic FIFO —
 * pathfinding is the one system that can blow the tick budget, and letting it
 * run unbounded would stall the lockstep turn for every peer.
 */
export const PATH_BUDGET_PER_TICK = 24;

/**
 * Move orders naming at least this many units use a shared flow field instead of
 * one A* search each.
 *
 * Below the threshold, per-unit A* is cheaper: a flow field costs a full-map
 * Dijkstra sweep, which is wasted effort for a single worker walking a few
 * tiles. Above it, the sweep is amortised across the whole group and wins by a
 * wide margin — an army-wide attack-move was measured at roughly 60ms of
 * pathfinding per tick before this split, which in lockstep stalls every peer.
 */
export const GROUP_PATH_THRESHOLD = 5;

/** Separation strength when units overlap, as a fraction of the overlap. */
export const SEPARATION_STRENGTH = fromFloat(0.35);
