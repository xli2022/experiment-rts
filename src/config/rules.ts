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

import { fromFloat, toFloat, type Fix } from '../sim/fixed.js';
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
  /**
   * How much of its top speed a unit can gain or shed in one tick.
   *
   * Expressed as a fraction of `speedPerTick`, so retuning a unit's speed does
   * not silently retune how sharply it starts and stops. A whole 1.0 restores
   * the old behaviour of snapping straight to full speed.
   */
  readonly accelFraction: Fix;
  /** Max chord step when turning toward a new facing, per tick. */
  readonly turnPerTick: Fix;
  readonly sightRange: Fix;
  /** Zero means this entity cannot attack. */
  readonly attackRange: Fix;
  /**
   * Closest a weapon can fire, in world units. Zero for everything but the
   * Sentry, whose barrel points at the sky: a mortar cannot depress onto
   * something standing next to it, and that dead zone is the whole reason a
   * siege unit needs an escort.
   */
  readonly minRange: Fix;
  /**
   * Whether this entity's weapon can reach a flying target.
   *
   * False makes air a hard counter rather than a soft one, which is what the
   * genre does with melee: a unit that swings a sword at something twenty feet
   * up is not missing by a little. It also stops melee units trailing after
   * beamdrones they could never hit.
   */
  readonly canHitAir: boolean;
  readonly damage: number;
  /**
   * Radius of the blast around an impact, in world units. Zero is a weapon that
   * hits one thing.
   *
   * Splash obeys `canHitAir` exactly as the direct hit does: a blast from
   * something that cannot reach a flyer does not reach it either.
   */
  readonly splashRadius: Fix;
  /**
   * Enemies one attack strikes at once, counting the primary target.
   *
   * One is an ordinary weapon. The Arclight's three coils each pick their own
   * enemy, which is a different shape from splash — the extra targets are
   * chosen near the *shooter*, so spreading out does not help against it.
   */
  readonly maxTargets: number;
  /** Whether a shot also damages everything standing on the line to its target. */
  readonly pierce: boolean;
  /**
   * Flat reduction on every point of damage this entity takes.
   *
   * The one stat on this table that is subtraction rather than a number in its
   * own right, and it earns that: it is what makes massed cheap units a bad
   * answer to a heavy one without reintroducing a hidden per-matchup
   * multiplier. A hit never falls below `MIN_DAMAGE`, so armour slows a swarm
   * down rather than making one immune to it, and the info panel prints it.
   */
  readonly armor: number;
  /** Ticks a struck target is slowed for. Zero is a weapon that does not chill. */
  readonly chillTicks: number;
  /**
   * HP restored per service to a damaged friendly unit. Zero is not a repairer.
   *
   * A repairer runs on the weapon clock — `attackRange` and `attackCooldown`
   * mean what they always did — so nothing else in combat, movement or the
   * renderer needs to know that this unit's beam is the helpful kind.
   */
  readonly repairAmount: number;
  /** Whether attacking destroys the attacker. The Boomwalker is the payload. */
  readonly detonates: boolean;
  /**
   * Whole ticks from attack start to impact.
   *
   * Must be zero when cooldown is zero; otherwise it is non-negative and
   * strictly shorter than cooldown. Equality would make the impact branch
   * consume the zero-cooldown tick, stretching repeat cadence by one tick.
   */
  readonly attackForeswing: number;
  /** Whole ticks between consecutive attack starts and impacts. */
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
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(7),
    attackRange: fromFloat(0.6),
    minRange: 0,
    canHitAir: false,
    damage: 5,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: seconds(1.0),
    mineralCost: 50,
    buildTicks: seconds(12),
    supplyCost: 1,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Burstbot,
    name: 'Burstbot',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 45,
    radius: fromFloat(0.4),
    footprint: 0,
    speedPerTick: speed(2.9),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(8),
    attackRange: fromFloat(5),
    minRange: 0,
    canHitAir: true,
    damage: 6,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: seconds(0.8),
    mineralCost: 50,
    buildTicks: seconds(17),
    supplyCost: 1,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Slicebot,
    name: 'Slicebot',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 90,
    radius: fromFloat(0.525),
    footprint: 0,
    speedPerTick: speed(3.6),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.6),
    sightRange: fromFloat(7),
    attackRange: fromFloat(0.9),
    minRange: 0,
    canHitAir: false,
    damage: 13,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    // The authored sword clip spends its opening beats drawing the blade back.
    // Start that motion before the authoritative hit instead of after it.
    attackForeswing: seconds(0.45),
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
    accelFraction: fromFloat(0.22),
    turnPerTick: 0,
    sightRange: fromFloat(9),
    attackRange: 0,
    minRange: 0,
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
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
    accelFraction: fromFloat(0.22),
    turnPerTick: 0,
    sightRange: fromFloat(6),
    attackRange: 0,
    minRange: 0,
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
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
    accelFraction: fromFloat(0.22),
    turnPerTick: 0,
    sightRange: fromFloat(7),
    attackRange: 0,
    minRange: 0,
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: 0,
    mineralCost: 150,
    buildTicks: seconds(45),
    supplyCost: 0,
    supplyProvided: 0,
    produces: [
      EntityType.Burstbot,
      EntityType.Slicebot,
      EntityType.Boomwalker,
      EntityType.Beamdrone,
      EntityType.Fixomatic,
      EntityType.Firespout,
    ],
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
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.9),
    sightRange: fromFloat(8),
    attackRange: fromFloat(6.5),
    minRange: 0,
    canHitAir: true,
    damage: 11,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
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
    accelFraction: fromFloat(0.22),
    turnPerTick: 0,
    sightRange: 0,
    attackRange: 0,
    minRange: 0,
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: 0,
    mineralCost: 0,
    buildTicks: 0,
    supplyCost: 0,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Beamdrone,
    name: 'Beamdrone',
    isBuilding: false,
    // Flies over everything, including its own army.
    collides: false,
    flying: true,
    maxHp: 70,
    radius: fromFloat(0.5625),
    footprint: 0,
    speedPerTick: speed(4.4),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.7),
    sightRange: fromFloat(9),
    attackRange: fromFloat(3.5),
    minRange: 0,
    canHitAir: true,
    damage: 10,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: seconds(1.0),
    mineralCost: 100,
    buildTicks: seconds(24),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Boomwalker,
    name: 'Boomwalker',
    isBuilding: false,
    collides: true,
    flying: false,
    // Two spindly legs under a canister half its own size: nothing about the
    // model suggests it survives contact, and nothing about it suggests it is
    // carrying a gun either. The whole unit is the warhead.
    maxHp: 50,
    radius: fromFloat(0.4),
    footprint: 0,
    // The fastest thing on the ground. A bomb that can be walked away from is
    // not a bomb.
    speedPerTick: speed(4.6),
    accelFraction: fromFloat(0.3),
    turnPerTick: fromFloat(0.7),
    sightRange: fromFloat(7),
    attackRange: fromFloat(0.7),
    minRange: 0,
    // It cannot reach a flyer, and neither can the blast. A charge that jumped
    // would be the faction's cheapest answer to air, which is not what the art
    // is of.
    canHitAir: false,
    damage: 45,
    splashRadius: fromFloat(2.0),
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: true,
    // 45 into one target is a bad trade for 75 minerals; 45 into six clumped
    // Burstbots is three dead units. That gap is the unit.
    attackForeswing: seconds(0.3),
    attackCooldown: seconds(1.0),
    mineralCost: 75,
    buildTicks: seconds(18),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Fixomatic,
    name: 'Fixomatic',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 60,
    radius: fromFloat(0.4),
    footprint: 0,
    speedPerTick: speed(3.4),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.6),
    sightRange: fromFloat(8),
    // A working range, not a weapon range: this is what the repair arms reach.
    attackRange: fromFloat(4.5),
    minRange: 0,
    // Nothing to shoot with, so nothing to say about air.
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    // 4 HP every half second — one Burstbot's worth of damage, undone. It
    // mends units only. Structures were deliberately left out: free repair on
    // buildings was removed from the Worker for making any attack that did not
    // outright kill a structure a waste of time, and handing it back to a
    // purpose-built unit would undo that decision rather than revisit it.
    repairAmount: 4,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: seconds(0.5),
    mineralCost: 100,
    buildTicks: seconds(22),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Foundry,
    name: 'Foundry',
    isBuilding: true,
    collides: true,
    flying: false,
    maxHp: 1100,
    radius: fromFloat(1.5),
    footprint: 3,
    speedPerTick: 0,
    accelFraction: fromFloat(0.22),
    turnPerTick: 0,
    sightRange: fromFloat(7),
    attackRange: 0,
    minRange: 0,
    canHitAir: true,
    damage: 0,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: 0,
    attackCooldown: 0,
    // Dearer and slower than a Barracks, which is the whole tech decision: the
    // minerals and the 55 seconds are an army you did not build meanwhile.
    mineralCost: 200,
    buildTicks: seconds(55),
    supplyCost: 0,
    supplyProvided: 0,
    produces: [
      EntityType.Piercebot,
      EntityType.Arclight,
      EntityType.Sentry,
      EntityType.DarkGolem,
      EntityType.IceGolem,
      EntityType.Plasmodrone,
    ],
  },
  {
    type: EntityType.Firespout,
    name: 'Firespout',
    isBuilding: false,
    collides: true,
    flying: false,
    // A barrel on crab legs with one wide nozzle out the front, and a plate
    // over the front of the barrel. Built to walk into things.
    //
    // The heaviest thing the Barracks makes, and the only splash weapon
    // available without teching. What keeps that honest is the 2.2 reach: a
    // Burstbot outranges it by more than double and a Piercebot by nearly
    // four times, so it only ever gets to do its job to something that let it
    // close.
    maxHp: 130,
    radius: fromFloat(0.5),
    footprint: 0,
    speedPerTick: speed(2.7),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(7),
    attackRange: fromFloat(2.2),
    minRange: 0,
    // The nozzle is level with the ground and does not tilt.
    canHitAir: false,
    damage: 14,
    splashRadius: fromFloat(1.6),
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.25),
    attackCooldown: seconds(0.9),
    mineralCost: 100,
    buildTicks: seconds(24),
    supplyCost: 2,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Arclight,
    name: 'Arclight',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 120,
    radius: fromFloat(0.525),
    footprint: 0,
    speedPerTick: speed(2.9),
    accelFraction: fromFloat(0.22),
    turnPerTick: fromFloat(0.5),
    sightRange: fromFloat(8),
    attackRange: fromFloat(4.5),
    minRange: 0,
    // Three coils on its back, pointing up and out. An arc does not care
    // whether what it earths through is standing on the ground.
    canHitAir: true,
    damage: 9,
    splashRadius: 0,
    // One coil, one enemy. Against a single target 8.2 damage per second for
    // 150 minerals is the worst rate in the game; against three it is the best.
    maxTargets: 3,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.3),
    attackCooldown: seconds(1.1),
    mineralCost: 150,
    buildTicks: seconds(28),
    supplyCost: 3,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Piercebot,
    name: 'Piercebot',
    isBuilding: false,
    collides: true,
    flying: false,
    // Flat and wide, almost all of it launcher. There is no armour on it.
    maxHp: 80,
    radius: fromFloat(0.5),
    footprint: 0,
    speedPerTick: speed(2.6),
    accelFraction: fromFloat(0.18),
    turnPerTick: fromFloat(0.35),
    sightRange: fromFloat(9),
    // The longest reach on the field, and two tiles past a Turret.
    attackRange: fromFloat(8.0),
    minRange: 0,
    canHitAir: true,
    damage: 20,
    splashRadius: 0,
    maxTargets: 1,
    // The rail is horizontal and the bolt does not stop. Everything standing
    // between it and what it aimed at takes the hit — which on a map made of
    // lanes is a decision about where to stand, not a bonus.
    pierce: true,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.5),
    attackCooldown: seconds(2.0),
    mineralCost: 125,
    buildTicks: seconds(26),
    supplyCost: 3,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Sentry,
    name: 'Sentry',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 100,
    radius: fromFloat(0.5),
    footprint: 0,
    speedPerTick: speed(2.4),
    accelFraction: fromFloat(0.16),
    turnPerTick: fromFloat(0.35),
    sightRange: fromFloat(9),
    // Outranges a Turret by 2.5 tiles, which is what makes it the base-cracker.
    attackRange: fromFloat(9.0),
    // The barrel points at the sky. It cannot be aimed at its own feet.
    minRange: fromFloat(2.5),
    canHitAir: false,
    damage: 30,
    splashRadius: fromFloat(2.2),
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.6),
    attackCooldown: seconds(2.6),
    mineralCost: 175,
    buildTicks: seconds(32),
    supplyCost: 3,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.DarkGolem,
    name: 'Dark Golem',
    isBuilding: false,
    collides: true,
    flying: false,
    // The heaviest thing either side can field: shoulder plate, two plasma
    // stacks, and arms that reach the floor.
    maxHp: 420,
    radius: fromFloat(0.7),
    footprint: 0,
    speedPerTick: speed(2.8),
    accelFraction: fromFloat(0.18),
    turnPerTick: fromFloat(0.4),
    sightRange: fromFloat(7),
    attackRange: fromFloat(1.1),
    minRange: 0,
    canHitAir: false,
    damage: 34,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    // 4 off every hit. A Burstbot's 6 becomes 2, so the cheap line unit needs
    // three times as long; a Sentry's 30 becomes 26 and barely notices.
    armor: 4,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.5),
    attackCooldown: seconds(1.5),
    mineralCost: 250,
    buildTicks: seconds(40),
    supplyCost: 5,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.IceGolem,
    name: 'Ice Golem',
    isBuilding: false,
    collides: true,
    flying: false,
    maxHp: 330,
    radius: fromFloat(0.7),
    footprint: 0,
    speedPerTick: speed(2.6),
    accelFraction: fromFloat(0.18),
    turnPerTick: fromFloat(0.4),
    sightRange: fromFloat(8),
    // Two cryo barrels over its shoulders, angled up. It shoots, and it can
    // shoot at things in the air.
    attackRange: fromFloat(4.0),
    minRange: 0,
    canHitAir: true,
    damage: 18,
    splashRadius: 0,
    maxTargets: 1,
    pierce: false,
    armor: 0,
    // What you buy is not the 12.9 damage per second — it is that whatever it
    // hits cannot leave. See CHILL_SPEED.
    chillTicks: seconds(2.5),
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.4),
    attackCooldown: seconds(1.4),
    mineralCost: 225,
    buildTicks: seconds(38),
    supplyCost: 5,
    supplyProvided: 0,
    produces: NONE,
  },
  {
    type: EntityType.Plasmodrone,
    name: 'Plasmodrone',
    isBuilding: false,
    // Flies over everything, including its own army.
    collides: false,
    flying: true,
    maxHp: 220,
    radius: fromFloat(0.7),
    footprint: 0,
    speedPerTick: speed(3.2),
    accelFraction: fromFloat(0.2),
    turnPerTick: fromFloat(0.45),
    sightRange: fromFloat(9),
    attackRange: fromFloat(4.5),
    minRange: 0,
    canHitAir: true,
    damage: 22,
    // Several emitters firing together, so it lands as one wide plasma bloom
    // rather than a beam. Slower and dearer than a Beamdrone in every respect
    // except what happens when it catches a crowd.
    splashRadius: fromFloat(1.8),
    maxTargets: 1,
    pierce: false,
    armor: 0,
    chillTicks: 0,
    repairAmount: 0,
    detonates: false,
    attackForeswing: seconds(0.4),
    attackCooldown: seconds(1.6),
    mineralCost: 225,
    buildTicks: seconds(36),
    supplyCost: 4,
    supplyProvided: 0,
    produces: NONE,
  },
];

/** The timing relation required by combat's wind-up-first tick ordering. */
export function isValidAttackTiming(foreswing: number, cooldown: number): boolean {
  if (!Number.isInteger(foreswing) || !Number.isInteger(cooldown)) return false;
  if (cooldown === 0) return foreswing === 0;
  return cooldown > 0 && foreswing >= 0 && foreswing < cooldown;
}

// Fail at module load rather than letting an invalid balance row subtly change
// cadence in a match. The definitions are literals, so every peer agrees.
for (let i = 0; i < DEFS.length; i++) {
  const def = DEFS[i]!;
  if (!isValidAttackTiming(def.attackForeswing, def.attackCooldown)) {
    throw new Error(
      `${def.name}: foreswing ${def.attackForeswing} must be shorter than cooldown ${def.attackCooldown}`,
    );
  }
  if (def.maxTargets < 1) {
    throw new Error(`${def.name}: maxTargets ${def.maxTargets} must be at least one`);
  }
  if (def.minRange > 0 && def.minRange >= def.attackRange) {
    throw new Error(`${def.name}: minimum range must be shorter than its attack range`);
  }
  // Every ability rides the weapon clock, so a row that has one and no clock
  // would sit there doing nothing with no sign of why.
  const armed = def.attackRange > 0 && def.attackCooldown > 0;
  if (!armed && (def.splashRadius > 0 || def.pierce || def.chillTicks > 0 || def.detonates)) {
    throw new Error(`${def.name}: has a weapon ability but no weapon`);
  }
  if (def.repairAmount > 0 && (!armed || def.damage > 0)) {
    throw new Error(`${def.name}: a repairer needs a working range and deals no damage`);
  }
}

export function defOf(type: EntityType): EntityDef {
  return DEFS[type]!;
}

/**
 * `damage` is the whole story: a unit deals it to everything it hits.
 *
 * There used to be a rock-paper-scissors triangle here — ranged/air/melee, each
 * dealing double to one other — applied as a percentage inside `combatSystem`.
 * It is gone, and deliberately so: the multiplier existed nowhere on screen, so
 * the number a player could see was never the number they got.
 *
 * The abilities above do not bring it back. Every one of them changes *how many
 * things a shot reaches* or *how much of a hit survives contact* — never how
 * much damage this attacker deals to that defender. `splashRadius`,
 * `maxTargets` and `pierce` each widen the set of things one attack lands on,
 * and every one of them takes the same `damage`; `armor` is subtracted from
 * every incoming hit whoever threw it. So a player can still read two panels
 * against each other and get the fight they expect, and `abilityText` puts each
 * of these on the panel beside the damage figure rather than leaving it to be
 * discovered.
 */

/**
 * Floor on a hit after armour.
 *
 * Armour that could zero a weapon out would make a Dark Golem literally
 * immune to Burstbots, and "this unit cannot be hurt by that one" is a rule
 * players discover by losing an army to it. One point a hit keeps the counter
 * lopsided without making it absolute.
 */
export const MIN_DAMAGE = 1;

/**
 * What a chilled unit's top speed is multiplied by.
 *
 * One constant rather than a per-weapon figure: only the Ice Golem chills, and
 * two sources with different strengths would need a rule for which one wins
 * that nothing on screen could explain. Half speed, for `chillTicks`.
 */
export const CHILL_SPEED = fromFloat(0.5);

/**
 * The abilities of a unit, as the info panel says them out loud.
 *
 * Kept next to the numbers it reads rather than in the HUD, so a row that gains
 * an ability gains its line here in the same edit. Empty for the units that do
 * exactly what their damage figure says.
 */
export function abilityText(def: EntityDef): string[] {
  const out: string[] = [];
  if (def.repairAmount > 0) {
    out.push(`repairs ${(def.repairAmount * TICKS_PER_SECOND) / def.attackCooldown} HP/s`);
  }
  if (def.detonates) out.push('detonates on contact');
  if (def.splashRadius > 0) out.push(`splash ${toFloat(def.splashRadius).toFixed(1)}`);
  if (def.maxTargets > 1) out.push(`hits ${def.maxTargets} at once`);
  if (def.pierce) out.push('pierces the line');
  if (def.chillTicks > 0) {
    out.push(`chills ${(def.chillTicks / TICKS_PER_SECOND).toFixed(1)}s`);
  }
  if (def.armor > 0) out.push(`armour ${def.armor}`);
  if (def.minRange > 0) out.push(`min range ${toFloat(def.minRange).toFixed(1)}`);
  if (def.attackRange > 0 && def.damage > 0 && !def.canHitAir) out.push('no air');
  return out;
}

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
 * This is not only a range test: movement clears a worker's path the tick the
 * test passes, so the slack *is* the standoff — a worker stops up to
 * `radius + slack` clear of the near face and builds from there. At 1.7 that
 * measured a gap of 0.7 to 1.5 units of open ground between a worker and the
 * wall it was supposedly raising, which is a worker's own width or two, and far
 * enough that the pair did not read as related. At 0.7 the same builds park it
 * against the wall, 0.0 to 0.5 out, arm first.
 *
 * The floor is the corner. Radii are circles but footprints are squares, and
 * movement measures the approach centre-to-centre, so a worker against the
 * corner of a 4x4 Command Post stands sqrt(2) x 2.0 = 2.83 from its centre
 * where the face is 2.0 away. Any slack below (sqrt(2) - 1) x 2.0 - 0.32, about
 * 0.51, leaves that worker counted as still walking with nowhere left to walk:
 * it slides along the face instead of settling. 0.7 keeps that margin.
 */
export const BUILD_REACH = fromFloat(0.7);

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
