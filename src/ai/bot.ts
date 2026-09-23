/**
 * The scripted bot.
 *
 * ## A player, not a system
 *
 * This used to run inside `Simulation.step`, on every peer, which is only
 * possible for a bot that is a pure function of the world. It is hosted now —
 * `ScriptedAgent` wraps it, `AgentDriver` calls it once a tick on one peer, and
 * its commands cross the wire and execute one input delay later, exactly like a
 * human's. See `agent.ts` for why every bot took that path.
 *
 * It is still a pure function of world state: given the same world and the same
 * tick, it emits the same commands, with no `Math.random`, no wall-clock, no
 * iteration over unordered collections and no state of its own. Nothing
 * structural depends on that any more, but the determinism and mirror probes
 * do — they drive whole matches with this bot and expect the same answer every
 * time, on every engine — so `tests/sealed-sim.test.ts` scans this file too.
 *
 * ## Two bots on a side are not two bots
 *
 * In co-op the AI holds both enemy slots, and the naive version of that is much
 * weaker than one bot with twice the economy: each half picks its own target,
 * arrives on its own schedule, and gets beaten twice in a row by an army that
 * never had to fight both at once. So the offensive decisions — when to commit,
 * and what to hit — are taken over the *team's* army rather than each bot's own.
 * Every bot on a side computes that from the same world in the same order and
 * therefore reaches the same answer, with no coordination channel and no shared
 * state to keep in step.
 */

import {
  buildingUpgrade,
  defOf,
  MAX_PRODUCTION_QUEUE,
  productionOptions,
  SUPPLY_MAX,
  type EntityDef,
} from '../config/rules.js';
import { CommandType, type Command } from '../sim/commands.js';
import { FIX_HALF, fromFloat, fromInt, toInt, sqRange, vecLenSqRaw } from '../sim/fixed.js';
import { mirrorTile } from '../sim/map.js';
import { mirroredHalf } from '../sim/mapgen.js';
import { standableTarget } from '../sim/systems/orders.js';
import {
  BuildState,
  ENTITY_TYPE_COUNT,
  EntityType,
  NEUTRAL,
  Order,
  type PlayerId,
} from '../sim/types.js';
import type { World } from '../sim/world.js';

/**
 * The bot reconsiders its plan this often.
 *
 * The same for every slot, deliberately. Staggering bots by player read as a
 * harmless way to keep command streams distinguishable in a replay — commands
 * carry their player anyway — but it hands whoever thinks first a whole think
 * interval of head start. Measured in a mirror matchup on a symmetric map, that
 * decided the game: player 0 spent its opening 50 minerals and had its workers
 * walking before player 1 had taken a turn at all.
 *
 * `ScriptedAgent` accepts a different interval, for tests that need a match
 * between unequal bots to resolve — never for play. Do not read it as a
 * strength dial: the cadence gates below (`beat % 2`, `beat % 6`) make its
 * effect anything but monotonic. Before the scouting and order-suppression
 * revision, an eight-seed test found the 20-tick bot won 8–0, the 30-tick bot
 * lost 8–0, and the 40-tick bot won 5–3. Those are historical measurements,
 * not strength guarantees for the current strategy.
 */
export const THINK_INTERVAL = 10;

/** How near a Command Post has to be for an expansion to count as taken. */
const CLAIMED_RANGE = fromInt(14);

/**
 * How far from a base a hostile has to be before it stops being an emergency.
 *
 * Generous, because the point is to catch an attack while the army can still
 * walk back and do something about it. Measured to a building, so a wide base
 * effectively defends a wider circle, which is right.
 */
const DEFEND_RANGE = fromInt(20);

/**
 * Patches further than this from one of our own Command Posts are somebody
 * else's.
 *
 * Idle workers used to be handed the nearest live patch on the map. With one
 * opponent across the map that was always home; with four bases it is
 * occasionally an ally's mineral line, or an enemy's, and a worker that walks
 * there is gone for good — it mines into the wrong bank or dies on arrival.
 */
const HOME_PATCH_RANGE = fromInt(24);

/**
 * The knobs the strategy turns on.
 *
 * There used to be three settings of these — Easy, Normal and Hard — and only
 * the last survives: one scripted bot, and the neural bot as the other choice.
 * Everything here is behavioural: no bonus income, no extra starting units, no
 * cheating on fog. It works its economy hard and commits early, which is a thing
 * a player could have done too.
 */
interface Tuning {
  /** Workers to saturate the mineral line before spending on army. */
  readonly targetWorkers: number;
  /** Team army size that triggers a push, then keeps triggering it. */
  readonly attackArmySize: number;
  /** Barracks the bot will run once minerals are spare. */
  readonly maxBarracks: number;
  /** Factories it will run. Two, and the second only off a deep bank. */
  readonly maxFactories: number;
  /** Turrets it will put up at home. */
  readonly maxTurrets: number;
  /** Command Posts it will run. */
  readonly maxBases: number;
  /** Minerals on hand before it considers another base. */
  readonly expandAtMinerals: number;
  /** Concurrent construction sites. */
  readonly maxSites: number;
  /** Whether it walks its army home when its base is attacked. */
  readonly defendsHome: boolean;
  /** Whether it commits with its partner rather than on its own count. */
  readonly coordinates: boolean;
}

// Saturates two bases, expands off a smaller bank, and commits on a small army
// — which against two humans means arriving before either of them has an army
// of their own. These are the values the Hard tier shipped with.
const TUNING: Tuning = {
  targetWorkers: 18,
  attackArmySize: 6,
  maxBarracks: 8,
  maxFactories: 2,
  maxTurrets: 3,
  maxBases: 3,
  expandAtMinerals: 450,
  maxSites: 3,
  defendsHome: true,
  coordinates: true,
};

/**
 * Minerals in the bank that mean production, not income, is the bottleneck.
 *
 * Past this the bot queues barracks to their cap instead of two deep. Measured
 * over a four-player match, two-deep queues left every bot floating six to
 * eight thousand minerals for the last five minutes — an army it had paid for
 * and never received, because the only other outlet was yet another Barracks.
 */
const DEEP_QUEUE_MINERALS = 700;

/**
 * Attack range at or below which a weapon only reaches what it can touch.
 *
 * Used to sort scouted enemies into melee and ranged without naming them. The
 * longest melee reach on the roster is the Dark Golem's 1.1.
 */
const MELEE_RANGE = fromFloat(1.2);

/**
 * Whether this is a thing the bot fights with.
 *
 * Structural rather than a list of types, so a unit added to the roster joins
 * the army — and the team's army count, which is what triggers a push — on the
 * tick it is trained rather than on the tick someone remembers this function.
 * A Fixomatic has no weapon and is army all the same: it walks with the push
 * and it is not a worker.
 */
function isArmyUnit(def: EntityDef): boolean {
  if (def.isBuilding || def.type === EntityType.Worker) return false;
  return def.damage > 0 || def.repairAmount > 0;
}

/**
 * Base supply headroom before building another depot.
 *
 * Scaled by production capacity below: a base with four barracks burns supply
 * far faster than one, and a fixed buffer leaves the bot permanently blocked
 * with minerals it cannot spend.
 */
const SUPPLY_BUFFER = 6;

/**
 * One think: everything the bot wants to order right now.
 *
 * Ungated on purpose — `ScriptedAgent` decides *when* to think, so that a
 * handicapped agent can think less often without this file knowing. Commands
 * may name any number of units; the agent chunks them to `MAX_COMMAND_UNITS`.
 */
export function botThink(world: World, player: PlayerId): Command[] {
  if (world.player(player).defeated) return [];
  if (world.matchOver) return [];

  const tuning = TUNING;
  const cmds: Command[] = [];
  const s = survey(world, player);

  manageProduction(world, player, s, tuning, cmds);
  manageConstruction(world, player, s, tuning, cmds);
  keepWorkersBusy(world, player, s, cmds);
  manageArmy(world, player, s, tuning, cmds);

  return cmds;
}

interface Survey {
  workers: number[];
  idleWorkers: number[];
  army: number[];
  /** Living and already queued units, including orders from this think. */
  planned: number[];
  fightersPlanned: number;
  commandPosts: number[];
  barracks: number[];
  factories: number[];
  airports: number[];
  airportsPlanned: number;
  /**
   * Factories standing *or* going up.
   *
   * `factories` holds only the finished ones, so gating on that alone can
   * exceed the cap while the first one is still being built, every time a
   * construction slot comes free.
   * A cap on a slow building has to count the ones that are not there yet.
   */
  factoriesPlanned: number;
  depots: number[];
  turrets: number[];
  sites: number[];
  /** Own structures, including construction sites, in creation order. */
  buildings: number[];
  /**
   * Observed patches to send an idle worker to: the ones near a base of ours,
   * or other observed patches when none of ours is left.
   */
  patches: number[];
  /**
   * How many of those were actually near a base of ours.
   *
   * Kept apart from `patches.length` because the fallback deliberately blurs the
   * two: "somewhere to mine" and "my own line is running out" are different
   * questions, and answering the second from the widened list reports the line
   * as healthy at exactly the moment it has run dry.
   */
  homePatches: number;
  /** Live patches currently observed by the team. */
  livePatches: number;
  /** Hostile structures currently observed by the team. */
  enemyTargets: number[];
  /** Visible enemies with a damaging weapon, including static defences. */
  visibleThreats: number[];
  /** Hostile combat units currently in allied sight, grouped by their role. */
  enemyRanged: number;
  enemyMelee: number;
  enemyAir: number;
  enemyArmored: number;
  /**
   * Combat units belonging to anyone on our side, including a partner's.
   *
   * The team's fist. Every bot on the side derives it from the same ascending
   * pass, so all of them agree on how big it is and where its middle is.
   */
  teamArmy: number[];
  /** An observed armed hostile near our buildings that our army can engage. */
  threatened: number;
  minerals: number;
  supplyUsed: number;
  supplyMax: number;
}

/**
 * Single ordered pass over the entity pool.
 *
 * One pass rather than several scattered scans, because every list here is built
 * in ascending index order and that ordering is what makes the bot's choices
 * reproducible.
 */
function survey(world: World, player: PlayerId): Survey {
  const pool = world.pool;
  const s: Survey = {
    workers: [],
    idleWorkers: [],
    army: [],
    planned: new Array<number>(ENTITY_TYPE_COUNT).fill(0),
    fightersPlanned: 0,
    commandPosts: [],
    barracks: [],
    factories: [],
    airports: [],
    airportsPlanned: 0,
    factoriesPlanned: 0,
    depots: [],
    turrets: [],
    sites: [],
    buildings: [],
    patches: [],
    homePatches: 0,
    livePatches: 0,
    enemyTargets: [],
    visibleThreats: [],
    enemyRanged: 0,
    enemyMelee: 0,
    enemyAir: 0,
    enemyArmored: 0,
    teamArmy: [],
    threatened: -1,
    minerals: world.player(player).minerals,
    supplyUsed: world.player(player).supplyUsed,
    supplyMax: world.player(player).supplyMax,
  };

  // Patches are collected before they can be filtered by distance to a base,
  // because the bases are found in this same pass. Two short passes rather than
  // one long one, both in ascending index order.
  const allPatches: number[] = [];
  const hostileUnits: number[] = [];
  const ownBuildings: number[] = [];

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;
    const owner = pool.owner[i]!;

    if (type === EntityType.MineralPatch) {
      if (inAlliedSight(world, player, i) && pool.resourceAmount[i]! > 0) allPatches.push(i);
      continue;
    }
    if (owner === NEUTRAL) continue;

    const def = defOf(type);
    const isArmy = isArmyUnit(def);

    if (!world.areAllied(owner, player)) {
      if (!inAlliedSight(world, player, i)) continue;
      if (def.damage > 0) s.visibleThreats.push(i);
      // Prefer structures as attack targets; killing buildings is what wins.
      if (def.isBuilding) s.enemyTargets.push(i);
      else {
        hostileUnits.push(i);
        // By shape, not by name. A roster that grows would otherwise leave the
        // bot scouting a dozen units it counted as nothing at all. Counters
        // respond to visible roles, not to the names of individual robots.
        if (!isArmy) continue;
        if (def.armor > 0) s.enemyArmored++;
        if (def.flying) s.enemyAir++;
        else if (def.attackRange > MELEE_RANGE) s.enemyRanged++;
        else if (def.damage > 0) s.enemyMelee++;
      }
      continue;
    }

    // Allied, which includes ourselves. A partner's army counts toward the
    // team's, but nothing else of theirs is ours to command.
    if (isArmy) s.teamArmy.push(i);
    if (owner !== player) continue;

    s.planned[type]!++;
    if (isArmy && def.damage > 0) s.fightersPlanned++;
    for (let q = 0; q < pool.prodCount[i]!; q++) {
      const queued = pool.prodAt(i, q);
      s.planned[queued]!++;
      if (isArmyUnit(defOf(queued)) && defOf(queued).damage > 0) s.fightersPlanned++;
    }

    const complete = pool.buildState[i] === BuildState.Complete;
    if (def.isBuilding) ownBuildings.push(i);

    if (isArmy) {
      s.army.push(i);
      continue;
    }

    switch (type) {
      case EntityType.Worker:
        s.workers.push(i);
        if (pool.order[i] === Order.None) s.idleWorkers.push(i);
        break;
      case EntityType.CommandPost:
        if (complete) s.commandPosts.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Barracks:
        if (complete) s.barracks.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Factory:
        s.factoriesPlanned++;
        if (complete) s.factories.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Airport:
        s.airportsPlanned++;
        if (complete) s.airports.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Depot:
        if (complete) s.depots.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Turret:
        if (complete) s.turrets.push(i);
        else s.sites.push(i);
        break;
      default:
        break;
    }
  }

  // Maintained beside the push it always equalled, this was a second copy of
  // one fact that an edit could put out of step. It is the list's length.
  s.livePatches = allPatches.length;

  // Every list in creation order rather than slot order. Ties everywhere
  // below break by position in a list, and slot order is not the same on both
  // halves of a mirrored match — the first player's entities take the low
  // slots at setup, and both halves recycle each other's slots after the
  // first death. Creation order is what the two halves share, so it is the
  // order a bot and its opposite number make the same choices in.
  const byCreation = creationOrder(world);
  s.workers.sort(byCreation);
  s.idleWorkers.sort(byCreation);
  s.army.sort(byCreation);
  s.commandPosts.sort(byCreation);
  s.barracks.sort(byCreation);
  s.factories.sort(byCreation);
  s.airports.sort(byCreation);
  s.depots.sort(byCreation);
  s.turrets.sort(byCreation);
  s.sites.sort(byCreation);
  s.enemyTargets.sort(byCreation);
  s.teamArmy.sort(byCreation);
  hostileUnits.sort(byCreation);
  ownBuildings.sort(byCreation);
  s.buildings = ownBuildings;
  // Patches are neutral and have no creation order of their own: by tile, in
  // this player's canonical frame, which is the same patch seen from either
  // side.
  allPatches.sort(canonicalTileOrder(world, player));

  // Patches worth walking to: near a base of ours. In canonical order, like
  // everything else, so ties in `keepWorkersBusy` break the same way on every
  // peer and the mirrored way on the other half.
  for (const p of allPatches) {
    const home = nearestOf(world, p, s.commandPosts);
    if (home >= 0 && distSqBetween(world, p, home) <= sqRange(HOME_PATCH_RANGE)) {
      s.patches.push(p);
    }
  }
  // Nothing near home is worth walking to, but something somewhere is: take it.
  //
  // The filter above is about *preference* — with four bases on the map, the
  // nearest live patch is occasionally an ally's line or an enemy's, and a
  // worker sent there mines into the wrong bank or dies on arrival. It is not a
  // reason to stop mining. Restricting the fallback to a bot with no Command
  // Post at all covered the fresh-expansion case and missed the far commoner
  // one: a home line that has run dry while patches remain elsewhere, where
  // every worker simply stood still for the rest of the match.
  s.homePatches = s.patches.length;
  if (s.patches.length === 0) {
    for (const p of allPatches) s.patches.push(p);
  }

  s.threatened = nearestThreat(world, hostileUnits, ownBuildings, s.army);
  return s;
}

/** Current sight only; hidden reinforcements must not change the next counter-unit. */
function inAlliedSight(world: World, player: PlayerId, target: number): boolean {
  return pointInAlliedSight(world, player, world.pool.posX[target]!, world.pool.posY[target]!);
}

/** Current visibility only, conservatively on both sides of an exact tile edge.
 * This agrees with the human's floor-based tile visibility without granting
 * the opposite seat extra information when the same point is rotated.
 */
function pointInAlliedSight(world: World, player: PlayerId, x: number, y: number): boolean {
  const tx = toInt(x);
  const ty = toInt(y);
  const minX = x === fromInt(tx) ? tx - 1 : tx;
  const minY = y === fromInt(ty) ? ty - 1 : ty;
  for (let py = minY; py <= ty; py++) {
    for (let px = minX; px <= tx; px++) {
      if (!tileInAlliedSight(world, player, px, py)) return false;
    }
  }
  return true;
}

function tileInAlliedSight(world: World, player: PlayerId, tx: number, ty: number): boolean {
  const pool = world.pool;
  const x = fromInt(tx) + FIX_HALF;
  const y = fromInt(ty) + FIX_HALF;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || !world.areAllied(pool.owner[i]!, player)) continue;
    const sight = defOf(pool.type[i]! as EntityType).sightRange;
    if (sight > 0 && vecLenSqRaw(pool.posX[i]! - x, pool.posY[i]! - y) <= sqRange(sight)) {
      return true;
    }
  }
  return false;
}

/**
 * Creation order shared by both halves of a match: seat within the half, then
 * the owner's creation ordinal, then the slot for the one case that agrees on
 * both — an entity and its own mirror image.
 */
function creationOrder(world: World): (a: number, b: number) => number {
  const pool = world.pool;
  return (a, b) =>
    world.ownerCanonical(pool.owner[a]!) - world.ownerCanonical(pool.owner[b]!) ||
    pool.serial[a]! - pool.serial[b]! ||
    pool.owner[a]! - pool.owner[b]! ||
    a - b;
}

/** Tile order in `player`'s canonical frame, for entities nobody owns. */
function canonicalTileOrder(world: World, player: PlayerId): (a: number, b: number) => number {
  const { map, pool } = world;
  const flip = world.flipOf(player);
  const key = (i: number): number =>
    map.canonicalIndex(map.index(pool.tileX[i]!, pool.tileY[i]!), flip);
  return (a, b) => key(a) - key(b) || a - b;
}

/**
 * The nearest of `others` to entity `i`, or -1 when the list is empty.
 *
 * Ties break by position in `others`, which every caller builds in creation
 * order — a strict total order the two halves of a match share, and the reason
 * this is one helper rather than the four hand-rolled copies of the same loop
 * it replaced.
 */
function nearestOf(world: World, i: number, others: readonly number[]): number {
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const j of others) {
    const d = vecLenSqRaw(pool.posX[j]! - pool.posX[i]!, pool.posY[j]! - pool.posY[i]!);
    if (d < bestDist) {
      bestDist = d;
      best = j;
    }
  }
  return best;
}

/** Squared distance between two entities, in the raw space `sqRange` compares. */
function distSqBetween(world: World, a: number, b: number): number {
  const pool = world.pool;
  return vecLenSqRaw(pool.posX[b]! - pool.posX[a]!, pool.posY[b]! - pool.posY[a]!);
}

/**
 * The observed hostile nearest one of our structures, if our army can engage it.
 *
 * A lone scouting worker is not an attack, and pulling an army home for one is
 * how a bot gets pulled out of position on purpose. Unarmed support and flyers
 * an entirely melee army cannot hit must not repeatedly recall that army either.
 * Aim at the threat: aiming at our building can walk away from the enemy until
 * it leaves sight, then send the army forward to discover the same threat again.
 */
function nearestThreat(
  world: World,
  hostileUnits: readonly number[],
  ownBuildings: readonly number[],
  army: readonly number[],
): number {
  if (ownBuildings.length === 0 || army.length === 0) return -1;
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const h of hostileUnits) {
    if (pool.type[h] === EntityType.Worker) continue;
    const hostile = defOf(pool.type[h]! as EntityType);
    if (hostile.damage === 0) continue;
    const building = nearestOf(world, h, ownBuildings);
    if (building < 0) continue;
    const distSq = distSqBetween(world, h, building);
    if (distSq > sqRange(DEFEND_RANGE) || distSq >= bestDist) continue;
    if (
      !army.some((i) => {
        const def = defOf(pool.type[i]! as EntityType);
        return def.damage > 0 && (!hostile.flying || def.canHitAir);
      })
    )
      continue;
    // Do not issue a ground order the executor will reject for a flyer over
    // deep cliffs. A flying defender can still reach such a target directly.
    if (
      hostile.flying &&
      !standableTarget(world, pool.owner[army[0]!]!, pool.posX[h]!, pool.posY[h]!, {
        x: 0,
        y: 0,
      }) &&
      !army.some((i) => {
        const def = defOf(pool.type[i]! as EntityType);
        return def.flying && def.damage > 0 && def.canHitAir;
      })
    )
      continue;
    bestDist = distSq;
    best = h;
  }
  return best;
}

/** Any worker with nothing to do goes back to the nearest live patch. */
function keepWorkersBusy(world: World, player: PlayerId, s: Survey, cmds: Command[]): void {
  if (s.patches.length === 0) return;
  const pool = world.pool;

  for (const w of s.idleWorkers) {
    const id = pool.idAt(w);
    if (
      cmds.some(
        (c) =>
          (c.type === CommandType.Build && c.worker === id) ||
          (c.type === CommandType.Move && c.units.includes(id)),
      )
    )
      continue;
    const best = nearestOf(world, w, s.patches);
    if (best < 0) continue;
    cmds.push({
      type: CommandType.Harvest,
      player,
      units: [id],
      target: pool.idAt(best),
    });
  }
}

/** Train workers up to saturation, then pour everything into army. */
function manageProduction(
  world: World,
  player: PlayerId,
  s: Survey,
  tuning: Tuning,
  cmds: Command[],
): void {
  const pool = world.pool;
  if (
    s.commandPosts.length === 0 &&
    s.workers.length > 0 &&
    !s.sites.some((i) => pool.type[i] === EntityType.CommandPost) &&
    s.minerals >= defOf(EntityType.CommandPost).mineralCost
  )
    return;
  // Give the first upgrade a deliberate queue-draining window. Otherwise a
  // two-deep producer never becomes idle and its second tier stays locked.
  // Other producers keep the army moving while this one invests in technology.
  const reserved = new Set<number>();
  for (const [buildings, threshold] of [
    [s.barracks, 6],
    [s.factories, 10],
  ] as const) {
    if (s.fightersPlanned < threshold) continue;
    const alreadyAdvanced = buildings.some(
      (i) => pool.buildingLevel[i]! >= 2 || pool.upgrading[i] === 1,
    );
    if (alreadyAdvanced && s.minerals < DEEP_QUEUE_MINERALS) continue;
    const candidate = buildings.find((i) => pool.buildingLevel[i] === 1 && pool.upgrading[i] === 0);
    if (candidate === undefined) continue;
    const upgrade = buildingUpgrade(pool.type[candidate]! as EntityType)!;
    // Do not idle our only producer to save for technology we cannot afford.
    if (s.minerals < upgrade.mineralCost) continue;
    reserved.add(candidate);
    if (pool.prodCount[candidate] === 0 && s.minerals >= upgrade.mineralCost) {
      cmds.push({ type: CommandType.UpgradeBuilding, player, building: pool.idAt(candidate) });
      s.minerals -= upgrade.mineralCost;
    }
  }

  const supplyFree = s.supplyMax - s.supplyUsed;
  if (supplyFree <= 0) return;

  // Worker target scales with how many bases there are to work: a second
  // Command Post with nobody mining at it is 400 minerals of decoration.
  const wantWorkers = Math.min(
    tuning.targetWorkers * Math.max(1, s.commandPosts.length),
    s.homePatches * 3,
  );
  if (s.planned[EntityType.Worker]! < wantWorkers) {
    // Every base trains, not just the first. One Command Post queueing all the
    // workers is what left an expansion's mineral line empty for minutes.
    for (const hq of s.commandPosts) {
      if (s.planned[EntityType.Worker]! >= wantWorkers) break;
      if (pool.prodCount[hq]! >= 2) continue;
      if (s.minerals < defOf(EntityType.Worker).mineralCost) break;
      cmds.push({
        type: CommandType.Train,
        player,
        building: pool.idAt(hq),
        unit: EntityType.Worker,
      });
      s.minerals -= defOf(EntityType.Worker).mineralCost;
      s.planned[EntityType.Worker]!++;
    }
  }

  // Two deep normally, so income remains available for buildings and new
  // technology; fill light queues only when production cannot spend the bank.
  const depth = s.minerals >= DEEP_QUEUE_MINERALS ? MAX_PRODUCTION_QUEUE : 2;
  for (const buildings of [s.barracks, s.factories, s.airports]) {
    for (const building of buildings) {
      if (reserved.has(building) || pool.upgrading[building] === 1) continue;
      const type = pool.type[building]! as EntityType;
      if (pool.prodCount[building]! >= (type === EntityType.Barracks ? depth : 2)) continue;
      const available = productionOptions(type, pool.buildingLevel[building]!);
      const unit = pickUnitToTrain(s, available);
      if (unit === null) continue;
      cmds.push({ type: CommandType.Train, player, building: pool.idAt(building), unit });
      s.minerals -= defOf(unit).mineralCost;
      s.planned[unit]!++;
      if (defOf(unit).damage > 0) s.fightersPlanned++;
    }
  }
}

/** Choose the least represented affordable role, including units already in queues.
 *
 * Tick-based rotations lock onto particular train durations and can omit whole
 * roles forever. Comparing integer count/weight ratios makes the mix survive a
 * timing change, and lets losses and scouting naturally shift the next choice.
 */
function balancedUnit(
  s: Survey,
  roster: readonly (readonly [EntityType, number])[],
): EntityType | null {
  let best: EntityType | null = null;
  let bestWeight = 1;
  for (const [type, weight] of roster) {
    if (defOf(type).mineralCost > s.minerals) continue;
    if (best === null || (s.planned[type]! + 1) * bestWeight < (s.planned[best]! + 1) * weight) {
      best = type;
      bestWeight = weight;
    }
  }
  return best;
}

function pickUnitToTrain(s: Survey, available: readonly EntityType[]): EntityType | null {
  // Repairers need both an existing fighting core and an upgraded Barracks.
  if (
    available.includes(EntityType.Fixomatic) &&
    s.fightersPlanned >= 6 &&
    s.planned[EntityType.Fixomatic]! < Math.max(1, Math.floor(s.fightersPlanned / 8)) &&
    s.minerals >= defOf(EntityType.Fixomatic).mineralCost
  )
    return EntityType.Fixomatic;

  const air = s.enemyAir >= 2 && s.enemyAir * 2 >= s.enemyRanged + s.enemyMelee;
  const swarm = s.enemyMelee >= 3 && s.enemyMelee > s.enemyRanged;
  const armor = s.enemyArmored >= 2;
  const roster: readonly (readonly [EntityType, number])[] = [
    [EntityType.Burstbot, air ? 6 : armor ? 2 : 4],
    [EntityType.Slicebot, armor ? 3 : 2],
    [EntityType.Firespout, swarm ? 3 : 1],
    [EntityType.Arclight, air || swarm ? 3 : 2],
    [EntityType.Boomwalker, swarm ? 3 : 1],
    [EntityType.Sentry, 2],
    [EntityType.Piercebot, air || armor ? 4 : 2],
    [EntityType.DarkGolem, armor ? 2 : 1],
    [EntityType.IceGolem, swarm ? 2 : 1],
    [EntityType.Beamdrone, 3],
    [EntityType.Plasmodrone, 1],
  ];
  // During an air threat, devote the available factory capacity to weapons
  // that can actually hit it. No unavailable cross-building choices escape.
  const options = roster.filter(
    ([type]) => available.includes(type) && (!air || defOf(type).canHitAir),
  );
  return balancedUnit(s, options);
}

/**
 * Keep construction moving: staff existing sites, then start new ones.
 *
 * Staffing comes first and is the important half. A builder whose order gets
 * cleared — a failed path, the site becoming momentarily unreachable — leaves an
 * orphaned site behind, and nothing else in the bot ever notices. That stalled
 * every subsequent building decision and left the bot sitting on thousands of
 * unspent minerals for the rest of the match.
 */
function manageConstruction(
  world: World,
  player: PlayerId,
  s: Survey,
  tuning: Tuning,
  cmds: Command[],
): void {
  const pool = world.pool;
  if (staffOrphanedSites(world, player, s, cmds)) return;

  if (s.commandPosts.length === 0) {
    if (s.sites.some((i) => pool.type[i] === EntityType.CommandPost)) return;
    const def = defOf(EntityType.CommandPost);
    if (s.minerals < def.mineralCost) return;
    const builder = pickBuilder(world, s);
    if (builder < 0) return;
    const { canonical, flip } = mirroredHalf(player, world.map.starts.length);
    const start = world.map.starts[canonical]!;
    const half = def.footprint >> 1;
    const x = flip
      ? mirrorTile(world.map.width, start.tileX - half, def.footprint)
      : start.tileX - half;
    const y = flip
      ? mirrorTile(world.map.height, start.tileY - half, def.footprint)
      : start.tileY - half;
    const spot =
      world.map.canPlace(x, y, def.footprint) &&
      safeConstruction(world, s.visibleThreats, x, y, def.footprint)
        ? { x, y }
        : findBuildSpot(world, x, y, def.footprint, def.footprint, flip, s.visibleThreats);
    if (spot && footprintInSight(world, player, spot.x, spot.y, def.footprint)) {
      cmds.push({
        type: CommandType.Build,
        player,
        worker: pool.idAt(builder),
        building: EntityType.CommandPost,
        tileX: spot.x,
        tileY: spot.y,
      });
    }
    return;
  }

  if (s.sites.length >= tuning.maxSites) return;

  let builder = pickBuilder(world, s);
  if (builder < 0) return;

  const hq = s.commandPosts[0]!;
  const supplyFree = s.supplyMax - s.supplyUsed;
  // More production capacity means supply drains faster, so keep more headroom.
  const buffer = SUPPLY_BUFFER + (s.barracks.length + s.factories.length + s.airports.length) * 4;
  // A home mineral line that is nearly out is its own reason to expand, whatever
  // the bank looks like: waiting for a threshold that income can no longer reach
  // is how a bot mines itself to a standstill on a full map.
  const patchesRunningOut = s.homePatches <= 2;

  let want: EntityType | null = null;
  if (s.supplyMax < SUPPLY_MAX && supplyFree < buffer) {
    want = EntityType.Depot;
  } else if (s.barracks.length < 1) {
    want = EntityType.Barracks;
  } else if (s.turrets.length < tuning.maxTurrets && s.minerals >= 300) {
    want = EntityType.Turret;
  } else if (s.factoriesPlanned >= 1 && s.airportsPlanned === 0 && s.fightersPlanned >= 6) {
    want = EntityType.Airport;
  } else if (
    s.barracks.length >= 2 &&
    s.factoriesPlanned < tuning.maxFactories &&
    (s.factoriesPlanned === 0 || s.minerals >= DEEP_QUEUE_MINERALS)
  ) {
    // Behind the first two Barracks on purpose. Tech buys no fighting units
    // on its own, so the light army needs to exist while it builds. The second one
    // waits on a bank the Barracks cannot spend: a Factory queues two units at
    // a time, so it is the outlet that absorbs a pile of minerals, and adding
    // one is worth more than a tenth Barracks queueing behind the same eight
    // patches.
    want = EntityType.Factory;
  } else if (
    (s.minerals >= tuning.expandAtMinerals || patchesRunningOut) &&
    expansionSite(world, player, s, tuning)
  ) {
    // Floating this much means the mineral line at home cannot absorb the
    // income any more. A second base is what turns it into more income rather
    // than more barracks queueing behind the same eight patches.
    want = EntityType.CommandPost;
  } else if (s.barracks.length < tuning.maxBarracks && s.minerals >= 300) {
    // Excess minerals are wasted minerals; convert them into production.
    want = EntityType.Barracks;
  }

  if (want === null) return;
  const def = defOf(want);
  if (s.minerals < def.mineralCost) return;

  // An expansion goes on its site; everything else goes next to the base it
  // supports.
  const site = want === EntityType.CommandPost ? expansionSite(world, player, s, tuning) : null;
  const spot =
    site ??
    findBuildSpot(
      world,
      pool.tileX[hq]!,
      pool.tileY[hq]!,
      defOf(EntityType.CommandPost).footprint,
      def.footprint,
      world.flipOf(player),
      s.visibleThreats,
    );
  if (!spot) return;

  if (site) {
    const x = fromInt(site.x) + fromFloat(def.footprint / 2);
    const y = fromInt(site.y) + fromFloat(def.footprint / 2);
    const scout = s.workers.find(
      (i) => pool.order[i] === Order.Move && pool.orderX[i] === x && pool.orderY[i] === y,
    );
    if (scout !== undefined) {
      if (!footprintInSight(world, player, site.x, site.y, def.footprint)) return;
      builder = scout;
    } else {
      builder = pickBuilder(world, s, x, y);
      if (builder < 0) return;
    }
  }

  if (site && !footprintInSight(world, player, site.x, site.y, def.footprint)) {
    // Public expansion coordinates are a scouting plan, not permission to
    // inspect whatever hidden structure or resource amount happens to be there.
    cmds.push({
      type: CommandType.Move,
      player,
      units: [pool.idAt(builder)],
      x: fromInt(site.x) + fromFloat(def.footprint / 2),
      y: fromInt(site.y) + fromFloat(def.footprint / 2),
    });
    return;
  }

  cmds.push({
    type: CommandType.Build,
    player,
    worker: pool.idAt(builder),
    building: want,
    tileX: spot.x,
    tileY: spot.y,
  });
}

/**
 * The nearest expansion worth taking, as a Command Post's top-left tile.
 *
 * "Worth taking" means nobody already has a Command Post on it — including this
 * player and a partner, so a team claims each site once rather than two bots
 * racing a worker each to the same tile — and it is nearer to home than to any
 * enemy. Walking a lone worker past the enemy's front door to build a base is
 * not an expansion, it is a donation.
 */
function expansionSite(
  world: World,
  player: PlayerId,
  s: Survey,
  tuning: Tuning,
): { x: number; y: number } | null {
  const { map, pool } = world;
  if (s.commandPosts.length === 0 || s.commandPosts.length >= tuning.maxBases) return null;

  const home = s.commandPosts[0]!;
  const homeX = pool.posX[home]!;
  const homeY = pool.posY[home]!;
  const def = defOf(EntityType.CommandPost);
  const half = def.footprint >> 1;
  const flip = world.flipOf(player);

  let best: { x: number; y: number } | null = null;
  let bestDist = Infinity;
  let bestKey = 0;

  for (let e = 0; e < map.expansions.length; e++) {
    // Sites are stored in mirrored halves, and a site's Command Post has to be
    // the exact rotation of the canonical one's: `site - 2` applied to the
    // rotated point is a tile off on both axes, the same footprint bug the
    // opening was cured of, and it put every second-half expansion a tile
    // further from its own minerals for the rest of the match.
    const mirrored = mirroredHalf(e, map.expansions.length);
    const canonical = map.expansions[mirrored.canonical]!;
    const x = mirrored.flip
      ? mirrorTile(map.width, canonical.tileX - half, def.footprint)
      : canonical.tileX - half;
    const y = mirrored.flip
      ? mirrorTile(map.height, canonical.tileY - half, def.footprint)
      : canonical.tileY - half;
    if (!safeConstruction(world, s.visibleThreats, x, y, def.footprint)) continue;
    if (footprintInSight(world, player, x, y, def.footprint) && !map.canPlace(x, y, def.footprint))
      continue;

    // Everything in world units from the centres of things, as the rest of
    // the simulation measures. Mixing a site's centre tile with a building's
    // top-left tile gave the two halves different distances to identical
    // sites, and the thresholds below went with them.
    const siteX = fromInt(x + half);
    const siteY = fromInt(y + half);
    const dHome = vecLenSqRaw(siteX - homeX, siteY - homeY);
    let contested = false;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (pool.type[i] !== EntityType.CommandPost) continue;
      if (!world.areAllied(pool.owner[i]!, player) && !inAlliedSight(world, player, i)) continue;
      if (vecLenSqRaw(pool.posX[i]! - siteX, pool.posY[i]! - siteY) < sqRange(CLAIMED_RANGE)) {
        contested = true;
        break;
      }
    }
    if (contested) continue;

    // Nearer to us than to any enemy base, or it is indefensible. An ally's
    // base does not count against it — a site behind a partner is safer than
    // one behind us, not less safe.
    let enemyCloser = false;
    for (const i of s.enemyTargets) {
      if (vecLenSqRaw(pool.posX[i]! - siteX, pool.posY[i]! - siteY) < dHome) {
        enemyCloser = true;
        break;
      }
    }
    if (enemyCloser) continue;

    // Starting positions are public map geometry. An unseen opponent still
    // owns its side of the map; scouting cannot justify a worker crossing it.
    for (let enemy = 0; enemy < world.players.length; enemy++) {
      if (world.areAllied(enemy, player)) continue;
      const start = map.starts[enemy]!;
      const ex = fromInt(start.tileX) + FIX_HALF;
      const ey = fromInt(start.tileY) + FIX_HALF;
      if (vecLenSqRaw(ex - siteX, ey - siteY) < dHome) {
        enemyCloser = true;
        break;
      }
    }
    if (enemyCloser) continue;

    // Two sites the same distance from home — the contested pair on the
    // four-corner map — are told apart in this player's canonical frame.
    const key = map.canonicalIndex(map.index(x, y), flip);
    if (dHome < bestDist || (dHome === bestDist && key < bestKey)) {
      bestDist = dHome;
      bestKey = key;
      best = { x, y };
    }
  }
  return best;
}

/** Send a worker to any construction site nobody is working on. */
function staffOrphanedSites(world: World, player: PlayerId, s: Survey, cmds: Command[]): boolean {
  if (s.sites.length === 0) return false;
  const pool = world.pool;

  // Which sites already have someone assigned.
  const staffed = new Set<number>();
  for (const w of s.workers) {
    if (pool.order[w] !== Order.Build) continue;
    const target = pool.orderTarget[w]!;
    if (target !== -1 && pool.isAlive(target)) staffed.add(target & 0xffff);
  }

  for (const site of s.sites) {
    if (staffed.has(site)) continue;
    if (
      !safeConstruction(
        world,
        s.visibleThreats,
        pool.tileX[site]!,
        pool.tileY[site]!,
        defOf(pool.type[site]! as EntityType).footprint,
      )
    )
      continue;
    const builder = pickBuilder(world, s);
    if (builder < 0) return false;
    cmds.push({
      type: CommandType.Build,
      player,
      worker: pool.idAt(builder),
      building: pool.type[site]! as EntityType,
      tileX: pool.tileX[site]!,
      tileY: pool.tileY[site]!,
    });
    // Reserve this worker for the orphaned site. A new build in the same think
    // would otherwise take the same worker and overwrite this assignment.
    return true;
  }
  return false;
}

function footprintInSight(
  world: World,
  player: PlayerId,
  x: number,
  y: number,
  size: number,
): boolean {
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      if (!tileInAlliedSight(world, player, x + dx, y + dy)) return false;
    }
  }
  return true;
}

/**
 * Choose a worker to construct with.
 *
 * With a destination, prefer the nearest available worker. Otherwise prefer an
 * idle worker, then the newest harvester. Preserve the last miner unless the
 * Command Post itself needs rebuilding.
 */
function pickBuilder(world: World, s: Survey, x?: number, y?: number): number {
  if (s.workers.length <= (s.commandPosts.length > 0 ? 1 : 0)) return -1;
  if (x !== undefined && y !== undefined) {
    let best = -1;
    let bestDistance = Infinity;
    for (const i of s.workers) {
      if (world.pool.order[i] !== Order.None && world.pool.order[i] !== Order.Harvest) continue;
      const distance = vecLenSqRaw(world.pool.posX[i]! - x, world.pool.posY[i]! - y);
      if (distance < bestDistance) {
        best = i;
        bestDistance = distance;
      }
    }
    return best;
  }
  if (s.idleWorkers.length > 0) return s.idleWorkers[0]!;
  for (let k = s.workers.length - 1; k >= 0; k--) {
    const w = s.workers[k]!;
    if (world.pool.order[w] === Order.Harvest) return w;
  }
  return -1;
}

/**
 * Spiral outward from the base for a legal placement.
 *
 * A fixed spiral rather than random probing, so every peer picks the same tile.
 *
 * The spiral is walked in the player's canonical frame and rotated for the
 * second half, so a bot and its opposite number lay out mirror-image bases.
 * Walked in absolute map directions — top edge first, from its left corner —
 * it filled one seat's *rear* first and the other seat's *front*: over the
 * first eight structures, the second seat put all eight, both turrets among
 * them, on its enemy-facing side and the first seat put four, with both
 * turrets behind. One side's defences fired on every attack and the other's
 * never did, which decided fifteen of sixteen mirror matches.
 *
 * `originFootprint` is the footprint of the building the spiral starts from,
 * because rotating a footprint about the origin's centre moves its top-left by
 * the difference in size: the mirror of offset `dx` for a footprint `f` from a
 * base of footprint `F` is `(F - f) - dx`.
 *
 * Every candidate must also keep a clear one-tile moat (see `hasClearMoat`).
 * Without that rule the bot packs its structures into a solid ring and seals its
 * own army inside the base — measured: 106 of 106 combat units unable to reach
 * the enemy, so neither side could ever win.
 */
function findBuildSpot(
  world: World,
  originX: number,
  originY: number,
  originFootprint: number,
  footprint: number,
  flip: boolean,
  visibleThreats: readonly number[],
): { x: number; y: number } | null {
  for (let ring = 4; ring <= 26; ring += 2) {
    for (let step = 0; step < ring * 8; step += 3) {
      const side = Math.floor(step / (ring * 2));
      const along = step % (ring * 2);
      let dx = 0;
      let dy = 0;
      switch (side) {
        case 0:
          dx = -ring + along;
          dy = -ring;
          break;
        case 1:
          dx = ring;
          dy = -ring + along;
          break;
        case 2:
          dx = ring - along;
          dy = ring;
          break;
        default:
          dx = -ring;
          dy = ring - along;
          break;
      }
      const x = originX + (flip ? originFootprint - footprint - dx : dx);
      const y = originY + (flip ? originFootprint - footprint - dy : dy);
      if (!safeConstruction(world, visibleThreats, x, y, footprint)) continue;
      if (!world.map.canPlace(x, y, footprint)) continue;
      if (!hasClearMoat(world, x, y, footprint)) continue;
      // The moat alone is not enough: enough moated buildings still ring the
      // base into a closed shell. Only commit to a spot that provably leaves the
      // base connected to the rest of the map.
      if (!keepsBaseConnected(world, x, y, footprint, originX, originY, flip)) continue;
      return { x, y };
    }
  }
  return null;
}

/** A foundation must not be placed directly under a weapon we can already see.
 * Measure to the whole footprint, so a large structure's exposed edge counts.
 * The survey supplies observed enemies only; hidden positions never veto a spot.
 */
function safeConstruction(
  world: World,
  threats: readonly number[],
  tileX: number,
  tileY: number,
  footprint: number,
): boolean {
  const pool = world.pool;
  const left = fromInt(tileX),
    right = fromInt(tileX + footprint);
  const top = fromInt(tileY),
    bottom = fromInt(tileY + footprint);
  for (const i of threats) {
    const def = defOf(pool.type[i]! as EntityType);
    const dx = Math.max(left - pool.posX[i]!, 0, pool.posX[i]! - right);
    const dy = Math.max(top - pool.posY[i]!, 0, pool.posY[i]! - bottom);
    if (vecLenSqRaw(dx, dy) <= sqRange(def.attackRange + def.radius)) return false;
  }
  return true;
}

/**
 * True when the ring of tiles immediately around a footprint is free.
 *
 * Requiring this of every building guarantees no two structures ever end up
 * flush against each other, so walkable lanes always survive between them and
 * the base cannot be accidentally sealed shut.
 */
function hasClearMoat(world: World, tileX: number, tileY: number, footprint: number): boolean {
  for (let y = tileY - 1; y <= tileY + footprint; y++) {
    for (let x = tileX - 1; x <= tileX + footprint; x++) {
      const onPerimeter =
        x === tileX - 1 || x === tileX + footprint || y === tileY - 1 || y === tileY + footprint;
      if (!onPerimeter) continue;
      if (!world.map.isWalkable(x, y)) return false;
    }
  }
  return true;
}

/**
 * Would placing here seal the base in?
 *
 * Tentatively marks the footprint occupied, floods outward from beside the
 * Command Post, and checks the middle of the map is still reachable. This is the
 * only check that actually guarantees the property — a purely local rule like
 * the moat cannot see that twenty individually-legal buildings have closed a
 * ring. Without it the bots walled their own armies in and no match could ever
 * be won.
 *
 * Costs one flood fill per building placed, a few times a minute, and uses only
 * integer state so it stays deterministic.
 */
function keepsBaseConnected(
  world: World,
  tileX: number,
  tileY: number,
  footprint: number,
  originX: number,
  originY: number,
  flip: boolean,
): boolean {
  const map = world.map;
  map.setOccupied(tileX, tileY, footprint, 1);
  try {
    const start = findOpenTileNear(world, originX, originY, flip);
    if (start < 0) return false;

    // The middle of the map, as seen from this half: on an even-sized map the
    // centre tile has no mirror image of its own, so each half floods toward
    // its own side of the centre and the two answers agree by symmetry.
    const midX = flip ? map.width - 1 - (map.width >> 1) : map.width >> 1;
    const midY = flip ? map.height - 1 - (map.height >> 1) : map.height >> 1;
    const target = findOpenTileNear(world, midX, midY, flip);
    if (target < 0) return true; // nothing to connect to; do not block building

    const w = map.width;
    const seen = new Uint8Array(w * map.height);
    const queue: number[] = [start];
    seen[start] = 1;

    for (let head = 0; head < queue.length; head++) {
      const cur = queue[head]!;
      if (cur === target) return true;
      const cx = cur % w;
      const cy = (cur / w) | 0;
      // Four-directional is deliberately conservative: anything connected
      // orthogonally is also connected under the eight-directional movement
      // rules, never the other way round.
      for (let d = 0; d < 4; d++) {
        const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
        const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
        if (!map.isWalkable(nx, ny)) continue;
        const ni = ny * w + nx;
        if (seen[ni] === 1) continue;
        seen[ni] = 1;
        queue.push(ni);
      }
    }
    return false;
  } finally {
    // Always undo the tentative placement, on every exit path.
    map.setOccupied(tileX, tileY, footprint, 0);
  }
}

/**
 * Nearest walkable tile to a point, searched in fixed ring order — in the
 * player's canonical frame, so the two halves search mirrored rings.
 */
function findOpenTileNear(world: World, tx: number, ty: number, flip: boolean): number {
  const map = world.map;
  if (map.isWalkable(tx, ty)) return map.index(tx, ty);
  const sign = flip ? -1 : 1;
  for (let r = 1; r <= 20; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx > -r && dx < r && dy > -r && dy < r) continue;
        const x = tx + sign * dx;
        const y = ty + sign * dy;
        if (map.isWalkable(x, y)) return map.index(x, y);
      }
    }
  }
  return -1;
}

/**
 * Defend, then attack.
 *
 * Order matters: an army walking across the map while its own Command Post is
 * being shot is the single most expensive thing a bot can do, and the old one
 * did it every match. Coming home is checked first and, when it applies, is the
 * only order issued.
 */
function manageArmy(
  world: World,
  player: PlayerId,
  s: Survey,
  tuning: Tuning,
  cmds: Command[],
): void {
  const pool = world.pool;
  const beat = Math.floor(world.tick / THINK_INTERVAL);

  // --- defence ------------------------------------------------------------
  //
  // Reacted to more often than an attack is re-aimed: an attack that is one
  // second stale costs a little walking, a defence that is one second stale
  // costs a building.
  if (tuning.defendsHome && s.threatened >= 0 && s.army.length > 0) {
    if (beat % 2 !== 0) return;
    orderArmy(world, player, s.army, pool.posX[s.threatened]!, pool.posY[s.threatened]!, cmds);
    return;
  }

  // Orders retain the last observed defensive position. Inspect it before
  // resuming an offensive plan; fog alone does not mean the threat is gone.
  const defence = armyObjective(world, s.army);
  if (
    tuning.defendsHome &&
    defence &&
    !pointInAlliedSight(world, player, defence.x, defence.y) &&
    s.buildings.some(
      (i) =>
        vecLenSqRaw(pool.posX[i]! - defence.x, pool.posY[i]! - defence.y) <= sqRange(DEFEND_RANGE),
    )
  ) {
    if (beat % 2 === 0) orderArmy(world, player, s.army, defence.x, defence.y, cmds);
    return;
  }

  // --- attack -------------------------------------------------------------

  // Normally wait for a critical mass before committing. But without observed
  // resources or minerals banked, there is no known way to make that army any
  // bigger, and holding out for a threshold it can no longer reach turns a won
  // position into a permanent draw — observed with a crippled opponent still
  // standing because the winner was one unit short of attacking. With no way to
  // reinforce, whatever is left goes in.
  const cheapest = defOf(EntityType.Burstbot).mineralCost;
  const canReinforce = s.livePatches > 0 || s.minerals >= cheapest;
  const required = canReinforce ? tuning.attackArmySize : 1;

  // With the economy dead and no army left, workers are the only pieces on the
  // board. They fight badly but they do fight, and a base full of them idling
  // next to exhausted patches while the opponent's last buildings stand is a
  // draw by inaction rather than a decision.
  const attackers = s.army.length > 0 ? s.army : !canReinforce ? s.workers : [];
  if (attackers.length === 0) return;

  // The count that decides whether to commit is the *team's*, not ours. Two
  // bots each waiting for their own eight units attack four seconds apart and
  // are beaten one at a time; counting together, they arrive together.
  //
  // Only while there is an army to count, though. `teamArmy` holds combat units
  // and nothing else, so measuring the worker last stand against it compares a
  // group of workers to a count that is necessarily zero — and the bot stands
  // in its dead base forever, which is the exact draw by inaction the paragraph
  // above exists to prevent.
  const marching = attackers === s.army;
  const force = tuning.coordinates && marching ? s.teamArmy : attackers;
  const committed = force.filter((i) => defOf(pool.type[i]! as EntityType).damage > 0).length;
  if (committed < required) return;

  // Re-issue occasionally rather than every think tick, so units get a chance
  // to actually walk somewhere before being redirected.
  //
  // Note the floor: every player thinks on the same tick now, but this used to
  // divide a staggered tick and produce a fraction for every player but the
  // first, so `% 6` could never equal zero and player 1 never attacked at all.
  if (beat % 6 !== 0) return;

  const target = pickAttackTarget(world, s, tuning, player);
  if (target >= 0) {
    orderArmy(world, player, attackers, pool.posX[target]!, pool.posY[target]!, cmds);
    return;
  }
  const scout = scoutingPoint(world, player, s);
  if (scout) orderArmy(world, player, attackers, scout.x, scout.y, cmds);
}

/** Reinforce and retarget without repeatedly resetting the same march or wind-up. */
function orderArmy(
  world: World,
  player: PlayerId,
  army: readonly number[],
  x: number,
  y: number,
  cmds: Command[],
): void {
  const pool = world.pool;
  const units = army
    .filter((i) => {
      if (pool.attackWindup[i]! > 0) return false;
      return (
        pool.order[i] !== Order.AttackMove ||
        vecLenSqRaw(pool.orderX[i]! - x, pool.orderY[i]! - y) > sqRange(fromInt(8))
      );
    })
    .map((i) => pool.idAt(i));
  if (units.length > 0) cmds.push({ type: CommandType.AttackMove, player, units, x, y });
}

/** Choose the main march, not a stale order on one unit finishing a swing.
 * Formation destinations near each other count as the same objective. Lists
 * arrive in creation order, which also breaks equal-sized group ties.
 */
function armyObjective(
  world: World,
  army: readonly number[],
  points: readonly { x: number; y: number }[] = [],
  includeCompleted = false,
): { x: number; y: number; index: number; active: boolean } | null {
  const pool = world.pool;
  const groups: { x: number; y: number; index: number; count: number; active: boolean }[] = [];
  for (const i of army) {
    const active = pool.order[i] === Order.AttackMove;
    if (!active && !(includeCompleted && pool.order[i] === Order.None)) continue;
    let x = pool.orderX[i]!,
      y = pool.orderY[i]!;
    // New units have never received a destination.
    if (!active && x === 0 && y === 0) continue;
    let index = -1;
    let distance = sqRange(fromInt(8));
    for (let k = 0; k < points.length; k++) {
      const d = vecLenSqRaw(x - points[k]!.x, y - points[k]!.y);
      if (d <= distance && (index < 0 || d < distance)) {
        index = k;
        distance = d;
      }
    }
    if (index >= 0) ({ x, y } = points[index]!);
    else if (!active) continue;
    const group = groups.find((g) =>
      index >= 0
        ? g.index === index
        : g.index < 0 && vecLenSqRaw(x - g.x, y - g.y) <= sqRange(fromInt(8)),
    );
    if (group) {
      group.count++;
      group.active ||= active;
    } else groups.push({ x, y, index, count: 1, active });
  }
  let best: (typeof groups)[number] | null = null;
  for (const group of groups) if (!best || group.count > best.count) best = group;
  return best;
}

/** Search public start/expansion positions; never inspect an unseen enemy entity.
 * The current (or just completed) order is the search cursor. Advance around
 * the public sites rather than restarting at the first site that left sight.
 * Orders carry this progress through command delays without private bot state.
 */
function scoutingPoint(world: World, player: PlayerId, s: Survey): { x: number; y: number } | null {
  const { map } = world;
  const points: { x: number; y: number }[] = [];
  for (let p = 0; p < world.players.length; p++) {
    if (world.areAllied(p, player)) continue;
    const start = map.starts[p]!;
    points.push({ x: fromInt(start.tileX) + FIX_HALF, y: fromInt(start.tileY) + FIX_HALF });
  }
  // Keep the search order in the requesting player's frame on either seat.
  const flip = world.flipOf(player);
  points.sort(
    (a, b) =>
      map.tileOfPosFor(a.x, a.y, flip) * (flip ? -1 : 1) -
      map.tileOfPosFor(b.x, b.y, flip) * (flip ? -1 : 1),
  );
  const expansions = map.expansions.map((site) => ({
    x: fromInt(site.tileX) + FIX_HALF,
    y: fromInt(site.tileY) + FIX_HALF,
  }));
  expansions.sort(
    (a, b) =>
      map.tileOfPosFor(a.x, a.y, flip) * (flip ? -1 : 1) -
      map.tileOfPosFor(b.x, b.y, flip) * (flip ? -1 : 1),
  );
  points.push(...expansions);
  if (points.length === 0) return null;
  const scouts = s.teamArmy.length > 0 ? s.teamArmy : s.workers;
  const objective = armyObjective(world, scouts, points, true);
  if (objective?.active && !pointInAlliedSight(world, player, objective.x, objective.y))
    return objective;
  const cursor = objective?.index ?? -1;
  const phase = cursor >= 0 ? cursor + 1 : Math.floor(Math.max(0, world.tick - 3600) / 900);
  for (let k = 0; k < points.length; k++) {
    const point = points[(phase + k) % points.length]!;
    if (!pointInAlliedSight(world, player, point.x, point.y)) return point;
  }
  return null;
}

/**
 * What the side is pushing at: the hostile structure nearest the team's army.
 *
 * Two properties matter and the old rule — "the lowest-index enemy building" —
 * had neither. It never changed, so an army that had fought its way into a base
 * would walk back out past a Barracks to keep pounding at a Command Post it had
 * already passed; and it was the same target from anywhere on the map, so two
 * allied bots on opposite flanks converged on one point by walking through each
 * other.
 *
 * Measuring from the team's centre of mass gives both: the nearest thing gets
 * killed first, and two bots whose armies are together pick the same target
 * while two whose armies are apart still agree — they compute one centroid, not
 * one each.
 */
function pickAttackTarget(world: World, s: Survey, tuning: Tuning, player: PlayerId): number {
  const pool = world.pool;
  const from = tuning.coordinates && s.teamArmy.length > 0 ? s.teamArmy : s.army;
  const flip = world.flipOf(player);

  // Summed in whole tiles of this player's canonical frame, and compared as
  // `(n * target - sum)^2` rather than against a rounded mean. A mean has to
  // be rounded, and no rounding of an absolute coordinate is its own mirror;
  // the scaled comparison never divides, so the two halves rank targets
  // identically. Tiles rather than world units keep `n * tile` well inside
  // float64's exact range for any army.
  let cx = 0;
  let cy = 0;
  let n = 0;
  for (const i of from) {
    cx += canonicalTileX(world, pool.posX[i]!, flip);
    cy += canonicalTileY(world, pool.posY[i]!, flip);
    n++;
  }
  if (n === 0) {
    // No army at all: workers are marching, and they start from home.
    if (s.commandPosts.length === 0) return s.enemyTargets[0]!;
    cx = canonicalTileX(world, pool.posX[s.commandPosts[0]!]!, flip);
    cy = canonicalTileY(world, pool.posY[s.commandPosts[0]!]!, flip);
    n = 1;
  }

  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const t of s.enemyTargets) {
    const dx = n * canonicalTileX(world, pool.posX[t]!, flip) - cx;
    const dy = n * canonicalTileY(world, pool.posY[t]!, flip) - cy;
    const d = dx * dx + dy * dy;
    // Strictly ordered: distance, then creation order, which the list order
    // already guarantees by only replacing on a strict improvement.
    if (d < bestDist) {
      bestDist = d;
      best = t;
    }
  }
  return best;
}

/**
 * The tile column a world coordinate falls in, seen from the canonical half.
 *
 * A building's centre sits exactly on a tile boundary, and a plain floor puts
 * a boundary in the higher tile on both halves — which, rotated back, is the
 * lower tile on one of them. Flooring in the canonical frame makes a mirrored
 * coordinate give the same canonical tile.
 */
function canonicalTileX(world: World, x: number, flip: boolean): number {
  return flip ? world.map.width - 1 - toInt(fromInt(world.map.width) - x) : toInt(x);
}

function canonicalTileY(world: World, y: number, flip: boolean): number {
  return flip ? world.map.height - 1 - toInt(fromInt(world.map.height) - y) : toInt(y);
}
