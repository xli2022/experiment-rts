/**
 * The AI opponent.
 *
 * ## Why this lives inside the simulation
 *
 * The bot is a pure function of world state: given the same world and the same
 * tick, it emits the same commands. That means it can run *on every peer* rather
 * than on one machine that broadcasts its decisions — costing zero bandwidth,
 * needing no ownership negotiation, and surviving the host disconnecting.
 *
 * It also means single-player and multiplayer are the same code path. A skirmish
 * is just a match where one slot's commands come from here instead of from a
 * network packet, so the multiplayer plumbing is exercised every time anyone
 * plays alone.
 *
 * The flip side is that this file lives under the same determinism rules as the
 * rest of the simulation: no `Math.random`, no wall-clock, no iteration over
 * unordered collections. It is deliberately *stateless* — every decision is
 * re-derived from the world each time — so there is no hidden bot state that
 * could drift between peers.
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

import { defOf, MAX_PRODUCTION_QUEUE, SUPPLY_MAX } from '../config/rules.js';
import { CommandType, type Command } from '../sim/commands.js';
import { fromInt, sqRange, vecLenSqRaw } from '../sim/fixed.js';
import {
  BotDifficulty,
  BuildState,
  EntityType,
  NEUTRAL,
  Order,
  type PlayerId,
} from '../sim/types.js';
import type { World } from '../sim/world.js';

/**
 * The bot reconsiders its plan this often.
 *
 * The same for every slot and every difficulty, deliberately. Staggering bots by
 * player read as a harmless way to keep command streams distinguishable in a
 * replay — commands carry their player anyway — but it hands whoever thinks
 * first a whole think interval of head start. Measured in a mirror matchup on a
 * symmetric map, that decided the game: player 0 spent its opening 50 minerals
 * and had its workers walking before player 1 had taken a turn at all. Making it
 * a difficulty knob would reintroduce exactly that, between the two sides.
 */
const THINK_INTERVAL = 10;

/** How near a Command Post has to be for an expansion to count as taken. */
const CLAIMED_RANGE_SQ = 14 * 14;

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
 * Everything difficulty actually changes.
 *
 * All of it is behavioural: no bonus income, no extra starting units, no
 * cheating on fog. A harder bot works its economy harder and commits sooner,
 * which is a thing a player could have done too.
 */
interface Tuning {
  /** Workers to saturate the mineral line before spending on army. */
  readonly targetWorkers: number;
  /** Team army size that triggers a push, then keeps triggering it. */
  readonly attackArmySize: number;
  /** Barracks the bot will run once minerals are spare. */
  readonly maxBarracks: number;
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

const TUNING: Readonly<Record<BotDifficulty, Tuning>> = {
  // Slow to saturate, slow to commit, and it fights one base at a time. It also
  // never comes home, which is what makes it beatable by walking around it.
  [BotDifficulty.Easy]: {
    targetWorkers: 9,
    attackArmySize: 12,
    maxBarracks: 2,
    maxTurrets: 0,
    maxBases: 1,
    expandAtMinerals: 900,
    maxSites: 1,
    defendsHome: false,
    coordinates: false,
  },
  // What the skirmish bot has always played like, plus the things that were
  // simply missing: it defends, and it pushes with its partner.
  [BotDifficulty.Normal]: {
    targetWorkers: 14,
    attackArmySize: 8,
    maxBarracks: 6,
    maxTurrets: 2,
    maxBases: 2,
    expandAtMinerals: 550,
    maxSites: 2,
    defendsHome: true,
    coordinates: true,
  },
  // Saturates two bases, expands off a smaller bank, and commits on a smaller
  // army — which against two humans means arriving before either of them has an
  // army of their own.
  [BotDifficulty.Hard]: {
    targetWorkers: 18,
    attackArmySize: 6,
    maxBarracks: 8,
    maxTurrets: 3,
    maxBases: 3,
    expandAtMinerals: 450,
    maxSites: 3,
    defendsHome: true,
    coordinates: true,
  },
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
 * Base supply headroom before building another depot.
 *
 * Scaled by production capacity below: a base with four barracks burns supply
 * far faster than one, and a fixed buffer leaves the bot permanently blocked
 * with minerals it cannot spend.
 */
const SUPPLY_BUFFER = 6;

export function generateBotCommands(
  world: World,
  player: PlayerId,
  difficulty: BotDifficulty = BotDifficulty.Normal,
): Command[] {
  // Every bot thinks on the same tick. See THINK_INTERVAL.
  if (world.tick % THINK_INTERVAL !== 0) return [];
  if (world.player(player).defeated) return [];
  if (world.matchOver) return [];

  const tuning = TUNING[difficulty] ?? TUNING[BotDifficulty.Normal];
  const cmds: Command[] = [];
  const s = survey(world, player);

  keepWorkersBusy(world, player, s, cmds);
  manageProduction(world, player, s, tuning, cmds);
  manageConstruction(world, player, s, tuning, cmds);
  manageArmy(world, player, s, tuning, cmds);

  return cmds;
}

interface Survey {
  workers: number[];
  idleWorkers: number[];
  army: number[];
  commandPosts: number[];
  barracks: number[];
  depots: number[];
  turrets: number[];
  sites: number[];
  /**
   * Patches to send an idle worker to: the ones near a base of ours, or every
   * live patch on the map when none of ours is left.
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
  /** Any live patch anywhere, for deciding whether the map is mined out. */
  livePatches: number;
  /** Hostile structures, ascending by slot. Killing these is what wins. */
  enemyTargets: number[];
  /** Hostile combat units by class. Only the air count changes what gets built. */
  enemyRanged: number;
  enemyMelee: number;
  enemyAir: number;
  /**
   * Combat units belonging to anyone on our side, including a partner's.
   *
   * The team's fist. Every bot on the side derives it from the same ascending
   * pass, so all of them agree on how big it is and where its middle is.
   */
  teamArmy: number[];
  /** The building of ours a hostile is nearest to, or -1 when none is. */
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
    commandPosts: [],
    barracks: [],
    depots: [],
    turrets: [],
    sites: [],
    patches: [],
    homePatches: 0,
    livePatches: 0,
    enemyTargets: [],
    enemyRanged: 0,
    enemyMelee: 0,
    enemyAir: 0,
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
      if (pool.resourceAmount[i]! > 0) allPatches.push(i);
      continue;
    }
    if (owner === NEUTRAL) continue;

    const def = defOf(type);
    const isArmy =
      type === EntityType.Burstbot || type === EntityType.Slicebot || type === EntityType.Beamdrone;

    if (!world.areAllied(owner, player)) {
      // Prefer structures as attack targets; killing buildings is what wins.
      if (def.isBuilding) s.enemyTargets.push(i);
      else {
        hostileUnits.push(i);
        if (type === EntityType.Burstbot) s.enemyRanged++;
        else if (type === EntityType.Slicebot) s.enemyMelee++;
        else if (type === EntityType.Beamdrone) s.enemyAir++;
      }
      continue;
    }

    // Allied, which includes ourselves. A partner's army counts toward the
    // team's, but nothing else of theirs is ours to command.
    if (isArmy) s.teamArmy.push(i);
    if (owner !== player) continue;

    const complete = pool.buildState[i] === BuildState.Complete;
    if (def.isBuilding) ownBuildings.push(i);

    switch (type) {
      case EntityType.Worker:
        s.workers.push(i);
        if (pool.order[i] === Order.None) s.idleWorkers.push(i);
        break;
      case EntityType.Burstbot:
      case EntityType.Slicebot:
      case EntityType.Beamdrone:
        s.army.push(i);
        break;
      case EntityType.CommandPost:
        if (complete) s.commandPosts.push(i);
        else s.sites.push(i);
        break;
      case EntityType.Barracks:
        if (complete) s.barracks.push(i);
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

  // Patches worth walking to: near a base of ours. Ascending, like everything
  // else, so ties in `keepWorkersBusy` break the same way on every peer.
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

  s.threatened = nearestThreat(world, hostileUnits, ownBuildings);
  return s;
}

/**
 * The nearest of `others` to entity `i`, or -1 when the list is empty.
 *
 * Ties break by position in `others`, which every caller builds by an ascending
 * pass over the pool — a strict total order, and the reason this is one helper
 * rather than the four hand-rolled copies of the same loop it replaced.
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
 * The structure of ours a hostile is nearest to, if one is close enough to
 * count as an attack.
 *
 * A lone scouting worker is not an attack, and pulling an army home for one is
 * how a bot gets pulled out of position on purpose. Anything armed is.
 *
 * The *building*, not the attacker, and for two reasons. An enemy position is
 * a moving goal tile, and a grouped order navigates by a flow field cached per
 * goal tile — measured over a four-bot match, aiming at the attacker made 66%
 * of defensive orders a fresh Dijkstra sweep against 22% for the attack orders
 * that aim at buildings, and rebuilding the cache cost about a tenth of total
 * simulation time. It is also the only aim point that is reliably reachable: a
 * flyer parked over a cliff mass has no walkable tile near it, so the order was
 * refused outright by `standableTarget` and the whole army stood still for as
 * long as one drone cared to hover there.
 *
 * Walking home is what defence means anyway; an attack-move picks the attacker
 * up on arrival.
 */
function nearestThreat(
  world: World,
  hostileUnits: readonly number[],
  ownBuildings: readonly number[],
): number {
  if (ownBuildings.length === 0) return -1;
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const h of hostileUnits) {
    if (pool.type[h] === EntityType.Worker) continue;
    const building = nearestOf(world, h, ownBuildings);
    if (building < 0) continue;
    const distSq = distSqBetween(world, h, building);
    if (distSq > sqRange(DEFEND_RANGE)) continue;
    if (distSq < bestDist) {
      bestDist = distSq;
      best = building;
    }
  }
  return best;
}

/** Any worker with nothing to do goes back to the nearest live patch. */
function keepWorkersBusy(world: World, player: PlayerId, s: Survey, cmds: Command[]): void {
  if (s.patches.length === 0) return;
  const pool = world.pool;

  for (const w of s.idleWorkers) {
    const best = nearestOf(world, w, s.patches);
    if (best < 0) continue;
    cmds.push({
      type: CommandType.Harvest,
      player,
      units: [pool.idAt(w)],
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
  const supplyFree = s.supplyMax - s.supplyUsed;
  if (supplyFree <= 0) return;

  // Worker target scales with how many bases there are to work: a second
  // Command Post with nobody mining at it is 400 minerals of decoration.
  const wantWorkers = tuning.targetWorkers * Math.max(1, s.commandPosts.length);
  if (s.workers.length < wantWorkers) {
    // Every base trains, not just the first. One Command Post queueing all the
    // workers is what left an expansion's mineral line empty for minutes.
    for (const hq of s.commandPosts) {
      if (pool.prodCount[hq]! >= 2) continue;
      if (s.minerals < defOf(EntityType.Worker).mineralCost) break;
      cmds.push({
        type: CommandType.Train,
        player,
        building: pool.idAt(hq),
        unit: EntityType.Worker,
      });
      s.minerals -= defOf(EntityType.Worker).mineralCost;
    }
  }

  // Two deep normally, so the bank stays available for buildings and the mix
  // can still react to what gets scouted; full depth once minerals are piling
  // up faster than they can be spent.
  const depth = s.minerals >= DEEP_QUEUE_MINERALS ? MAX_PRODUCTION_QUEUE : 2;
  for (const b of s.barracks) {
    if (pool.prodCount[b]! >= depth) continue;
    const unit = pickUnitToTrain(world, s, b);
    const cost = defOf(unit).mineralCost;
    if (s.minerals < cost) break;
    cmds.push({ type: CommandType.Train, player, building: pool.idAt(b), unit });
    s.minerals -= cost;
  }
}

/**
 * Choose the next unit: a rotating spread, skewed by the one matchup rule left.
 *
 * There used to be a damage triangle to play against, and this read the enemy
 * composition to counter it. With damage now a single number per unit, that
 * reasoning would be picking units against a mechanic that no longer exists —
 * and would get it backwards, since it answered massed Burstbots with Slicebots.
 *
 * What survives is structural rather than numeric: a Slicebot cannot reach a
 * flyer at all. So a scouted air force forces something that can shoot back,
 * and otherwise the bot keeps a mix, which is what stops one lucky read
 * deciding a match.
 */
function pickUnitToTrain(world: World, s: Survey, building: number): EntityType {
  const phase = (Math.floor(world.tick / THINK_INTERVAL) + building) % 4;

  // Enough air out there to matter: only build what can answer it.
  if (s.enemyAir >= 2 && s.enemyAir * 2 >= s.enemyRanged + s.enemyMelee) {
    return phase === 1 ? EntityType.Beamdrone : EntityType.Burstbot;
  }

  if (phase === 0) return EntityType.Slicebot;
  if (phase === 1) return EntityType.Beamdrone;
  return EntityType.Burstbot;
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
  if (s.commandPosts.length === 0) return;

  staffOrphanedSites(world, player, s, cmds);

  if (s.sites.length >= tuning.maxSites) return;

  const builder = pickBuilder(world, s);
  if (builder < 0) return;

  const hq = s.commandPosts[0]!;
  const supplyFree = s.supplyMax - s.supplyUsed;
  // More production capacity means supply drains faster, so keep more headroom.
  const buffer = SUPPLY_BUFFER + s.barracks.length * 4;
  // A home mineral line that is nearly out is its own reason to expand, whatever
  // the bank looks like: waiting for a threshold that income can no longer reach
  // is how a bot mines itself to a standstill on a full map.
  const patchesRunningOut = s.homePatches <= 2 && s.livePatches > s.homePatches;

  let want: EntityType | null = null;
  if (s.supplyMax < SUPPLY_MAX && supplyFree < buffer) {
    want = EntityType.Depot;
  } else if (s.barracks.length < 1) {
    want = EntityType.Barracks;
  } else if (s.turrets.length < tuning.maxTurrets && s.minerals >= 300) {
    want = EntityType.Turret;
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
  const spot = site ?? findBuildSpot(world, pool.tileX[hq]!, pool.tileY[hq]!, def.footprint);
  if (!spot) return;

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
  const homeX = pool.tileX[home]!;
  const homeY = pool.tileY[home]!;
  const def = defOf(EntityType.CommandPost);
  const half = def.footprint >> 1;

  let best: { x: number; y: number } | null = null;
  let bestDist = Infinity;

  for (let e = 0; e < map.expansions.length; e++) {
    const site = map.expansions[e]!;
    const x = site.tileX - half;
    const y = site.tileY - half;
    if (!map.canPlace(x, y, def.footprint)) continue;

    // Distance is squared throughout; nothing here needs the actual length.
    const dHome = sq(site.tileX - homeX) + sq(site.tileY - homeY);
    let contested = false;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (pool.type[i] !== EntityType.CommandPost) continue;
      if (sq(pool.tileX[i]! - site.tileX) + sq(pool.tileY[i]! - site.tileY) < CLAIMED_RANGE_SQ) {
        contested = true;
        break;
      }
    }
    if (contested) continue;

    // Nearer to us than to any enemy base, or it is indefensible. An ally's
    // base does not count against it — a site behind a partner is safer than
    // one behind us, not less safe.
    let enemyCloser = false;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (!world.isHostile(i, player)) continue;
      if (!defOf(pool.type[i]! as EntityType).isBuilding) continue;
      if (sq(pool.tileX[i]! - site.tileX) + sq(pool.tileY[i]! - site.tileY) < dHome) {
        enemyCloser = true;
        break;
      }
    }
    if (enemyCloser) continue;

    if (dHome < bestDist) {
      bestDist = dHome;
      best = { x, y };
    }
  }
  return best;
}

/**
 * Squared, in *tile* space.
 *
 * Distances in Q16.16 go through `vecLenSqRaw` and `sqRange` instead, which is
 * the pair the rest of the simulation uses and what keeps a reader from having
 * to work out which coordinate system a comparison is in. This one is left for
 * `expansionSite`, which reasons about whole tiles.
 */
function sq(v: number): number {
  return v * v;
}

/** Send a worker to any construction site nobody is working on. */
function staffOrphanedSites(world: World, player: PlayerId, s: Survey, cmds: Command[]): void {
  if (s.sites.length === 0) return;
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
    const builder = pickBuilder(world, s);
    if (builder < 0) return;
    cmds.push({
      type: CommandType.Build,
      player,
      worker: pool.idAt(builder),
      building: pool.type[site]! as EntityType,
      tileX: pool.tileX[site]!,
      tileY: pool.tileY[site]!,
    });
    // Only one reassignment per think tick; the command may be rejected (the
    // tile is already occupied by the site itself), in which case the direct
    // repair below still applies.
    return;
  }
}

/**
 * Choose a worker to construct with.
 *
 * Prefers an idle one; otherwise takes the highest-index harvester so the same
 * worker is not repeatedly pulled off minerals. Never takes the last worker.
 */
function pickBuilder(world: World, s: Survey): number {
  if (s.workers.length <= 1) return -1;
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
 * Every candidate must also keep a clear one-tile moat (see `hasClearMoat`).
 * Without that rule the bot packs its structures into a solid ring and seals its
 * own army inside the base — measured: 106 of 106 combat units unable to reach
 * the enemy, so neither side could ever win.
 */
function findBuildSpot(
  world: World,
  originX: number,
  originY: number,
  footprint: number,
): { x: number; y: number } | null {
  for (let ring = 4; ring <= 26; ring += 2) {
    for (let step = 0; step < ring * 8; step += 3) {
      const angleIndex = step % (ring * 8);
      const side = Math.floor(angleIndex / (ring * 2));
      const along = angleIndex % (ring * 2);
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
      const x = originX + dx;
      const y = originY + dy;
      if (!world.map.canPlace(x, y, footprint)) continue;
      if (!hasClearMoat(world, x, y, footprint)) continue;
      // The moat alone is not enough: enough moated buildings still ring the
      // base into a closed shell. Only commit to a spot that provably leaves the
      // base connected to the rest of the map.
      if (!keepsBaseConnected(world, x, y, footprint, originX, originY)) continue;
      return { x, y };
    }
  }
  return null;
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
): boolean {
  const map = world.map;
  map.setOccupied(tileX, tileY, footprint, 1);
  try {
    const start = findOpenTileNear(world, originX, originY);
    if (start < 0) return false;

    const target = findOpenTileNear(world, map.width >> 1, map.height >> 1);
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

/** Nearest walkable tile to a point, searched in fixed ring order. */
function findOpenTileNear(world: World, tx: number, ty: number): number {
  const map = world.map;
  if (map.isWalkable(tx, ty)) return map.index(tx, ty);
  for (let r = 1; r <= 20; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx > -r && dx < r && dy > -r && dy < r) continue;
        if (map.isWalkable(tx + dx, ty + dy)) return map.index(tx + dx, ty + dy);
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
    cmds.push({
      type: CommandType.AttackMove,
      player,
      units: s.army.map((i) => pool.idAt(i)),
      x: pool.posX[s.threatened]!,
      y: pool.posY[s.threatened]!,
    });
    return;
  }

  // --- attack -------------------------------------------------------------
  if (s.enemyTargets.length === 0) return;

  // Normally wait for a critical mass before committing. But once the map is
  // mined out and there are no minerals banked, that army is never getting any
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
  const committed = tuning.coordinates && marching ? s.teamArmy.length : attackers.length;
  if (committed < required) return;

  // Re-issue occasionally rather than every think tick, so units get a chance
  // to actually walk somewhere before being redirected.
  //
  // Note the floor: every player thinks on the same tick now, but this used to
  // divide a staggered tick and produce a fraction for every player but the
  // first, so `% 6` could never equal zero and player 1 never attacked at all.
  if (beat % 6 !== 0) return;

  const target = pickAttackTarget(world, s, tuning);
  if (target < 0) return;

  cmds.push({
    type: CommandType.AttackMove,
    player,
    units: attackers.map((i) => pool.idAt(i)),
    x: pool.posX[target]!,
    y: pool.posY[target]!,
  });
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
function pickAttackTarget(world: World, s: Survey, tuning: Tuning): number {
  const pool = world.pool;
  const from = tuning.coordinates && s.teamArmy.length > 0 ? s.teamArmy : s.army;

  let cx = 0;
  let cy = 0;
  let n = 0;
  for (const i of from) {
    cx += pool.posX[i]!;
    cy += pool.posY[i]!;
    n++;
  }
  if (n === 0) {
    // No army at all: workers are marching, and they start from home.
    if (s.commandPosts.length === 0) return s.enemyTargets[0]!;
    cx = pool.posX[s.commandPosts[0]!]!;
    cy = pool.posY[s.commandPosts[0]!]!;
    n = 1;
  }
  // Exact integer arithmetic: the sums stay far inside float64's exact range
  // (a position is at most 152 << 16, and there are at most a few hundred of
  // them), and flooring a correctly-rounded division of exact integers is the
  // same on every engine.
  const ax = Math.floor(cx / n);
  const ay = Math.floor(cy / n);

  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const t of s.enemyTargets) {
    const dx = pool.posX[t]! - ax;
    const dy = pool.posY[t]! - ay;
    const d = dx * dx + dy * dy;
    // Strictly ordered: distance, then ascending slot index, which the scan
    // order already guarantees by only replacing on a strict improvement.
    if (d < bestDist) {
      bestDist = d;
      best = t;
    }
  }
  return best;
}
