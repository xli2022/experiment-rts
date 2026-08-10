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
 */

import { defOf, SUPPLY_MAX } from '../config/rules.js';
import { CommandType, type Command } from '../sim/commands.js';
import { BuildState, EntityType, NEUTRAL, Order, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';

/** The bot reconsiders its plan this often. */
const THINK_INTERVAL = 10;

/** Workers to saturate the mineral line before spending on army. */
const TARGET_WORKERS = 14;

/** Attack once the army reaches this size, then keep attacking. */
const ATTACK_ARMY_SIZE = 8;

/**
 * Base supply headroom before building another depot.
 *
 * Scaled by production capacity below: a base with four barracks burns supply
 * far faster than one, and a fixed buffer leaves the bot permanently blocked
 * with minerals it cannot spend.
 */
const SUPPLY_BUFFER = 6;

/** Concurrent construction sites the bot will run. */
const MAX_SITES = 2;

/**
 * Minerals on hand before the bot considers a second base.
 *
 * Comfortably above a Command Post's 400, because expanding at exactly the cost
 * would stall army production for as long as the build takes. Floating this
 * much is the signal that the home mineral line has stopped being the
 * bottleneck.
 */
const EXPAND_AT_MINERALS = 550;
/** Command Posts the bot will run. Two bases is plenty to manage well. */
const MAX_BASES = 2;
/** How near a Command Post has to be for an expansion to count as taken. */
const CLAIMED_RANGE_SQ = 14 * 14;

export function generateBotCommands(world: World, player: PlayerId): Command[] {
  // Every bot thinks on the same tick.
  //
  // Staggering them by player read as a harmless way to keep command streams
  // distinguishable in a replay — commands carry their player anyway — but it
  // hands whoever thinks first a whole think interval of head start. Measured in
  // a mirror matchup on a symmetric map, that decided the game: player 0 spent
  // its opening 50 minerals and had its workers walking before player 1 had
  // taken a turn at all.
  if (world.tick % THINK_INTERVAL !== 0) return [];
  if (world.player(player).defeated) return [];
  if (world.matchOver) return [];

  const cmds: Command[] = [];
  const s = survey(world, player);

  keepWorkersBusy(world, player, s, cmds);
  manageProduction(world, player, s, cmds);
  manageConstruction(world, player, s, cmds);
  manageArmy(world, player, s, cmds);

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
  patches: number[];
  enemyTargets: number[];
  /** Enemy combat units by class. Only the air count changes what gets built. */
  enemyRanged: number;
  enemyMelee: number;
  enemyAir: number;
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
    enemyTargets: [],
    enemyRanged: 0,
    enemyMelee: 0,
    enemyAir: 0,
    minerals: world.player(player).minerals,
    supplyUsed: world.player(player).supplyUsed,
    supplyMax: world.player(player).supplyMax,
  };

  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    const type = pool.type[i]! as EntityType;
    const owner = pool.owner[i]!;

    if (type === EntityType.MineralPatch) {
      if (pool.resourceAmount[i]! > 0) s.patches.push(i);
      continue;
    }
    if (owner === NEUTRAL) continue;

    if (owner !== player) {
      // Prefer structures as attack targets; killing buildings is what wins.
      if (defOf(type).isBuilding) s.enemyTargets.push(i);
      else if (type === EntityType.Burstbot) s.enemyRanged++;
      else if (type === EntityType.Slicebot) s.enemyMelee++;
      else if (type === EntityType.Beamdrone) s.enemyAir++;
      continue;
    }

    const complete = pool.buildState[i] === BuildState.Complete;
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
  return s;
}

/** Any worker with nothing to do goes back to the nearest live patch. */
function keepWorkersBusy(
  world: World,
  player: PlayerId,
  s: Survey,
  cmds: Command[],
): void {
  if (s.patches.length === 0) return;
  const pool = world.pool;

  for (const w of s.idleWorkers) {
    let best = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const p of s.patches) {
      const dx = pool.posX[p]! - pool.posX[w]!;
      const dy = pool.posY[p]! - pool.posY[w]!;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
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
  cmds: Command[],
): void {
  const pool = world.pool;
  const supplyFree = s.supplyMax - s.supplyUsed;
  if (supplyFree <= 0) return;

  if (s.workers.length < TARGET_WORKERS && s.commandPosts.length > 0) {
    const hq = s.commandPosts[0]!;
    if (pool.prodCount[hq]! < 2 && s.minerals >= defOf(EntityType.Worker).mineralCost) {
      cmds.push({
        type: CommandType.Train,
        player,
        building: pool.idAt(hq),
        unit: EntityType.Worker,
      });
      s.minerals -= defOf(EntityType.Worker).mineralCost;
    }
  }

  for (const b of s.barracks) {
    if (pool.prodCount[b]! >= 2) continue;
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
  cmds: Command[],
): void {
  const pool = world.pool;
  if (s.commandPosts.length === 0) return;

  staffOrphanedSites(world, player, s, cmds);

  if (s.sites.length >= MAX_SITES) return;

  const builder = pickBuilder(world, s);
  if (builder < 0) return;

  const hq = s.commandPosts[0]!;
  const supplyFree = s.supplyMax - s.supplyUsed;
  // More production capacity means supply drains faster, so keep more headroom.
  const buffer = SUPPLY_BUFFER + s.barracks.length * 4;

  let want: EntityType | null = null;
  if (s.supplyMax < SUPPLY_MAX && supplyFree < buffer) {
    want = EntityType.Depot;
  } else if (s.barracks.length < 1) {
    want = EntityType.Barracks;
  } else if (s.turrets.length < 2 && s.minerals >= 300) {
    want = EntityType.Turret;
  } else if (s.minerals >= EXPAND_AT_MINERALS && expansionSite(world, player, s)) {
    // Floating this much means the mineral line at home cannot absorb the
    // income any more. A second base is what turns it into more income rather
    // than more barracks queueing behind the same eight patches.
    want = EntityType.CommandPost;
  } else if (s.barracks.length < 6 && s.minerals >= 300) {
    // Excess minerals are wasted minerals; convert them into production.
    want = EntityType.Barracks;
  }

  if (want === null) return;
  const def = defOf(want);
  if (s.minerals < def.mineralCost) return;

  // An expansion goes on its site; everything else goes next to the base it
  // supports.
  const site = want === EntityType.CommandPost ? expansionSite(world, player, s) : null;
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
 * player, so the bot claims each site once rather than stacking bases — and it
 * is nearer to home than to the enemy. Walking a lone worker past the enemy's
 * front door to build a base is not an expansion, it is a donation.
 */
function expansionSite(
  world: World,
  player: PlayerId,
  s: Survey,
): { x: number; y: number } | null {
  const { map, pool } = world;
  if (s.commandPosts.length === 0 || s.commandPosts.length >= MAX_BASES) return null;

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

    // Nearer to us than to any enemy base, or it is indefensible.
    let enemyCloser = false;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || pool.owner[i] === player || pool.owner[i] === NEUTRAL) continue;
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

function sq(v: number): number {
  return v * v;
}

/** Send a worker to any construction site nobody is working on. */
function staffOrphanedSites(
  world: World,
  player: PlayerId,
  s: Survey,
  cmds: Command[],
): void {
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
function hasClearMoat(
  world: World,
  tileX: number,
  tileY: number,
  footprint: number,
): boolean {
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

/** Send the army at the enemy once it is big enough to matter. */
function manageArmy(world: World, player: PlayerId, s: Survey, cmds: Command[]): void {
  if (s.enemyTargets.length === 0) return;

  // Normally wait for a critical mass before committing. But once the map is
  // mined out and there are no minerals banked, that army is never getting any
  // bigger, and holding out for a threshold it can no longer reach turns a won
  // position into a permanent draw — observed with a crippled opponent still
  // standing because the winner was one unit short of attacking. With no way to
  // reinforce, whatever is left goes in.
  const cheapest = defOf(EntityType.Burstbot).mineralCost;
  const canReinforce = s.patches.length > 0 || s.minerals >= cheapest;
  const required = canReinforce ? ATTACK_ARMY_SIZE : 1;

  // With the economy dead and no army left, workers are the only pieces on the
  // board. They fight badly but they do fight, and a base full of them idling
  // next to exhausted patches while the opponent's last buildings stand is a
  // draw by inaction rather than a decision.
  const attackers = s.army.length > 0 ? s.army : !canReinforce ? s.workers : [];
  if (attackers.length < required) return;
  // Re-issue occasionally rather than every think tick, so units get a chance
  // to actually walk somewhere before being redirected.
  //
  // Note the floor: players think on staggered ticks, so `world.tick /
  // THINK_INTERVAL` is a fraction for every player but the first, and `% 6`
  // could then never equal zero. Player 1 never attacked at all.
  if (Math.floor(world.tick / THINK_INTERVAL) % 6 !== 0) return;

  const pool = world.pool;
  const target = s.enemyTargets[0]!;
  const units = attackers.map((i) => pool.idAt(i));

  cmds.push({
    type: CommandType.AttackMove,
    player,
    units,
    x: pool.posX[target]!,
    y: pool.posY[target]!,
  });
}
