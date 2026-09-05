/**
 * The complete simulation state.
 *
 * Everything that influences future ticks lives on this object and is covered by
 * `checksum()`. Anything not covered by the checksum must be either derived
 * (recomputable from checksummed state) or purely cosmetic — if it is neither,
 * it is a desync waiting to happen.
 *
 * The world knows nothing about rendering, input, or the network. It is stepped
 * by `tick.ts` with a set of commands and advances one fixed timestep.
 */

import { checksumInit, checksumU32 } from './checksum.js';
import { EntityPool } from './entities.js';
import { fromInt, type Fix } from './fixed.js';
import { GameMap, generateMap, OCCUPIED_RESERVED, OCCUPIED_SOLID } from './map.js';
import { duelMatch } from './match.js';
import { mirroredHalf } from './mapgen.js';
import { Rng } from './rng.js';
import { SpatialGrid } from './spatial.js';
import {
  EntityType,
  NEUTRAL,
  NO_ENTITY,
  type EntityId,
  type MatchConfig,
  type PlayerId,
  type TeamId,
} from './types.js';
import {
  defOf,
  PATCH_AMOUNT,
  PATCHES_PER_BASE,
  PATCHES_PER_EXPANSION,
  STARTING_MINERALS,
  STARTING_WORKERS,
  SUPPLY_MAX,
} from '../config/rules.js';

export interface PlayerState {
  minerals: number;
  /** Supply consumed by living units. */
  supplyUsed: number;
  /** Supply provided by completed buildings, capped at SUPPLY_MAX. */
  supplyMax: number;
  defeated: boolean;
}

/**
 * Transient per-tick events for the renderer and audio.
 *
 * Deliberately NOT part of the checksum: these are outputs of the tick, not
 * state carried into the next one. They are cleared at the start of every step,
 * so a peer that ignores them behaves identically to one that draws them.
 */
export interface SimEvents {
  /** (attacker, target) pairs whose authoritative attack wind-up began. */
  attackStarts: number[];
  /** Attackers whose wind-up reached its resolve tick, whether it hit or whiffed. */
  attackImpacts: number[];
  /** (attacker, target) pairs that fired this tick, for tracers. */
  shots: number[];
  /** Entity slot indices that died this tick. */
  deaths: number[];
  /** Entity slot indices that finished construction this tick. */
  completed: number[];
}

export class World {
  /** What every peer agreed to play before the first tick. Never mutated. */
  readonly config: MatchConfig;
  readonly map: GameMap;
  readonly pool = new EntityPool();
  readonly grid: SpatialGrid;
  readonly rng: Rng;
  readonly players: PlayerState[] = [];

  /** Ticks elapsed since the match began. */
  tick = 0;

  /**
   * -1 while the match is running or after a draw; otherwise the winning team.
   *
   * A *team*, not a player: co-op is won together or not at all. In a 1v1 team
   * ids are equal to player ids, so this reads exactly as it always did.
   */
  winner: TeamId = NO_ENTITY;

  /**
   * True once the match has ended, win or draw.
   *
   * Distinct from `winner` because a draw is a real outcome here: both sides
   * can be eliminated on the same tick, and "winner is nobody" must still stop
   * the match rather than leave it running forever.
   */
  matchOver = false;

  /**
   * Units waiting for an A* path, in request order.
   *
   * A plain array used as a FIFO. Pathfinding is budgeted per tick (see
   * `PATH_BUDGET_PER_TICK`), and this queue is what makes the overflow
   * deterministic: every peer defers exactly the same requests to the same
   * later tick, so a busy tick slows everyone identically rather than letting
   * one machine's spare CPU change the outcome.
   */
  readonly pathQueue: number[] = [];

  readonly events: SimEvents = {
    attackStarts: [],
    attackImpacts: [],
    shots: [],
    deaths: [],
    completed: [],
  };

  /**
   * A bare seed still means "the 1v1 map", so every existing caller and test
   * reads the same and produces the same world. Anything else — a different map,
   * more than two players, teams — comes in as a `MatchConfig`, which is the
   * thing peers actually agree on.
   */
  constructor(config: MatchConfig | number) {
    const cfg = typeof config === 'number' ? duelMatch(config) : config;
    this.config = cfg;
    this.map = generateMap(cfg.seed, cfg.mapSize, cfg.layout);
    this.grid = new SpatialGrid(cfg.mapSize);
    this.rng = new Rng(cfg.seed);
    for (let p = 0; p < cfg.teams.length; p++) {
      this.players.push({
        minerals: STARTING_MINERALS,
        supplyUsed: 0,
        supplyMax: 0,
        defeated: false,
      });
    }
  }

  player(id: PlayerId): PlayerState {
    return this.players[id]!;
  }

  /** Which side a player is on. */
  teamOf(player: PlayerId): TeamId {
    return this.config.teams[player] ?? player;
  }

  /**
   * Do these two slots fight for the same side?
   *
   * True for a player and itself, which is what every caller wants: the
   * question is always "may I shoot this", and the answer for your own units is
   * the same no as for a partner's.
   */
  areAllied(a: PlayerId, b: PlayerId): boolean {
    if (a === NEUTRAL || b === NEUTRAL) return false;
    return this.teamOf(a) === this.teamOf(b);
  }

  /**
   * May `player` shoot this entity?
   *
   * The one place hostility is decided. It used to live on the entity pool,
   * where it could only ever compare owners — correct while every player was
   * everyone else's enemy, and quietly wrong the moment two of them share a
   * side. Neutral entities are hostile to nobody, so mineral patches are
   * excluded here rather than at every call site.
   */
  isHostile(index: number, toPlayer: PlayerId): boolean {
    const owner = this.pool.owner[index]!;
    if (owner === NEUTRAL) return false;
    return !this.areAllied(owner, toPlayer);
  }

  /** Every slot on `team`, ascending. Allocates; not for per-tick use. */
  playersOnTeam(team: TeamId): PlayerId[] {
    const out: PlayerId[] = [];
    for (let p = 0; p < this.players.length; p++) {
      if (this.teamOf(p) === team) out.push(p);
    }
    return out;
  }

  /** Centre position of an entity, in world units. */
  centreX(index: number): Fix {
    return this.pool.posX[index]!;
  }

  centreY(index: number): Fix {
    return this.pool.posY[index]!;
  }

  /**
   * Place a building and mark its tiles occupied.
   *
   * Buildings are positioned by their top-left tile and their stored position is
   * the footprint centre, which is what movement and combat measure against.
   */
  placeBuilding(type: EntityType, owner: PlayerId, tileX: number, tileY: number): EntityId {
    const def = defOf(type);
    const half = def.footprint / 2;
    const cx = fromInt(tileX) + Math.round(half * 65536);
    const cy = fromInt(tileY) + Math.round(half * 65536);
    const id = this.pool.spawn(type, owner, cx, cy);
    if (id === NO_ENTITY) return NO_ENTITY;
    const i = id & 0xffff;
    this.pool.tileX[i] = tileX;
    this.pool.tileY[i] = tileY;
    // Buildings face the middle of the map rather than always +Y. Purely
    // cosmetic — nothing consults a building's facing — but it is the same rule
    // units spawn by, so a base and its opposite number are mirror images
    // instead of two copies pointing the same way.
    this.pool.faceY[i] = cy < fromInt(this.map.height >> 1) ? 65536 : -65536;
    // A structure that does not collide still reserves its ground — nothing can
    // be built on a mineral patch, but everything can walk over one.
    this.map.setOccupied(
      tileX,
      tileY,
      def.footprint,
      def.collides ? OCCUPIED_SOLID : OCCUPIED_RESERVED,
    );
    return id;
  }

  /**
   * Fingerprint the entire simulation.
   *
   * Field order is fixed and must never change casually — the golden replay test
   * pins expected values, and reordering would invalidate them for no reason.
   */
  checksum(): number {
    let h = checksumInit();
    // The roster is agreed out of band, so a lobby that disagreed about it would
    // otherwise show up as an inexplicable divergence some seconds in. Folded in
    // here, a mismatch is a desync on the very first comparison, which names the
    // real problem instead of a symptom.
    h = checksumU32(h, this.config.layout);
    h = checksumU32(h, this.config.teams.length);
    for (let p = 0; p < this.config.teams.length; p++) {
      h = checksumU32(h, this.config.teams[p]!);
    }
    h = checksumU32(h, this.config.bots.length);
    for (let b = 0; b < this.config.bots.length; b++) {
      const bot = this.config.bots[b]!;
      h = checksumU32(h, bot.player);
      h = checksumU32(h, bot.difficulty);
    }
    h = checksumU32(h, this.tick);
    h = checksumU32(h, this.rng.state);
    h = checksumU32(h, this.winner);
    h = checksumU32(h, this.matchOver ? 1 : 0);
    for (let p = 0; p < this.players.length; p++) {
      const ps = this.players[p]!;
      h = checksumU32(h, ps.minerals);
      h = checksumU32(h, ps.supplyUsed);
      h = checksumU32(h, ps.supplyMax);
      h = checksumU32(h, ps.defeated ? 1 : 0);
    }
    h = checksumU32(h, this.pathQueue.length);
    for (let i = 0; i < this.pathQueue.length; i++) {
      h = checksumU32(h, this.pathQueue[i]!);
    }
    h = this.pool.checksum(h);
    h = this.map.checksum(h);
    return h;
  }

  /** Recompute supply totals from scratch. Cheap, and immune to drift. */
  recomputeSupply(): void {
    for (let p = 0; p < this.players.length; p++) {
      this.players[p]!.supplyUsed = 0;
      this.players[p]!.supplyMax = 0;
    }
    const pool = this.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const owner = pool.owner[i]!;
      if (owner === NEUTRAL) continue;
      const ps = this.players[owner];
      if (!ps) continue;
      const def = defOf(pool.type[i]! as EntityType);
      if (def.isBuilding) {
        // Only finished buildings contribute supply.
        if (pool.buildState[i] === 2) ps.supplyMax += def.supplyProvided;
      } else {
        ps.supplyUsed += def.supplyCost;
      }
    }
    for (let p = 0; p < this.players.length; p++) {
      const ps = this.players[p]!;
      if (ps.supplyMax > SUPPLY_MAX) ps.supplyMax = SUPPLY_MAX;
    }
  }
}

/**
 * Offsets of each mineral patch from a base's centre tile, as top-left corners.
 *
 * Distance here is the single biggest lever on the pace of the whole game.
 * Patches scattered even a few tiles too far leave workers walking instead of
 * mining and the economy never gets going, so these sit just clear of the
 * Command Post footprint.
 */
const PATCH_OFFSETS: readonly (readonly [number, number])[] = [
  [-6, -2],
  [-6, 0],
  [-6, 2],
  [-5, -4],
  [-3, -5],
  [-1, -6],
  [1, -6],
  [3, -5],
];

/**
 * Lay a mineral line around a base site.
 *
 * `anchor` is always the *canonical* site — the first half of `map.starts` or
 * of `map.expansions`. With `flip` set, each patch is placed at the 180-degree
 * rotation of where it would otherwise go, so the second line is a true
 * rotation of the first rather than the same shape in the same orientation.
 *
 * That distinction is the whole point. Laid out the same way round, one
 * player's workers start on the far side of their own Command Post from their
 * own patches, and their minerals sit between their base and the attack.
 *
 * `faceOut` reflects the authored line horizontally about the base's own
 * centre, for a canonical site that sits in the right half of the map. The
 * offsets below run up and to the left, which tucks the line into the corner
 * for a base on the left and shoves it out toward the middle for one on the
 * right — and on the four-start map both are canonical sites, so one rule for
 * all of them is not enough. This is a reflection about the base's *centre*,
 * not its centre tile: half a tile out here is a whole tile of extra walking on
 * every trip for the rest of the match.
 */
function placeMineralLine(
  world: World,
  anchorX: number,
  anchorY: number,
  count: number,
  flip: boolean,
  faceOut = false,
): number {
  const { map, pool } = world;
  const patchDef = defOf(EntityType.MineralPatch);
  let placed = 0;
  for (let i = 0; i < PATCH_OFFSETS.length && placed < count; i++) {
    const [dx, dy] = PATCH_OFFSETS[i]!;
    let canonX = anchorX + dx;
    const canonY = anchorY + dy;
    if (faceOut) canonX = reflectTile(anchorX, canonX, patchDef.footprint);
    const tx = flip ? mirrorTile(map.width, canonX, patchDef.footprint) : canonX;
    const ty = flip ? mirrorTile(map.height, canonY, patchDef.footprint) : canonY;
    if (!map.canPlace(tx, ty, patchDef.footprint)) continue;
    const patch = world.placeBuilding(EntityType.MineralPatch, NEUTRAL, tx, ty);
    if (patch === NO_ENTITY) continue;
    const pi = patch & 0xffff;
    pool.buildState[pi] = 2;
    pool.resourceAmount[pi] = PATCH_AMOUNT;
    placed++;
  }
  return placed;
}

/**
 * Should a base at this tile have its mineral line reflected?
 *
 * True for a canonical site in the right half of the map, so the line ends up
 * on the outside rather than between the base and the middle. Applied to start
 * locations only: an opening mineral line is mined from tick zero with no say
 * in the matter, whereas an expansion is walked to by choice and has no
 * sheltered side to speak of once it is out in the open.
 */
function facesOutward(map: GameMap, tileX: number): boolean {
  return tileX * 2 >= map.width;
}

/**
 * Build the opening position: a finished Command Post, starting workers, and a
 * mineral line for each player.
 *
 * Runs identically on every peer from the shared seed, so no starting state ever
 * crosses the network.
 */
export function setupMatch(world: World): void {
  const { map, pool } = world;
  const hqDef = defOf(EntityType.CommandPost);
  const half = hqDef.footprint >> 1;

  // Openings are authored for the first half of the starts; the second half is
  // each of those rotated 180 degrees about the centre of the map. Player `p`'s
  // opposite number is therefore always `p + n/2`, and the two sides are exact
  // rotations of each other whether there are two of them or four.
  //
  // Applying the same offsets to a mirrored start instead is *nearly* the same
  // thing and was wrong: a footprint of 4 centred by `start - 2` cannot be
  // symmetric about a tile, so the mirrored base landed a tile nearer the
  // middle of the map — closer to the centre, to its expansion, and to the
  // enemy, on every axis, from tick zero.
  for (let p = 0; p < world.players.length; p++) {
    const { canonical, flip } = mirroredHalf(p, map.starts.length);
    const site = map.starts[canonical]!;
    const homeX = site.tileX - half;
    const homeY = site.tileY - half;
    const faceOut = facesOutward(map, site.tileX);
    const hqTileX = flip ? mirrorTile(map.width, homeX, hqDef.footprint) : homeX;
    const hqTileY = flip ? mirrorTile(map.height, homeY, hqDef.footprint) : homeY;

    const hq = world.placeBuilding(EntityType.CommandPost, p, hqTileX, hqTileY);
    if (hq !== NO_ENTITY) {
      const hi = hq & 0xffff;
      pool.buildState[hi] = 2; // starts complete
      pool.buildProgress[hi] = hqDef.buildTicks;
    }

    // Everything else is placed relative to the Command Post's centre, so it
    // rotates with it.
    const centreX = hqTileX + half;
    const centreY = hqTileY + half;

    // Mineral line: a tight arc beside the base, on the side away from the
    // enemy. Always anchored on the canonical start, and rotated for the half
    // that is a rotation of it.
    placeMineralLine(world, homeX + half, homeY + half, PATCHES_PER_BASE, flip, faceOut);

    // Starting workers, fanned out on the near side of the Command Post.
    const facing = flip ? -1 : 1;
    for (let wIdx = 0; wIdx < STARTING_WORKERS; wIdx++) {
      const ox = fromInt(centreX) + fromInt((wIdx - (STARTING_WORKERS >> 1)) * facing);
      const oy = fromInt(centreY + 3 * facing);
      const id = pool.spawn(EntityType.Worker, p, ox, oy);
      // Units spawn facing +Y by default, which is one more thing that has to
      // rotate: otherwise one player's opening six all have to turn around
      // before they can walk to their own minerals.
      if (id !== NO_ENTITY) pool.faceY[id & 0xffff] = fromInt(facing);
    }
  }

  // Expansions: a mineral line and nothing else. They belong to whoever gets a
  // Command Post up on them, which is the whole point — a second base is worth
  // the 400 minerals and the exposure, or it is not, and that is a decision
  // rather than something handed out at spawn.
  for (let e = 0; e < map.expansions.length; e++) {
    // Anchored on the canonical site of each pair, so the two lines are exact
    // rotations of one another rather than each laid out from its own site.
    const { canonical, flip } = mirroredHalf(e, map.expansions.length);
    const site = map.expansions[canonical]!;
    placeMineralLine(world, site.tileX, site.tileY, PATCHES_PER_EXPANSION, flip);
  }

  world.recomputeSupply();
  world.grid.rebuild(pool);
}

/** Top-left tile of a footprint rotated 180 degrees about the map centre. */
function mirrorTile(size: number, tile: number, footprint: number): number {
  return size - 1 - tile - (footprint - 1);
}

/**
 * Top-left tile of a footprint reflected about a base's centre on one axis.
 *
 * The same formula as `mirrorTile`, over a span of `2 * centre` rather than the
 * map — reflecting `[t, t+f)` about `c` gives `[2c-t-f, 2c-t)` either way. Spelt
 * as a call rather than repeated, because CLAUDE.md records that a half-tile
 * error in exactly this reflection cost a whole tile of walking on every trip
 * and took three attempts to get right; two copies of it is two places to get
 * it wrong again.
 *
 * `centre` is a base's centre tile, which for an even footprint is also its
 * exact geometric centre — a Command Post spans `[c-2, c+2)`.
 */
function reflectTile(centre: number, tile: number, footprint: number): number {
  return mirrorTile(2 * centre, tile, footprint);
}
