/**
 * Core simulation types shared across systems.
 *
 * Everything here is plain data with no behaviour, so it can be imported by the
 * sealed simulation, the renderer, and the UI without dragging dependencies
 * across the determinism boundary.
 */

import type { Fix } from './fixed.js';

/** Simulation runs at a fixed rate. Rendering interpolates between ticks. */
export const TICKS_PER_SECOND = 20;
export const MS_PER_TICK = 1000 / TICKS_PER_SECOND; // 50ms

/** How often peers compare state fingerprints. */
export const CHECKSUM_INTERVAL = TICKS_PER_SECOND; // once per second

/** Convert a duration in seconds to whole ticks. */
export function seconds(s: number): number {
  return Math.round(s * TICKS_PER_SECOND);
}

/** Stable handle to an entity. Index into the entity pool; see `entities.ts`. */
export type EntityId = number;

/** Sentinel for "no entity". Never a valid id. */
export const NO_ENTITY: EntityId = -1;

/** Player slot. Neutral owns resources and terrain decoration. */
export type PlayerId = number;
export const NEUTRAL: PlayerId = -1;

/**
 * Upper bound on player slots, not the size of a match.
 *
 * A match's actual roster is `MatchConfig.teams`, and `world.players.length` is
 * the number that exist. This constant only bounds what the rest of the engine
 * must be able to cope with — four, so two players can share a side against two
 * bots.
 */
export const MAX_PLAYERS = 4;

/**
 * Which side a player is on.
 *
 * Players on the same team never fight each other, share vision, and win or
 * lose together. A 1v1 is the degenerate case where every team has one member,
 * and team ids there are equal to player ids — which is why nothing about the
 * duel changes when teams are introduced.
 */
export type TeamId = number;

/**
 * The maps we know how to generate.
 *
 * The layout decides how many start locations exist, and therefore how many
 * players a match on it can hold.
 */
export enum MapLayout {
  /** Three lanes between two opposite corners. Two starts: the 1v1 map. */
  Lanes = 0,
  /**
   * Four starts, a team to each side of the map.
   *
   * Allies sit in adjacent corners joined by a back lane of their own, so
   * reinforcing a partner does not mean walking through the middle.
   */
  Quarters = 1,
}

/** How hard an AI slot plays. Agreed in the lobby, like everything else. */
export enum BotDifficulty {
  Easy = 0,
  Normal = 1,
  Hard = 2,
}

/** One AI-controlled slot. */
export interface BotSlot {
  readonly player: PlayerId;
  readonly difficulty: BotDifficulty;
}

/**
 * Everything two peers must agree on before the first tick.
 *
 * The seed alone used to be enough, because there was exactly one map and
 * exactly two players. It is not any more: the roster, the sides, and which
 * slots the AI plays all change what the simulation computes, and a peer that
 * disagreed about any of them would desync on tick zero rather than diverge
 * subtly later. So they travel together as one value that the lobby settles and
 * nothing afterwards mutates.
 */
export interface MatchConfig {
  readonly seed: number;
  readonly mapSize: number;
  readonly layout: MapLayout;
  /**
   * Team of each player slot; its length is the number of players.
   *
   * Indexed by `PlayerId`, so `teams[2] === 1` means slot 2 plays for team 1.
   */
  readonly teams: readonly TeamId[];
  /**
   * Slots the AI plays, in ascending player order.
   *
   * The bot is deterministic and runs inside the simulation on every peer, so
   * this is part of the agreed setup rather than a local choice — see
   * `Simulation.step`.
   */
  readonly bots: readonly BotSlot[];
}

export enum EntityType {
  // Units
  Worker = 0,
  Burstbot = 1,
  Slicebot = 2,
  // Buildings
  CommandPost = 3,
  Depot = 4,
  Barracks = 5,
  Turret = 6,
  // Neutral
  MineralPatch = 7,
  // Appended rather than slotted in with the other units, so existing numeric
  // ids stay put — the defs table is indexed by this enum.
  Beamdrone = 8,
}

export const ENTITY_TYPE_COUNT = 9;

/** What an entity is currently trying to do. Drives the system dispatch. */
export enum Order {
  /** No orders. Idle combat units still acquire nearby targets. */
  None = 0,
  /** Walk to a point, then stop. */
  Move = 1,
  /** Walk to a point, engaging anything hostile encountered on the way. */
  AttackMove = 2,
  /** Chase and attack one specific entity. */
  Attack = 3,
  /** Full harvest cycle: go to patch, mine, return to drop-off, repeat. */
  Harvest = 4,
  /** Walk to a construction site and build it. */
  Build = 5,
  /** Hold position: fire at range but never move. */
  Hold = 6,
}

/** Lifecycle of a building. */
export enum BuildState {
  /** Placed, but no worker has arrived; renders as a translucent shell. */
  Site = 0,
  /** A worker is actively building it; `buildProgress` climbs. */
  UnderConstruction = 1,
  /** Finished and functional. */
  Complete = 2,
}

/** Terrain classification per tile. */
export enum Tile {
  /** Walkable and buildable. */
  Ground = 0,
  /** Impassable rock/cliff. Blocks movement, vision is unaffected in v1. */
  Cliff = 1,
  /** Walkable but not buildable — keeps mineral lines clear of walls. */
  Resource = 2,
}

/** A two-component fixed-point point, used for command payloads. */
export interface FixPoint {
  x: Fix;
  y: Fix;
}
