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
export const MAX_PLAYERS = 2;

export enum EntityType {
  // Units
  Worker = 0,
  Rifleman = 1,
  Brawler = 2,
  // Buildings
  CommandPost = 3,
  Depot = 4,
  Barracks = 5,
  Turret = 6,
  // Neutral
  MineralPatch = 7,
  // Appended rather than slotted in with the other units, so existing numeric
  // ids stay put — the defs table is indexed by this enum.
  Gunship = 8,
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
