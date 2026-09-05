/**
 * The 180-degree rotation the map is symmetric under, applied to the things
 * that live on it.
 *
 * Player `p`'s opposite number is `p + n/2`, whose opening is the exact
 * rotation of `p`'s (see `mirroredHalf`). A position, a tile and a player all
 * have rotations, and both the mirror tests and the neural bot's canonical
 * frame need them — the bot plays every seat as if it were the first, and
 * un-rotates its orders on the way out.
 */

import { fromInt, type Fix } from './fixed.js';
import type { PlayerId } from './types.js';
import type { World } from './world.js';

/** The player whose opening is the rotation of `player`'s. */
export function mirrorPlayer(world: World, player: PlayerId): PlayerId {
  const n = world.players.length;
  return (player + (n >> 1)) % n;
}

/** A world coordinate rotated 180 degrees about the map centre. */
export function mirrorX(world: World, x: Fix): Fix {
  return fromInt(world.map.width) - x;
}

export function mirrorY(world: World, y: Fix): Fix {
  return fromInt(world.map.height) - y;
}

/** A tile index rotated 180 degrees; -1 stays -1. */
export function mirrorTileIndex(world: World, tile: number): number {
  return tile < 0 ? tile : world.map.width * world.map.height - 1 - tile;
}
