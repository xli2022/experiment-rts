/**
 * Match setup: the agreed description of what is about to be played.
 *
 * A `MatchConfig` is the whole of what two peers must settle before the first
 * tick — seed, map, roster, sides, and which slots the AI plays. Building one is
 * kept here, away from both the lobby and the simulation, because the lobby must
 * not be able to invent a roster the map cannot seat, and the simulation must
 * not have to guess one.
 *
 * ## Sides come from the map, not from a menu
 *
 * Start locations are stored in mirrored halves (see `Layout.bases`), so the
 * only team split that is exactly fair is the one that follows that structure:
 * the first half of the slots against the second. Team ids are therefore
 * *derived* from the layout rather than chosen, which also means a 1v1 keeps
 * team ids equal to player ids and nothing about the duel changes.
 */

import { layoutStarts, layoutSize } from './mapgen.js';
import {
  BotDifficulty,
  MapLayout,
  MAX_PLAYERS,
  type BotSlot,
  type MatchConfig,
  type PlayerId,
  type TeamId,
} from './types.js';

export interface MatchOptions {
  /** Override the layout's own size, for a caller that wants a smaller map. */
  readonly mapSize?: number;
  /** Slots the AI plays. Sorted and de-duplicated. */
  readonly botPlayers?: readonly PlayerId[];
  /** How hard those bots play. */
  readonly difficulty?: BotDifficulty;
}

/**
 * Team of each player slot on a layout.
 *
 * The first half of the roster is team 0 and the second half team 1, matching
 * the mirrored halves the starts are stored in — so player `p` and player
 * `p + n/2` are always on opposite sides and always begin on ground that is an
 * exact rotation of the other's.
 */
export function teamsFor(layout: MapLayout): TeamId[] {
  const count = layoutStarts(layout);
  // `MAX_PLAYERS` is what the rest of the engine is sized against — entity
  // capacity, the palette, the renderer's instanced pools. A layout that seats
  // more than that would overrun all three quietly, so it is refused here, at
  // the one place a roster is built, rather than discovered later.
  if (count > MAX_PLAYERS) {
    throw new Error(`layout ${layout} seats ${count} players, over the ${MAX_PLAYERS} cap`);
  }
  const half = count >> 1;
  const teams: TeamId[] = [];
  for (let p = 0; p < count; p++) teams.push(p < half ? 0 : 1);
  return teams;
}

/** Build a config for any layout. */
export function matchConfig(
  layout: MapLayout,
  seed: number,
  options: MatchOptions = {},
): MatchConfig {
  const teams = teamsFor(layout);
  const difficulty = options.difficulty ?? BotDifficulty.Normal;

  // Ascending, unique, and inside the roster. The bot roster is checksummed
  // input in every practical sense — two peers holding different lists would
  // simulate different games — so it is normalised here rather than trusted in
  // whatever order a caller happened to build it.
  const bots: BotSlot[] = [];
  const seen = new Set<PlayerId>();
  const requested = [...(options.botPlayers ?? [])].sort((a, b) => a - b);
  for (const player of requested) {
    if (player < 0 || player >= teams.length || seen.has(player)) continue;
    seen.add(player);
    bots.push({ player, difficulty });
  }

  return {
    seed: seed >>> 0,
    mapSize: options.mapSize ?? layoutSize(layout),
    layout,
    teams,
    bots,
  };
}

/** The 1v1 map: two players, one to a team. */
export function duelMatch(seed: number, options: MatchOptions = {}): MatchConfig {
  return matchConfig(MapLayout.Lanes, seed, options);
}

/**
 * The co-op map: two players a side.
 *
 * Slots 0 and 1 are team 0 and slots 2 and 3 are team 1, so the humans take the
 * low slots and the AI the high ones by default. That is not arbitrary: the
 * lockstep scheduler indexes its per-turn buffer by player id and only the
 * human slots ever appear on the wire, so the humans must occupy a contiguous
 * prefix of the roster.
 */
export function coopMatch(seed: number, options: MatchOptions = {}): MatchConfig {
  // Spread first, then re-apply the default for anything the caller left out.
  // A plain `{ botPlayers: [2, 3], ...options }` looks equivalent and is not:
  // an explicit `botPlayers: undefined` — which a caller assembling options
  // programmatically will produce sooner or later — overwrites the default, and
  // `matchConfig` then reads it as "no bots at all". That is a four-human
  // roster on a transport carrying two, which stalls forever on slots nobody
  // is sending for.
  return matchConfig(MapLayout.Quarters, seed, {
    ...options,
    botPlayers: options.botPlayers ?? [2, 3],
  });
}

/**
 * The same match on a different seed.
 *
 * Used by the `?seed=` URL override, which exists so that a desync report can
 * be reproduced: everything about the match is kept and only the roll changes.
 */
export function withSeed(config: MatchConfig, seed: number): MatchConfig {
  return { ...config, seed: seed >>> 0 };
}

/** How many human slots a config expects on the wire. */
export function humanCount(config: MatchConfig): number {
  return config.teams.length - config.bots.length;
}
