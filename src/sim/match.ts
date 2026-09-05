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
  BotKind,
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
  /** Slots the AI plays, all of `kind`. Sorted and de-duplicated. */
  readonly botPlayers?: readonly PlayerId[];
  /** Which brain plays them. */
  readonly kind?: BotKind;
  /**
   * Per-slot brains, for a roster that mixes them — a scripted partner beside
   * neural opponents. Wins over `botPlayers` for any slot both name.
   */
  readonly botSlots?: readonly BotSlot[];
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
  const kind = options.kind ?? BotKind.Scripted;

  // Ascending, unique, and inside the roster. The bot roster is checksummed
  // input in every practical sense — the peers deal these slots out among
  // themselves by their position in this list — so it is normalised here rather
  // than trusted in whatever order a caller happened to build it.
  const requested: BotSlot[] = [
    ...(options.botSlots ?? []),
    ...(options.botPlayers ?? []).map((player) => ({ player, kind })),
  ];
  const byPlayer = new Map<PlayerId, BotSlot>();
  for (const slot of requested) {
    if (slot.player < 0 || slot.player >= teams.length || byPlayer.has(slot.player)) continue;
    byPlayer.set(slot.player, { player: slot.player, kind: slot.kind });
  }
  const bots = [...byPlayer.values()].sort((a, b) => a.player - b.player);

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
 * humans must occupy a contiguous prefix of the roster, because a bot is dealt
 * to a human peer by `hostOf` and the transports number their peers from zero.
 */
export function coopMatch(seed: number, options: MatchOptions = {}): MatchConfig {
  // Spread first, then re-apply the default for anything the caller left out.
  // A plain `{ botPlayers: [2, 3], ...options }` looks equivalent and is not:
  // an explicit `botPlayers: undefined` — which a caller assembling options
  // programmatically will produce sooner or later — overwrites the default, and
  // `matchConfig` then reads it as "no bots at all". That is a four-human
  // roster on a transport carrying two, which stalls forever on slots nobody
  // is sending for. A caller naming slots either way gets exactly what it named.
  const named = options.botPlayers !== undefined || options.botSlots !== undefined;
  return matchConfig(MapLayout.Quarters, seed, {
    ...options,
    botPlayers: named ? options.botPlayers : [2, 3],
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

/** How many slots a person sits in — the peers a transport must carry. */
export function humanCount(config: MatchConfig): number {
  return config.teams.length - config.bots.length;
}

/**
 * Why `config` cannot be played on a transport carrying `playerCount` peers,
 * or null when it can.
 *
 * The transports number their peers from zero and every bot slot is dealt to
 * one of those peers by `hostOf`, so the humans must be exactly the low slots:
 * one per peer, with no bot below them. The count alone does not imply the
 * shape — `botPlayers: [0, 1]` on the four-corner map leaves two humans against
 * a two-peer transport and passes a count test while seating them in slots 2
 * and 3, which is a slot nobody sends for and a stall forever, behind a
 * "waiting for player" banner naming someone who is not playing.
 */
export function rosterProblem(config: MatchConfig, playerCount: number): string | null {
  const humans = humanCount(config);
  const botsAreHigh = config.bots.every((bot) => bot.player >= humans);
  if (humans === playerCount && botsAreHigh) return null;
  return (
    `roster has ${humans} human slots but the transport carries ${playerCount}, and ` +
    `the AI holds slots [${config.bots.map((b) => b.player).join(', ')}]; the humans ` +
    `must be the low slots and the AI the high ones`
  );
}

/** Does the AI play this slot? */
export function isBotSlot(config: MatchConfig, player: PlayerId): boolean {
  return config.bots.some((bot) => bot.player === player);
}

/** Which brain plays a bot slot, or undefined for a human's. */
export function botKindOf(config: MatchConfig, player: PlayerId): BotKind | undefined {
  return config.bots.find((bot) => bot.player === player)?.kind;
}

/**
 * The peer that sends for a slot.
 *
 * A human sends for their own slot. A bot is hosted by a human peer, and the
 * bots are dealt to the peers round-robin in roster order — so single-player
 * hosts every bot on peer 0 and two-tab co-op gives each tab one. Derived from
 * the agreed config rather than negotiated, for the same reason slot assignment
 * is (`slotFromPeerIds`): two peers that both believed they hosted a slot would
 * each apply whichever copy arrived first, which is a desync without a bug.
 *
 * A match with no humans has no wire, so its bots all answer 0; a headless
 * driver then owns them all.
 */
export function hostOf(config: MatchConfig, slot: PlayerId): PlayerId {
  const index = config.bots.findIndex((bot) => bot.player === slot);
  if (index < 0) return slot;
  return index % Math.max(humanCount(config), 1);
}

/** The bot slots `localPlayer` sends for, ascending. */
export function hostedBy(config: MatchConfig, localPlayer: PlayerId): PlayerId[] {
  const out: PlayerId[] = [];
  for (const bot of config.bots) {
    if (hostOf(config, bot.player) === localPlayer) out.push(bot.player);
  }
  return out;
}
