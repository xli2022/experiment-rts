/**
 * Reports where a mirrored match first stops being a mirror.
 *
 * The map is symmetric under a 180-degree rotation and both openings are exact
 * rotations of each other, so a match whose second-half commands are the exact
 * rotations of the first half's should stay mirrored to the last tick. Where
 * it does not, the simulation is treating the two seats differently — which is
 * a seat advantage, and on this map it decided 15 of 16 bot mirror matches.
 *
 *   npm run mirror:probe
 *
 * Each scenario prints the first tick, entity and field that broke the mirror.
 * The scenarios build on each other: harvesting alone, then everything a
 * scripted match can reach, then the bots, then the same match with the seats
 * swapped. `tests/mirror.test.ts` asserts that all of them stay mirrored; this
 * is the tool for finding out why one does not.
 */

import { coopMatch, duelMatch } from '../src/sim/match.js';
import { BotDifficulty } from '../src/sim/types.js';
import {
  describeMismatch,
  fullScript,
  harvestScript,
  probeBots,
  probePair,
  probeScript,
  type MirrorReport,
} from '../tests/helpers/mirror.js';

const SEED = 0x51ce7a11;

function report(r: MirrorReport & { winnerB?: number }): void {
  const outcome = r.matchOver ? `over, winner ${r.winner}` : 'running';
  const extra = r.winnerB === undefined ? '' : `, swapped winner ${r.winnerB}`;
  const broken = r.first === null ? '' : ` (${r.brokenTicks} of ${r.ticks} ticks broken)`;
  console.log(`${r.name.padEnd(28)} ${String(r.ticks).padStart(5)} ticks  ${outcome}${extra}`);
  console.log(`    ${describeMismatch(r.first)}${broken}`);
}

const t0 = performance.now();
report(probeScript('harvest only', duelMatch(SEED, { botPlayers: [] }), harvestScript, 3000));
report(probeScript('scripted match', duelMatch(SEED, { botPlayers: [] }), fullScript, 6500));
report(
  probeBots(
    'Hard mirror',
    duelMatch(SEED, { botPlayers: [0, 1], difficulty: BotDifficulty.Hard }),
    20000,
  ),
);
const base = duelMatch(SEED, { botPlayers: [0, 1] });
report(
  probePair(
    'seats swapped',
    {
      ...base,
      bots: [
        { player: 0, difficulty: BotDifficulty.Hard },
        { player: 1, difficulty: BotDifficulty.Normal },
      ],
    },
    {
      ...base,
      bots: [
        { player: 0, difficulty: BotDifficulty.Normal },
        { player: 1, difficulty: BotDifficulty.Hard },
      ],
    },
    20000,
  ),
);
report(
  probeBots(
    'Quarters, four Hard bots',
    coopMatch(SEED, { botPlayers: [0, 1, 2, 3], difficulty: BotDifficulty.Hard }),
    6000,
  ),
);
console.log(`${((performance.now() - t0) / 1000).toFixed(1)}s`);
