/**
 * The simulation is rotation-equivariant.
 *
 * The map is an exact 180-degree rotation of itself and every opening is the
 * exact rotation of its opposite number's. A match whose second-half commands
 * are the rotations of the first half's must therefore stay an exact mirror:
 * every entity with a twin at the rotated position and identical in every
 * other field, both banks equal, on every tick, to the end.
 *
 * That is the property that makes the two seats fair, and it is checked the
 * way determinism is: mechanically, on every tick, naming the first place it
 * breaks. `scripts/mirror-probe.ts` runs the same scenarios for diagnosis.
 */

import { describe, expect, it } from 'vitest';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { NO_ENTITY } from '../src/sim/types.js';
import { unequalAgents } from './helpers/agents.js';
import {
  describeMismatch,
  fullScript,
  harvestScript,
  probeBots,
  probePair,
  probeScript,
} from './helpers/mirror.js';

const SEED = 0x51ce7a11;

describe('mirror symmetry', () => {
  it('holds for mirrored harvest orders', () => {
    const r = probeScript('harvest', duelMatch(SEED, { botPlayers: [] }), harvestScript, 3000);
    expect(describeMismatch(r.first)).toBe('mirrored');
  });

  it('holds through building, training, pathing, a fight and an expansion', () => {
    const r = probeScript('scripted', duelMatch(SEED, { botPlayers: [] }), fullScript, 6500);
    expect(describeMismatch(r.first)).toBe('mirrored');
  });

  it('holds for a whole bot-driven mirror match, which can then only draw', () => {
    const r = probeBots('scripted mirror', duelMatch(SEED, { botPlayers: [0, 1] }), 20000);
    expect(describeMismatch(r.first)).toBe('mirrored');
    expect(r.winner).toBe(NO_ENTITY);
  });

  it('gives the same match with the seats swapped, so the winner swaps too', () => {
    // The same roster both times; what swaps is which seat holds the
    // full-speed bot and which the one that thinks half as often.
    const config = duelMatch(SEED, { botPlayers: [0, 1] });
    const r = probePair(
      'swapped',
      config,
      config,
      20000,
      unequalAgents(config, 0),
      unequalAgents(config, 1),
    );
    expect(describeMismatch(r.first)).toBe('mirrored');
    expect(r.matchOver).toBe(true);
    expect(r.winner).not.toBe(NO_ENTITY);
    expect(r.winnerB).toBe(1 - r.winner);
  });

  it('holds on the four-corner map with two bots a side', () => {
    const r = probeBots('Quarters', coopMatch(SEED, { botPlayers: [0, 1, 2, 3] }), 6000);
    expect(describeMismatch(r.first)).toBe('mirrored');
  });
});
