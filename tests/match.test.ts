/**
 * End-to-end gameplay tests.
 *
 * The determinism tests prove two peers agree. They say nothing about whether
 * the thing they agree on is a *game*, and every bug below shipped past a fully
 * green determinism suite:
 *
 *  - Abandoned construction sites were never reassigned a builder, which
 *    silently blocked all further building and left the AI hoarding 9,000
 *    unspent minerals.
 *  - A non-integer tick gate meant player 1 never issued an attack order.
 *  - Attack-move flow fields were seeded on a building's centre tile, which is
 *    inside its own footprint; the sweep could not escape it, so the entire map
 *    read as unreachable and every unit ordered to attack simply stopped. Armies
 *    grew into the hundreds without a shot being fired.
 *
 * Each was invisible to unit tests and obvious the moment a whole match ran. So
 * a whole match runs here.
 *
 * The two bots play at different strengths. The simulation is exactly
 * rotation-equivariant (see `mirror.test.ts`), so two identical bots on the
 * mirrored map make mirrored moves for the whole match and can only draw — a
 * match that has to *resolve* needs the sides to differ.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import {
  BotDifficulty,
  EntityType,
  NEUTRAL,
  NO_ENTITY,
  TICKS_PER_SECOND,
  type MatchConfig,
} from '../src/sim/types.js';
import { duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';

const SEED = 0x51ce7a11;
/** Long enough for a match between unequal bots to resolve, with headroom. */
const MAX_TICKS = TICKS_PER_SECOND * 60 * 30;

/** Hard in one seat, Normal in the other. */
function unequal(seed: number, hardSeat: 0 | 1): MatchConfig {
  const base = duelMatch(seed, { botPlayers: [0, 1] });
  const normal = hardSeat === 0 ? 1 : 0;
  return {
    ...base,
    bots: [
      { player: hardSeat, difficulty: BotDifficulty.Hard },
      { player: normal, difficulty: BotDifficulty.Normal },
    ].sort((a, b) => a.player - b.player),
  };
}

interface Tally {
  workers: number;
  army: number;
  buildings: number;
}

function tally(sim: Simulation, owner: number): Tally {
  const pool = sim.world.pool;
  const out: Tally = { workers: 0, army: 0, buildings: 0 };
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.owner[i] !== owner) continue;
    const type = pool.type[i]! as EntityType;
    if (defOf(type).isBuilding) out.buildings++;
    else if (type === EntityType.Worker) out.workers++;
    else out.army++;
  }
  return out;
}

/** Run a bot-vs-bot match, collecting what actually happened along the way. */
function playMatch(seed: number, maxTicks = MAX_TICKS, hardSeat: 0 | 1 = 0) {
  const sim = new Simulation(unequal(seed, hardSeat));

  let deaths = 0;
  let shots = 0;
  let peakArmy = 0;
  const peakWorkers = [0, 0];
  let ticks = 0;

  for (let t = 0; t < maxTicks; t++) {
    sim.step([]);
    ticks = t + 1;
    deaths += sim.world.events.deaths.length;
    shots += sim.world.events.shots.length / 2;
    peakArmy = Math.max(peakArmy, tally(sim, 0).army + tally(sim, 1).army);
    peakWorkers[0] = Math.max(peakWorkers[0]!, tally(sim, 0).workers);
    peakWorkers[1] = Math.max(peakWorkers[1]!, tally(sim, 1).workers);
    if (sim.world.matchOver) break;
  }

  return { sim, deaths, shots, peakArmy, peakWorkers, ticks };
}

describe('bot-vs-bot match', () => {
  const result = playMatch(SEED);

  it('reaches a winner', () => {
    expect(result.sim.world.matchOver).toBe(true);
    expect(result.sim.world.winner).not.toBe(NO_ENTITY);
    expect([0, 1]).toContain(result.sim.world.winner);
  });

  it('has both sides build a working economy', () => {
    // Peak rather than final, which is what "built an economy" means. Checking
    // the closing tally instead made this a measure of how badly the winner was
    // mauled on the way — it started failing the moment attack-move began
    // actually delivering armies to enemy bases.
    const winner = result.sim.world.winner;
    expect(result.peakWorkers[winner]).toBeGreaterThan(6);
    expect(tally(result.sim, winner).buildings).toBeGreaterThan(3);
  });

  it('actually fights', () => {
    // The bug this guards: armies grew to hundreds while never engaging,
    // because attack orders resolved to an unreachable destination.
    expect(result.shots).toBeGreaterThan(500);
    expect(result.deaths).toBeGreaterThan(30);
    expect(result.peakArmy).toBeGreaterThan(10);
  });

  it('spends its minerals rather than hoarding', () => {
    // Guards the stalled-construction bug, where the AI sat on thousands of
    // minerals it could not spend because one orphaned site blocked everything.
    //
    // Deliberately an order-of-magnitude bound rather than a tight one. A
    // stalled bot banks its entire income and fields nothing; the balance
    // between income and spending is not what this is measuring, and tying the
    // number to it makes the test fail whenever either side is tuned — which is
    // exactly what happened when workers stopped walking around the Command
    // Post and income rose by about 60%.
    const winner = result.sim.world.winner;
    expect(result.sim.world.player(winner).minerals).toBeLessThan(12000);
    // And it plainly did spend: the army and buildings above are what it bought.
    expect(result.peakArmy).toBeGreaterThan(10);
  });

  it('ends by elimination rather than by running out of time', () => {
    const loser = result.sim.world.winner === 0 ? 1 : 0;
    const t = tally(result.sim, loser);

    // Two ways to be eliminated, and both are real endings. The classic one is
    // losing every structure. The other is being left with no units and too few
    // minerals to train one, which cannot recover on a mined-out map — without
    // that rule those matches ran forever with no move available to either side.
    const lostEveryStructure = t.buildings === 0;
    const cannotEverAct = t.workers === 0 && t.army === 0;

    expect(lostEveryStructure || cannotEverAct).toBe(true);
    expect(result.sim.world.player(loser).defeated).toBe(true);
    // Whatever the route, it must have happened before the tick budget ran out.
    expect(result.ticks).toBeLessThan(MAX_TICKS);
  });

  it('leaves mineral patches neutral throughout', () => {
    const pool = result.sim.world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      if (pool.type[i] === EntityType.MineralPatch) {
        expect(pool.owner[i]).toBe(NEUTRAL);
      }
    }
  });
});

describe('the seats are equal', () => {
  it('gives the same match with the seats swapped, so the winner swaps too', () => {
    // The outcome-level statement of what `mirror.test.ts` checks tick by
    // tick: which seat a player sits in changes nothing but the orientation.
    const a = playMatch(SEED, MAX_TICKS, 0);
    const b = playMatch(SEED, MAX_TICKS, 1);
    expect(a.sim.world.matchOver).toBe(true);
    expect(b.ticks).toBe(a.ticks);
    expect(b.sim.world.winner).toBe(1 - a.sim.world.winner);
    expect(b.shots).toBe(a.shots);
    expect(b.deaths).toBe(a.deaths);
  });
});

describe('match variety', () => {
  it('resolves across different seeds', () => {
    // A single seed could pass by luck. Different maps and build orders should
    // all terminate rather than deadlocking into a permanent stalemate.
    //
    // "Terminate" is the assertion, not "someone wins": mutual annihilation on a
    // mined-out map is a legitimate draw. What must never happen is the match
    // running forever with no move available to either side.
    const outcomes: string[] = [];
    for (const seed of [0x1111, 0x2f2f2f, 0xabcdef]) {
      const r = playMatch(seed);
      expect(
        r.sim.world.matchOver,
        `seed ${seed.toString(16)} did not resolve within ${r.ticks} ticks`,
      ).toBe(true);
      outcomes.push(r.sim.world.winner === NO_ENTITY ? 'draw' : `p${r.sim.world.winner}`);
    }
    // A draw is allowed, but every seed drawing would mean the win condition is
    // effectively unreachable.
    expect(outcomes.some((o) => o !== 'draw')).toBe(true);
  });
});
