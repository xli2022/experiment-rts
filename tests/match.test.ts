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
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { EntityType, NEUTRAL, NO_ENTITY, TICKS_PER_SECOND } from '../src/sim/types.js';
import { Simulation } from '../src/sim/tick.js';

const SEED = 0x51ce7a11;
/** Long enough for a mirror match to resolve, with headroom. */
const MAX_TICKS = TICKS_PER_SECOND * 60 * 30;

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
function playMatch(seed: number, maxTicks = MAX_TICKS) {
  const sim = new Simulation(seed);
  sim.botPlayers.add(0);
  sim.botPlayers.add(1);

  let deaths = 0;
  let shots = 0;
  let peakArmy = 0;
  let ticks = 0;

  for (let t = 0; t < maxTicks; t++) {
    sim.step([]);
    ticks = t + 1;
    deaths += sim.world.events.deaths.length;
    shots += sim.world.events.shots.length / 2;
    peakArmy = Math.max(peakArmy, tally(sim, 0).army + tally(sim, 1).army);
    if (sim.world.winner !== NO_ENTITY) break;
  }

  return { sim, deaths, shots, peakArmy, ticks };
}

describe('bot-vs-bot match', () => {
  const result = playMatch(SEED);

  it('reaches a winner', () => {
    expect(result.sim.world.winner).not.toBe(NO_ENTITY);
    expect([0, 1]).toContain(result.sim.world.winner);
  });

  it('has both sides build a working economy', () => {
    // Checked at the end, so the loser having been wiped out is expected; the
    // winner must have grown well past its six starting workers at some point.
    const winner = result.sim.world.winner;
    const w = tally(result.sim, winner);
    expect(w.workers).toBeGreaterThan(6);
    expect(w.buildings).toBeGreaterThan(3);
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
    const winner = result.sim.world.winner;
    expect(result.sim.world.player(winner).minerals).toBeLessThan(6000);
  });

  it('ends because one side lost every structure', () => {
    const loser = result.sim.world.winner === 0 ? 1 : 0;
    expect(tally(result.sim, loser).buildings).toBe(0);
    expect(result.sim.world.player(loser).defeated).toBe(true);
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

describe('match variety', () => {
  it('resolves across different seeds', () => {
    // A single seed could pass by luck. Different maps and build orders should
    // all terminate rather than deadlocking into a permanent stalemate.
    for (const seed of [0x1111, 0x2f2f2f, 0xabcdef]) {
      const r = playMatch(seed);
      expect(
        r.sim.world.winner !== NO_ENTITY,
        `seed ${seed.toString(16)} did not resolve within ${r.ticks} ticks`,
      ).toBe(true);
    }
  });
});
