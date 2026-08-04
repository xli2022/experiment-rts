/**
 * The most important test in the project.
 *
 * Peer-to-peer lockstep works only if two machines running the same commands
 * reach byte-identical state. Everything else — the netcode, the lobby, the AI —
 * is built on that assumption, so it gets checked here every tick rather than
 * once at the end. When it fails, the assertion names the exact tick, which is
 * the difference between a five-minute fix and a two-day bisect.
 */

import { describe, expect, it } from 'vitest';
import { checksumToHex } from '../src/sim/checksum.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType } from '../src/sim/types.js';
import { cloneCommands, describeWorld, recordMatch, replayMatch } from './helpers/scripted.js';

const SEED = 0x1234abcd;
/**
 * Long enough to reach combat, production, and exhausted mineral patches.
 *
 * That claim used to be false and nothing said so: the scripted match could
 * never afford a Barracks, so no army existed and 2400 ticks of "the most
 * important test in the project" checked economy and pathfinding only. First
 * shot by a combat unit now lands around tick 2800, hence the raise — and
 * `covers combat` below asserts it rather than trusting this comment.
 */
const TICKS = 4000; // three and a bit simulated minutes at 20Hz

describe('deterministic simulation', () => {
  it('replays a recorded match to identical state at every tick', () => {
    const { log, checksums } = recordMatch(SEED, TICKS);
    const replayed = replayMatch(SEED, log);

    expect(replayed.length).toBe(checksums.length);
    for (let t = 0; t < checksums.length; t++) {
      if (replayed[t] !== checksums[t]) {
        throw new Error(
          `desync at tick ${t}: recorded ${checksumToHex(checksums[t]!)} ` +
            `!= replayed ${checksumToHex(replayed[t]!)}`,
        );
      }
    }
  });

  it('keeps two simulations stepped side by side identical', () => {
    // This mirrors the real topology: two independent worlds, same seed, fed the
    // same commands each tick, never exchanging state.
    const { log } = recordMatch(SEED, TICKS);

    const a = new Simulation(SEED);
    const b = new Simulation(SEED);

    for (let t = 0; t < log.length; t++) {
      a.step(cloneCommands(log[t]!));
      b.step(cloneCommands(log[t]!));
      const ca = a.checksum();
      const cb = b.checksum();
      if (ca !== cb) {
        throw new Error(
          `peers diverged at tick ${t}: ${checksumToHex(ca)} != ${checksumToHex(cb)}\n` +
            `  A: ${describeWorld(a.world)}\n  B: ${describeWorld(b.world)}`,
        );
      }
    }
  });

  it('produces different outcomes for different seeds', () => {
    // Guards against the checksum being trivially constant, which would make
    // every other assertion in this file vacuous.
    const a = recordMatch(SEED, 300);
    const b = recordMatch(SEED + 1, 300);
    expect(a.checksums[a.checksums.length - 1]).not.toBe(
      b.checksums[b.checksums.length - 1],
    );
  });

  it('actually simulates something', () => {
    // A determinism test over an empty world would pass trivially. Assert the
    // match reached a state with real activity before trusting the above.
    const sim = new Simulation(SEED);
    const { log } = recordMatch(SEED, 600);
    for (const commands of log) sim.step(cloneCommands(commands));

    const world = sim.world;
    expect(world.tick).toBe(600);
    // Both players should have gathered minerals and built something by now.
    expect(world.player(0).supplyMax).toBeGreaterThan(0);
    expect(world.player(0).supplyUsed).toBeGreaterThan(0);
  });

  it('covers combat, not just economy', () => {
    // The coverage assertion for everything above it. A determinism check that
    // never fires a shot proves nothing about target acquisition, damage or
    // death, and that was the real state of this file: one shot in 6000 ticks,
    // worker on worker, because the scripted match could not afford a Barracks.
    // Nothing failed to say so — silent gaps in coverage never do.
    const sim = new Simulation(SEED);
    const { log } = recordMatch(SEED, TICKS);

    let combatShots = 0;
    const shooters = new Set<number>();
    for (const commands of log) {
      sim.step(cloneCommands(commands));
      const shots = sim.world.events.shots;
      for (let k = 0; k < shots.length; k += 2) {
        const type = sim.world.pool.type[shots[k]!]!;
        if (type === EntityType.Rifleman || type === EntityType.Brawler) {
          combatShots++;
          shooters.add(sim.world.pool.owner[shots[k]!]!);
        }
      }
    }

    expect(combatShots).toBeGreaterThan(20);
    // Both sides, or one player is simply being farmed and the losing side's
    // combat code never runs.
    expect(shooters.size).toBe(2);
  });

  it('is unaffected by how commands are batched across ticks', () => {
    // Same commands, but the arrays are freshly allocated and shuffled before
    // sorting. `sortCommands` must impose a canonical order regardless.
    const { log, checksums } = recordMatch(SEED, 400);
    const sim = new Simulation(SEED);
    for (let t = 0; t < log.length; t++) {
      const shuffled = cloneCommands(log[t]!).reverse();
      sim.step(shuffled);
    }
    expect(sim.checksum()).toBe(checksums[checksums.length - 1]);
  });
});
