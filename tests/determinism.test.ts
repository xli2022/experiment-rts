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
import { coopMatch } from '../src/sim/match.js';
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

/**
 * Ticks for the four-player legs.
 *
 * The same length as the duel's, and for the same reason. It was set shorter on
 * the theory that four players on a larger map cost roughly twice as much per
 * tick and that these legs were about co-op code paths rather than length —
 * with a comment asserting contact happened inside it anyway. Measured, the two
 * sides do not meet until around tick 2800 on this map, so at 2500 every co-op
 * leg stopped before a single shot was fired between the teams: cross-team
 * damage, `isHostile`, and every combat path on the four-player map were
 * checksummed exactly never. This is the blindness the file's own duel coverage
 * test exists to prevent, in a new place. `covers combat on both maps` below
 * asserts it rather than trusting this comment.
 */
const COOP_TICKS = 4000;

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
    expect(a.checksums[a.checksums.length - 1]).not.toBe(b.checksums[b.checksums.length - 1]);
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
        if (type === EntityType.Burstbot || type === EntityType.Slicebot) {
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

  it('replays a four-player co-op match to identical state at every tick', () => {
    // Twice the players, a different map, and — unlike the duel — two of the
    // four command streams belong to a side rather than to an individual. All
    // of that is new arithmetic reaching new code paths, so it gets the same
    // per-tick treatment as the match above rather than a spot check at the end.
    const config = coopMatch(SEED, { botPlayers: [] });
    const { log, checksums } = recordMatch(config, COOP_TICKS);
    const replayed = replayMatch(config, log);

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

  it('keeps four co-op peers stepped side by side identical', () => {
    const config = coopMatch(SEED, { botPlayers: [] });
    const { log } = recordMatch(config, COOP_TICKS);

    const a = new Simulation(config);
    const b = new Simulation(config);
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

  it('reproduces a bot-driven co-op match from the config alone', () => {
    // The bots are the whole command stream here: nothing is recorded, nothing
    // is replayed, and two simulations built from the same agreed config have to
    // arrive at the same state anyway. That is the property single-player and
    // co-op both actually rely on — bot commands never cross the wire.
    const config = coopMatch(SEED, { botPlayers: [0, 1, 2, 3] });
    const a = new Simulation(config);
    const b = new Simulation(config);
    for (let t = 0; t < COOP_TICKS; t++) {
      a.step([]);
      b.step([]);
      if (a.checksum() !== b.checksum()) {
        throw new Error(`bot-driven co-op peers diverged at tick ${t}`);
      }
    }
    // And the match was worth checksumming: four bots that never built anything
    // would agree perfectly about nothing.
    expect(a.world.player(0).supplyMax).toBeGreaterThan(10);
    expect(a.world.player(3).supplyMax).toBeGreaterThan(10);
  });

  it('covers combat on the four-player map, not just economy', () => {
    // The same coverage assertion as the duel's, for the same reason and after
    // the same failure. `COOP_TICKS` was 2500 and the two sides do not meet
    // until around 2800, so every co-op leg above agreed perfectly about a match
    // in which nobody had yet fired at anybody — while a comment on the constant
    // claimed contact happened inside it. Hostility on this map is a *team*
    // question rather than an owner one, which is precisely the new arithmetic
    // these legs exist to checksum.
    const sim = new Simulation(coopMatch(SEED, { botPlayers: [0, 1, 2, 3] }));
    const world = sim.world;

    let crossTeamShots = 0;
    const shooters = new Set<number>();
    for (let t = 0; t < COOP_TICKS; t++) {
      sim.step([]);
      const shots = world.events.shots;
      for (let k = 0; k + 1 < shots.length; k += 2) {
        const attacker = world.pool.owner[shots[k]!]!;
        const victim = world.pool.owner[shots[k + 1]!]!;
        // Allies never shoot each other; a shot that crossed the front line is
        // the only kind that exercises team hostility.
        expect(world.areAllied(attacker, victim)).toBe(false);
        crossTeamShots++;
        shooters.add(attacker);
      }
    }

    expect(crossTeamShots).toBeGreaterThan(20);
    // All four slots, or half the roster's combat code never runs.
    expect(shooters.size).toBe(4);
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
