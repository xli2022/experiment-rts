/**
 * The selection panel's account of what a unit is doing.
 *
 * A worker spends nearly the whole match on jobs that look, on screen, exactly
 * like standing about: walking to a construction site, building it, mining,
 * walking back with a load, repairing a wall. The panel used to say nothing for
 * any of them unless the worker happened to be holding minerals at that instant,
 * so the one state a player has to act on — genuinely idle — was
 * indistinguishable from the four they must not touch.
 *
 * These tests drive a real worker through a real build and a real harvest cycle
 * rather than poking state, because the interesting failure is the status
 * disagreeing with the simulation: reporting "building" while the worker is
 * still walking is worse than reporting nothing.
 */

import { describe, expect, it } from 'vitest';
import { defOf, HARVEST_TICKS } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, Order } from '../src/sim/types.js';
import { activityOf } from '../src/ui/status.js';

/** Derived, not spelled out: the Depot's display name is "Supply Depot". */
const DEPOT = defOf(EntityType.Depot).name;

/** A match with one worker of player 0 picked out, and money to spend. */
function withWorker(): { sim: Simulation; worker: number } {
  const sim = new Simulation(0x51ce7a11);
  sim.world.players[0]!.minerals = 5000;
  const pool = sim.world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Worker) {
      return { sim, worker: i };
    }
  }
  throw new Error('no worker at match start');
}

/** Run until `done`, returning every distinct status seen along the way. */
function statusesUntil(
  sim: Simulation,
  worker: number,
  ticks: number,
  done: () => boolean,
): string[] {
  const seen: string[] = [];
  for (let t = 0; t < ticks; t++) {
    const s = activityOf(sim.world, worker);
    if (s !== null && s !== seen[seen.length - 1]) seen.push(s);
    if (done()) break;
    sim.step([]);
  }
  return seen;
}

/** Strip the trailing percentage, which changes every tick. */
function shape(status: string): string {
  return status.replace(/ \d+%$/, '');
}

describe('what a worker is doing', () => {
  it('reports building, and says so only once it has arrived', () => {
    // The bug report: a worker told to build looked exactly like one forgotten
    // about. This is the whole request.
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;
    const start = sim.world.map.starts[0]!;

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.Depot,
        tileX: start.tileX + 4,
        tileY: start.tileY + 6,
      },
    ]);

    let site = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Depot) site = i;
    }
    expect(site).toBeGreaterThanOrEqual(0);

    const seen = statusesUntil(
      sim,
      worker,
      defOf(EntityType.Depot).buildTicks + 400,
      () => pool.buildState[site] === BuildState.Complete,
    ).map(shape);

    // Walking first, building second, and never the other way round.
    expect(seen[0]).toBe(`walking to ${DEPOT} site`);
    expect(seen).toContain(`building ${DEPOT}`);
    expect(seen.indexOf(`building ${DEPOT}`)).toBeGreaterThan(seen.indexOf(`walking to ${DEPOT} site`));

    // And it never claimed to be building before it got there.
    expect(seen.slice(0, seen.indexOf(`building ${DEPOT}`))).not.toContain(`building ${DEPOT}`);
  });

  it('never says building while the worker is still out of reach', () => {
    // The status must agree with the simulation tick for tick — it asks the
    // same reach question the economy system does, rather than its own.
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;
    const start = sim.world.map.starts[0]!;

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.Depot,
        tileX: start.tileX + 4,
        tileY: start.tileY + 6,
      },
    ]);

    let site = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Depot) site = i;
    }

    let checked = 0;
    for (let t = 0; t < 300; t++) {
      const before = pool.buildProgress[site]!;
      sim.step([]);
      // The completing tick both advances progress and releases the worker, so
      // there is no longer a build to report; it is not a disagreement.
      if (pool.buildState[site] === BuildState.Complete) break;
      const advanced = pool.buildProgress[site]! > before;
      const claims = shape(activityOf(sim.world, worker) ?? '') === `building ${DEPOT}`;
      // The simulation advancing progress is the ground truth for "working".
      expect(`tick ${t}: progressing ${advanced}, claims ${claims}`).toBe(
        `tick ${t}: progressing ${advanced}, claims ${advanced}`,
      );
      checked++;
    }
    expect(checked).toBeGreaterThan(20);
  });

  it('reports each leg of the harvest cycle', () => {
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;

    let patch = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.MineralPatch) {
        patch = i;
        break;
      }
    }
    sim.step([
      {
        type: CommandType.Harvest,
        player: 0,
        units: [pool.idAt(worker)],
        target: pool.idAt(patch),
      },
    ]);

    const seen = new Set<string>();
    let delivered = 0;
    const before = sim.world.players[0]!.minerals;
    for (let t = 0; t < 600; t++) {
      seen.add(shape(activityOf(sim.world, worker) ?? ''));
      sim.step([]);
      if (sim.world.players[0]!.minerals > before) delivered++;
      if (delivered > 0 && seen.size >= 3) break;
    }

    expect([...seen].sort()).toEqual(
      expect.arrayContaining(['walking to minerals', 'mining', 'returning 8 minerals']),
    );
  });

  it('says idle, and only when the worker really has nothing to do', () => {
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;
    pool.order[worker] = Order.None;
    pool.clearPath(worker);
    sim.step([]);
    expect(activityOf(sim.world, worker)).toBe('idle');

    // A worker on any job is not idle, whatever else it is doing.
    pool.order[worker] = Order.Move;
    expect(activityOf(sim.world, worker)).toBe('moving');
  });

  it('does not repair a finished building, however damaged', () => {
    // Repair was removed for being too strong: any attack that failed to kill a
    // structure outright was wasted effort. A worker sent at a hurt building
    // must be released rather than quietly healing it.
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;
    const start = sim.world.map.starts[0]!;
    const id = sim.world.placeBuilding(EntityType.Depot, 0, start.tileX + 4, start.tileY + 6);
    const depot = id & 0xffff;
    const def = defOf(EntityType.Depot);
    pool.buildState[depot] = BuildState.Complete;
    pool.buildProgress[depot] = def.buildTicks;
    pool.hp[depot] = Math.floor(def.maxHp / 2);

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: pool.idAt(worker),
        building: EntityType.Depot,
        tileX: start.tileX + 4,
        tileY: start.tileY + 6,
      },
    ]);

    const hurt = pool.hp[depot]!;
    const seen = statusesUntil(sim, worker, 200, () => false).map(shape);
    expect(seen).not.toContain(`repairing ${DEPOT}`);
    expect(seen).not.toContain(`building ${DEPOT}`);
    expect(`depot hp after 200 ticks: ${pool.hp[depot]}`).toBe(`depot hp after 200 ticks: ${hurt}`);
    expect(seen).toContain('idle');
  });

  it('reports no activity for a building', () => {
    // Buildings have their own lines in the panel; a Barracks reading "idle"
    // would be noise.
    const { sim } = withWorker();
    const pool = sim.world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.CommandPost) {
        expect(activityOf(sim.world, i)).toBeNull();
        return;
      }
    }
    throw new Error('no Command Post');
  });

  it('keeps every percentage inside 0-100', () => {
    // A mining counter is a fraction of HARVEST_TICKS and a build counter a
    // fraction of buildTicks; either overrunning reads as a bug on screen.
    const { sim, worker } = withWorker();
    const pool = sim.world.pool;
    let patch = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.MineralPatch) {
        patch = i;
        break;
      }
    }
    sim.step([
      {
        type: CommandType.Harvest,
        player: 0,
        units: [pool.idAt(worker)],
        target: pool.idAt(patch),
      },
    ]);

    let sawPercent = 0;
    for (let t = 0; t < 600; t++) {
      sim.step([]);
      const status = activityOf(sim.world, worker) ?? '';
      const m = /(\d+)%$/.exec(status);
      if (m) {
        const pct = Number(m[1]);
        expect(pct).toBeGreaterThanOrEqual(0);
        expect(pct).toBeLessThanOrEqual(100);
        sawPercent++;
      }
    }
    // HARVEST_TICKS of mining per trip, several trips: percentages must appear.
    expect(sawPercent).toBeGreaterThan(HARVEST_TICKS);
  });
});
