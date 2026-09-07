/**
 * Abandoning a structure that never got finished.
 *
 * The cost of a building is committed the instant its site is placed, which is
 * the right rule — it stops a player from spending the same minerals twice
 * while a foundation goes up — but until now there was no way back out of it.
 * A site placed in the wrong spot, or one whose worker was killed on the way,
 * was minerals gone for the rest of the match: nothing in the player's own base
 * was ever going to knock it down for them.
 *
 * The rules worth pinning are the ones a player would notice: the refund is
 * whole, the ground becomes buildable again, and a finished building is not
 * something a stray keypress can dissolve.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, NO_ENTITY, Order, type PlayerId } from '../src/sim/types.js';
import { duelMatch } from '../src/sim/match.js';
import { describeMismatch, probeScript, type MirrorScript } from './helpers/mirror.js';

const DEPOT = defOf(EntityType.Depot);

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

/** The player's only Depot, finished or not. */
function depotOf(sim: Simulation): number {
  const pool = sim.world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Depot) return i;
  }
  return -1;
}

/** Place a Depot site and return the world it sits in. */
function withSite(): { sim: Simulation; worker: number; site: number; tile: [number, number] } {
  const { sim, worker } = withWorker();
  const start = sim.world.map.starts[0]!;
  const tileX = start.tileX + 4;
  const tileY = start.tileY + 6;
  sim.step([
    {
      type: CommandType.Build,
      player: 0,
      worker: sim.world.pool.idAt(worker),
      building: EntityType.Depot,
      tileX,
      tileY,
    },
  ]);
  const site = depotOf(sim);
  expect(site).toBeGreaterThanOrEqual(0);
  return { sim, worker, site, tile: [tileX, tileY] };
}

/**
 * Cancel without advancing a tick.
 *
 * The mineral assertions below are the point of these tests, and a whole tick
 * would fold a harvest delivery into them on whichever tick a worker happened
 * to reach the base — a passing test that depends on the map seed. Commands
 * are applied at the top of a tick in any case, so this is what the simulation
 * itself does with one.
 */
function cancel(sim: Simulation, building: number, player = 0): void {
  executeCommand(sim.world, {
    type: CommandType.CancelBuild,
    player,
    building: sim.world.pool.idAt(building),
  });
}

describe('cancelling an unfinished building', () => {
  it('refunds the whole cost and removes the site', () => {
    const { sim, site } = withSite();
    const after = sim.world.players[0]!.minerals;

    cancel(sim, site);

    expect(sim.world.pool.alive[site]).toBe(0);
    expect(sim.world.players[0]!.minerals).toBe(after + DEPOT.mineralCost);
  });

  it('refunds in full even when the site is nearly finished', () => {
    // Deliberate: nothing on screen tells a player how many minerals of
    // progress a foundation holds, so charging for it would be a rule they
    // could only discover by losing money to it.
    const { sim, site } = withSite();
    for (let t = 0; t < 400 && sim.world.pool.buildProgress[site]! < DEPOT.buildTicks / 2; t++) {
      sim.step([]);
    }
    expect(sim.world.pool.buildProgress[site]).toBeGreaterThan(0);

    const before = sim.world.players[0]!.minerals;
    cancel(sim, site);
    expect(sim.world.players[0]!.minerals).toBe(before + DEPOT.mineralCost);
  });

  it('gives the ground back, so the spot can be built on again', () => {
    const { sim, worker, site, tile } = withSite();
    cancel(sim, site);
    expect(sim.world.map.canPlace(tile[0], tile[1], DEPOT.footprint)).toBe(true);

    sim.step([
      {
        type: CommandType.Build,
        player: 0,
        worker: sim.world.pool.idAt(worker),
        building: EntityType.Depot,
        tileX: tile[0],
        tileY: tile[1],
      },
    ]);
    expect(depotOf(sim)).toBeGreaterThanOrEqual(0);
  });

  it('releases the worker instead of leaving it walking to a ghost', () => {
    const { sim, worker, site } = withSite();
    expect(sim.world.pool.order[worker]).toBe(Order.Build);

    cancel(sim, site);

    expect(sim.world.pool.order[worker]).toBe(Order.None);
    expect(sim.world.pool.orderTarget[worker]).toBe(NO_ENTITY);
  });

  it('leaves a finished building alone', () => {
    const { sim, site } = withSite();
    for (let t = 0; t < DEPOT.buildTicks + 600; t++) {
      if (sim.world.pool.buildState[site] === BuildState.Complete) break;
      sim.step([]);
    }
    expect(sim.world.pool.buildState[site]).toBe(BuildState.Complete);

    const before = sim.world.players[0]!.minerals;
    cancel(sim, site);

    expect(sim.world.pool.alive[site]).toBe(1);
    expect(sim.world.players[0]!.minerals).toBe(before);
  });

  it('will not cancel someone else’s site', () => {
    const { sim, site } = withSite();
    const before = sim.world.players[1]!.minerals;

    cancel(sim, site, 1);

    expect(sim.world.pool.alive[site]).toBe(1);
    expect(sim.world.players[1]!.minerals).toBe(before);
  });

  it('is a no-op on a stale handle', () => {
    const { sim, site } = withSite();
    const stale = sim.world.pool.idAt(site);
    cancel(sim, site);
    const after = sim.world.players[0]!.minerals;

    executeCommand(sim.world, { type: CommandType.CancelBuild, player: 0, building: stale });
    expect(sim.world.players[0]!.minerals).toBe(after);
  });

  it('survives the rest of the tick it was cancelled on', () => {
    // The site is freed in the middle of a tick, before movement, combat and
    // the economy run over a world that still holds handles to it. Nothing
    // downstream may trip over the hole that leaves.
    const { sim, worker, site } = withSite();
    sim.step([{ type: CommandType.CancelBuild, player: 0, building: sim.world.pool.idAt(site) }]);
    for (let t = 0; t < 60; t++) sim.step([]);

    expect(depotOf(sim)).toBe(-1);
    expect(sim.world.pool.alive[worker]).toBe(1);
    expect(sim.world.pool.order[worker]).not.toBe(Order.Build);
  });
});

/**
 * Place a Depot, cancel it, and place another one on the same ground.
 *
 * Cancelling frees an entity slot mid-tick, and a freed slot is reused by the
 * next thing spawned. That is the part of this worth checking against the
 * rotation: if the two halves of a mirrored match recycled slots in different
 * orders, every entity id after the cancel would drift apart and the match
 * would stop being fair — silently, and long after the cancel.
 */
const cancelScript: MirrorScript = (world, tick) => {
  const pool = world.pool;
  const player = 0 as PlayerId;
  // Both seats, equally, so the banks stay each other's mirror. Without it a
  // Depot is unaffordable this early and the script quietly builds nothing.
  if (tick === 0) for (const p of world.players) p.minerals = 5000;
  const workers: number[] = [];
  const depots: number[] = [];
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.owner[i] !== player) continue;
    if (pool.type[i] === EntityType.Worker) workers.push(i);
    if (pool.type[i] === EntityType.Depot) depots.push(i);
  }
  const start = world.map.starts[0]!;
  const tileX = start.tileX + 4;
  const tileY = start.tileY + 6;

  if ((tick === 50 || tick === 400) && workers[0] !== undefined) {
    return [
      {
        type: CommandType.Build,
        player,
        worker: pool.idAt(workers[0]),
        building: EntityType.Depot,
        tileX,
        tileY,
      },
    ];
  }
  if (tick === 300 && depots[0] !== undefined) {
    return [{ type: CommandType.CancelBuild, player, building: pool.idAt(depots[0]) }];
  }
  return [];
};

describe('mirror symmetry', () => {
  it('holds across a cancelled site and the one built in its place', () => {
    const r = probeScript('cancel', duelMatch(0x51ce7a11, { botPlayers: [] }), cancelScript, 900);
    expect(describeMismatch(r.first)).toBe('mirrored');
  });
});
