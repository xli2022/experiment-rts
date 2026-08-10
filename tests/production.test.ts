/**
 * Training, and the way it stalls.
 *
 * A finished unit waits inside the building until there is supply room for it,
 * which is what the genre does. The trap is that "room" is per unit: with
 * exactly one free supply a Burstbot still pops out every time while a Beamdrone
 * never does, and the supply counter reads 19/20 the whole while. That looks
 * from the outside like one unit type being broken, and it was reported as
 * exactly that.
 *
 * So the behaviour is pinned here — including the way out, since a player whose
 * queue is jammed needs to be able to cancel it.
 */

import { describe, expect, it } from 'vitest';
import { defOf, MAX_PRODUCTION_QUEUE } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType } from '../src/sim/types.js';

/** A match with a finished Barracks and money to spend. */
function withBarracks(): { sim: Simulation; barracks: number } {
  const sim = new Simulation(0x51ce7a11);
  const world = sim.world;
  world.players[0]!.minerals = 5000;

  const start = world.map.starts[0]!;
  const id = world.placeBuilding(EntityType.Barracks, 0, start.tileX + 5, start.tileY + 6);
  const barracks = id & 0xffff;
  world.pool.buildState[barracks] = BuildState.Complete;
  world.pool.buildProgress[barracks] = defOf(EntityType.Barracks).buildTicks;
  world.recomputeSupply();
  return { sim, barracks };
}

function train(sim: Simulation, barracks: number, unit: EntityType): void {
  sim.step([
    {
      type: CommandType.Train,
      player: 0,
      building: sim.world.pool.idAt(barracks),
      unit,
    },
  ]);
}

function countOf(sim: Simulation, unit: EntityType): number {
  const pool = sim.world.pool;
  let n = 0;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === unit) n++;
  }
  return n;
}

/** Run `ticks`, holding the player at exactly `free` spare supply throughout. */
function runPinned(sim: Simulation, ticks: number, free: number): void {
  for (let t = 0; t < ticks; t++) {
    const ps = sim.world.players[0]!;
    ps.supplyMax = ps.supplyUsed + free;
    sim.step([]);
  }
}

describe('training', () => {
  it('delivers every unit type when supply is plentiful', () => {
    for (const unit of [EntityType.Burstbot, EntityType.Slicebot, EntityType.Beamdrone]) {
      const { sim, barracks } = withBarracks();
      sim.world.players[0]!.supplyMax = 200;
      train(sim, barracks, unit);
      for (let t = 0; t < defOf(unit).buildTicks + 20; t++) {
        sim.world.players[0]!.supplyMax = 200;
        sim.step([]);
      }
      expect(`${defOf(unit).name}: ${countOf(sim, unit)}`).toBe(`${defOf(unit).name}: 1`);
    }
  });

  it('holds a unit that does not fit, while a smaller one still comes out', () => {
    // One free supply. This is the state behind the bug report: the counter
    // shows headroom, so it does not look like being supply blocked at all.
    for (const unit of [EntityType.Burstbot, EntityType.Slicebot, EntityType.Beamdrone]) {
      const { sim, barracks } = withBarracks();
      train(sim, barracks, unit);
      runPinned(sim, defOf(unit).buildTicks + 200, 1);

      const fits = defOf(unit).supplyCost <= 1;
      expect(`${defOf(unit).name} spawned: ${countOf(sim, unit) > 0}`).toBe(
        `${defOf(unit).name} spawned: ${fits}`,
      );
      if (!fits) {
        // Parked at exactly 100%, not overrunning, and still holding its slot.
        expect(sim.world.pool.prodProgress[barracks]).toBe(defOf(unit).buildTicks);
        expect(sim.world.pool.prodCount[barracks]).toBe(1);
      }
    }
  });

  it('releases the held unit as soon as the supply arrives', () => {
    const { sim, barracks } = withBarracks();
    train(sim, barracks, EntityType.Beamdrone);
    runPinned(sim, defOf(EntityType.Beamdrone).buildTicks + 100, 1);
    expect(countOf(sim, EntityType.Beamdrone)).toBe(0);

    // A depot finishes: the wait was the only thing holding it.
    sim.world.players[0]!.supplyMax = sim.world.players[0]!.supplyUsed + 10;
    sim.step([]);
    expect(countOf(sim, EntityType.Beamdrone)).toBe(1);
  });

  it('does not let a held unit block the ones behind it forever', () => {
    // The head of the queue is what stalls; the rest must still be there, and
    // must start moving again the moment it clears.
    const { sim, barracks } = withBarracks();
    train(sim, barracks, EntityType.Beamdrone);
    train(sim, barracks, EntityType.Burstbot);
    runPinned(sim, defOf(EntityType.Beamdrone).buildTicks + 100, 1);
    expect(sim.world.pool.prodCount[barracks]).toBe(2);

    sim.world.players[0]!.supplyMax = sim.world.players[0]!.supplyUsed + 10;
    for (let t = 0; t < defOf(EntityType.Burstbot).buildTicks + 40; t++) sim.step([]);
    expect(countOf(sim, EntityType.Beamdrone)).toBe(1);
    expect(countOf(sim, EntityType.Burstbot)).toBe(1);
  });

  it('refunds a cancelled unit in full, even one already finished', () => {
    const { sim, barracks } = withBarracks();
    const cost = defOf(EntityType.Beamdrone).mineralCost;
    const before = sim.world.players[0]!.minerals;

    train(sim, barracks, EntityType.Beamdrone);
    expect(sim.world.players[0]!.minerals).toBe(before - cost);

    runPinned(sim, defOf(EntityType.Beamdrone).buildTicks + 60, 1);
    expect(sim.world.pool.prodCount[barracks]).toBe(1);

    sim.step([
      {
        type: CommandType.CancelTrain,
        player: 0,
        building: sim.world.pool.idAt(barracks),
        slot: 0,
      },
    ]);
    expect(sim.world.pool.prodCount[barracks]).toBe(0);
    expect(sim.world.players[0]!.minerals).toBe(before);
  });

  it('will not queue past the limit, or another player’s building', () => {
    const { sim, barracks } = withBarracks();
    for (let k = 0; k < MAX_PRODUCTION_QUEUE + 3; k++) train(sim, barracks, EntityType.Burstbot);
    expect(sim.world.pool.prodCount[barracks]).toBe(MAX_PRODUCTION_QUEUE);

    const spent = sim.world.players[1]!.minerals;
    sim.step([
      {
        type: CommandType.Train,
        player: 1,
        building: sim.world.pool.idAt(barracks),
        unit: EntityType.Burstbot,
      },
    ]);
    expect(sim.world.players[1]!.minerals).toBe(spent);
  });
});
