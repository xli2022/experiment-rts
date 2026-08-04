/**
 * Rally points, and flyers sharing the sky.
 *
 * Both are ordinary simulation state — the rally lives on the building, not on
 * the player, because a barracks at the front and one at home want different
 * answers, and that is the whole point of having one.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { fromInt, toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, Order, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

/** A match with a finished Barracks and money to spend. */
function withBarracks(): { sim: Simulation; barracks: number } {
  const sim = new Simulation(0x51ce7a11);
  const world = sim.world;
  world.players[0]!.minerals = 5000;
  world.players[0]!.supplyMax = 200;

  const start = world.map.starts[0]!;
  const id = world.placeBuilding(EntityType.Barracks, 0, start.tileX + 5, start.tileY + 6);
  const barracks = id & 0xffff;
  world.pool.buildState[barracks] = BuildState.Complete;
  world.pool.buildProgress[barracks] = defOf(EntityType.Barracks).buildTicks;
  world.recomputeSupply();
  return { sim, barracks };
}

/** A walkable tile well away from the base. */
function openTile(sim: Simulation, from: number): { x: number; y: number } {
  const map = sim.world.map;
  const bx = Math.round(toFloat(sim.world.pool.posX[from]!));
  const by = Math.round(toFloat(sim.world.pool.posY[from]!));
  for (let r = 8; r < 30; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        if (map.isWalkable(bx + dx, by + dy)) return { x: bx + dx, y: by + dy };
      }
    }
  }
  throw new Error('no open tile');
}

describe('rally points', () => {
  it('sends a trained unit toward the rally instead of leaving it at the door', () => {
    const { sim, barracks } = withBarracks();
    const pool = sim.world.pool;
    const rally = openTile(sim, barracks);

    sim.step([
      {
        type: CommandType.SetRally,
        player: 0,
        building: pool.idAt(barracks),
        x: fromInt(rally.x) + 32768,
        y: fromInt(rally.y) + 32768,
      },
      {
        type: CommandType.Train,
        player: 0,
        building: pool.idAt(barracks),
        unit: EntityType.Rifleman,
      },
    ]);
    expect(pool.hasRally[barracks]).toBe(1);

    let unit = -1;
    for (let t = 0; t < defOf(EntityType.Rifleman).buildTicks + 30 && unit < 0; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Rifleman) {
          unit = i;
        }
      }
    }
    expect(unit).toBeGreaterThanOrEqual(0);
    // It leaves with a move order pointed at the rally, not standing idle.
    expect(pool.order[unit]).toBe(Order.Move);

    const before = Math.hypot(
      toFloat(pool.posX[unit]!) - (rally.x + 0.5),
      toFloat(pool.posY[unit]!) - (rally.y + 0.5),
    );
    for (let t = 0; t < 200; t++) sim.step([]);
    const after = Math.hypot(
      toFloat(pool.posX[unit]!) - (rally.x + 0.5),
      toFloat(pool.posY[unit]!) - (rally.y + 0.5),
    );
    expect(`closed the gap: ${after < before - 1}`).toBe('closed the gap: true');
  });

  it('leaves output alone when no rally is set', () => {
    const { sim, barracks } = withBarracks();
    const pool = sim.world.pool;
    expect(pool.hasRally[barracks]).toBe(0);

    sim.step([
      { type: CommandType.Train, player: 0, building: pool.idAt(barracks), unit: EntityType.Rifleman },
    ]);
    let unit = -1;
    for (let t = 0; t < defOf(EntityType.Rifleman).buildTicks + 30 && unit < 0; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.Rifleman) {
          unit = i;
        }
      }
    }
    expect(pool.order[unit]).toBe(Order.None);
  });

  it('refuses another player’s building, and refuses points off the map', () => {
    const { sim, barracks } = withBarracks();
    const pool = sim.world.pool;
    const rally = openTile(sim, barracks);

    sim.step([
      {
        type: CommandType.SetRally,
        player: 1,
        building: pool.idAt(barracks),
        x: fromInt(rally.x),
        y: fromInt(rally.y),
      },
    ]);
    expect(pool.hasRally[barracks]).toBe(0);

    sim.step([
      {
        type: CommandType.SetRally,
        player: 0,
        building: pool.idAt(barracks),
        x: fromInt(-40),
        y: fromInt(-40),
      },
    ]);
    expect(pool.hasRally[barracks]).toBe(0);
  });
});

describe('flyers occupy the sky', () => {
  /** Two gunships spawned on the same spot, and how far apart they end up. */
  function twoFlyers(sameOwner: boolean): number {
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const start = map.starts[0]!;
    const x = Math.round((start.tileX + 6.5) * FIX);
    const y = Math.round((start.tileY + 6.5) * FIX);
    const a = pool.spawn(EntityType.Gunship, 0 as PlayerId, x, y) & 0xffff;
    const b = pool.spawn(EntityType.Gunship, (sameOwner ? 0 : 1) as PlayerId, x, y) & 0xffff;
    for (let t = 0; t < 40; t++) sim.step([]);
    return Math.hypot(
      toFloat(pool.posX[a]!) - toFloat(pool.posX[b]!),
      toFloat(pool.posY[a]!) - toFloat(pool.posY[b]!),
    );
  }

  it('pushes two coincident gunships apart', () => {
    const gap = twoFlyers(true);
    // They start exactly on top of each other; separation must resolve that.
    expect(`gap ${gap.toFixed(2)} > 0.3: ${gap > 0.3}`).toBe(`gap ${gap.toFixed(2)} > 0.3: true`);
  });

  it('never shoulders a ground unit aside', () => {
    // Air and ground share the map but not the space. A gunship hovering over a
    // brawler must not shove it, or flyers become a battering ram.
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const start = map.starts[0]!;
    let spot: { x: number; y: number } | null = null;
    for (let r = 4; r < 24 && !spot; r++) {
      for (let dx = -r; dx <= r && !spot; dx++) {
        if (map.isWalkable(start.tileX + dx, start.tileY + r)) {
          spot = { x: start.tileX + dx, y: start.tileY + r };
        }
      }
    }
    const x = Math.round((spot!.x + 0.5) * FIX);
    const y = Math.round((spot!.y + 0.5) * FIX);
    const ground = pool.spawn(EntityType.Brawler, 0 as PlayerId, x, y) & 0xffff;
    pool.spawn(EntityType.Gunship, 0 as PlayerId, x, y);

    const gx = pool.posX[ground]!;
    const gy = pool.posY[ground]!;
    for (let t = 0; t < 30; t++) sim.step([]);
    const moved = Math.hypot(pool.posX[ground]! - gx, pool.posY[ground]! - gy) / FIX;
    expect(`ground unit shoved ${moved.toFixed(3)} tiles`).toBe('ground unit shoved 0.000 tiles');
  });
});
