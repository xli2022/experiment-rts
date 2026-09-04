/**
 * Nothing may end up standing where it cannot stand.
 *
 * Three separate routes put a unit on a solid tile, and every one of them shows
 * up to the player the same way: `clampToMap` teleports the unit to the middle
 * of the nearest open tile, and if whatever put it there does it again next
 * tick, the unit snaps between two positions for as long as the cause lasts.
 *
 *   - production placed trained units at a point computed purely geometrically;
 *   - a rally point was validated for being on the map but not for being ground;
 *   - separation and direct steering wrote positions with no idea walls exist.
 *
 * The invariants at the bottom are the real guard: they run whole matches and
 * assert the property rather than any one of its causes.
 */

import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt, toFloat } from '../src/sim/fixed.js';
import { duelMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, Order, type PlayerId } from '../src/sim/types.js';

const FIX = 65536;

/** Open ground big enough to work in, anywhere on the map. */
function openArea(sim: Simulation, w: number, h: number): { x: number; y: number } {
  const { map } = sim.world;
  for (let y = 1; y < map.height - h - 1; y++) {
    for (let x = 1; x < map.width - w - 1; x++) {
      let ok = true;
      for (let a = 0; a < w && ok; a++) {
        for (let b = 0; b < h && ok; b++) ok = map.isWalkable(x + a, y + b);
      }
      if (ok) return { x, y };
    }
  }
  throw new Error(`no open ${w}x${h} area`);
}

/** A finished building on the map, footprint occupied like a real one. */
function place(sim: Simulation, type: EntityType, tx: number, ty: number): number {
  const { pool, map } = sim.world;
  const def = defOf(type);
  const id = pool.spawn(
    type,
    0 as PlayerId,
    Math.round((tx + def.footprint / 2) * FIX),
    Math.round((ty + def.footprint / 2) * FIX),
  );
  const i = id & 0xffff;
  pool.buildState[i] = BuildState.Complete;
  pool.tileX[i] = tx;
  pool.tileY[i] = ty;
  map.setOccupied(tx, ty, def.footprint, 1);
  return i;
}

function standable(sim: Simulation, i: number): boolean {
  const { map, pool } = sim.world;
  const tile = map.tileOfPos(pool.posX[i]!, pool.posY[i]!);
  return tile >= 0 && map.isWalkable(map.tileXOf(tile), map.tileYOf(tile));
}

/**
 * Solid rock `depth` tiles thick, but with open ground within `reach`.
 *
 * The `reach` bound matters: the map's border is a band of rock more than ten
 * tiles from anything walkable, and an order into the middle of *that* is
 * refused rather than snapped — correctly, since there is nowhere near it to
 * send anyone. Testing the snap needs a cliff, not the edge of the world.
 */
function rockNearGround(sim: Simulation, depth: number, reach = 5): { x: number; y: number } {
  const { map } = sim.world;
  const openWithin = (x: number, y: number): boolean => {
    for (let a = -reach; a <= reach; a++) {
      for (let b = -reach; b <= reach; b++) if (map.isWalkable(x + a, y + b)) return true;
    }
    return false;
  };
  for (let y = depth; y < map.height - depth; y++) {
    for (let x = depth; x < map.width - depth; x++) {
      let solid = true;
      for (let a = -depth; a <= depth && solid; a++) {
        for (let b = -depth; b <= depth && solid; b++) {
          if (map.isWalkable(x + a, y + b)) solid = false;
        }
      }
      if (solid && openWithin(x, y)) return { x, y };
    }
  }
  throw new Error(`no rock ${depth} deep near open ground`);
}

/** The middle of the map's border band — rock with nothing standable near it. */
function unreachableRock(sim: Simulation): { x: number; y: number } {
  const { map } = sim.world;
  for (let y = 1; y < map.height; y++) {
    for (let x = 1; x < map.width; x++) {
      if (map.isWalkable(x, y)) continue;
      let clear = false;
      for (let a = -9; a <= 9 && !clear; a++) {
        for (let b = -9; b <= 9 && !clear; b++) if (map.isWalkable(x + a, y + b)) clear = true;
      }
      if (!clear) return { x, y };
    }
  }
  throw new Error('no unreachable rock');
}

function setRally(sim: Simulation, b: number, to: { x: number; y: number }): void {
  sim.step([
    {
      type: CommandType.SetRally,
      player: 0,
      building: sim.world.pool.idAt(b),
      x: fromInt(to.x) + 32768,
      y: fromInt(to.y) + 32768,
    } as Command,
  ]);
}

describe('trained units are placed somewhere they can stand', () => {
  it('does not drop a unit inside a neighbouring building', () => {
    // Two barracks side by side: the geometric spawn point of the first lands
    // squarely inside the second, which is how 3 of 89 units in a bot match
    // came out standing in a wall.
    const sim = new Simulation(0x51ce7a11);
    const { pool } = sim.world;
    const area = openArea(sim, 14, 14);
    const b = place(sim, EntityType.Barracks, area.x + 4, area.y + 4);
    // Directly on the side units come out of.
    place(sim, EntityType.Barracks, area.x + 4, area.y + 7);

    const before = new Set<number>();
    for (let i = 0; i < pool.count; i++) if (pool.alive[i] === 1) before.add(i);
    for (let k = 0; k < 3; k++) pool.prodPush(b, EntityType.Burstbot);

    // Every tick, not just at the end. `clampToMap` ejects a unit standing in a
    // wall within a tick or two, so by the end of the run a badly placed unit
    // looks exactly like a correctly placed one — the damage is the ejection,
    // and it is only visible while it is happening.
    const seen = new Set<number>();
    let worstTick = -1;
    for (let t = 0; t < 900; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1 || before.has(i)) continue;
        if (defOf(pool.type[i]! as EntityType).isBuilding) continue;
        seen.add(i);
        if (!standable(sim, i) && worstTick < 0) worstTick = t;
      }
    }
    expect(seen.size).toBeGreaterThan(0);
    expect(worstTick < 0 ? 'never stood in a wall' : `stood in a wall at tick ${worstTick}`).toBe(
      'never stood in a wall',
    );
  });

  it('holds production rather than placing a unit with nowhere to go', () => {
    // Walled in on every side: there is no legal spot, so the finished unit
    // waits in the queue exactly as it does when supply runs out.
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const area = openArea(sim, 14, 14);
    const b = place(sim, EntityType.Barracks, area.x + 5, area.y + 5);

    // Make everything around the barracks solid.
    for (let y = area.y; y < area.y + 14; y++) {
      for (let x = area.x; x < area.x + 14; x++) {
        const insideRax = x >= area.x + 5 && x < area.x + 8 && y >= area.y + 5 && y < area.y + 8;
        if (!insideRax) map.setOccupied(x, y, 1, 1);
      }
    }

    const before = new Set<number>();
    for (let i = 0; i < pool.count; i++) if (pool.alive[i] === 1) before.add(i);
    pool.prodPush(b, EntityType.Burstbot);
    for (let t = 0; t < 900; t++) sim.step([]);

    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || before.has(i)) continue;
      if (defOf(pool.type[i]! as EntityType).isBuilding) continue;
      // Anything that did get out must at least be standing legally.
      expect(standable(sim, i)).toBe(true);
    }
  });
});

describe('a rally point names ground', () => {
  it('snaps a rally set on a cliff to the ground beside it', () => {
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const area = openArea(sim, 12, 12);
    const b = place(sim, EntityType.Barracks, area.x + 4, area.y + 4);

    setRally(sim, b, rockNearGround(sim, 1));

    expect(pool.hasRally[b]).toBe(1);
    const tile = map.tileOfPos(pool.rallyX[b]!, pool.rallyY[b]!);
    expect(tile).toBeGreaterThanOrEqual(0);
    expect(map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))).toBe(true);
  });

  it('refuses a rally with no ground anywhere near it', () => {
    // Deep inside the map's border band. Snapping here would send units to
    // somewhere nowhere near where the player pointed, so the order is dropped.
    const sim = new Simulation(0x51ce7a11);
    const { pool } = sim.world;
    const area = openArea(sim, 12, 12);
    const b = place(sim, EntityType.Barracks, area.x + 4, area.y + 4);

    setRally(sim, b, unreachableRock(sim));
    expect(pool.hasRally[b]).toBe(0);
  });

  it('sends trained units to a destination they can reach', () => {
    // The bug this covers is not the rally itself but the path production takes
    // to apply it: it writes the order straight onto the unit and never goes
    // near the command validation, so a bad rally became a bad order forever.
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;
    const area = openArea(sim, 12, 12);
    const b = place(sim, EntityType.Barracks, area.x + 4, area.y + 4);

    setRally(sim, b, rockNearGround(sim, 1));
    expect(pool.hasRally[b]).toBe(1);

    const before = new Set<number>();
    for (let i = 0; i < pool.count; i++) if (pool.alive[i] === 1) before.add(i);
    for (let k = 0; k < 3; k++) pool.prodPush(b, EntityType.Burstbot);

    // Checked while they are still walking, since an order that has completed
    // no longer says anything about where it pointed.
    let checked = 0;
    for (let t = 0; t < 1500; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1 || before.has(i)) continue;
        if (pool.order[i] !== Order.Move) continue;
        checked++;
        const tile = map.tileOfPos(pool.orderX[i]!, pool.orderY[i]!);
        expect(tile).toBeGreaterThanOrEqual(0);
        expect(map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))).toBe(true);
      }
    }
    expect(checked).toBeGreaterThan(0);
  });
});

describe('nothing is ever pushed into a wall', () => {
  it('keeps a crowd jammed against rock out of the rock', () => {
    // Separation had no idea walls existed. A crowd shoved against a cliff put
    // its outer members inside it, and clampToMap teleported them back to the
    // middle of the nearest open tile — every tick, for as long as the crowd
    // lasted. 22 of the 25 relocations in a bot match were this.
    const sim = new Simulation(0x51ce7a11);
    const { pool, map } = sim.world;

    // A tile with rock on one side, and room to pack units against it.
    let spot: { x: number; y: number } | null = null;
    for (let y = 2; y < map.height - 2 && !spot; y++) {
      for (let x = 2; x < map.width - 2 && !spot; x++) {
        if (!map.isWalkable(x, y)) continue;
        if (map.isWalkable(x + 1, y)) continue; // rock to the east
        let room = true;
        for (let d = 1; d <= 3 && room; d++) room = map.isWalkable(x - d, y);
        if (room) spot = { x, y };
      }
    }
    expect(spot).not.toBeNull();

    const ids: number[] = [];
    for (let k = 0; k < 8; k++) {
      ids.push(
        pool.spawn(
          EntityType.Burstbot,
          0 as PlayerId,
          Math.round((spot!.x + 0.5 - k * 0.15) * FIX),
          Math.round((spot!.y + 0.5) * FIX),
        ) & 0xffff,
      );
    }
    // Drive them all at the rock so separation has to resolve a real jam.
    for (const i of ids) {
      pool.order[i] = Order.Move;
      pool.orderX[i] = Math.round((spot!.x + 0.5) * FIX);
      pool.orderY[i] = Math.round((spot!.y + 0.5) * FIX);
    }

    // Counting raw displacement would not work here: eight units stacked on one
    // point each take a push from every neighbour in the same tick, and the sum
    // legitimately exceeds a tick of walking. What is being looked for is the
    // ejection specifically, and `clampToMap` has a signature — it drops the
    // unit exactly on a tile centre.
    let relocations = 0;
    const prev = ids.map((i) => ({ x: toFloat(pool.posX[i]!), y: toFloat(pool.posY[i]!) }));
    for (let t = 0; t < 400; t++) {
      sim.step([]);
      for (let k = 0; k < ids.length; k++) {
        const i = ids[k]!;
        if (pool.alive[i] !== 1) continue;
        expect(standable(sim, i)).toBe(true);
        const x = toFloat(pool.posX[i]!);
        const y = toFloat(pool.posY[i]!);
        const onCentre = Math.abs(x % 1) === 0.5 && Math.abs(y % 1) === 0.5;
        const moved = Math.hypot(x - prev[k]!.x, y - prev[k]!.y) > 0.01;
        if (onCentre && moved) relocations++;
        prev[k] = { x, y };
      }
    }
    expect(relocations).toBe(0);
  });
});

describe('pathing invariants over a whole match', () => {
  it('never leaves a unit standing on, or ordered onto, a solid tile', () => {
    const sim = new Simulation(duelMatch(0x51ce7a11, { botPlayers: [0, 1] }));
    const { pool, map } = sim.world;

    let onRock = 0;
    let badDest = 0;
    for (let t = 0; t < 3000; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1) continue;
        const def = defOf(pool.type[i]! as EntityType);
        if (def.isBuilding || def.speedPerTick === 0 || def.flying) continue;
        if (!standable(sim, i)) onRock++;
        const order = pool.order[i]!;
        if (order !== Order.Move && order !== Order.AttackMove) continue;
        const dt = map.tileOfPos(pool.orderX[i]!, pool.orderY[i]!);
        if (dt < 0 || !map.isWalkable(map.tileXOf(dt), map.tileYOf(dt))) badDest++;
      }
    }
    expect(`${onRock} on rock, ${badDest} ordered onto rock`).toBe('0 on rock, 0 ordered onto rock');
  });

  it('never leaves a unit holding one move order for a whole match', () => {
    // Crossing the entire map takes well under a thousand ticks, so an order
    // held longer than this is one that can never complete.
    const sim = new Simulation(duelMatch(0x51ce7a11, { botPlayers: [0, 1] }));
    const pool = sim.world.pool;

    const held = new Int32Array(pool.posX.length);
    const key = new Float64Array(pool.posX.length);
    const gen = new Int32Array(pool.posX.length);
    let worst = 0;

    for (let t = 0; t < 3000; t++) {
      sim.step([]);
      for (let i = 0; i < pool.count; i++) {
        if (pool.alive[i] !== 1 || gen[i] !== pool.generation[i]!) {
          gen[i] = pool.generation[i]!;
          held[i] = 0;
          continue;
        }
        const order = pool.order[i]!;
        if (order !== Order.Move && order !== Order.AttackMove) {
          held[i] = 0;
          continue;
        }
        const k = pool.orderX[i]! * 1e6 + pool.orderY[i]! + order;
        if (key[i] === k) {
          held[i]! += 1;
          if (held[i]! > worst) worst = held[i]!;
        } else {
          key[i] = k;
          held[i] = 0;
        }
      }
    }
    expect(`longest held move order: ${worst < 1500 ? 'under 1500' : worst} ticks`).toBe(
      'longest held move order: under 1500 ticks',
    );
  });
});
