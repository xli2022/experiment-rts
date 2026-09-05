/**
 * Pathfinding tests.
 *
 * These matter beyond ordinary correctness: pathfinding is where determinism is
 * easiest to lose, because equal-cost routes are everywhere on a grid and any
 * arbitrary tie-break diverges peers. The tests below therefore check not just
 * that a route is found, but that the *same* route is found every time.
 *
 * The flow field also needs its own coverage because the gameplay determinism
 * test never reaches the group-size threshold that activates it — profiling
 * showed it running zero times across a full simulated match.
 */

import { describe, expect, it } from 'vitest';
import { AStar, nearestWalkable } from '../src/sim/pathing/astar.js';
import { FlowField, FlowFieldCache, UNREACHABLE } from '../src/sim/pathing/flowfield.js';
import { GameMap } from '../src/sim/map.js';
import { Tile } from '../src/sim/types.js';

/** Open square map with no obstacles. */
function openMap(size = 32): GameMap {
  return new GameMap(size);
}

/** Map split in two by a full-height wall, with an optional gap. */
function walledMap(size = 32, gapY = -1): GameMap {
  const map = new GameMap(size);
  const wallX = size >> 1;
  for (let y = 0; y < size; y++) {
    if (y === gapY) continue;
    map.tiles[map.index(wallX, y)] = Tile.Cliff;
  }
  return map;
}

describe('A*', () => {
  it('finds a route across an open map', () => {
    const map = openMap();
    const astar = new AStar(map);
    const path = astar.find(map, map.index(2, 2), map.index(20, 14));
    expect(path.length).toBeGreaterThan(0);
    expect(path[path.length - 1]).toBe(map.index(20, 14));
  });

  it('returns a contiguous route', () => {
    const map = openMap();
    const astar = new AStar(map);
    const start = map.index(3, 3);
    const path = astar.find(map, start, map.index(25, 19));

    let prev = start;
    for (const tile of path) {
      const dx = Math.abs(map.tileXOf(tile) - map.tileXOf(prev));
      const dy = Math.abs(map.tileYOf(tile) - map.tileYOf(prev));
      expect(Math.max(dx, dy)).toBe(1); // each step moves exactly one tile
      prev = tile;
    }
  });

  it('never routes through a wall', () => {
    const map = walledMap(32, 8);
    const astar = new AStar(map);
    const path = astar.find(map, map.index(2, 2), map.index(29, 2));
    expect(path.length).toBeGreaterThan(0);
    for (const tile of path) {
      expect(map.tiles[tile]).not.toBe(Tile.Cliff);
    }
    // The only opening is at y = 8, so the route must pass through it.
    expect(path.some((t) => map.tileYOf(t) === 8)).toBe(true);
  });

  it('reports no route when the goal is sealed off', () => {
    const map = walledMap(32); // no gap
    const astar = new AStar(map);
    const path = astar.find(map, map.index(2, 2), map.index(29, 2));
    expect(path).toEqual([]);
  });

  it('produces byte-identical routes across repeated searches', () => {
    // The tile-index tie-break is what guarantees this. Without it, equal-cost
    // routes would be chosen by heap history and two peers would disagree.
    const map = walledMap(32, 20);
    const first = new AStar(map).find(map, map.index(1, 1), map.index(30, 30));
    for (let i = 0; i < 20; i++) {
      const again = new AStar(map).find(map, map.index(1, 1), map.index(30, 30));
      expect(again).toEqual(first);
    }
  });

  it('finds the nearest walkable tile beside an obstacle', () => {
    const map = openMap();
    map.tiles[map.index(10, 10)] = Tile.Cliff;
    const found = nearestWalkable(map, 10, 10);
    expect(found).toBeGreaterThanOrEqual(0);
    expect(map.tiles[found]).not.toBe(Tile.Cliff);
  });
});

describe('flow field', () => {
  it('assigns zero distance at the goal and grows outward', () => {
    const map = openMap();
    const field = new FlowField(map.width * map.height);
    const goal = map.index(16, 16);
    field.build(map, goal);

    expect(field.dist[goal]).toBe(0);
    // Orthogonal neighbour costs 10, diagonal costs 14.
    expect(field.dist[map.index(17, 16)]).toBe(10);
    expect(field.dist[map.index(17, 17)]).toBe(14);
    // Further away is strictly greater.
    expect(field.dist[map.index(24, 16)]!).toBeGreaterThan(field.dist[map.index(20, 16)]!);
  });

  it('marks sealed-off regions unreachable', () => {
    const map = walledMap(32);
    const field = new FlowField(map.width * map.height);
    field.build(map, map.index(2, 2));

    expect(field.dist[map.index(4, 4)]).not.toBe(UNREACHABLE);
    expect(field.dist[map.index(29, 4)]).toBe(UNREACHABLE);
    expect(field.isStranded(map.index(29, 4))).toBe(true);
  });

  it('descends monotonically to the goal from anywhere reachable', () => {
    // This is the property units rely on: following stepFrom must terminate at
    // the goal rather than looping between two tiles.
    const map = walledMap(48, 30);
    const field = new FlowField(map.width * map.height);
    const goal = map.index(44, 40);
    field.build(map, goal);

    const start = map.index(2, 2);
    let cur = start;
    let guard = 0;
    let last = field.dist[cur]!;
    while (cur !== goal) {
      const next = field.stepFrom(map, cur);
      expect(next).toBeGreaterThanOrEqual(0);
      const d = field.dist[next]!;
      expect(d).toBeLessThan(last); // strictly closer every step
      last = d;
      cur = next;
      expect(++guard).toBeLessThan(map.width * map.height);
    }
    expect(cur).toBe(goal);
  });

  it('agrees with A* on shortest-path cost', () => {
    // Two independent algorithms should measure the same distance; a mismatch
    // means one of them is wrong.
    const map = walledMap(32, 12);
    const goal = map.index(30, 30);
    const start = map.index(1, 1);

    const field = new FlowField(map.width * map.height);
    field.build(map, goal);

    const path = new AStar(map).find(map, start, goal);
    // Recompute the A* route's cost using the same 10/14 step weights.
    let cost = 0;
    let prev = start;
    for (const tile of path) {
      const dx = Math.abs(map.tileXOf(tile) - map.tileXOf(prev));
      const dy = Math.abs(map.tileYOf(tile) - map.tileYOf(prev));
      cost += dx !== 0 && dy !== 0 ? 14 : 10;
      prev = tile;
    }
    expect(cost).toBe(field.dist[start]);
  });

  it('builds identical fields from independent instances', () => {
    const map = walledMap(32, 7);
    const goal = map.index(28, 28);
    const a = new FlowField(map.width * map.height);
    const b = new FlowField(map.width * map.height);
    a.build(map, goal);
    b.build(map, goal);
    expect(Array.from(a.dist)).toEqual(Array.from(b.dist));
  });

  it('does not cut diagonally between two blocked tiles', () => {
    const map = openMap(16);
    // A diagonal pinch: (5,4) and (4,5) blocked, so (4,4)->(5,5) must go around.
    map.tiles[map.index(5, 4)] = Tile.Cliff;
    map.tiles[map.index(4, 5)] = Tile.Cliff;
    const field = new FlowField(map.width * map.height);
    field.build(map, map.index(5, 5));
    // Squeezing through would cost 14; going around costs more.
    expect(field.dist[map.index(4, 4)]!).toBeGreaterThan(14);
  });
});

describe('flow field cache', () => {
  it('reuses a field for a repeated goal', () => {
    const map = openMap();
    const cache = new FlowFieldCache(map.width * map.height);
    const first = cache.get(map, map.index(8, 8));
    const second = cache.get(map, map.index(8, 8));
    expect(second).toBe(first); // same instance, not rebuilt
  });

  it('rebuilds when a building changes the map', () => {
    const map = openMap();
    const cache = new FlowFieldCache(map.width * map.height);
    const goal = map.index(8, 8);
    const field = cache.get(map, goal);
    const versionBefore = field.builtVersion;

    map.setOccupied(4, 4, 2, 1); // place a building
    const after = cache.get(map, goal);
    expect(after.builtVersion).not.toBe(versionBefore);
    // The occupied tiles are no longer walkable, so they drop out of the field.
    expect(after.dist[map.index(4, 4)]).toBe(UNREACHABLE);
  });

  it('evicts without corrupting other entries', () => {
    const map = openMap();
    const cache = new FlowFieldCache(map.width * map.height, 3);
    const goals = [map.index(2, 2), map.index(20, 20), map.index(5, 25), map.index(28, 3)];
    for (const g of goals) {
      const f = cache.get(map, g);
      expect(f.dist[g]).toBe(0);
    }
    // Re-requesting an evicted goal must give a correctly rebuilt field.
    const again = cache.get(map, goals[0]!);
    expect(again.dist[goals[0]!]).toBe(0);
    expect(again.goalTile).toBe(goals[0]);
  });
});
