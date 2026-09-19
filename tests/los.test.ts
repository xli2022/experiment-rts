/**
 * Straight-line visibility and path smoothing.
 *
 * Smoothing is what stops units walking the grid's staircase, and it is allowed
 * to shorten a route but never to invent one. So the tests below care about two
 * things in roughly equal measure: that a clear line is recognised as clear,
 * and that every rule the tile search enforces — walls, and the corner rule
 * that stops a diagonal slipping between two blocked tiles — survives being
 * string-pulled.
 *
 * The determinism properties get their own coverage because they are the ones
 * that fail silently: a visibility test that answered differently depending on
 * which end it was asked from, or on which half of the map asked, would desync
 * two peers rather than look wrong.
 */

import { describe, expect, it } from 'vitest';
import { GameMap } from '../src/sim/map.js';
import { AStar } from '../src/sim/pathing/astar.js';
import { lineOfSightClear, smoothPath } from '../src/sim/pathing/los.js';
import { FIX_HALF, fromInt, type Fix } from '../src/sim/fixed.js';
import { Tile } from '../src/sim/types.js';

/** Centre of a tile, in world coordinates. */
function centre(t: number): Fix {
  return fromInt(t) + FIX_HALF;
}

function openMap(size = 32): GameMap {
  return new GameMap(size);
}

describe('line of sight', () => {
  it('sees straight across open ground', () => {
    const map = openMap();
    expect(lineOfSightClear(map, centre(2), centre(2), centre(20), centre(14))).toBe(true);
  });

  it('is blocked by a wall between the ends', () => {
    const map = openMap();
    for (let y = 0; y < 32; y++) map.tiles[map.index(10, y)] = Tile.Cliff;
    expect(lineOfSightClear(map, centre(2), centre(5), centre(20), centre(5))).toBe(false);
  });

  it('is blocked by a single tile it passes straight through', () => {
    const map = openMap();
    map.tiles[map.index(5, 5)] = Tile.Cliff;
    expect(lineOfSightClear(map, centre(2), centre(5), centre(9), centre(5))).toBe(false);
    // One row over, the same span is clear.
    expect(lineOfSightClear(map, centre(2), centre(6), centre(9), centre(6))).toBe(true);
  });

  it('refuses a start or end inside a wall', () => {
    const map = openMap();
    map.tiles[map.index(5, 5)] = Tile.Cliff;
    expect(lineOfSightClear(map, centre(5), centre(5), centre(9), centre(5))).toBe(false);
    expect(lineOfSightClear(map, centre(2), centre(5), centre(5), centre(5))).toBe(false);
  });

  it('refuses to leave the map', () => {
    const map = openMap();
    expect(lineOfSightClear(map, centre(2), centre(2), fromInt(40), fromInt(40))).toBe(false);
  });

  it('will not slip diagonally between two blocked tiles', () => {
    // The classic corner: (4,5) and (5,4) blocked, so the diagonal from (4,4)
    // to (5,5) touches both and must be refused — the same rule A* and the flow
    // field apply to a single diagonal step.
    const map = openMap();
    map.tiles[map.index(4, 5)] = Tile.Cliff;
    map.tiles[map.index(5, 4)] = Tile.Cliff;
    expect(lineOfSightClear(map, centre(4), centre(4), centre(5), centre(5))).toBe(false);
    // One blocked shoulder is still refused: a unit has a body.
    const half = openMap();
    half.tiles[half.index(4, 5)] = Tile.Cliff;
    expect(lineOfSightClear(half, centre(4), centre(4), centre(5), centre(5))).toBe(false);
  });

  it('allows a diagonal with both shoulders open', () => {
    const map = openMap();
    expect(lineOfSightClear(map, centre(4), centre(4), centre(8), centre(8))).toBe(true);
  });

  it('gives the same answer from either end', () => {
    // A segment is a segment; nothing may depend on which way it is walked.
    const map = openMap();
    for (let y = 4; y < 28; y++) map.tiles[map.index(13, y)] = Tile.Cliff;
    map.tiles[map.index(7, 9)] = Tile.Cliff;
    map.tiles[map.index(20, 21)] = Tile.Cliff;

    for (let ax = 1; ax < 31; ax += 3) {
      for (let ay = 1; ay < 31; ay += 5) {
        for (let bx = 1; bx < 31; bx += 7) {
          for (let by = 1; by < 31; by += 4) {
            const fwd = lineOfSightClear(map, centre(ax), centre(ay), centre(bx), centre(by));
            const back = lineOfSightClear(map, centre(bx), centre(by), centre(ax), centre(ay));
            expect(back, `(${ax},${ay}) -> (${bx},${by})`).toBe(fwd);
          }
        }
      }
    }
  });

  it('answers the mirrored question the mirrored way', () => {
    // The whole point of `flip`: a player on the rotated half asks about the
    // rotated segment and must get the same answer, including where an endpoint
    // sits exactly on a tile boundary and flooring would disagree.
    const size = 32;
    const map = new GameMap(size);
    for (let y = 6; y < 20; y++) {
      map.tiles[map.index(11, y)] = Tile.Cliff;
      map.tiles[map.index(size - 1 - 11, size - 1 - y)] = Tile.Cliff;
    }

    const points: Fix[][] = [];
    for (let x = 2; x < 30; x += 3) {
      for (let y = 2; y < 30; y += 3) {
        points.push([centre(x), centre(y)]);
        points.push([fromInt(x), fromInt(y)]); // exactly on a tile boundary
      }
    }

    for (const [ax, ay] of points) {
      for (const [bx, by] of points) {
        const plain = lineOfSightClear(map, ax!, ay!, bx!, by!, false);
        // The same segment, seen from the rotated half.
        const w = fromInt(size);
        const h = fromInt(size);
        const flipped = lineOfSightClear(map, w - ax!, h - ay!, w - bx!, h - by!, true);
        expect(flipped).toBe(plain);
      }
    }
  });
});

describe('path smoothing', () => {
  it('collapses an open-ground staircase to one leg', () => {
    const map = openMap();
    const astar = new AStar(map);
    const start = map.index(3, 3);
    const path = astar.find(map, start, map.index(25, 19));
    expect(path.length).toBeGreaterThan(20);

    const smooth = smoothPath(map, path, centre(3), centre(3));
    expect(smooth).toEqual([map.index(25, 19)]);
  });

  it('keeps the destination', () => {
    const map = openMap();
    const astar = new AStar(map);
    for (let y = 0; y < 24; y++) map.tiles[map.index(16, y)] = Tile.Cliff;
    const path = astar.find(map, map.index(3, 3), map.index(28, 3));
    const smooth = smoothPath(map, path, centre(3), centre(3));
    expect(smooth[smooth.length - 1]).toBe(map.index(28, 3));
    expect(smooth.length).toBeLessThan(path.length);
  });

  it('keeps every leg walkable', () => {
    // The property that matters: a unit walking the smoothed polyline must
    // never cross anything the tile route would have gone around.
    const map = openMap(48);
    for (let y = 0; y < 30; y++) map.tiles[map.index(20, y)] = Tile.Cliff;
    for (let x = 20; x < 40; x++) map.tiles[map.index(x, 30)] = Tile.Cliff;
    map.tiles[map.index(33, 12)] = Tile.Cliff;
    map.tiles[map.index(34, 12)] = Tile.Cliff;

    const astar = new AStar(map);
    for (let sy = 2; sy < 46; sy += 5) {
      for (let gy = 2; gy < 46; gy += 5) {
        const path = astar.find(map, map.index(2, sy), map.index(45, gy));
        if (path.length === 0) continue;
        const smooth = smoothPath(map, path, centre(2), centre(sy));
        expect(smooth[smooth.length - 1]).toBe(path[path.length - 1]);

        let px = centre(2);
        let py = centre(sy);
        for (const node of smooth) {
          const nx = centre(map.tileXOf(node));
          const ny = centre(map.tileYOf(node));
          expect(
            lineOfSightClear(map, px, py, nx, ny),
            `leg to (${map.tileXOf(node)},${map.tileYOf(node)}) crosses something`,
          ).toBe(true);
          px = nx;
          py = ny;
        }
      }
    }
  });

  it('is never longer than the route it smooths', () => {
    const map = openMap(48);
    for (let y = 8; y < 40; y++) map.tiles[map.index(24, y)] = Tile.Cliff;
    const astar = new AStar(map);
    const path = astar.find(map, map.index(4, 20), map.index(44, 20));
    const smooth = smoothPath(map, path, centre(4), centre(20));
    expect(smooth.length).toBeLessThanOrEqual(path.length);
    expect(smooth.length).toBeGreaterThan(0);
  });

  it('smooths the mirrored route into the mirrored corners', () => {
    const size = 48;
    const map = new GameMap(size);
    for (let y = 8; y < 34; y++) {
      map.tiles[map.index(24, y)] = Tile.Cliff;
      map.tiles[map.index(size - 1 - 24, size - 1 - y)] = Tile.Cliff;
    }
    const astar = new AStar(map);

    const start = map.index(6, 20);
    const goal = map.index(42, 26);
    const plain = smoothPath(
      map,
      astar.find(map, start, goal, [], false),
      centre(6),
      centre(20),
      false,
    );

    const mStart = map.mirrorIndex(start);
    const mGoal = map.mirrorIndex(goal);
    const mirrored = smoothPath(
      map,
      astar.find(map, mStart, mGoal, [], true),
      centre(size - 1 - 6),
      centre(size - 1 - 20),
      true,
    );

    expect(plain.length).toBeGreaterThan(1);
    expect(mirrored.map((t) => map.mirrorIndex(t))).toEqual(plain);
  });

  it('returns an empty result for an empty route', () => {
    const map = openMap();
    expect(smoothPath(map, [], centre(3), centre(3))).toEqual([]);
  });

  it('keeps a single-node route', () => {
    const map = openMap();
    expect(smoothPath(map, [map.index(4, 3)], centre(3), centre(3))).toEqual([map.index(4, 3)]);
  });
});
