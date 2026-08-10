/**
 * Terrain height, and the one thing that has to stay true about it.
 *
 * Cliff height and flight altitude used to be two independent numbers, and they
 * drifted the moment the map gained real elevation: ridges grew past 3 units
 * while Beamdrones stayed at 2.4, so air units flew through rock. The altitude is
 * now derived from the terrain's own maximum, and this pins the relationship so
 * a future change to either one fails here instead of in a screenshot.
 */

import { describe, expect, it } from 'vitest';
import { FLIGHT_ALTITUDE, FLYER_BOB, FLYER_UNDERHANG } from '../src/render/entities.js';
import { MAX_CLIFF_HEIGHT, cliffHeightFor } from '../src/render/terrain.js';
import { generateMap } from '../src/sim/map.js';
import { Tile } from '../src/sim/types.js';

/** The lowest point of a flying model, at the bottom of its hover. */
const flyerBelly = FLIGHT_ALTITUDE - FLYER_UNDERHANG - FLYER_BOB;

/** Tallest geometry on a Command Post: the cap at y=2.05, half a unit high. */
const TALLEST_BUILDING_GEOMETRY = 2.3;

describe('cliff height', () => {
  it('rises with distance from open ground, then saturates', () => {
    expect(cliffHeightFor(1)).toBeLessThan(cliffHeightFor(2));
    expect(cliffHeightFor(2)).toBeLessThan(cliffHeightFor(3));
    expect(cliffHeightFor(5)).toBe(cliffHeightFor(4));
    expect(cliffHeightFor(99)).toBe(MAX_CLIFF_HEIGHT);
  });

  it('treats a tile touching open ground as the shortest step', () => {
    expect(cliffHeightFor(1)).toBeGreaterThan(0);
    // Elevation 0 is ground; it must not produce a negative or zero-height box
    // if it ever reaches this function.
    expect(cliffHeightFor(0)).toBe(cliffHeightFor(1));
  });

  it('never exceeds what the generator can produce', () => {
    const map = generateMap(0x51ce7a11);
    let tallest = 0;
    for (let i = 0; i < map.tiles.length; i++) {
      if (map.tiles[i] !== Tile.Cliff) continue;
      const h = cliffHeightFor(map.elevation[i]!);
      if (h > tallest) tallest = h;
    }
    expect(tallest).toBe(MAX_CLIFF_HEIGHT);
  });
});

describe('air units clear the world', () => {
  it('flies over the tallest cliff with daylight to spare', () => {
    expect(flyerBelly).toBeGreaterThan(MAX_CLIFF_HEIGHT);
    // Not merely above it — visibly above it.
    expect(flyerBelly - MAX_CLIFF_HEIGHT).toBeGreaterThanOrEqual(0.25);
  });

  it('flies over the tallest building', () => {
    expect(flyerBelly).toBeGreaterThan(TALLEST_BUILDING_GEOMETRY);
  });

  it('stays low enough to read as part of the battlefield', () => {
    // The camera looks down at 52 degrees, so altitude H displaces a flyer about
    // 0.78H tiles from the ground it is over. Past a few tiles it stops looking
    // like it is above anything in particular.
    expect(FLIGHT_ALTITUDE * 0.78).toBeLessThan(3);
  });
});
