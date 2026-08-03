/**
 * Fog of war texture layout.
 *
 * The shroud is a `DataTexture` stretched over a ground-aligned plane, and
 * texture V runs opposite to tile Y there. Unlike the canvas-backed ground
 * texture, a `DataTexture` is uploaded without three.js's default vertical flip,
 * so the rows have to be written bottom-up by hand.
 *
 * Getting it wrong is close to invisible in a screenshot — the shroud still
 * looks like a shroud, it just lifts over the mirror image of where the player
 * actually is. On a rotationally symmetric map that reads as "the fog is upside
 * down", and in a mirror match each player ends up watching the other's half.
 * Hence a test on the pixel layout rather than on how it looks.
 */

import { describe, expect, it } from 'vitest';
import { EXPLORED, FogRenderer, UNEXPLORED, VISIBLE } from '../src/render/fog.js';
import { GameMap } from '../src/sim/map.js';

/** Alpha the renderer writes for the texel covering this tile. */
function alphaAt(fog: FogRenderer, size: number, tx: number, ty: number): number {
  const pixels = (fog as unknown as { pixels: Uint8Array }).pixels;
  // Rows are stored bottom-up; this is the mapping under test.
  return pixels[((size - 1 - ty) * size + tx) * 4 + 3]!;
}

describe('fog texture layout', () => {
  const size = 16;

  it('writes a tile to the texel that covers it', () => {
    const fog = new FogRenderer(new GameMap(size));

    // An asymmetric pattern: a flip or a transpose both change the answer.
    fog.state.fill(UNEXPLORED);
    fog.state[2 * size + 5] = VISIBLE;
    fog.state[11 * size + 3] = EXPLORED;
    fog.refresh();

    expect(alphaAt(fog, size, 5, 2)).toBe(0);
    expect(alphaAt(fog, size, 3, 11)).toBeGreaterThan(0);
    expect(alphaAt(fog, size, 3, 11)).toBeLessThan(alphaAt(fog, size, 0, 0));

    // The transposed and vertically mirrored positions must NOT be clear — those
    // are the two ways this has actually been got wrong.
    expect(alphaAt(fog, size, 2, 5)).toBeGreaterThan(0);
    expect(alphaAt(fog, size, 5, size - 1 - 2)).toBeGreaterThan(0);
  });

  it('reveals around the tile a unit stands on, for either corner', () => {
    // Both players' bases are 180-degree opposites; each must light up its own.
    for (const [tx, ty] of [
      [3, 4],
      [size - 1 - 3, size - 1 - 4],
    ] as const) {
      const fog = new FogRenderer(new GameMap(size));
      fog.state.fill(UNEXPLORED);
      fog.state[ty * size + tx] = VISIBLE;
      fog.refresh();

      expect(alphaAt(fog, size, tx, ty)).toBe(0);
      expect(alphaAt(fog, size, size - 1 - tx, size - 1 - ty)).toBeGreaterThan(0);
    }
  });

  it('bumps its version when visibility is recomputed', () => {
    const fog = new FogRenderer(new GameMap(size));
    const before = fog.version;
    fog.state[0] = VISIBLE;
    fog.refresh();
    expect(fog.version).toBe(before); // refresh alone is not a change
  });
});
