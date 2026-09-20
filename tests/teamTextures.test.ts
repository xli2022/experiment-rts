import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
// @ts-expect-error Node build scripts do not have TypeScript declarations.
import { tintTeamPixels } from '../scripts/lib/team-tint.mjs';
// @ts-expect-error Node build scripts do not have TypeScript declarations.
import { decodeKTX2 } from '../scripts/lib/ktx2-rgba.mjs';

interface Decoded {
  width: number;
  height: number;
  data: Uint8Array;
}

const pixels = (...rgba: number[][]): Uint8Array => Uint8Array.from(rgba.flat());
const rgbAt = (data: Uint8Array, pixel: number): number[] =>
  Array.from(data.subarray(pixel * 4, pixel * 4 + 3));

describe('paired team-paint tinting', () => {
  it.each(['teal', 'orange'])(
    'keeps %s pixel layout and source alpha without mutating the originals',
    (team) => {
      const blue = pixels([15, 60, 200, 0], [40, 90, 220, 37], [30, 70, 180, 255]);
      const red = pixels([200, 35, 15, 0], [220, 60, 40, 91], [180, 50, 30, 255]);
      const beforeBlue = blue.slice();
      const beforeRed = red.slice();
      const source = team === 'teal' ? blue : red;
      const { data } = tintTeamPixels(blue, red, team) as { data: Uint8Array };
      expect(data).not.toBe(source);
      expect(data).toHaveLength(source.length);
      expect(data.subarray(0, 4)).toEqual(source.subarray(0, 4));
      for (let i = 3; i < data.length; i += 4) expect(data[i]).toBe(source[i]);
      expect(rgbAt(data, 1)).not.toEqual(rgbAt(source, 1));
      expect(blue).toEqual(beforeBlue);
      expect(red).toEqual(beforeRed);
    },
  );

  it.each(['teal', 'orange'])(
    'leaves shared ice, gold, neutral metal and small compression noise unchanged for %s',
    (team) => {
      const blue = pixels(
        [40, 170, 235, 255],
        [210, 170, 30, 255],
        [80, 80, 80, 255],
        [81, 80, 79, 255],
      );
      const red = pixels(
        [40, 170, 235, 255],
        [210, 170, 30, 255],
        [80, 80, 80, 255],
        [80, 81, 81, 255],
      );
      const result = tintTeamPixels(blue, red, team);
      expect(result.data).toEqual(team === 'teal' ? blue : red);
      expect(result.changedPixels).toBe(0);
    },
  );

  it.each(['teal', 'orange'])(
    'preserves shadow and highlight value and saturation in %s paint',
    (team) => {
      const blue = pixels([8, 20, 64, 255], [35, 80, 160, 255], [165, 193, 248, 255]);
      const red = pixels([64, 12, 8, 255], [160, 55, 35, 255], [248, 179, 165, 255]);
      const source = team === 'teal' ? blue : red;
      const { data } = tintTeamPixels(blue, red, team) as { data: Uint8Array };
      for (let pixel = 0; pixel < 3; pixel++) {
        const before = rgbAt(source, pixel);
        const after = rgbAt(data, pixel);
        expect(Math.max(...after)).toBe(Math.max(...before));
        expect(Math.min(...after)).toBe(Math.min(...before));
        if (team === 'teal') {
          expect(after[1]).toBeGreaterThan(after[2]);
          expect(after[2]).toBeGreaterThan(after[0]);
        } else {
          expect(after[0]).toBeGreaterThan(after[1]);
          expect(after[1]).toBeGreaterThan(after[2]);
        }
      }
    },
  );

  it('recognizes red paint even when the paired blue panel is almost neutral', () => {
    // An actual Revolver atlas panel: requiring both skins to be highly
    // saturated would leave this red region unchanged in the orange unit.
    const blue = pixels([71, 54, 79, 255]);
    const red = pixels([128, 29, 21, 255]);
    const { data, changedPixels } = tintTeamPixels(blue, red, 'orange');
    expect(changedPixels).toBe(1);
    expect(data[0]).toBe(128);
    expect(data[1]).toBeGreaterThan(red[1] + 40);
    expect(data[1]).toBeLessThan(data[0]);
    expect(data[2]).toBe(21);
  });

  it('rejects unequal or incomplete RGBA buffers and unknown teams', () => {
    expect(() => tintTeamPixels(new Uint8Array(4), new Uint8Array(8), 'teal')).toThrow(
      /matching RGBA/,
    );
    expect(() => tintTeamPixels(new Uint8Array(5), new Uint8Array(5), 'orange')).toThrow(
      /matching RGBA/,
    );
    for (const team of ['purple', 'toString', '__proto__']) {
      expect(() => tintTeamPixels(new Uint8Array(4), new Uint8Array(4), team)).toThrow(
        /Unknown derived team/,
      );
    }
  });
});

describe('KTX2 derived texture round trip', () => {
  let sourceBytes: Uint8Array;
  let blue: Decoded;
  let red: Decoded;
  let teal: Decoded;

  beforeAll(async () => {
    sourceBytes = await readFile(new URL('../public/units/revolver-blue.ktx2', import.meta.url));
    [blue, red, teal] = await Promise.all([
      decodeKTX2(sourceBytes),
      readFile(new URL('../public/units/revolver-red.ktx2', import.meta.url)).then(decodeKTX2),
      readFile(new URL('../public/units/revolver-teal.ktx2', import.meta.url)).then(decodeKTX2),
    ]);
  });

  it('decodes original and generated textures to the same RGBA dimensions', () => {
    expect(blue.width).toBeGreaterThan(0);
    expect(blue.height).toBeGreaterThan(0);
    for (const image of [blue, red, teal]) {
      expect([image.width, image.height]).toEqual([blue.width, blue.height]);
      expect(image.data).toHaveLength(image.width * image.height * 4);
    }
  });

  it('retains atlas orientation and paint detail through compressed encoding', () => {
    const expected: Uint8Array = tintTeamPixels(blue.data, red.data, 'teal').data;
    let error = 0;
    let flippedError = 0;
    let channels = 0;
    // Sample the entire atlas, comparing to independent decoded source data.
    // ETC1S is lossy; the correct orientation must be close and much better
    // than another vertical flip, which would put paint on unrelated UVs.
    for (let y = 0; y < blue.height; y += 3) {
      for (let x = 0; x < blue.width; x += 3) {
        const i = (y * blue.width + x) * 4;
        const flipped = ((blue.height - 1 - y) * blue.width + x) * 4;
        for (let channel = 0; channel < 3; channel++) {
          error += Math.abs(teal.data[i + channel] - expected[i + channel]);
          flippedError += Math.abs(teal.data[i + channel] - expected[flipped + channel]);
          channels++;
        }
      }
    }
    expect(error / channels).toBeLessThan(4);
    expect(flippedError).toBeGreaterThan(error * 8);
  });

  it('rejects empty, truncated and invalid KTX2 before returning pixel data', async () => {
    const corrupt = sourceBytes.slice();
    corrupt[0] = 0;
    for (const bytes of [
      new Uint8Array(),
      new Uint8Array([1, 2, 3]),
      sourceBytes.subarray(0, 80),
      corrupt,
    ]) {
      await expect(decodeKTX2(bytes)).rejects.toThrow(/valid.*Basis KTX2/);
    }
  });
});
