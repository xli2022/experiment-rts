/**
 * Encode the unit skins to KTX2.
 *
 * Source art lives in `assets/textures/` and is not served; only the encoded
 * `.ktx2` files under `public/models/` ship. Run after changing a skin:
 *
 *   npm run textures
 *
 * ETC1S rather than UASTC. UASTC is the higher-quality mode and roughly eight
 * times the size, which is the wrong trade for a unit that is sixty pixels tall
 * on screen — these are read at a distance, not inspected.
 *
 * The vertical flip is baked in. A compressed texture cannot be flipped as it
 * uploads the way a PNG can, so the flip that the models' UVs need has to happen
 * here instead; the loader then takes the file as-is.
 */

import { readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { join, basename } from 'node:path';
import sharp from 'sharp';
import { encodeToKTX2 } from 'ktx2-encoder';

const SOURCE = new URL('../assets/textures/', import.meta.url).pathname;
const OUT = new URL('../public/models/', import.meta.url).pathname;

await mkdir(OUT, { recursive: true });
const files = (await readdir(SOURCE)).filter((f) => f.endsWith('.png')).sort();
if (files.length === 0) {
  console.error(`no PNGs in ${SOURCE}`);
  process.exit(1);
}

let before = 0;
let after = 0;
for (const file of files) {
  const png = await readFile(join(SOURCE, file));
  const ktx2 = await encodeToKTX2(new Uint8Array(png), {
    // Node has no built-in image decoding, so the encoder asks for one.
    imageDecoder: async (buffer) => {
      const image = sharp(Buffer.from(buffer)).ensureAlpha();
      const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
      return { width: info.width, height: info.height, data: new Uint8Array(data) };
    },
    isUASTC: false,
    isKTX2File: true,
    generateMipmap: true,
    // Colour art, so the encoder should weigh error perceptually and the file
    // should declare sRGB. Getting this wrong washes the skins out.
    isPerceptual: true,
    isSetKTX2SRGBTransferFunc: true,
    isYFlip: true,
    qualityLevel: 210,
    compressionLevel: 4,
  });
  const out = join(OUT, `${basename(file, '.png')}.ktx2`);
  await writeFile(out, ktx2);
  before += png.byteLength;
  after += ktx2.byteLength;
  const pct = (100 * (1 - ktx2.byteLength / png.byteLength)).toFixed(1);
  console.log(
    `${file.padEnd(26)} ${(png.byteLength / 1024).toFixed(0).padStart(6)} KB -> ` +
      `${(ktx2.byteLength / 1024).toFixed(0).padStart(5)} KB  (-${pct}%)`,
  );
}
console.log(
  `\ntotal ${(before / 1024 / 1024).toFixed(2)} MB -> ${(after / 1024 / 1024).toFixed(2)} MB ` +
    `(-${(100 * (1 - after / before)).toFixed(1)}%)`,
);
