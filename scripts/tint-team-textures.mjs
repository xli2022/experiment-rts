/** Generate teal/orange from committed blue/red KTX2; source PNGs are optional. */
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodeToKTX2 } from 'ktx2-encoder';
import sharp from 'sharp';
import { decodeKTX2 } from './lib/ktx2-rgba.mjs';
import { tintTeamPixels } from './lib/team-tint.mjs';

const DEFAULT_DIRECTORY = fileURLToPath(new URL('../public/units/', import.meta.url));

export async function generateTeamTextures({
  directory = DEFAULT_DIRECTORY,
  basenames,
  shard,
  previewDirectory,
} = {}) {
  const files = await readdir(directory);
  let names =
    basenames ??
    files
      .filter((file) => file.endsWith('-blue.ktx2'))
      .map((file) => file.slice(0, -10))
      .sort();
  if (shard) {
    const match = /^(\d+)\/(\d+)$/.exec(shard);
    if (!match || Number(match[2]) < 1 || Number(match[1]) >= Number(match[2]))
      throw new Error('--shard must be INDEX/COUNT (e.g. 0/3)');
    names = names.filter((_, i) => i % Number(match[2]) === Number(match[1]));
  }
  if (names.length === 0) throw new Error('No blue team KTX2 textures found');
  if (previewDirectory) await mkdir(previewDirectory, { recursive: true });
  let bytesWritten = 0;
  for (const name of names) {
    if (!/^[a-z0-9-]+$/.test(name)) throw new Error(`Invalid skin basename: ${name}`);
    const [blue, red] = await Promise.all(
      ['blue', 'red'].map(async (team) =>
        decodeKTX2(await readFile(join(directory, `${name}-${team}.ktx2`))),
      ),
    );
    if (blue.width !== red.width || blue.height !== red.height)
      throw new Error(`${name}: source dimensions differ`);
    for (const team of ['teal', 'orange']) {
      const { data, changedPixels } = tintTeamPixels(blue.data, red.data, team);
      if (!changedPixels) throw new Error(`${name}: no authored team paint found`);
      const ktx2 = await encodeToKTX2(data, {
        imageDecoder: async () => ({ data, width: blue.width, height: blue.height }),
        isUASTC: false,
        isKTX2File: true,
        generateMipmap: true,
        isPerceptual: true,
        isSetKTX2SRGBTransferFunc: true,
        // Existing KTX2 texels already have the UV flip baked in.
        isYFlip: false,
        qualityLevel: 210,
        compressionLevel: 4,
      });
      const output = join(directory, `${name}-${team}.ktx2`);
      const temporary = `${output}.${process.pid}.tmp`;
      try {
        await writeFile(temporary, ktx2);
        await rename(temporary, output);
      } finally {
        await rm(temporary, { force: true });
      }
      bytesWritten += ktx2.byteLength;
      if (previewDirectory) {
        await sharp(data, { raw: { width: blue.width, height: blue.height, channels: 4 } })
          .flip()
          .png()
          .toFile(join(previewDirectory, `${name}-${team}.png`));
      }
      console.log(
        `${name}-${team}.ktx2: ${(ktx2.byteLength / 1024).toFixed(0)} KB, ${((100 * changedPixels) / (blue.width * blue.height)).toFixed(1)}% team paint`,
      );
    }
  }
  console.log(
    `Generated ${names.length * 2} skins (${(bytesWritten / 1024 / 1024).toFixed(2)} MB).`,
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const options = {};
  for (let i = 0; i < args.length; i += 2) {
    const value = args[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${args[i]}`);
    if (args[i] === '--unit') options.basenames = [value];
    else if (args[i] === '--shard') options.shard = value;
    else if (args[i] === '--preview') options.previewDirectory = resolve(value);
    else throw new Error(`Unknown argument: ${args[i]}`);
  }
  await generateTeamTextures(options);
}
