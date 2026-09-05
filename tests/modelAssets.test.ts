/**
 * Authored model files are loaded by URL and fail softly at runtime, so neither
 * TypeScript nor Vite notices a missing skin, malformed GLB, or misspelled clip.
 * This test makes the asset contract explicit instead of accepting the
 * procedural fallback as proof that an import worked.
 */

import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { EXPECTED_FACTIONS, type ExpectedFaction } from './athena2Factions.js';

const MODEL_ROOT = fileURLToPath(new URL('../public/models/', import.meta.url));
const EXPECTED_UNITS = [
  'Alligator',
  'Archer',
  'Arclight',
  'Banshee',
  'Beamdrone',
  'Blunderbuss',
  'BoneArcher',
  'Boomerang',
  'Boomwalker',
  'Burstbot',
  'Cleric',
  'DarkGolem',
  'DarkMage',
  'DeathKnight',
  'DragonPup',
  'FallenMage',
  'FireDragon',
  'FireMage',
  'Firespout',
  'Fixomatic',
  'Footman',
  'FrostLich',
  'GriffinRider',
  'IceDragon',
  'IceGolem',
  'IceMage',
  'Infector',
  'Knight',
  'Mushroom',
  'Necromancer',
  'Parasite',
  'FlyingParasite',
  'Piercebot',
  'Plasmodrone',
  'Reaper',
  'Rider',
  'Sentry',
  'Wolf',
  'SkeletalDragon',
  'Slicebot',
  'Slime',
  'BigSlime',
  'GiantSlime',
  'Sniper',
  'Soulfire',
  'Sphinx',
  'Spider',
  'Squirrel',
  'Succubus',
  'Toad',
  'Treant',
  'Valkyrie',
  'Zombie',
  'Skeleton',
];
const REQUIRED_ATTRIBUTES = ['JOINTS_0', 'NORMAL', 'POSITION', 'TEXCOORD_0', 'WEIGHTS_0'];
const KTX2_IDENTIFIER = Buffer.from([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

interface CatalogModel {
  unit: string;
  faction: ExpectedFaction;
  file: string;
  skins: [string, string];
  runSize: [number, number, number];
  runGroundY?: number;
  clips: Record<'run' | 'attack' | 'die', CatalogClip>;
}

interface CatalogClip {
  static: boolean;
  duration: number;
  frames: number;
  frameRate: number;
}

interface Catalog {
  version: number;
  models: CatalogModel[];
}

interface GlbJson {
  asset?: { version?: string };
  accessors?: {
    componentType?: number;
    count?: number;
    min?: number[];
    max?: number[];
  }[];
  meshes?: {
    weights?: number[];
    primitives?: {
      attributes?: Record<string, number>;
      indices?: number;
      targets?: Record<string, number>[];
    }[];
  }[];
  nodes?: {
    name?: string;
    mesh?: number;
    extras?: {
      animationFormat?: string;
      boundsVertexCount?: number;
      clipFrames?: Record<string, number>;
      sourceVertexCount?: number;
    };
  }[];
  skins?: { joints?: number[] }[];
  animations?: {
    name?: string;
    channels?: {
      target?: { node?: number; path?: string };
    }[];
    samplers?: { input?: number; output?: number }[];
  }[];
}

interface Ktx2Info {
  width: number;
  height: number;
  levels: number;
  supercompression: number;
}

const catalog = await readCatalog();
const EXPLICITLY_GROUNDED_UNITS = new Set([
  'FireDragon',
  'GriffinRider',
  'IceDragon',
  'SkeletalDragon',
  'Sphinx',
  'Treant',
]);
const UNITY_SAMPLED_SIZE_LIMITS = {
  Arclight: 250_000,
  Beamdrone: 100_000,
  BigSlime: 110_000,
  DarkGolem: 245_000,
  FallenMage: 395_000,
  FireDragon: 650_000,
  Firespout: 155_000,
  Fixomatic: 230_000,
  FrostLich: 300_000,
  GiantSlime: 130_000,
  Parasite: 190_000,
  Piercebot: 160_000,
  Plasmodrone: 130_000,
  Slime: 75_000,
  Soulfire: 355_000,
  Spider: 340_000,
  Succubus: 445_000,
  Wolf: 250_000,
} as const;

describe('Athena2 authored model catalog', () => {
  it('contains unique complete entries', () => {
    expect(catalog.version).toBe(2);
    expect(catalog.models.map((model) => model.unit).sort()).toEqual([...EXPECTED_UNITS].sort());
    expect(new Set(catalog.models.map((model) => model.unit)).size).toBe(catalog.models.length);
    expect(new Set(catalog.models.map((model) => model.file)).size).toBe(catalog.models.length);
    for (const model of catalog.models) {
      const expectedKeys = ['clips', 'faction', 'file', 'runSize', 'skins', 'unit'];
      if (EXPLICITLY_GROUNDED_UNITS.has(model.unit)) {
        expectedKeys.push('runGroundY');
        expect(Number.isFinite(model.runGroundY), model.unit).toBe(true);
      } else {
        expect(model.runGroundY, model.unit).toBeUndefined();
      }
      expect(Object.keys(model).sort(), model.unit).toEqual(expectedKeys.sort());
      expect(model.skins).toHaveLength(2);
      expect(model.skins[0], `${model.unit} blue skin`).toMatch(/-blue\.ktx2$/);
      expect(model.skins[1], `${model.unit} red skin`).toMatch(/-red\.ktx2$/);
      expect(model.runSize, `${model.unit} authored run size`).toHaveLength(3);
      for (const size of model.runSize) {
        expect(Number.isFinite(size), `${model.unit} run size`).toBe(true);
        expect(size, `${model.unit} run size`).toBeGreaterThan(0);
      }
      for (const name of ['run', 'attack', 'die'] as const) {
        const clip = model.clips[name];
        expect(clip.duration, `${model.unit} ${name} duration`).toBeGreaterThan(0);
        expect(clip.static, `${model.unit} ${name} must be animated`).toBe(false);
        expect(clip.frames, `${model.unit} ${name} baked frames`).toBeGreaterThan(1);
        expect(clip.frameRate, `${model.unit} ${name} frame rate`).toBeGreaterThan(0);
        expect(clip.duration).toBeCloseTo(clip.frames / clip.frameRate, 8);
      }
    }
  });

  it('contains the exact supplied faction lists with no duplicate assignment', () => {
    const grouped = Object.fromEntries(
      (Object.keys(EXPECTED_FACTIONS) as ExpectedFaction[]).map((faction) => [
        faction,
        catalog.models.filter((model) => model.faction === faction).map((model) => model.unit),
      ]),
    );
    expect(grouped).toEqual(EXPECTED_FACTIONS);

    const assigned = Object.values(grouped).flat();
    expect(assigned).toHaveLength(54);
    expect(new Set(assigned).size).toBe(54);
    expect([...assigned].sort()).toEqual([...EXPECTED_UNITS].sort());
  });

  it('preserves authored relative sizes for the shared-scale gallery', () => {
    const extent = (unit: string): number => {
      const model = catalog.models.find((candidate) => candidate.unit === unit);
      expect(model, unit).toBeDefined();
      return Math.max(...model!.runSize);
    };

    expect(extent('Parasite')).toBeLessThan(extent('Slicebot'));
    expect(extent('Slicebot')).toBeLessThan(extent('DeathKnight'));
    expect(extent('DeathKnight')).toBeLessThan(extent('FireDragon'));
  });

  it('ships no model or skin outside the complete catalog', async () => {
    const files = await readdir(MODEL_ROOT);
    const actual = files.filter((file) => file.endsWith('.glb') || file.endsWith('.ktx2')).sort();
    const expected = catalog.models.flatMap((model) => [model.file, ...model.skins]).sort();
    expect(actual).toEqual(expected);
    expect(files.filter((file) => file.endsWith('.json'))).toEqual(['all-units.json']);
  });

  it('uses Athena2 Variant 0 baked timing instead of combining variants', () => {
    const clip = (unit: string, name: 'run' | 'attack' | 'die') => {
      const model = catalog.models.find((candidate) => candidate.unit === unit);
      expect(model, unit).toBeDefined();
      return model!.clips[name];
    };

    expect(clip('Sniper', 'attack')).toEqual({
      static: false,
      duration: 60 / 30,
      frames: 60,
      frameRate: 30,
    });
    expect(clip('Necromancer', 'attack').duration).toBe(29 / 30);
    expect(clip('Skeleton', 'run')).toEqual({
      static: false,
      duration: 24 / 10,
      frames: 24,
      frameRate: 10,
    });
  });

  it("keeps Treant's attack prop while binding clips to its skin joints", async () => {
    const model = catalog.models.find((candidate) => candidate.unit === 'Treant');
    expect(model).toBeDefined();
    const json = glbJson(await readFile(join(MODEL_ROOT, model!.file)));
    const positionAccessor = json.meshes?.[0]?.primitives?.[0]?.attributes?.POSITION;
    expect(
      positionAccessor === undefined ? undefined : json.accessors?.[positionAccessor]?.count,
      'Treant full body-and-Log vertex count',
    ).toBe(6240);
    const meshNode = json.nodes?.find((node) => node.mesh === 0);
    expect(meshNode?.extras?.boundsVertexCount, 'Treant body-only bounds prefix').toBe(6060);
    const joints = new Set(json.skins?.[0]?.joints ?? []);
    const jointNames = new Set([...joints].map((index) => json.nodes?.[index]?.name));
    expect(jointNames.has(undefined), 'Treant skin-joint names').toBe(false);
    const shadowNames = (json.nodes ?? [])
      .map((node, index) => ({ index, name: node.name }))
      .filter(({ index, name }) => !joints.has(index) && jointNames.has(name))
      .map(({ name }) => name);
    expect(shadowNames, 'Treant non-joint name shadows').toEqual([]);

    const run = json.animations?.find((animation) => animation.name === 'run');
    expect(run, 'Treant run clip').toBeDefined();
    const animatedJoints = new Set(
      (run?.channels ?? [])
        .map((channel) => channel.target?.node)
        .filter((node): node is number => node !== undefined && joints.has(node))
        .map((node) => json.nodes?.[node]?.name),
    );
    expect([...animatedJoints].sort(), 'Treant animated skin joints').toEqual(
      [...jointNames].sort(),
    );
  });

  it('ships FireDragon as a compact Unity-sampled skeletal GLB', async () => {
    const model = catalog.models.find((candidate) => candidate.unit === 'FireDragon');
    expect(model).toBeDefined();
    const glb = await readFile(join(MODEL_ROOT, model!.file));
    expect(glb.byteLength, 'FireDragon compact skeletal size').toBeLessThanOrEqual(650_000);
    const json = glbJson(glb);
    const mesh = json.meshes?.[0];
    const primitive = mesh?.primitives?.[0];
    const positionAccessor = primitive?.attributes?.POSITION;
    expect(
      positionAccessor === undefined ? undefined : json.accessors?.[positionAccessor]?.count,
      'FireDragon indexed source vertex count',
    ).toBe(2868);
    expect(
      json.accessors?.[primitive?.indices ?? -1]?.count,
      'FireDragon indexed triangle corners',
    ).toBe(8751);
    expect(
      json.accessors?.[primitive?.indices ?? -1]?.componentType,
      'FireDragon 16-bit topology',
    ).toBe(5123);
    expect(mesh?.weights, 'FireDragon has no morph weights').toBeUndefined();
    expect(primitive?.targets, 'FireDragon has no morph targets').toBeUndefined();
    expect(json.skins?.[0]?.joints, 'FireDragon sampled skin joints').toHaveLength(66);

    const meshNode = json.nodes?.find((node) => node.mesh === 0);
    expect(meshNode?.extras).toMatchObject({
      animationFormat: 'athena2-unity-sampled-skeleton-v1',
      sourceVertexCount: 2868,
      clipFrames: { run: 40, attack: 60, die: 29 },
    });
    for (const [name, keys, channels] of [
      ['run', 41, 84],
      ['attack', 61, 91],
      ['die', 30, 89],
    ] as const) {
      const animation = json.animations?.find((candidate) => candidate.name === name);
      expect(animation?.channels, `FireDragon ${name} skeletal channels`).toHaveLength(channels);
      expect(new Set(animation?.channels?.map((channel) => channel.target?.path))).toEqual(
        new Set(['rotation', 'translation']),
      );
      const inputCounts = (animation?.samplers ?? []).map(
        (sampler) => json.accessors?.[sampler.input ?? -1]?.count,
      );
      expect(Math.max(...inputCounts.map((count) => count ?? 0))).toBe(keys);
    }
  });

  it.each(Object.entries(UNITY_SAMPLED_SIZE_LIMITS))(
    '%s stays a standard compact Unity-sampled skeletal GLB',
    async (unit, maximumBytes) => {
      const model = catalog.models.find((candidate) => candidate.unit === unit);
      expect(model, unit).toBeDefined();
      const glb = await readFile(join(MODEL_ROOT, model!.file));
      expect(glb.byteLength, `${unit} compact skeletal size`).toBeLessThanOrEqual(maximumBytes);
      const json = glbJson(glb);
      const meshNode = json.nodes?.find(
        (node) =>
          node.mesh !== undefined &&
          node.extras?.animationFormat === 'athena2-unity-sampled-skeleton-v1',
      );
      expect(meshNode, `${unit} sampled-format mesh node`).toBeDefined();
      const mesh = json.meshes?.[meshNode!.mesh!];
      const primitive = mesh?.primitives?.[0];
      const position = json.accessors?.[primitive?.attributes?.POSITION ?? -1];
      const indices = json.accessors?.[primitive?.indices ?? -1];
      expect(position?.count, `${unit} indexed source vertices`).toBe(
        meshNode?.extras?.sourceVertexCount,
      );
      expect(indices?.count, `${unit} triangle corners`).toBeGreaterThan(0);
      expect(indices?.componentType, `${unit} 16-bit topology`).toBe(5123);
      expect(mesh?.weights, `${unit} has no morph weights`).toBeUndefined();
      expect(primitive?.targets, `${unit} has no morph targets`).toBeUndefined();
      expect(meshNode?.extras?.clipFrames).toEqual({
        run: model!.clips.run.frames,
        attack: model!.clips.attack.frames,
        die: model!.clips.die.frames,
      });
      expect(json.animations?.map((animation) => animation.name).sort(), `${unit} clips`).toEqual([
        'attack',
        'die',
        'run',
      ]);
    },
  );

  it.each(catalog.models)(
    '$unit has one compatible skinned GLB with all three clips',
    async (model) => {
      const glb = await readFile(join(MODEL_ROOT, model.file));
      const json = glbJson(glb);
      expect(json.asset?.version).toBe('2.0');
      expect(json.meshes, `${model.unit} mesh count`).toHaveLength(1);
      expect(json.meshes?.[0]?.primitives, `${model.unit} primitive count`).toHaveLength(1);
      expect(json.skins, `${model.unit} skin count`).toHaveLength(1);
      expect(json.skins?.[0]?.joints?.length, `${model.unit} bone count`).toBeGreaterThan(0);

      const attributes = Object.keys(json.meshes?.[0]?.primitives?.[0]?.attributes ?? {}).sort();
      expect(attributes, `${model.unit} vertex attributes`).toEqual(
        expect.arrayContaining(REQUIRED_ATTRIBUTES),
      );

      const animations = json.animations ?? [];
      expect(
        animations.map((animation) => animation.name).sort(),
        `${model.unit} clip names`,
      ).toEqual(['attack', 'die', 'run']);
      for (const animation of animations) {
        expect(
          animation.channels?.length,
          `${model.unit} ${animation.name} channels`,
        ).toBeGreaterThan(0);
        expect(
          animation.samplers?.length,
          `${model.unit} ${animation.name} samplers`,
        ).toBeGreaterThan(0);
        const channelTargets = (animation.channels ?? []).map(
          (channel) => `${channel.target?.node}|${channel.target?.path}`,
        );
        expect(
          new Set(channelTargets).size,
          `${model.unit} ${animation.name} duplicate animation targets`,
        ).toBe(channelTargets.length);
        const name = animation.name as 'run' | 'attack' | 'die';
        const duration = Math.max(
          ...(animation.samplers ?? []).map((sampler) =>
            sampler.input === undefined
              ? Number.NaN
              : (json.accessors?.[sampler.input]?.max?.[0] ?? Number.NaN),
          ),
        );
        expect(duration, `${model.unit} ${animation.name} exported duration`).toBeCloseTo(
          model.clips[name].duration,
          5,
        );
      }
    },
  );

  it.each(catalog.models)(
    '$unit has matching blue and red ETC1S skins with mipmaps',
    async (model) => {
      const [blue, red] = await Promise.all(
        model.skins.map(async (skin) => ktx2Info(await readFile(join(MODEL_ROOT, skin)))),
      );
      expect(red, `${model.unit} team skin dimensions`).toEqual(blue);
      expect(blue.width, `${model.unit} skin width`).toBeGreaterThan(0);
      expect(blue.height, `${model.unit} skin height`).toBeGreaterThan(0);
      expect(blue.levels, `${model.unit} complete mip chain`).toBe(
        Math.floor(Math.log2(Math.max(blue.width, blue.height))) + 1,
      );
      // Basis Universal ETC1S data uses KTX_SS_BASIS_LZ (1).
      expect(blue.supercompression, `${model.unit} supercompression`).toBe(1);
    },
  );
});

async function readCatalog(): Promise<Catalog> {
  return JSON.parse(await readFile(join(MODEL_ROOT, 'all-units.json'), 'utf8')) as Catalog;
}

function glbJson(buffer: Buffer): GlbJson {
  expect(buffer.readUInt32LE(0), 'GLB magic').toBe(0x46546c67);
  expect(buffer.readUInt32LE(4), 'GLB version').toBe(2);
  expect(buffer.readUInt32LE(8), 'GLB declared length').toBe(buffer.byteLength);

  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const length = buffer.readUInt32LE(offset);
    const type = buffer.readUInt32LE(offset + 4);
    if (type === 0x4e4f534a) {
      return JSON.parse(
        buffer
          .subarray(offset + 8, offset + 8 + length)
          .toString('utf8')
          .trim(),
      ) as GlbJson;
    }
    offset += 8 + length;
  }
  throw new Error('GLB has no JSON chunk');
}

function ktx2Info(buffer: Buffer): Ktx2Info {
  expect(buffer.subarray(0, KTX2_IDENTIFIER.length), 'KTX2 identifier').toEqual(KTX2_IDENTIFIER);
  expect(buffer.readUInt32LE(12), 'KTX2 vkFormat for Basis data').toBe(0);
  expect(buffer.readUInt32LE(16), 'KTX2 type size').toBe(1);
  expect(buffer.readUInt32LE(28), 'KTX2 pixel depth').toBe(0);
  expect(buffer.readUInt32LE(32), 'KTX2 layer count').toBe(0);
  expect(buffer.readUInt32LE(36), 'KTX2 face count').toBe(1);
  return {
    width: buffer.readUInt32LE(20),
    height: buffer.readUInt32LE(24),
    levels: buffer.readUInt32LE(40),
    supercompression: buffer.readUInt32LE(44),
  };
}
