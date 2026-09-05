import { describe, expect, it } from 'vitest';
import { EXPECTED_FACTIONS, type ExpectedFaction } from './athena2Factions.js';

interface Athena2ModelSpec {
  unit: string;
  animationAsset: string;
  source: string;
  slug: string;
  legacySlugs: string[];
  publish: boolean;
  faction: ExpectedFaction | null;
  unitySampledSkeleton: { controller: string } | null;
  joinInBlender: boolean;
  geometryParts: unknown[];
  geometry: 'run' | 'attack' | 'die';
  geometryFile: string | null;
  files: {
    attack: string;
    run: string;
    die: string;
  };
  rotateX: number;
  rotateY: number;
}

const manifest = (await import(new URL('../scripts/athena2-models.mjs', import.meta.url).href)) as {
  ATHENA2_FACTIONS: Record<ExpectedFaction, readonly string[]>;
  ATHENA2_MODELS: Athena2ModelSpec[];
};

describe('Athena2 import orientation corrections', () => {
  it('keeps the full source inventory while replacing the original Skeleton', () => {
    expect(manifest.ATHENA2_MODELS).toHaveLength(64);
    const unpublished = manifest.ATHENA2_MODELS.filter((model) => !model.publish);
    expect(unpublished.map((model) => model.unit).sort()).toEqual(
      ['Mercenary', 'SkeletonOriginal'].sort(),
    );
    expect(unpublished.find((model) => model.unit === 'SkeletonOriginal')).toMatchObject({
      animationAsset: 'Skeleton',
      source: 'Skeleton',
      slug: 'skeleton-original',
    });

    const replacement = manifest.ATHENA2_MODELS.find((model) => model.unit === 'Skeleton');
    expect(replacement).toMatchObject({
      animationAsset: 'ZombieRespawned',
      publish: true,
      source: 'ZombieSkeleton',
      slug: 'skeleton',
      legacySlugs: ['zombie-skeleton'],
    });
    expect(manifest.ATHENA2_MODELS.some((model) => model.unit === 'ZombieSkeleton')).toBe(false);
  });

  it('assigns exactly the supplied 54 public units to factions', () => {
    expect(manifest.ATHENA2_FACTIONS).toEqual(EXPECTED_FACTIONS);
    const assignedUnits = Object.values(manifest.ATHENA2_FACTIONS).flat();
    expect(assignedUnits).toHaveLength(54);
    expect(new Set(assignedUnits).size).toBe(54);

    const assignedModels = manifest.ATHENA2_MODELS.filter((model) => model.faction !== null);
    expect(assignedModels).toHaveLength(54);
    expect(assignedModels.map((model) => model.unit).sort()).toEqual([...assignedUnits].sort());
    for (const model of assignedModels) {
      expect(manifest.ATHENA2_FACTIONS[model.faction!], `${model.unit} manifest faction`).toContain(
        model.unit,
      );
    }
    expect(
      manifest.ATHENA2_MODELS.filter((model) => model.faction === null)
        .map((model) => model.unit)
        .sort(),
    ).toEqual(
      [
        'Diana',
        'King',
        'Mercenary',
        'Molten',
        'Overseer',
        'Prince',
        'RockBuddy',
        'SentryFixed',
        'SkeletonOriginal',
        'Thorne',
      ].sort(),
    );
  });

  it('limits direct Unity skeleton sampling to the parity-proven allowlist', () => {
    const sampled = manifest.ATHENA2_MODELS.filter((model) => model.unitySampledSkeleton);
    expect(
      Object.fromEntries(
        sampled.map((model) => [model.unit, model.unitySampledSkeleton?.controller]),
      ),
    ).toEqual({
      Arclight: 'TeslaCoil.controller',
      Beamdrone: 'BeamShip.controller',
      BigSlime: 'BigSlime.controller',
      DarkGolem: 'DarkGolem.controller',
      FallenMage: 'SkeletonMage.controller',
      FireDragon: 'FireDragon.controller',
      Firespout: 'Flamethrower.controller',
      Fixomatic: 'HealingMachine.controller',
      FrostLich: 'FrostLich.controller',
      GiantSlime: 'KingSlime.controller',
      Parasite: 'Parasite.controller',
      Piercebot: 'Ballista.controller',
      Plasmodrone: 'FlyingMachine.controller',
      Slime: 'Slime.controller',
      Soulfire: 'Inferno.controller',
      Spider: 'Spider.controller',
      Succubus: 'Vampire.controller',
      Wolf: 'ShadowWolf.controller',
    });
    expect(sampled.every((model) => !model.joinInBlender && model.geometryParts.length === 0)).toBe(
      true,
    );
    expect(sampled.find((model) => model.unit === 'FireDragon')?.geometry).toBe('attack');
  });

  it('keeps audited attack-geometry overrides explicit', () => {
    const attackGeometry = manifest.ATHENA2_MODELS.filter((model) => model.geometry === 'attack');

    expect(attackGeometry.map((model) => model.unit).sort()).toEqual(
      [
        'BigSlime',
        'FireDragon',
        'FlyingParasite',
        'GiantSlime',
        'Molten',
        'Slime',
        'Treant',
      ].sort(),
    );
  });

  it("keeps Footman's recorder geometry separate from its canonical clips", () => {
    const footman = manifest.ATHENA2_MODELS.find((model) => model.unit === 'Footman');

    expect(footman?.geometry).toBe('run');
    expect(footman?.geometryFile).toBe('kevinNosheild@attack.fbx');
    expect(footman?.files).toEqual({
      attack: 'kevin@attack.fbx',
      run: 'kevin@walk.fbx',
      die: 'kevin@die.fbx',
    });
  });

  it('keeps the audited up-axis and facing correction sets explicit', () => {
    const rotateX = manifest.ATHENA2_MODELS.filter((model) => model.rotateX !== 0);
    const rotateY = manifest.ATHENA2_MODELS.filter((model) => model.rotateY !== 0);

    expect(rotateX.map((model) => model.unit).sort()).toEqual(
      ['Blunderbuss', 'Knight', 'Sniper', 'Valkyrie'].sort(),
    );
    expect(rotateX.every((model) => model.rotateX === -Math.PI / 2)).toBe(true);

    expect(rotateY.map((model) => model.unit).sort()).toEqual(
      [
        'Banshee',
        'BoneArcher',
        'FallenMage',
        'FrostLich',
        'Reaper',
        'Soulfire',
        'Sphinx',
        'Succubus',
        'Skeleton',
        'Zombie',
      ].sort(),
    );
    expect(rotateY.every((model) => model.rotateY === Math.PI)).toBe(true);
  });
});
