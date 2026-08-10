import { describe, expect, it } from "vitest";

interface Athena2ModelSpec {
  unit: string;
  publish: boolean;
  unitySampledSkeleton: { controller: string } | null;
  joinInBlender: boolean;
  geometryParts: unknown[];
  geometry: "run" | "attack" | "die";
  geometryFile: string | null;
  files: {
    attack: string;
    run: string;
    die: string;
  };
  rotateX: number;
  rotateY: number;
}

const manifest = (await import(
  new URL("../scripts/athena2-models.mjs", import.meta.url).href
)) as { ATHENA2_MODELS: Athena2ModelSpec[] };

describe("Athena2 import orientation corrections", () => {
  it("keeps the full source inventory while excluding redundant Skeleton", () => {
    expect(manifest.ATHENA2_MODELS).toHaveLength(64);
    const unpublished = manifest.ATHENA2_MODELS.filter(
      (model) => !model.publish,
    );
    expect(unpublished.map((model) => model.unit)).toEqual(["Skeleton"]);
    expect(
      manifest.ATHENA2_MODELS.find((model) => model.unit === "ZombieSkeleton")
        ?.publish,
    ).toBe(true);
  });

  it("limits direct Unity skeleton sampling to the parity-proven allowlist", () => {
    const sampled = manifest.ATHENA2_MODELS.filter(
      (model) => model.unitySampledSkeleton,
    );
    expect(
      Object.fromEntries(
        sampled.map((model) => [
          model.unit,
          model.unitySampledSkeleton?.controller,
        ]),
      ),
    ).toEqual({
      Arclight: "TeslaCoil.controller",
      Beamdrone: "BeamShip.controller",
      BigSlime: "BigSlime.controller",
      DarkGolem: "DarkGolem.controller",
      FallenMage: "SkeletonMage.controller",
      FireDragon: "FireDragon.controller",
      Firespout: "Flamethrower.controller",
      Fixomatic: "HealingMachine.controller",
      FrostLich: "FrostLich.controller",
      GiantSlime: "KingSlime.controller",
      Parasite: "Parasite.controller",
      Piercebot: "Ballista.controller",
      Plasmodrone: "FlyingMachine.controller",
      Slime: "Slime.controller",
      Soulfire: "Inferno.controller",
      Spider: "Spider.controller",
      Succubus: "Vampire.controller",
      Wolf: "ShadowWolf.controller",
    });
    expect(
      sampled.every(
        (model) => !model.joinInBlender && model.geometryParts.length === 0,
      ),
    ).toBe(true);
    expect(sampled.find((model) => model.unit === "FireDragon")?.geometry).toBe(
      "attack",
    );
  });

  it("keeps audited attack-geometry overrides explicit", () => {
    const attackGeometry = manifest.ATHENA2_MODELS.filter(
      (model) => model.geometry === "attack",
    );

    expect(attackGeometry.map((model) => model.unit).sort()).toEqual(
      [
        "BigSlime",
        "FireDragon",
        "FlyingParasite",
        "GiantSlime",
        "Molten",
        "Slime",
        "Treant",
      ].sort(),
    );
  });

  it("keeps Footman's recorder geometry separate from its canonical clips", () => {
    const footman = manifest.ATHENA2_MODELS.find(
      (model) => model.unit === "Footman",
    );

    expect(footman?.geometry).toBe("run");
    expect(footman?.geometryFile).toBe("kevinNosheild@attack.fbx");
    expect(footman?.files).toEqual({
      attack: "kevin@attack.fbx",
      run: "kevin@walk.fbx",
      die: "kevin@die.fbx",
    });
  });

  it("keeps the audited up-axis and facing correction sets explicit", () => {
    const rotateX = manifest.ATHENA2_MODELS.filter(
      (model) => model.rotateX !== 0,
    );
    const rotateY = manifest.ATHENA2_MODELS.filter(
      (model) => model.rotateY !== 0,
    );

    expect(rotateX.map((model) => model.unit).sort()).toEqual(
      ["Blunderbuss", "Knight", "Sniper", "Valkyrie"].sort(),
    );
    expect(rotateX.every((model) => model.rotateX === -Math.PI / 2)).toBe(true);

    expect(rotateY.map((model) => model.unit).sort()).toEqual(
      [
        "Banshee",
        "BoneArcher",
        "FallenMage",
        "FrostLich",
        "Reaper",
        "Soulfire",
        "Sphinx",
        "Succubus",
        "Zombie",
        "ZombieSkeleton",
      ].sort(),
    );
    expect(rotateY.every((model) => model.rotateY === Math.PI)).toBe(true);
  });
});
