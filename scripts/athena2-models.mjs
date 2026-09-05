/**
 * Athena2's authored unit source inventory.
 *
 * The public unit name, Animation asset, and FBX folder name are not always the
 * same. These aliases come from `00_CharacterAnimation.unity`: the recorder's
 * OutputData points at `Animations/<animationAsset>.asset`, while its Animator
 * points at the FBX under `Imports/<source>/`.
 *
 * `Assets/Art/Units/Animations` is the authority for which source units exist.
 * The importer excludes entries whose required run, attack, or death slot has
 * only one baked frame. A complete source unit can also remain in this inventory
 * with `publish: false` when it is intentionally omitted from the game's public
 * gallery. Complete source rigs may still have several skinned parts or old FBXs
 * that need normalization before Three can read them.
 */
export const ATHENA2_FACTIONS = Object.freeze({
  Human: Object.freeze([
    'Footman',
    'Archer',
    'Valkyrie',
    'DarkMage',
    'IceMage',
    'Blunderbuss',
    'Boomerang',
    'Cleric',
    'Knight',
    'GriffinRider',
    'Rider',
    'FireMage',
    'Sniper',
  ]),
  Robot: Object.freeze([
    'Boomwalker',
    'Burstbot',
    'Slicebot',
    'Firespout',
    'Sentry',
    'Beamdrone',
    'Arclight',
    'Fixomatic',
    'Plasmodrone',
    'Piercebot',
    'DarkGolem',
    'IceGolem',
  ]),
  Monster: Object.freeze([
    'Alligator',
    'Squirrel',
    'Wolf',
    'DragonPup',
    'Infector',
    'Parasite',
    'FlyingParasite',
    'Mushroom',
    'Toad',
    'Spider',
    'GiantSlime',
    'BigSlime',
    'Slime',
    'Treant',
    'FireDragon',
    'IceDragon',
  ]),
  Undead: Object.freeze([
    'Zombie',
    'BoneArcher',
    'Succubus',
    'Necromancer',
    'Skeleton',
    'Soulfire',
    'Reaper',
    'FallenMage',
    'DeathKnight',
    'Banshee',
    'Sphinx',
    'SkeletalDragon',
    'FrostLich',
  ]),
});

const FACTION_ASSIGNMENTS = Object.values(ATHENA2_FACTIONS).flat();
if (new Set(FACTION_ASSIGNMENTS).size !== FACTION_ASSIGNMENTS.length) {
  throw new Error('Athena2 faction assignments contain a duplicate unit');
}
const FACTION_BY_UNIT = new Map(
  Object.entries(ATHENA2_FACTIONS).flatMap(([faction, units]) =>
    units.map((unit) => [unit, faction]),
  ),
);

export const ATHENA2_MODELS = [
  model('Alligator', 'Lizard', 'lizard', 'Lizard@attack.FBX', 'Lizard@run.FBX', 'Lizard@die.FBX'),
  model('Archer', 'Archer', 'archer', 'Archer@attack.fbx', 'archer@run.FBX', 'archer@die.FBX', {
    joinInBlender: false,
  }),
  model(
    'Arclight',
    'TeslaCoil',
    'tesla-coil',
    'TeslaCoil@attack.FBX',
    'TeslaCoil@run.FBX',
    'TeslaCoil@die.FBX',
    { unitySampledSkeleton: { controller: 'TeslaCoil.controller' } },
  ),
  model(
    'Banshee',
    'Banshee',
    'banshee',
    'Banshee@attack.FBX',
    'Banshee@run.FBX',
    'Banshee@die.FBX',
    { rotateY: Math.PI },
  ),
  model(
    'Beamdrone',
    'BeamShip',
    'beam-ship',
    'BeamShip@attack.FBX',
    'BeamShip@run.FBX',
    'BeamShip@die.FBX',
    {
      existing: true,
      unitySampledSkeleton: { controller: 'BeamShip.controller' },
    },
  ),
  model(
    'Blunderbuss',
    'Shotgun',
    'blunderbuss',
    'Shotgun@attack.FBX',
    'Shotgun@run.FBX',
    'Shotgun@die.FBX',
    { normalize: 'unity', rotateX: -Math.PI / 2 },
  ),
  model(
    'BoneArcher',
    'SkeletonArcher',
    'bone-archer',
    'SkeletonArcher@attack.FBX',
    'SkeletonArcher@run.FBX',
    'SkeletonArcher@die.FBX',
    { joinInBlender: false, rotateY: Math.PI },
  ),
  model(
    'Boomerang',
    'Boomerang',
    'boomerang',
    'boomerang@attack.fbx',
    'boomerang@run.fbx',
    'Boomerang@die.FBX',
    { joinInBlender: false },
  ),
  model('Boomwalker', 'Bomb', 'bomb', 'Bomb@attack.FBX', 'Bomb@run.FBX', 'Bomb@die.FBX'),
  model(
    'Burstbot',
    'Revolver',
    'revolver',
    'Revolver@attack.FBX',
    'Revolver@run.FBX',
    'Revolver@die.FBX',
    true,
  ),
  model('Cleric', 'Priest', 'priest', 'Priest@attack.FBX', 'Priest@run.FBX', 'Priest@die.FBX'),
  model(
    'DarkGolem',
    'DarkGolem',
    'dark-golem',
    'DarkGolem@attack.FBX',
    'DarkGolem@run.FBX',
    'DarkGolem@die.FBX',
    { unitySampledSkeleton: { controller: 'DarkGolem.controller' } },
  ),
  model(
    'DarkMage',
    'DarkMage',
    'dark-mage',
    'Dark Mage@attack.FBX',
    'Dark Mage@run.FBX',
    'Dark Mage@die.FBX',
  ),
  model(
    'DeathKnight',
    'SkeletonKnight',
    'death-knight',
    'SkeletonKnight@Attack.fbx',
    'SkeletonKnight@Walk.fbx',
    'SkeletonKnight@Die.fbx',
    { joinInBlender: false },
  ),
  model('Diana', 'fee', 'diana', 'fee@attack.fbx', 'fee@idle.fbx', 'fee@idle.fbx', {
    joinInBlender: false,
    skins: ['Diana/Diana_Albedo.png', 'Diana/Diana_Albedo.png'],
  }),
  model(
    'DragonPup',
    'BabyDragon',
    'baby-dragon',
    'BabyDragon@attack.FBX',
    'BabyDragon@run.FBX',
    'BabyDragon@die.FBX',
  ),
  model(
    'FallenMage',
    'SkeletonMage',
    'skeleton-mage',
    'SkeletonMage@attack.FBX',
    'SkeletonMage@run.FBX',
    'SkeletonMage@die.FBX',
    {
      rotateY: Math.PI,
      unitySampledSkeleton: { controller: 'SkeletonMage.controller' },
    },
  ),
  model(
    'FireDragon',
    'FireDragon',
    'fire-dragon',
    'FireDragon@attack.FBX',
    'FireDragon@run.FBX',
    'FireDragon@die.FBX',
    {
      geometry: 'attack',
      joinInBlender: false,
      runGround: { bones: ['foot', 'toe'], margin: 0.02 },
      unitySampledSkeleton: { controller: 'FireDragon.controller' },
    },
  ),
  model(
    'FireMage',
    'FireMage',
    'fire-mage',
    'FireMage@attack.FBX',
    'FireMage@run.FBX',
    'FireMage@die.FBX',
    { normalize: 'unity' },
  ),
  model(
    'Firespout',
    'Flamethrower',
    'flamethrower',
    'Flamethrower@attack.FBX',
    'Flamethrower@run.FBX',
    'Flamethrower@die.FBX',
    { unitySampledSkeleton: { controller: 'Flamethrower.controller' } },
  ),
  model(
    'Fixomatic',
    'HealingMachine',
    'healing-machine',
    'HealingMachine@attack.FBX',
    'HealingMachine@run.FBX',
    'HealingMachine@die.FBX',
    { unitySampledSkeleton: { controller: 'HealingMachine.controller' } },
  ),
  model('Footman', 'Kevin', 'kevin', 'kevin@attack.fbx', 'kevin@walk.fbx', 'kevin@die.fbx', {
    geometryFile: 'kevinNosheild@attack.fbx',
  }),
  model(
    'FrostLich',
    'FrostLich',
    'frost-lich',
    'FrostLich@attack.FBX',
    'FrostLich@run.FBX',
    'FrostLich@die.FBX',
    {
      rotateY: Math.PI,
      unitySampledSkeleton: { controller: 'FrostLich.controller' },
    },
  ),
  model(
    'GriffinRider',
    'GriffinRider',
    'griffin-rider',
    'GriffinRider@attack.FBX',
    'GriffinRider@run.FBX',
    'GriffinRider@die.FBX',
    {
      joinInBlender: false,
      runGround: { bones: ['foot', 'toe'], margin: 0.02 },
    },
  ),
  model(
    'IceDragon',
    'IceDragon',
    'ice-dragon',
    'IceDragon@attack.fbx',
    'IceDragon@run.fbx',
    'IceDragon@die.fbx',
    { runGround: { bones: ['foot', 'toe'], margin: 0.02 } },
  ),
  model(
    'IceGolem',
    'IceGolem',
    'ice-golem',
    'IceGolem@attack.FBX',
    'IceGolem@run.FBX',
    'IceGolem@die.FBX',
  ),
  model(
    'IceMage',
    'IceMage',
    'ice-mage',
    'IceMage@attack.FBX',
    'IceMage@run.FBX',
    'IceMage@die.FBX',
  ),
  model(
    'Infector',
    'Infestor',
    'infestor',
    'Infestor@attack.FBX',
    'Infestor@run.FBX',
    'Infestor@die.FBX',
  ),
  model('King', 'King', 'king', '_ing@Attack.fbx', '_ing@Idle.fbx', '_ing@Idle.fbx', {
    joinInBlender: false,
  }),
  model('Knight', 'Knight', 'knight', 'knight@attack.FBX', 'knight@run.FBX', 'knight@die.FBX', {
    normalize: 'unity',
    rotateX: -Math.PI / 2,
  }),
  model(
    'Mercenary',
    'DarkKnight',
    'mercenary',
    'dark_knight@attack.fbx',
    'dark_knight@walk.fbx',
    'dark_knight@die.fbx',
    { normalize: 'blender', publish: false },
  ),
  model(
    'Molten',
    'obsidian',
    'molten',
    'obsidian@attack.fbx',
    'obsidian@idle.fbx',
    'obsidian@idle.fbx',
    {
      normalize: 'blender',
      geometry: 'attack',
      skins: ['Molten/Molten_Albedo.png', 'Molten/Molten_Albedo.png'],
    },
  ),
  model(
    'Mushroom',
    'Mushroom',
    'mushroom',
    'Mushroom@attack.FBX',
    'Mushroom@run.FBX',
    'Mushroom@die.FBX',
  ),
  model(
    'Necromancer',
    'Necromancer',
    'necromancer',
    'Necromancer@Attack.fbx',
    'Necromancer@Walk.fbx',
    'Necromancer@Die.fbx',
    { joinInBlender: false },
  ),
  model(
    'Overseer',
    'Overseer',
    'overseer',
    'Overseer@attack.FBX',
    'Overseer@idle.FBX',
    'Overseer@idle.FBX',
    {
      skins: ['Overseer/Overseer_Albedo.tga', 'Overseer/Overseer_Albedo.tga'],
    },
  ),
  model(
    'Parasite',
    'Parasite',
    'parasite',
    'Parasite@attack.FBX',
    'Parasite@run.FBX',
    'Parasite@die.FBX',
    {
      skins: ['Infector_Blue.png', 'Infector_Red.png'],
      unitySampledSkeleton: { controller: 'Parasite.controller' },
    },
  ),
  model(
    'FlyingParasite',
    'ParasiteFly',
    'flying-parasite',
    'ParasiteFly@attack.FBX',
    'ParasiteFly@run.FBX',
    'ParasiteFly@die.FBX',
    {
      animationAsset: 'ParasiteFlying',
      geometry: 'attack',
      skins: ['Infector_Blue.png', 'Infector_Red.png'],
    },
  ),
  model(
    'Piercebot',
    'Ballista',
    'ballista',
    'Ballista@attack.FBX',
    'Ballista@run.FBX',
    'Ballista@die.FBX',
    { unitySampledSkeleton: { controller: 'Ballista.controller' } },
  ),
  model(
    'Plasmodrone',
    'FlyingMachine',
    'flying-machine',
    'FlyingMachine@attack.FBX',
    'FlyingMachine@run.FBX',
    'FlyingMachine@die.FBX',
    { unitySampledSkeleton: { controller: 'FlyingMachine.controller' } },
  ),
  model('Prince', 'King2', 'prince', 'King@Attack.fbx', 'King@Idle.fbx', 'King@Idle.fbx', {
    joinInBlender: false,
  }),
  model('Reaper', 'Reaper', 'reaper', 'Reaper@attack.FBX', 'Reaper@run.FBX', 'Reaper@die.FBX', {
    rotateY: Math.PI,
  }),
  model(
    'Rider',
    'GriffinRider_Rider',
    'rider',
    'Rider@attack1.FBX',
    'Rider@run.FBX',
    'Rider@die.FBX',
    {
      geometryParts: [{ name: 'Sword', source: 'attack', bindAnchor: 'Weap01' }],
      joinInBlender: false,
      skins: ['GriffinRider_Blue.png', 'GriffinRider_Red.png'],
    },
  ),
  model(
    'RockBuddy',
    'RockBuddy',
    'rock-buddy',
    'RockBuddy@attack.fbx',
    'RockBuddy@walk.fbx',
    'RockBuddy@idle.fbx',
    {
      skins: ['Molten/Molten_RockBuddy.png', 'Molten/Molten_RockBuddy.png'],
    },
  ),
  model('Sentry', 'Cannon', 'cannon', 'Cannon@attack.FBX', 'Cannon@run.FBX', 'Cannon@die01.FBX'),
  model(
    'SentryFixed',
    'Cannon',
    'sentry-fixed',
    'Cannon@attack.FBX',
    'Cannon@die02.FBX',
    'Cannon@die02.FBX',
    {
      skins: ['Sentry_Blue.png', 'Sentry_Red.png'],
    },
  ),
  model(
    'Wolf',
    'ShadowWolf',
    'wolf',
    'ShadowWolf@attack.FBX',
    'ShadowWolf@run.FBX',
    'ShadowWolf@die.FBX',
    {
      animationAsset: 'ShadowWolf',
      unitySampledSkeleton: { controller: 'ShadowWolf.controller' },
    },
  ),
  model(
    'SkeletalDragon',
    'SkeletonDragon',
    'skeletal-dragon',
    'SkeletonDragon@Attack.fbx',
    'SkeletonDragon@Walk.fbx',
    'SkeletonDragon@Die.fbx',
    {
      joinInBlender: false,
      runGround: { bones: ['ankle', 'toe'], margin: 0.02 },
    },
  ),
  model(
    'SkeletonOriginal',
    'Skeleton',
    'skeleton-original',
    'Skeleton@Attack.fbx',
    'Skeleton@Walk.fbx',
    'Skeleton@Die.fbx',
    {
      animationAsset: 'Skeleton',
      joinInBlender: false,
      publish: false,
    },
  ),
  model(
    'Slicebot',
    'SwordMachine',
    'sword-machine',
    'SwordMachine@attack.FBX',
    'SwordMachine@run.FBX',
    'SwordMachine@die.FBX',
    true,
  ),
  model('Slime', 'Slime', 'slime', 'slime@attack.fbx', 'Slime@run.FBX', 'Slime@die.FBX', {
    geometry: 'attack',
    skins: ['SlimeGiant_Blue.png', 'SlimeGiant_Red.png'],
    unitySampledSkeleton: { controller: 'Slime.controller' },
  }),
  model(
    'BigSlime',
    'BigSlime',
    'big-slime',
    'BigSlime@attack.FBX',
    'BigSlime@run.FBX',
    'BigSlime@die.FBX',
    {
      animationAsset: 'SlimeBig',
      geometry: 'attack',
      skins: ['SlimeGiant_Blue.png', 'SlimeGiant_Red.png'],
      unitySampledSkeleton: { controller: 'BigSlime.controller' },
    },
  ),
  model(
    'GiantSlime',
    'KingSlime',
    'giant-slime',
    'Kingslime@attack.FBX',
    'KingSlime@run.FBX',
    'KingSlime@die.FBX',
    {
      animationAsset: 'SlimeGiant',
      geometry: 'attack',
      unitySampledSkeleton: { controller: 'KingSlime.controller' },
    },
  ),
  model('Sniper', 'sniper', 'sniper', 'sniper@attack.fbx', 'Sniper@run.FBX', 'Sniper@die.FBX', {
    normalize: 'unity',
    rotateX: -Math.PI / 2,
  }),
  model(
    'Soulfire',
    'Inferno',
    'inferno',
    'Inferno@attack.FBX',
    'Inferno@run.FBX',
    'Inferno@die.FBX',
    {
      rotateY: Math.PI,
      unitySampledSkeleton: { controller: 'Inferno.controller' },
    },
  ),
  model('Sphinx', 'Sphinx', 'sphinx', 'Sphinx@attack.FBX', 'Sphinx@run.FBX', 'Sphinx@die.FBX', {
    rotateY: Math.PI,
    runGround: { bones: ['foot', 'toe'], margin: 0.02 },
  }),
  model('Spider', 'Spider', 'spider', 'Spider@attack.FBX', 'Spider@run.FBX', 'Spider@die.FBX', {
    unitySampledSkeleton: { controller: 'Spider.controller' },
  }),
  model(
    'Squirrel',
    'Squirrel',
    'squirrel',
    'Squirrel@attack.FBX',
    'Squirrel@run.FBX',
    'Squirrel@die.FBX',
    { joinInBlender: false },
  ),
  model(
    'Succubus',
    'Vampire',
    'vampire',
    'Vampire@attack.FBX',
    'Vampire@run.FBX',
    'Vampire@die.FBX',
    {
      rotateY: Math.PI,
      unitySampledSkeleton: { controller: 'Vampire.controller' },
    },
  ),
  model('Thorne', 'bolton', 'thorne', 'bolton@attack2.fbx', 'bolton@idle.fbx', 'bolton@idle.fbx', {
    joinInBlender: false,
    skins: ['Thorne/Thorne_Albedo.png', 'Thorne/Thorne_Albedo.png'],
  }),
  model('Toad', 'Frog', 'frog', 'Frog@attack.FBX', 'Frog@run.FBX', 'Frog@die.FBX'),
  model('Treant', 'Treant', 'treant', 'Treant@Attack.fbx', 'Treant@run.fbx', 'Treant@Die.fbx', {
    boundsVertexCount: 6060,
    geometry: 'attack',
    normalize: 'blender',
    preferSkinJointTracks: true,
    pristineClips: true,
    runGround: { bones: ['ankle'], margin: 0.02 },
  }),
  model('Valkyrie', 'Thor', 'valkyrie', 'Thor@attack.FBX', 'Thor@run.FBX', 'Thor@die.FBX', {
    normalize: 'unity',
    rotateX: -Math.PI / 2,
  }),
  model('Zombie', 'Zombie', 'zombie', 'Zombie@attack.FBX', 'Zombie@run.FBX', 'Zombie@die.FBX', {
    rotateY: Math.PI,
  }),
  model(
    'Skeleton',
    'ZombieSkeleton',
    'skeleton',
    'ZombieSkeleton@attack.FBX',
    'ZombieSkeleton@run.FBX',
    'ZombieSkeleton@die.FBX',
    {
      animationAsset: 'ZombieRespawned',
      legacySlugs: ['zombie-skeleton'],
      rotateY: Math.PI,
    },
  ),
];

function model(unit, source, slug, attack, run, die, options = {}) {
  if (typeof options === 'boolean') options = { existing: options };
  const animationAsset = options.animationAsset ?? unit;
  const publish = options.publish ?? true;
  const faction = FACTION_BY_UNIT.get(unit) ?? null;
  const legacySlugs = options.legacySlugs ?? [];
  if (
    !Array.isArray(legacySlugs) ||
    legacySlugs.some(
      (legacySlug) =>
        typeof legacySlug !== 'string' || legacySlug.length === 0 || legacySlug === slug,
    ) ||
    new Set(legacySlugs).size !== legacySlugs.length
  ) {
    throw new Error(`${unit}: invalid legacy slug metadata`);
  }
  const runGround = options.runGround ?? null;
  if (
    runGround !== null &&
    (!Array.isArray(runGround.bones) ||
      runGround.bones.length === 0 ||
      !runGround.bones.every((bone) => typeof bone === 'string' && bone.length > 0) ||
      !Number.isFinite(runGround.margin) ||
      runGround.margin < 0)
  ) {
    throw new Error(`${unit}: invalid runGround metadata`);
  }
  return Object.freeze({
    unit,
    animationAsset,
    source,
    slug,
    legacySlugs: Object.freeze([...legacySlugs]),
    skins: Object.freeze(
      options.skins ?? [`${animationAsset}_Blue.png`, `${animationAsset}_Red.png`],
    ),
    files: Object.freeze({ attack, run, die }),
    existing: options.existing ?? false,
    publish,
    faction,
    geometry: options.geometry ?? 'run',
    geometryFile: options.geometryFile ?? null,
    geometryParts: Object.freeze(
      (options.geometryParts ?? []).map((part) => Object.freeze({ ...part })),
    ),
    boundsVertexCount: options.boundsVertexCount ?? null,
    runGround: runGround
      ? Object.freeze({
          bones: Object.freeze([...runGround.bones]),
          margin: runGround.margin,
        })
      : null,
    unitySampledSkeleton: options.unitySampledSkeleton
      ? Object.freeze({ ...options.unitySampledSkeleton })
      : null,
    // Unity-sampled rigs consume the original prefab/controller in an isolated
    // project and must never enter the FBX multipart join pipeline first.
    joinInBlender: options.unitySampledSkeleton ? false : (options.joinInBlender ?? true),
    normalize: options.normalize ?? false,
    preferSkinJointTracks: options.preferSkinJointTracks ?? false,
    pristineClips: options.pristineClips ?? false,
    rotateX: options.rotateX ?? 0,
    rotateY: options.rotateY ?? 0,
  });
}
