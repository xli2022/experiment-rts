/**
 * Convert complete Athena2 unit rigs into the format used by this game.
 *
 * Usage:
 *   npm run import:athena2 -- --meshes <Imports> --textures <Textures> \
 *     --animations <Animations>
 *
 * Each Athena2 unit stores run, attack, and death as separate FBXs sharing one
 * logical rig. The run FBX normally supplies the geometry and skeleton. A
 * manifest entry can replace an individual part when one clip contains extra
 * authored deformation for it. Multipart skins are joined, old FBXs are
 * normalized, and all three files supply the authored clips. Direct clips use
 * their FBX AnimStack windows to discard out-of-take curves. Repaired clips are
 * already rebased, so their curve envelope is capped by the baked duration.
 * Both paths are resampled to the exact duration and cadence recorded in the
 * baked Animation asset. Rigs that cannot retain Unity's deformation semantics
 * through FBX can opt into direct Unity mesh/bind-pose/transform sampling. A
 * required slot with only one baked frame is missing, so that unit and its stale
 * outputs are excluded. Team art is copied or decoded to the ignored
 * `assets/textures` staging directory, then encoded like the existing skins with
 * `npm run textures`.
 */

import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { TGALoader } from 'three/examples/jsm/loaders/TGALoader.js';
import { ATHENA2_FACTIONS, ATHENA2_MODELS } from './athena2-models.mjs';
import { readFbxTakeWindows } from './fbx-take-window.mjs';
import { buildUnitySampledSkeletonModel } from './unity-sampled-skeleton.mjs';

installBrowserShims();
const warnedTrajectoryCycles = new Set();

const args = parseArgs(process.argv.slice(2));
const modelRoot = resolve(required(args, 'meshes'));
const textureRoot = resolve(required(args, 'textures'));
const animationRoot = resolve(args.animations ?? join(modelRoot, '..', 'Animations'));
const outputRoot = resolve(args.out ?? fileURLToPath(new URL('../public/units/', import.meta.url)));
const textureStage = resolve(
  args['texture-stage'] ?? fileURLToPath(new URL('../assets/textures/', import.meta.url)),
);
const only = args.only ? new Set(args.only.split(',').map((value) => value.trim())) : null;
const includeExisting = args['include-existing'] === 'true';

await validateAnimationInventory(animationRoot);
console.log(`Reading baked clip timings for ${ATHENA2_MODELS.length} Athena2 units...`);
const authoredTimings = new Map();
const authoredRunSizes = new Map();
for (const spec of ATHENA2_MODELS) {
  const authored = await readAuthoredAnimation(
    join(animationRoot, `${spec.animationAsset}.asset`),
    spec,
  );
  authoredTimings.set(spec.animationAsset, authored.clips);
  authoredRunSizes.set(spec.animationAsset, authored.runSize);
}

const completeModels = ATHENA2_MODELS.filter((spec) =>
  hasCompleteAnimationSet(authoredTimings.get(spec.animationAsset)),
);
const incompleteModels = ATHENA2_MODELS.filter(
  (spec) => !hasCompleteAnimationSet(authoredTimings.get(spec.animationAsset)),
);
const excludedModels = completeModels.filter((spec) => !spec.publish);
const publishedModels = completeModels.filter((spec) => spec.publish);
const unfactionedModels = publishedModels.filter((spec) => !spec.faction);
if (unfactionedModels.length > 0) {
  throw new Error(
    `Published units have no faction assignment: ${unfactionedModels
      .map((model) => model.unit)
      .join(', ')}`,
  );
}
const selected = publishedModels.filter(
  (model) => (!model.existing || includeExisting) && (!only || only.has(model.slug)),
);

await mkdir(outputRoot, { recursive: true });
await mkdir(textureStage, { recursive: true });
await pruneIncompleteOutputs(incompleteModels, outputRoot, textureStage);
await pruneExcludedOutputs(excludedModels, outputRoot, textureStage);
await pruneLegacyOutputs(ATHENA2_MODELS, outputRoot, textureStage);

if (selected.length === 0) {
  const requestedExcluded = excludedModels.filter((model) => only?.has(model.slug));
  if (requestedExcluded.length > 0) {
    throw new Error(
      `Requested units are explicitly excluded from public output: ${requestedExcluded
        .map((model) => model.unit)
        .join(', ')}`,
    );
  }
  const requestedIncomplete = incompleteModels.filter((model) => only?.has(model.slug));
  if (requestedIncomplete.length > 0) {
    throw new Error(
      `Requested units have missing required animations: ${requestedIncomplete
        .map((model) => model.unit)
        .join(', ')}`,
    );
  }
  throw new Error('No complete models selected');
}

let normalized = { paths: new Map(), tempRoot: null };
let joined = { paths: new Map(), tempRoot: null };
try {
  // Direct Unity sampling consumes the original FBX/controller pair itself.
  // Sending those specs through the legacy normalizer or multipart probe is
  // both wasted work and risks failing before Unity can author the final rig.
  const fbxSelected = selected.filter((spec) => !spec.unitySampledSkeleton);
  normalized = await normalizeLegacyFbx(fbxSelected, modelRoot, args.unity, args.blender);
  joined = await joinMultipartGeometry(fbxSelected, modelRoot, normalized.paths, args.blender);
  for (const [index, spec] of selected.entries()) {
    console.log(`[${index + 1}/${selected.length}] ${spec.unit} <- ${spec.source}`);
    await importModel(
      spec,
      modelRoot,
      textureRoot,
      outputRoot,
      textureStage,
      normalized.paths,
      joined.paths,
      authoredTimings.get(spec.animationAsset),
      args.unity,
    );
  }
} finally {
  for (const tempRoot of [joined.tempRoot, normalized.tempRoot]) {
    if (tempRoot) await rm(tempRoot, { recursive: true, force: true });
  }
}

await writeCatalog(outputRoot, authoredTimings, authoredRunSizes, publishedModels);
console.log(
  `\nImported ${selected.length} model${selected.length === 1 ? '' : 's'}. ` +
    'Run `npm run textures` to encode the staged team skins.',
);

function hasCompleteAnimationSet(timings) {
  return ['run', 'attack', 'die'].every(
    (name) => timings?.[name]?.frames > 1 && timings[name].frameRate > 0,
  );
}

function missingAnimationNames(timings) {
  return ['run', 'attack', 'die'].filter(
    (name) => !(timings?.[name]?.frames > 1 && timings[name].frameRate > 0),
  );
}

async function pruneIncompleteOutputs(models, output, stage) {
  if (models.length === 0) return;
  console.warn(`Excluding ${models.length} units with missing required animations:`);
  for (const spec of models) {
    const missing = missingAnimationNames(authoredTimings.get(spec.animationAsset));
    console.warn(`  ${spec.unit}: ${missing.join(', ')}`);
    await Promise.all([
      rm(join(output, `${spec.slug}.glb`), { force: true }),
      rm(join(output, `${spec.slug}-blue.ktx2`), { force: true }),
      rm(join(output, `${spec.slug}-red.ktx2`), { force: true }),
      rm(join(stage, `${spec.slug}-blue.png`), { force: true }),
      rm(join(stage, `${spec.slug}-red.png`), { force: true }),
    ]);
  }
}

async function pruneExcludedOutputs(models, output, stage) {
  if (models.length === 0) return;
  console.warn(
    `Excluding ${models.length} complete units from public output: ${models
      .map((model) => model.unit)
      .join(', ')}`,
  );
  await Promise.all(
    models.flatMap((spec) => [
      rm(join(output, `${spec.slug}.glb`), { force: true }),
      rm(join(output, `${spec.slug}-blue.ktx2`), { force: true }),
      rm(join(output, `${spec.slug}-red.ktx2`), { force: true }),
      rm(join(stage, `${spec.slug}-blue.png`), { force: true }),
      rm(join(stage, `${spec.slug}-red.png`), { force: true }),
    ]),
  );
}

async function pruneLegacyOutputs(models, output, stage) {
  await Promise.all(
    models.flatMap((spec) =>
      spec.legacySlugs.flatMap((slug) => [
        rm(join(output, `${slug}.glb`), { force: true }),
        rm(join(output, `${slug}-blue.ktx2`), { force: true }),
        rm(join(output, `${slug}-red.ktx2`), { force: true }),
        rm(join(stage, `${slug}-blue.png`), { force: true }),
        rm(join(stage, `${slug}-red.png`), { force: true }),
      ]),
    ),
  );
}

async function importModel(
  spec,
  meshes,
  textures,
  output,
  stage,
  normalizedPaths,
  geometryPaths,
  timings,
  unityOverride,
) {
  const source = join(meshes, spec.source);
  const originalPaths = {
    run: join(source, spec.files.run),
    attack: join(source, spec.files.attack),
    die: join(source, spec.files.die),
  };
  const originalGeometryPath = spec.geometryFile
    ? join(source, spec.geometryFile)
    : originalPaths[spec.geometry ?? 'run'];
  const [blue, red] = spec.skins.map((skin) => join(textures, skin));
  if (spec.unitySampledSkeleton) {
    await Promise.all(
      [
        ...Object.values(originalPaths),
        originalGeometryPath,
        join(source, spec.unitySampledSkeleton.controller),
        blue,
        red,
      ].map((path) => access(path)),
    );
    const data = await buildUnitySampledSkeletonModel(source, spec, timings, unityOverride);
    validateGlb(data, spec);
    await Promise.all([
      writeFile(join(output, `${spec.slug}.glb`), new Uint8Array(data)),
      stageTexture(blue, join(stage, `${spec.slug}-blue.png`)),
      stageTexture(red, join(stage, `${spec.slug}-red.png`)),
    ]);
    return;
  }
  const paths = Object.fromEntries(
    Object.entries(originalPaths).map(([name, path]) => [name, normalizedPaths.get(path) ?? path]),
  );
  const animationPaths = spec.pristineClips ? originalPaths : paths;
  const takeWindows = Object.fromEntries(
    await Promise.all(
      Object.entries(originalPaths).map(async ([name, path]) => [
        name,
        await readFbxTakeWindows(path),
      ]),
    ),
  );
  await Promise.all(
    [...Object.values(originalPaths), originalGeometryPath, blue, red].map((path) => access(path)),
  );

  const geometryPath =
    geometryPaths.get(spec.slug) ??
    normalizedPaths.get(originalGeometryPath) ??
    originalGeometryPath;
  const geometryParts = spec.geometryParts ?? [];
  const partSources = [
    ...new Set(
      geometryParts.map((part) => {
        if (!(part.source in paths)) {
          throw new Error(`${spec.unit} geometry part ${part.name}: unknown source ${part.source}`);
        }
        return part.source;
      }),
    ),
  ];
  const [runScene, runClipScene, attackScene, dieScene, geometryPartEntries] = await Promise.all([
    loadFbx(geometryPath),
    geometryPath === animationPaths.run && geometryParts.length === 0
      ? Promise.resolve(null)
      : loadFbx(animationPaths.run),
    loadFbx(animationPaths.attack),
    loadFbx(animationPaths.die),
    Promise.all(partSources.map(async (name) => [name, await loadFbx(paths[name])])),
  ]);
  const geometryPartScenes = new Map(geometryPartEntries);
  const runAnimationScene = runClipScene ?? runScene;
  const runSourceClip = authoredClip(runAnimationScene, runScene, spec, 'run', timings.run);
  const attackSourceClip = authoredClip(attackScene, runScene, spec, 'attack', timings.attack);
  const dieSourceClip = authoredClip(dieScene, runScene, spec, 'die', timings.die);
  const clipScenes = {
    run: runAnimationScene,
    attack: attackScene,
    die: dieScene,
  };
  const geometryCorrections = applyGeometryPartOverrides(
    runScene,
    geometryPartScenes,
    clipScenes,
    spec,
  );
  const runMesh = mergeRenderableMeshes(runScene, spec);
  applyBoundsVertexCount(runMesh, spec);
  removeUnsupportedSceneObjects(runScene);
  if (spec.preferSkinJointTracks) {
    renameConflictingAnimationHelpers(runScene, spec);
  }

  // A few source FBXs omit the up-axis or facing transform that Athena2 applies
  // above the renderer. Keep the geometry, skin, and animation data untouched
  // and carry that manifest correction on the exported model node instead.
  if (spec.rotateX !== 0) {
    runScene.quaternion.premultiply(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), spec.rotateX),
    );
  }
  if (spec.rotateY !== 0) {
    runScene.quaternion.premultiply(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), spec.rotateY),
    );
  }

  const clips = [
    namedClip(runScene, runSourceClip, 'run', spec, timings.run, runAnimationScene, {
      completeRestPose: geometryParts.length > 0,
      corrections: geometryCorrections.get('run'),
      takeWindow: takeWindowForClip(
        takeWindows.run,
        runSourceClip,
        spec,
        'run',
        timings.run,
        animationPaths.run !== originalPaths.run,
      ),
    }),
    namedClip(runScene, attackSourceClip, 'attack', spec, timings.attack, attackScene, {
      completeRestPose: geometryParts.length > 0,
      corrections: geometryCorrections.get('attack'),
      takeWindow: takeWindowForClip(
        takeWindows.attack,
        attackSourceClip,
        spec,
        'attack',
        timings.attack,
        animationPaths.attack !== originalPaths.attack,
      ),
    }),
    namedClip(runScene, dieSourceClip, 'die', spec, timings.die, dieScene, {
      completeRestPose: geometryParts.length > 0,
      corrections: geometryCorrections.get('die'),
      takeWindow: takeWindowForClip(
        takeWindows.die,
        dieSourceClip,
        spec,
        'die',
        timings.die,
        animationPaths.die !== originalPaths.die,
      ),
    }),
  ];
  validateClipTargets(runScene, clips, spec);
  replaceMaterials(runMesh);
  runScene.animations = clips;

  const data = await new GLTFExporter().parseAsync(runScene, {
    animations: clips,
    binary: true,
    onlyVisible: false,
  });
  if (!(data instanceof ArrayBuffer))
    throw new Error(`${spec.unit}: exporter did not return GLB data`);
  validateGlb(data, spec);

  await Promise.all([
    writeFile(join(output, `${spec.slug}.glb`), new Uint8Array(data)),
    stageTexture(blue, join(stage, `${spec.slug}-blue.png`)),
    stageTexture(red, join(stage, `${spec.slug}-red.png`)),
  ]);
}

function applyGeometryPartOverrides(targetScene, geometryPartScenes, clipScenes, spec) {
  const corrections = new Map();
  for (const part of spec.geometryParts ?? []) {
    if (!part.name || !part.source || !part.bindAnchor) {
      throw new Error(`${spec.unit}: geometry parts require name, source, and bindAnchor`);
    }
    const sourceScene = geometryPartScenes.get(part.source);
    if (!sourceScene) {
      throw new Error(
        `${spec.unit} geometry part ${part.name}: source ${part.source} was not loaded`,
      );
    }
    const targetPart = uniqueRenderableByName(targetScene, part.name, spec);
    const sourcePart = uniqueRenderableByName(sourceScene, part.name, spec);
    if (!targetPart.isSkinnedMesh || !sourcePart.isSkinnedMesh) {
      throw new Error(
        `${spec.unit} geometry part ${part.name}: both versions must be skinned meshes`,
      );
    }

    const sourceAnchorIndex = sourcePart.skeleton.bones.findIndex(
      (bone) => bone.name === part.bindAnchor,
    );
    if (sourceAnchorIndex < 0) {
      throw new Error(
        `${spec.unit} geometry part ${part.name}: ${part.source} skin lacks bind anchor ${part.bindAnchor}`,
      );
    }
    const sourceInverse = sourcePart.skeleton.boneInverses[sourceAnchorIndex].clone();

    // Each non-owning clip used the part from its own FBX at a potentially
    // different bind pose. Derive the local anchor pose that makes the
    // replacement skin matrix identical:
    //   local * overrideInverse = sourceLocal * sourceInverse
    for (const [clipName, clipScene] of Object.entries(clipScenes)) {
      if (clipName === part.source) continue;
      const clipPart = uniqueRenderableByName(clipScene, part.name, spec);
      if (!clipPart.isSkinnedMesh) {
        throw new Error(
          `${spec.unit} ${clipName} geometry part ${part.name}: expected a skinned mesh`,
        );
      }
      const clipAnchorIndex = clipPart.skeleton.bones.findIndex(
        (bone) => bone.name === part.bindAnchor,
      );
      if (clipAnchorIndex < 0) {
        throw new Error(
          `${spec.unit} ${clipName} geometry part ${part.name}: skin lacks bind anchor ${part.bindAnchor}`,
        );
      }
      clipScene.updateMatrixWorld(true);
      const clipAnchor = clipPart.skeleton.bones[clipAnchorIndex];
      const correction = clipAnchor.matrix
        .clone()
        .multiply(clipPart.skeleton.boneInverses[clipAnchorIndex])
        .multiply(sourceInverse.clone().invert());
      addGeometryCorrection(corrections, clipName, part.bindAnchor, correction, spec);
    }

    const targetNodes = attachMissingSkinBones(targetScene, sourcePart, spec, part);
    const bones = sourcePart.skeleton.bones.map((bone) => {
      const target = targetNodes.get(bone.name);
      if (!target?.isBone) {
        throw new Error(
          `${spec.unit} geometry part ${part.name}: target rig lacks bone ${bone.name}`,
        );
      }
      return target;
    });
    const bindMatrix = sourcePart.bindMatrix.clone();
    const boneInverses = sourcePart.skeleton.boneInverses.map((inverse) => inverse.clone());
    targetPart.removeFromParent();
    sourcePart.removeFromParent();
    targetScene.add(sourcePart);
    sourcePart.bind(new THREE.Skeleton(bones, boneInverses), bindMatrix);
    targetScene.updateMatrixWorld(true);
    console.log(
      `${spec.unit}: replacing ${part.name} geometry from ${part.source} ` +
        `(${sourcePart.geometry.getAttribute('position').count} vertices)`,
    );
  }
  return corrections;
}

function uniqueRenderableByName(scene, name, spec) {
  const found = [];
  scene.traverse((object) => {
    if (object.isMesh && object.name === name) found.push(object);
  });
  if (found.length !== 1) {
    throw new Error(
      `${spec.unit} geometry part ${name}: expected one renderable, found ${found.length}`,
    );
  }
  return found[0];
}

function namedNodeMap(scene) {
  const nodes = new Map();
  scene.traverse((object) => {
    if (object.name && !nodes.has(object.name)) nodes.set(object.name, object);
  });
  return nodes;
}

function attachMissingSkinBones(targetScene, sourcePart, spec, part) {
  const targetNodes = namedNodeMap(targetScene);
  const missingRoots = new Set();
  for (const bone of sourcePart.skeleton.bones) {
    if (targetNodes.has(bone.name)) continue;
    let root = bone;
    while (root.parent?.isBone && !targetNodes.has(root.parent.name)) {
      root = root.parent;
    }
    missingRoots.add(root);
  }

  for (const root of missingRoots) {
    if (targetNodes.has(root.name)) continue;
    const targetParent = targetNodes.get(root.parent?.name);
    if (!targetParent?.isBone) {
      throw new Error(
        `${spec.unit} geometry part ${part.name}: cannot attach missing bone subtree ${root.name}`,
      );
    }
    targetParent.add(root);
    root.traverse((object) => {
      if (object.name) targetNodes.set(object.name, object);
    });
  }
  return targetNodes;
}

function addGeometryCorrection(corrections, clipName, anchor, matrix, spec) {
  let clipCorrections = corrections.get(clipName);
  if (!clipCorrections) {
    clipCorrections = new Map();
    corrections.set(clipName, clipCorrections);
  }
  const existing = clipCorrections.get(anchor);
  if (existing && matrixMaxElementDelta(existing, matrix) > 1e-5) {
    throw new Error(
      `${spec.unit} ${clipName}: geometry parts require conflicting ${anchor} bind corrections`,
    );
  }
  clipCorrections.set(anchor, matrix.clone());
}

function matrixMaxElementDelta(left, right) {
  let maximum = 0;
  for (let index = 0; index < 16; index++) {
    maximum = Math.max(maximum, Math.abs(left.elements[index] - right.elements[index]));
  }
  return maximum;
}

async function normalizeLegacyFbx(specs, meshes, unityOverride, blenderOverride) {
  const sourcesFor = (method) => [
    ...new Set(
      specs
        .filter((spec) => spec.normalize === method)
        .flatMap((spec) =>
          [...Object.values(spec.files), spec.geometryFile]
            .filter(Boolean)
            .map((file) => join(meshes, spec.source, file)),
        ),
    ),
  ];
  const unitySources = sourcesFor('unity');
  const blenderSources = sourcesFor('blender');
  const originals = [...unitySources, ...blenderSources];
  if (originals.length === 0) return { paths: new Map(), tempRoot: null };
  await Promise.all(originals.map((path) => access(path)));

  const tempRoot = await mkdtemp(join(tmpdir(), 'rts-athena2-normalize-'));
  const outputRoot = join(tempRoot, 'Normalized');
  await mkdir(outputRoot, { recursive: true });
  const paths = new Map();
  let outputIndex = 0;

  if (unitySources.length > 0) {
    const unity = await firstAvailablePath([
      unityOverride ? resolve(unityOverride) : null,
      'C:\\Program Files\\Unity\\Hub\\Editor\\2022.3.62f3\\Editor\\Unity.exe',
    ]);
    if (!unity) {
      throw new Error(
        'The selected Athena2 units contain legacy FBXs. Install Unity 2022.3 ' +
          'or pass its editor path with --unity.',
      );
    }
    const localAppData = process.env.LOCALAPPDATA;
    if (!localAppData) throw new Error("LOCALAPPDATA is required to locate Unity's FBX SDK");
    const fbxPackage = join(
      localAppData,
      'Unity',
      'cache',
      'packages',
      'packages.unity.com',
      'com.autodesk.fbx@5.1.1',
    );
    await access(join(fbxPackage, 'package.json'));

    const editorRoot = join(tempRoot, 'Assets', 'Editor');
    const packagesRoot = join(tempRoot, 'Packages');
    const settingsRoot = join(tempRoot, 'ProjectSettings');
    await Promise.all(
      [editorRoot, packagesRoot, settingsRoot].map((path) => mkdir(path, { recursive: true })),
    );
    const sourceScript = fileURLToPath(new URL('./unity-normalize-fbx.cs', import.meta.url));
    await Promise.all([
      copyFile(sourceScript, join(editorRoot, 'RtsFbxNormalizer.cs')),
      writeFile(
        join(packagesRoot, 'manifest.json'),
        `${JSON.stringify(
          {
            dependencies: {
              'com.autodesk.fbx': `file:${fbxPackage.replaceAll('\\', '/')}`,
            },
          },
          null,
          2,
        )}\n`,
      ),
      writeFile(
        join(settingsRoot, 'ProjectVersion.txt'),
        'm_EditorVersion: 2022.3.62f3\nm_EditorVersionWithRevision: 2022.3.62f3\n',
      ),
    ]);

    const jobs = unitySources.map((source) => {
      const destination = join(outputRoot, `${outputIndex++}.fbx`);
      paths.set(source, destination);
      return `${source}\t${destination}`;
    });
    const manifest = join(tempRoot, 'unity-fbx-jobs.tsv');
    await writeFile(manifest, `${jobs.join('\n')}\n`);
    console.log(`Normalizing ${jobs.length} legacy FBX files with Unity...`);
    await runProcess(unity, [
      '-batchmode',
      '-nographics',
      '-quit',
      '-projectPath',
      tempRoot,
      '-executeMethod',
      'RtsFbxNormalizer.Run',
      '-rtsFbxManifest',
      manifest,
      '-logFile',
      join(tempRoot, 'unity.log'),
    ]);
  }

  if (blenderSources.length > 0) {
    const blender = await findBlender(blenderOverride);
    if (!blender) {
      throw new Error(
        'Some Athena2 animation curves require Blender 4.5. Install it or ' +
          'pass its executable path with --blender.',
      );
    }
    const jobs = blenderSources.map((source) => {
      const destination = join(outputRoot, `${outputIndex++}.fbx`);
      paths.set(source, destination);
      return { source, destination };
    });
    const manifest = join(tempRoot, 'blender-fbx-jobs.json');
    await writeFile(manifest, `${JSON.stringify(jobs, null, 2)}\n`);
    const script = fileURLToPath(new URL('./blender-roundtrip-fbx.py', import.meta.url));
    console.log(`Repairing ${jobs.length} FBX animation files with Blender...`);
    await runProcess(blender, [
      '--background',
      '--factory-startup',
      '--python',
      script,
      '--',
      manifest,
    ]);
  }

  await Promise.all([...paths.values()].map((path) => access(path)));
  return { paths, tempRoot };
}

async function joinMultipartGeometry(specs, meshes, normalizedPaths, blenderOverride) {
  const jobs = [];
  for (const spec of specs) {
    if (!spec.joinInBlender) continue;
    const original = join(meshes, spec.source, spec.geometryFile ?? spec.files[spec.geometry]);
    const source = normalizedPaths.get(original) ?? original;
    const scene = await loadFbx(source);
    let skinnedCount = 0;
    scene.traverse((object) => {
      if (object.isSkinnedMesh) skinnedCount++;
    });
    if (skinnedCount > 1) jobs.push({ spec, source });
  }
  if (jobs.length === 0) return { paths: new Map(), tempRoot: null };

  const blender = await findBlender(blenderOverride);
  if (!blender) {
    throw new Error(
      'The selected Athena2 units contain multipart rigs. Install Blender 4.5 ' +
        'or pass its executable path with --blender.',
    );
  }

  const tempRoot = await mkdtemp(join(tmpdir(), 'rts-athena2-blender-'));
  const outputRoot = join(tempRoot, 'Joined');
  await mkdir(outputRoot, { recursive: true });
  const paths = new Map();
  const manifestJobs = jobs.map(({ spec, source }, index) => {
    const destination = join(outputRoot, `${index}.fbx`);
    paths.set(spec.slug, destination);
    return { source, destination };
  });
  const manifest = join(tempRoot, 'fbx-jobs.json');
  await writeFile(manifest, `${JSON.stringify(manifestJobs, null, 2)}\n`);

  const script = fileURLToPath(new URL('./blender-normalize-fbx.py', import.meta.url));
  console.log(`Joining ${jobs.length} multipart model rigs with Blender...`);
  await runProcess(blender, [
    '--background',
    '--factory-startup',
    '--python',
    script,
    '--',
    manifest,
  ]);
  await Promise.all([...paths.values()].map((path) => access(path)));
  return { paths, tempRoot };
}

async function findBlender(override) {
  const localAppData = process.env.LOCALAPPDATA;
  return firstAvailablePath([
    override ? resolve(override) : null,
    localAppData
      ? join(localAppData, 'Programs', 'Blender Foundation', 'Blender 4.5', 'blender.exe')
      : null,
  ]);
}

async function firstAvailablePath(candidates) {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      await access(candidate);
      return candidate;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  return null;
}

async function runProcess(command, commandArgs) {
  await new Promise((resolvePromise, reject) => {
    const child = spawn(command, commandArgs, {
      stdio: ['ignore', 'inherit', 'inherit'],
      windowsHide: true,
    });
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} exited with ${code ?? `signal ${signal ?? 'unknown'}`}`));
    });
  });
}

async function stageTexture(source, destination) {
  const extension = extname(source).toLowerCase();
  if (extension === '.png') {
    await copyFile(source, destination);
    return;
  }
  if (extension !== '.tga') {
    throw new Error(`Unsupported team texture format: ${source}`);
  }

  const bytes = await readFile(source);
  const array = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const image = new TGALoader().parse(array);
  await sharp(image.data, {
    raw: { width: image.width, height: image.height, channels: 4 },
  })
    .png()
    .toFile(destination);
}

async function loadFbx(path) {
  const bytes = await readFile(path);
  const array = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const originalError = console.error;
  let ignoredTrajectoryCycle = false;
  console.error = (message, object, ...rest) => {
    if (
      message === "THREE.Object3D.add: object can't be added as a child of itself." &&
      object?.name?.startsWith('z_DTrajEdit_')
    ) {
      // Sphinx's run FBX contains a circular 3ds Max trajectory-helper link.
      // Object3D correctly refuses that editor-only edge; the rig and mesh are
      // already parented correctly, so avoid dumping the entire cyclic object.
      ignoredTrajectoryCycle = true;
      return;
    }
    originalError(message, object, ...rest);
  };
  try {
    const scene = new FBXLoader().parse(array, `${dirname(path).replaceAll('\\', '/')}/`);
    if (ignoredTrajectoryCycle && !warnedTrajectoryCycles.has(path)) {
      warnedTrajectoryCycles.add(path);
      console.warn(`${basename(path)}: ignored a cyclic editor-only trajectory helper`);
    }
    return scene;
  } finally {
    console.error = originalError;
  }
}

function authoredClip(scene, bindScene, spec, clipName, timing) {
  if (timing.static) {
    if (scene.animations.length === 1 && scene.animations[0].duration > 0) {
      return scene.animations[0];
    }
    let bone = null;
    bindScene.traverse((object) => {
      if (!bone && object.isBone) bone = object;
    });
    if (!bone) throw new Error(`${spec.unit} ${clipName}: no bone for static pose`);
    const values = new Float32Array(8);
    bone.quaternion.toArray(values, 0);
    bone.quaternion.toArray(values, 4);
    return new THREE.AnimationClip(clipName, 1 / 30, [
      new THREE.QuaternionKeyframeTrack(
        `${bone.name}.quaternion`,
        new Float32Array([0, 1 / 30]),
        values,
        THREE.InterpolateDiscrete,
      ),
    ]);
  }
  const nonEmpty = scene.animations.filter((clip) => clip.duration > 0);
  if (nonEmpty.length === 0) {
    throw new Error(
      `${spec.unit} ${clipName}: expected one non-empty animation, found ` +
        `${scene.animations.length} (${scene.animations.map((clip) => clip.duration).join(', ')})`,
    );
  }
  if (nonEmpty.length === 1) return nonEmpty[0];

  // A Unity-baked rig intentionally stores the complete run/attack/die set in
  // one modern FBX. Prefer its named take before falling back to the legacy
  // Blender case where many per-object takes together form one logical clip.
  const exact = nonEmpty.filter((clip) => clip.name === clipName);
  if (exact.length === 1) return exact[0];

  // Blender exposes some old FBXs as one take per animated object. They share
  // a time span and together form Athena2's single authored clip.
  return new THREE.AnimationClip(
    clipName,
    Math.max(...nonEmpty.map((clip) => clip.duration)),
    nonEmpty.flatMap((clip) => clip.tracks),
  );
}

function takeWindowForClip(windows, clip, spec, clipName, timing, normalized) {
  if (normalized) {
    let start = Infinity;
    let stop = -Infinity;
    for (const track of clip.tracks) {
      if (track.times.length === 0) continue;
      start = Math.min(start, track.times[0]);
      stop = Math.max(stop, track.times.at(-1));
    }
    if (!(stop > start)) {
      throw new Error(`${spec.unit} ${clipName}: normalized clip has an empty time window`);
    }
    // Blender and Unity export repaired actions in a zero-based time domain,
    // while the source FBX stack and Unity's per-file clip override can refer
    // to the original timeline. Athena2's baked duration is the authoritative
    // stop in the rebased clip; clamp only when the repaired action is shorter.
    return Object.freeze({
      name: clip.name,
      start,
      stop: Math.min(stop, start + timing.duration),
    });
  }
  if (windows.length === 0) return null;
  if (windows.length === 1) return windows[0];
  const exact = windows.filter((window) => window.name === clip.name);
  if (exact.length === 1) return exact[0];
  throw new Error(
    `${spec.unit} ${clipName}: could not select ${clip.name} from FBX takes ` +
      `(${windows.map((window) => window.name).join(', ')})`,
  );
}

function mergeRenderableMeshes(scene, spec) {
  const skinnedMeshes = [];
  const renderableMeshes = [];
  scene.traverse((object) => {
    if (object.isSkinnedMesh) skinnedMeshes.push(object);
    if (object.isMesh) renderableMeshes.push(object);
  });
  if (skinnedMeshes.length === 0) {
    throw new Error(`${spec.unit} geometry: expected a skinned mesh after joining`);
  }

  const mesh = skinnedMeshes.reduce((largest, candidate) =>
    candidate.geometry.getAttribute('position').count >
    largest.geometry.getAttribute('position').count
      ? candidate
      : largest,
  );
  if (renderableMeshes.length > 1) {
    mergeMeshParts(mesh, renderableMeshes, scene, spec);
  }

  // Source material slots are irrelevant: the renderer replaces them with the
  // selected KTX2 team skin. Groups would nevertheless become multiple glTF
  // primitives, so flatten them explicitly.
  mesh.geometry.clearGroups();
  const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  mesh.material = material ?? new THREE.MeshStandardMaterial();
  validateSkinnedMesh(mesh, spec);

  const remaining = [];
  scene.traverse((object) => {
    if (object.isMesh) remaining.push(object);
  });
  if (remaining.length !== 1 || remaining[0] !== mesh) {
    throw new Error(
      `${spec.unit} geometry: expected one renderable mesh after merging, found ${remaining.length}`,
    );
  }
  return mesh;
}

function applyBoundsVertexCount(mesh, spec) {
  const retained = spec.boundsVertexCount;
  if (retained === null) return;
  const position = mesh.geometry.getAttribute('position');
  if (!Number.isSafeInteger(retained) || retained <= 0 || retained > position.count) {
    throw new Error(
      `${spec.unit} geometry: invalid bounds vertex count ${retained} for ${position.count} vertices`,
    );
  }

  // Treant's Log is an authored attack-only part. Its run and death tracks
  // collapse it to a zero-area point well outside the body, which is invisible
  // but would poison gallery framing if it participated in sampled bounds.
  // Standard glTF node extras preserve this body-prefix hint without removing
  // the Log vertices or any of their animation tracks.
  mesh.userData.boundsVertexCount = retained;
  console.warn(
    `${spec.unit} geometry: using the first ${retained} of ${position.count} vertices for bounds`,
  );
}

function validateSkinnedMesh(mesh, spec) {
  for (const attribute of ['position', 'normal', 'uv', 'skinIndex', 'skinWeight']) {
    if (!mesh.geometry.getAttribute(attribute)) {
      throw new Error(`${spec.unit} geometry: missing ${attribute} geometry attribute`);
    }
  }
  if (mesh.skeleton.bones.length === 0) {
    throw new Error(`${spec.unit} geometry: skeleton has no bones`);
  }
}

function mergeMeshParts(base, renderableMeshes, scene, spec) {
  scene.updateMatrixWorld(true);
  const bones = [];
  const boneInverses = [];
  const jointVariants = new Map();
  const primaryJoint = new Map();
  const canonicalBones = new Map();
  for (const object of renderableMeshes) {
    if (!object.isSkinnedMesh) continue;
    for (const bone of object.skeleton.bones) {
      if (canonicalBones.has(bone.name)) continue;
      const found = THREE.PropertyBinding.findNode(scene, bone.name);
      canonicalBones.set(bone.name, found?.isBone ? found : bone);
    }
  }

  const jointFor = (bone, inverse) => {
    const logicalName = bone.name || `unnamed_${bone.uuid}`;
    const key = `${logicalName}|${matrixKey(inverse)}`;
    const existing = jointVariants.get(key);
    if (existing !== undefined) return existing;

    const canonical = canonicalBones.get(bone.name) ?? bone;
    let joint = canonical;
    if (primaryJoint.has(logicalName)) {
      joint = new THREE.Bone();
      joint.name = `${logicalName}_skin_${primaryJoint.get(logicalName)}`;
      canonical.add(joint);
      joint.updateMatrixWorld(true);
    } else {
      primaryJoint.set(logicalName, bones.length);
    }
    const index = bones.length;
    bones.push(joint);
    boneInverses.push(inverse.clone());
    jointVariants.set(key, index);
    return index;
  };

  const sources = [];
  const inverseBind = base.bindMatrix.clone().invert();
  let rigidIndex = 0;
  for (const object of renderableMeshes) {
    if (object.isSkinnedMesh) {
      sources.push({
        geometry: object.geometry,
        jointMap: object.skeleton.bones.map((bone, index) =>
          jointFor(bone, object.skeleton.boneInverses[index]),
        ),
        rigidJoint: null,
        transform: inverseBind.clone().multiply(object.bindMatrix),
      });
    } else {
      const proxy = new THREE.Bone();
      proxy.name = `${spec.slug}_rigid_${rigidIndex++}`;
      object.add(proxy);
      proxy.updateMatrixWorld(true);
      const joint = bones.length;
      bones.push(proxy);
      boneInverses.push(proxy.matrixWorld.clone().invert());
      sources.push({
        geometry: object.geometry,
        jointMap: null,
        rigidJoint: joint,
        transform: inverseBind.clone().multiply(object.matrixWorld),
      });
    }
  }

  base.geometry = combineGeometry(sources, spec);
  base.bind(new THREE.Skeleton(bones, boneInverses), base.bindMatrix);
  for (const object of renderableMeshes) {
    if (object !== base) demoteMesh(object);
  }
}

function matrixKey(matrix) {
  return matrix.elements.map((value) => value.toFixed(5)).join(',');
}

function combineGeometry(sources, spec) {
  const chunks = [];
  let totalVertices = 0;
  for (const source of sources) {
    let geometry = source.geometry.clone();
    if (geometry.index) geometry = geometry.toNonIndexed();
    if (source.transform) geometry.applyMatrix4(source.transform);
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');
    const uv = geometry.getAttribute('uv');
    if (!position || !normal || !uv) {
      throw new Error(`${spec.unit} geometry: a rigid part lacks position/normal/uv`);
    }
    const skinIndex = geometry.getAttribute('skinIndex');
    const skinWeight = geometry.getAttribute('skinWeight');
    if (
      source.rigidJoint === null &&
      (!skinIndex || !skinWeight || skinIndex.itemSize !== 4 || skinWeight.itemSize !== 4)
    ) {
      throw new Error(`${spec.unit} geometry: the deforming mesh lacks skin weights`);
    }
    chunks.push({
      position,
      normal,
      uv,
      skinIndex,
      skinWeight,
      jointMap: source.jointMap,
      rigidJoint: source.rigidJoint,
    });
    totalVertices += position.count;
  }

  const positions = new Float32Array(totalVertices * 3);
  const normals = new Float32Array(totalVertices * 3);
  const uvs = new Float32Array(totalVertices * 2);
  const skinIndices = new Uint16Array(totalVertices * 4);
  const skinWeights = new Float32Array(totalVertices * 4);
  let vertexOffset = 0;
  for (const chunk of chunks) {
    for (let vertex = 0; vertex < chunk.position.count; vertex++) {
      const destination = vertexOffset + vertex;
      positions[destination * 3] = chunk.position.getX(vertex);
      positions[destination * 3 + 1] = chunk.position.getY(vertex);
      positions[destination * 3 + 2] = chunk.position.getZ(vertex);
      normals[destination * 3] = chunk.normal.getX(vertex);
      normals[destination * 3 + 1] = chunk.normal.getY(vertex);
      normals[destination * 3 + 2] = chunk.normal.getZ(vertex);
      uvs[destination * 2] = chunk.uv.getX(vertex);
      uvs[destination * 2 + 1] = chunk.uv.getY(vertex);
      if (chunk.rigidJoint !== null) {
        skinIndices[destination * 4] = chunk.rigidJoint;
        skinWeights[destination * 4] = 1;
      } else {
        for (let component = 0; component < 4; component++) {
          const sourceJoint = chunk.skinIndex.getComponent(vertex, component);
          const destinationJoint = chunk.jointMap[sourceJoint];
          if (destinationJoint === undefined) {
            throw new Error(`${spec.unit} geometry: skin index ${sourceJoint} has no merged joint`);
          }
          skinIndices[destination * 4 + component] = destinationJoint;
          skinWeights[destination * 4 + component] = chunk.skinWeight.getComponent(
            vertex,
            component,
          );
        }
      }
    }
    vertexOffset += chunk.position.count;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function demoteMesh(object) {
  object.isMesh = false;
  object.isSkinnedMesh = false;
  object.type = 'Group';
  object.geometry = undefined;
  object.material = undefined;
  object.skeleton = undefined;
}

function validateGlb(data, spec) {
  const view = new DataView(data);
  if (
    view.byteLength < 20 ||
    view.getUint32(0, true) !== 0x46546c67 ||
    view.getUint32(4, true) !== 2 ||
    view.getUint32(8, true) !== view.byteLength
  ) {
    throw new Error(`${spec.unit}: exporter returned an invalid GLB header`);
  }

  const bytes = new Uint8Array(data);
  let offset = 12;
  let json = null;
  while (offset + 8 <= bytes.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    if (type === 0x4e4f534a) {
      json = JSON.parse(
        new TextDecoder().decode(bytes.subarray(offset + 8, offset + 8 + length)).trim(),
      );
      break;
    }
    offset += 8 + length;
  }
  if (!json) throw new Error(`${spec.unit}: exported GLB has no JSON chunk`);

  if (json.meshes?.length !== 1 || json.meshes[0].primitives?.length !== 1) {
    throw new Error(`${spec.unit}: exported GLB must contain one mesh with one primitive`);
  }
  if (json.skins?.length !== 1) {
    throw new Error(`${spec.unit}: exported GLB must contain one skin`);
  }
  const attributes = json.meshes[0].primitives[0].attributes ?? {};
  for (const attribute of ['POSITION', 'NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0']) {
    if (!(attribute in attributes)) {
      throw new Error(`${spec.unit}: exported GLB is missing ${attribute}`);
    }
  }
  const clipNames = (json.animations ?? [])
    .map((clip) => clip.name)
    .sort()
    .join(',');
  if (clipNames !== 'attack,die,run') {
    throw new Error(`${spec.unit}: exported GLB clips are ${clipNames || 'absent'}`);
  }
}

function validateClipTargets(scene, clips, spec) {
  for (const clip of clips) {
    if (!clip.validate()) throw new Error(`${spec.unit} ${clip.name}: invalid animation clip`);
    for (const track of clip.tracks) {
      const parsed = THREE.PropertyBinding.parseTrackName(track.name);
      if (!THREE.PropertyBinding.findNode(scene, parsed.nodeName)) {
        throw new Error(
          `${spec.unit} ${clip.name}: track target ${parsed.nodeName} is absent from the geometry rig`,
        );
      }
    }
  }
}

function namedClip(scene, clip, name, spec, timing, sourceScene, options = {}) {
  const copy = clip.clone();
  copy.name = name;
  copy.tracks = copy.tracks.map(strictlyIncreasingTrack);
  copy.tracks = canonicalizeDuplicateTracks(
    copy.tracks,
    scene,
    options.takeWindow?.start ?? 0,
    spec,
    name,
  );
  const kept = [];
  const dropped = [];
  for (const track of copy.tracks) {
    const parsed = THREE.PropertyBinding.parseTrackName(track.name);
    if (THREE.PropertyBinding.findNode(scene, parsed.nodeName)) {
      kept.push(track);
    } else {
      // A few FBXs animate editor-only helpers (for example
      // `Bip001_Footsteps`) that another clip omits. They are not bones and do
      // not deform the model. Three's exporter would drop them anyway; doing it
      // explicitly keeps the GLB clean and makes the omission visible here.
      dropped.push(parsed.nodeName);
    }
  }
  if (dropped.length > 0) {
    const sourceJoints = skinJointNames(sourceScene);
    const missingJoints = [...new Set(dropped)].filter((node) => sourceJoints.has(node));
    if (missingJoints.length > 0) {
      throw new Error(
        `${spec.unit} ${name}: geometry rig is missing animated skin joints ` +
          `(${missingJoints.join(', ')})`,
      );
    }
    const examples = [...new Set(dropped)].slice(0, 4).join(', ');
    console.warn(
      `${spec.unit} ${name}: dropping ${dropped.length} absent helper tracks` +
        (examples ? ` (${examples}${dropped.length > 4 ? ', ...' : ''})` : ''),
    );
  }
  copy.tracks = kept;
  applyClipGeometryCorrections(copy, options.corrections, spec, name);
  if (options.completeRestPose) {
    completeMissingJointTracks(scene, sourceScene, copy, spec, name);
  }
  if (copy.tracks.length === 0)
    throw new Error(`${spec.unit} ${name}: animation has no usable tracks`);
  if (timing.static) {
    copy.tracks = copy.tracks.map(frozenTrack);
    copy.resetDuration();
  } else {
    if (options.takeWindow) {
      resampleTakeWindow(copy, timing, options.takeWindow);
    } else {
      retimeClip(copy, timing.duration);
    }
  }
  return copy;
}

function canonicalizeDuplicateTracks(tracks, scene, sourceTime, spec, clipName) {
  const groups = new Map();
  for (const track of tracks) {
    const group = groups.get(track.name) ?? [];
    group.push(track);
    groups.set(track.name, group);
  }
  const weightedBones = weightedBonesByName(scene);
  const weightedBoneSet = new Set(weightedBones.values());
  const canonical = [];
  let resolved = 0;
  let droppedHelpers = 0;
  for (const [trackName, candidates] of groups) {
    if (candidates.length === 1) {
      canonical.push(candidates[0]);
      continue;
    }
    const parsed = THREE.PropertyBinding.parseTrackName(trackName);
    const target =
      weightedBones.get(parsed.nodeName) ?? THREE.PropertyBinding.findNode(scene, parsed.nodeName);
    if (!target) {
      canonical.push(candidates[0]);
      resolved += candidates.length - 1;
      continue;
    }
    if (!affectsWeightedBones(target, weightedBoneSet)) {
      droppedHelpers += candidates.length;
      continue;
    }
    const rest = target[parsed.propertyName];
    if (!rest?.toArray) {
      throw new Error(
        `${spec.unit} ${clipName}: duplicate ${trackName} cannot be matched to a rest transform`,
      );
    }
    const restValue = rest.toArray();
    let best = candidates[0];
    let bestDistance = Infinity;
    for (const candidate of candidates) {
      const sampled = candidate.createInterpolant().evaluate(sourceTime);
      const distance = transformDistance(sampled, restValue, parsed.propertyName === 'quaternion');
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    canonical.push(best);
    resolved += candidates.length - 1;
  }
  if (resolved > 0 || droppedHelpers > 0) {
    console.warn(
      `${spec.unit} ${clipName}: resolved ${resolved} duplicate animation track${resolved === 1 ? '' : 's'}` +
        (droppedHelpers > 0 ? ` and dropped ${droppedHelpers} duplicate helper tracks` : ''),
    );
  }
  return canonical;
}

function weightedBonesByName(scene) {
  const bones = new Map();
  scene.traverse((object) => {
    if (!object.isSkinnedMesh) return;
    for (const bone of object.skeleton.bones) {
      if (!bones.has(bone.name)) bones.set(bone.name, bone);
    }
  });
  return bones;
}

function renameConflictingAnimationHelpers(scene, spec) {
  const weightedBones = weightedBonesByName(scene);
  const usedNames = new Set();
  scene.traverse((object) => {
    if (object.name) usedNames.add(object.name);
  });
  let renamed = 0;
  scene.traverse((object) => {
    const bone = weightedBones.get(object.name);
    if (!bone || object === bone) return;
    const original = object.name;
    let suffix = 1;
    let replacement = `${original}_helper`;
    while (usedNames.has(replacement)) {
      replacement = `${original}_helper_${++suffix}`;
    }
    object.name = replacement;
    usedNames.add(replacement);
    renamed++;
  });
  if (renamed === 0) {
    throw new Error(`${spec.unit}: expected conflicting helper and skin-joint names`);
  }
  console.warn(`${spec.unit}: renamed ${renamed} helpers that shadow animated skin joints`);
}

function affectsWeightedBones(target, weightedBones) {
  let affects = false;
  target.traverse((object) => {
    if (weightedBones.has(object)) affects = true;
  });
  return affects;
}

function transformDistance(value, rest, quaternion) {
  let direct = 0;
  let negated = 0;
  for (let index = 0; index < rest.length; index++) {
    direct += (value[index] - rest[index]) ** 2;
    if (quaternion) negated += (value[index] + rest[index]) ** 2;
  }
  return Math.sqrt(quaternion ? Math.min(direct, negated) : direct);
}

function strictlyIncreasingTrack(track) {
  let increasing = true;
  for (let index = 1; index < track.times.length; index++) {
    if (!(track.times[index] > track.times[index - 1])) {
      increasing = false;
      break;
    }
  }
  if (increasing) return track;

  const valueSize = track.getValueSize();
  const indices = [...track.times.keys()].sort(
    (left, right) => track.times[left] - track.times[right] || left - right,
  );
  const unique = [];
  for (const index of indices) {
    if (unique.length > 0 && track.times[unique[unique.length - 1]] === track.times[index]) {
      unique[unique.length - 1] = index;
    } else {
      unique.push(index);
    }
  }
  const Times = track.times.constructor;
  const Values = track.values.constructor;
  const times = new Times(unique.length);
  const values = new Values(unique.length * valueSize);
  unique.forEach((sourceIndex, outputIndex) => {
    times[outputIndex] = track.times[sourceIndex];
    values.set(
      track.values.subarray(sourceIndex * valueSize, (sourceIndex + 1) * valueSize),
      outputIndex * valueSize,
    );
  });
  return new track.constructor(track.name, times, values, track.getInterpolation());
}

function resampleTakeWindow(clip, timing, window) {
  const sourceDuration = window.stop - window.start;
  if (!(sourceDuration > 0)) {
    throw new Error(`${clip.name}: FBX take has an empty time window`);
  }
  const sampleCount = timing.frames + 1;
  clip.tracks = clip.tracks.map((track) => {
    const valueSize = track.getValueSize();
    const Times = track.times.constructor;
    const Values = track.values.constructor;
    const times = new Times(sampleCount);
    const values = new Values(sampleCount * valueSize);
    const interpolant = track.createInterpolant();
    for (let frame = 0; frame < sampleCount; frame++) {
      const ratio = frame / timing.frames;
      times[frame] = ratio * timing.duration;
      values.set(interpolant.evaluate(window.start + ratio * sourceDuration), frame * valueSize);
    }
    return new track.constructor(track.name, times, values, track.getInterpolation());
  });
  clip.duration = timing.duration;
}

function applyClipGeometryCorrections(clip, corrections, spec, clipName) {
  if (!corrections || corrections.size === 0) return;
  const duration = Math.max(clip.duration, 1 / 30);
  for (const [anchor, matrix] of corrections) {
    const animated = clip.tracks.filter((track) => {
      const parsed = THREE.PropertyBinding.parseTrackName(track.name);
      return parsed.nodeName === anchor;
    });
    if (animated.length > 0) {
      throw new Error(
        `${spec.unit} ${clipName}: bind anchor ${anchor} is animated and cannot use a static geometry correction`,
      );
    }
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    clip.tracks.push(
      constantTransformTrack(anchor, 'position', position, duration),
      constantTransformTrack(anchor, 'quaternion', quaternion, duration),
      constantTransformTrack(anchor, 'scale', scale, duration),
    );
  }
}

function completeMissingJointTracks(scene, sourceScene, clip, spec, clipName) {
  const tracked = new Set(
    clip.tracks.map((track) => {
      const parsed = THREE.PropertyBinding.parseTrackName(track.name);
      return `${parsed.nodeName}|${parsed.propertyName}`;
    }),
  );
  const duration = Math.max(clip.duration, 1 / 30);
  for (const joint of skinJointNames(sourceScene)) {
    const source = THREE.PropertyBinding.findNode(sourceScene, joint);
    const target = THREE.PropertyBinding.findNode(scene, joint);
    if (!source?.isBone) {
      throw new Error(
        `${spec.unit} ${clipName}: source skin joint ${joint} is absent from its scene`,
      );
    }
    if (!target?.isBone) {
      throw new Error(`${spec.unit} ${clipName}: geometry rig lacks source skin joint ${joint}`);
    }
    for (const property of ['position', 'quaternion', 'scale']) {
      const key = `${joint}|${property}`;
      if (tracked.has(key)) continue;
      clip.tracks.push(constantTransformTrack(joint, property, source[property], duration));
      tracked.add(key);
    }
  }
}

function constantTransformTrack(node, property, value, duration) {
  const item = value.toArray();
  const values = new Float32Array([...item, ...item]);
  const Track =
    property === 'quaternion' ? THREE.QuaternionKeyframeTrack : THREE.VectorKeyframeTrack;
  return new Track(`${node}.${property}`, new Float32Array([0, duration]), values);
}

function skinJointNames(scene) {
  const names = new Set();
  scene.traverse((object) => {
    if (!object.isSkinnedMesh) return;
    for (const bone of object.skeleton.bones) names.add(bone.name);
  });
  return names;
}

function frozenTrack(track) {
  const valueSize = track.getValueSize();
  const Values = track.values.constructor;
  const values = new Values(valueSize * 2);
  const first = track.values.subarray(0, valueSize);
  values.set(first, 0);
  values.set(first, valueSize);
  return new track.constructor(
    track.name,
    new Float32Array([0, 1 / 30]),
    values,
    THREE.InterpolateDiscrete,
  );
}

function retimeClip(clip, duration) {
  let firstTime = Infinity;
  for (const track of clip.tracks) {
    if (track.times.length > 0) firstTime = Math.min(firstTime, track.times[0]);
  }
  if (Number.isFinite(firstTime) && firstTime !== 0) {
    for (const track of clip.tracks) {
      for (let index = 0; index < track.times.length; index++) {
        track.times[index] -= firstTime;
      }
    }
  }
  clip.resetDuration();
  if (!(clip.duration > 0)) throw new Error(`${clip.name}: cannot retime an empty clip`);
  const scale = duration / clip.duration;
  for (const track of clip.tracks) track.scale(scale);
  clip.duration = duration;
}

function replaceMaterials(mesh) {
  const source = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  const clean = source.map(
    (material) =>
      new THREE.MeshStandardMaterial({
        name: material.name,
        color: material.color ?? 0xffffff,
        roughness: 1,
        metalness: 0,
      }),
  );
  mesh.material = Array.isArray(mesh.material) ? clean : clean[0];
}

function removeUnsupportedSceneObjects(scene) {
  const remove = [];
  scene.traverse((object) => {
    if (object.isLight || object.isCamera) {
      remove.push(object);
    } else if (object.isLine || object.isPoints) {
      // FBXLoader creates line objects as hierarchy helpers in a few rigs. They
      // can parent animated groups and even bones, so removing the object would
      // also remove real animation. Demote it to a transform-only node instead:
      // the hierarchy remains, while GLTFExporter no longer emits line geometry
      // the runtime would ignore.
      object.isLine = false;
      object.isLineSegments = false;
      object.isPoints = false;
      object.type = 'Group';
    }
  });
  for (const object of remove) object.removeFromParent();
}

async function validateAnimationInventory(root) {
  const actual = (await readdir(root, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && extname(entry.name).toLowerCase() === '.asset')
    .map((entry) => entry.name.slice(0, -'.asset'.length))
    .sort();
  const expected = ATHENA2_MODELS.map((spec) => spec.animationAsset).sort();
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  const missing = expected.filter((unit) => !actualSet.has(unit));
  const unexpected = actual.filter((unit) => !expectedSet.has(unit));
  if (missing.length > 0 || unexpected.length > 0) {
    throw new Error(
      `Athena2 animation inventory does not match the ${ATHENA2_MODELS.length}-unit manifest.` +
        (missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : '') +
        (unexpected.length > 0 ? ` Unexpected: ${unexpected.join(', ')}.` : ''),
    );
  }
}

async function readAuthoredAnimation(path, spec) {
  const wantedSlots = new Map([
    [1, 'run'],
    [2, 'attack'],
    [4, 'die'],
  ]);
  const records = new Map();
  let slot = -1;
  let variant = -1;
  let firstVariant = null;
  let readingRunVertices = false;
  const runMin = [Infinity, Infinity, Infinity];
  const runMax = [-Infinity, -Infinity, -Infinity];

  const finishSlot = () => {
    const name = wantedSlots.get(slot);
    if (name && firstVariant) records.set(name, firstVariant);
  };

  const lines = createInterface({
    input: createReadStream(path),
    crlfDelay: Infinity,
  });
  for await (const line of lines) {
    if (line.startsWith('  - Variants:')) {
      finishSlot();
      slot++;
      variant = -1;
      firstVariant = null;
      readingRunVertices = false;
      continue;
    }
    if (!wantedSlots.has(slot)) continue;
    const variantStart = /^    - (.*)$/.exec(line);
    if (variantStart) {
      variant++;
      readingRunVertices = false;
      if (variant === 0) firstVariant = { frames: 0, frameRate: 0 };
      const rate = /^FrameRate:\s*([\d.+-]+)\s*$/.exec(variantStart[1]);
      if (variant === 0 && rate) firstVariant.frameRate = Number(rate[1]);
      continue;
    }
    if (variant !== 0 || !firstVariant) continue;
    if (line.startsWith('      - Vertices:')) {
      firstVariant.frames++;
      // Frame zero is the stable same-pose scale reference. A union across the
      // full run can include authored root motion and would make movement
      // distance masquerade as body size in the gallery.
      readingRunVertices = slot === 1 && firstVariant.frames === 1 && !line.endsWith('[]');
      continue;
    }
    if (readingRunVertices) {
      const vertex = /^        - \{x:\s*([^,]+), y:\s*([^,]+), z:\s*([^}]+)\}\s*$/.exec(line);
      if (vertex) {
        const values = vertex.slice(1).map(Number);
        if (!values.every(Number.isFinite)) {
          throw new Error(`${spec.unit} run: invalid baked vertex in ${path}`);
        }
        for (let axis = 0; axis < 3; axis++) {
          runMin[axis] = Math.min(runMin[axis], values[axis]);
          runMax[axis] = Math.max(runMax[axis], values[axis]);
        }
        continue;
      }
      readingRunVertices = false;
    }
    const rate = /^      FrameRate:\s*([\d.+-]+)\s*$/.exec(line);
    if (rate) firstVariant.frameRate = Number(rate[1]);
  }
  finishSlot();

  const timings = {};
  for (const name of wantedSlots.values()) {
    const record = records.get(name);
    if (!record) {
      throw new Error(`${spec.unit} ${name}: missing Variant 0 in ${path}`);
    }
    if (record.frames > 1 && !(record.frameRate > 0)) {
      throw new Error(
        `${spec.unit} ${name}: ${record.frames} baked frames have an invalid ` +
          `frame rate (${record.frameRate})`,
      );
    }
    const isStatic = record.frames <= 1 || !(record.frameRate > 0);
    timings[name] = Object.freeze({
      static: isStatic,
      duration: isStatic ? 1 / 30 : record.frames / record.frameRate,
      frames: record.frames,
      frameRate: record.frameRate,
    });
  }
  const runSize = runMin.map((minimum, axis) => runMax[axis] - minimum);
  if (!runSize.every((size) => Number.isFinite(size) && size > 0)) {
    throw new Error(`${spec.unit} run: invalid baked bounds in ${path}`);
  }
  return Object.freeze({
    clips: Object.freeze(timings),
    runSize: Object.freeze(runSize),
  });
}

async function writeCatalog(output, timings, runSizes, completeModels) {
  const models = [];
  for (const spec of completeModels) {
    const modelPath = join(output, `${spec.slug}.glb`);
    try {
      await access(modelPath);
    } catch (error) {
      if (error.code === 'ENOENT') continue;
      throw error;
    }
    const runSize = runSizes.get(spec.animationAsset);
    const runGroundY = spec.runGround
      ? await deriveRunGroundY(modelPath, spec, timings.get(spec.animationAsset).run, runSize)
      : null;
    models.push({
      unit: spec.unit,
      faction: spec.faction,
      file: `${spec.slug}.glb`,
      skins: [`${spec.slug}-blue.ktx2`, `${spec.slug}-red.ktx2`],
      clips: timings.get(spec.animationAsset),
      runSize,
      ...(runGroundY === null ? {} : { runGroundY }),
    });
  }
  const catalogOrder = new Map(
    Object.values(ATHENA2_FACTIONS)
      .flat()
      .map((unit, index) => [unit, index]),
  );
  models.sort((left, right) => catalogOrder.get(left.unit) - catalogOrder.get(right.unit));
  const catalog = {
    version: 2,
    models,
  };
  await writeFile(join(output, 'all-units.json'), `${JSON.stringify(catalog, null, 2)}\n`);
}

/**
 * Resolve an authored foot margin into the coordinate system of the final GLB.
 * Source FBXs, Blender-normalized files, and Unity-sampled exports do not share
 * one origin, so a source-space pivot cannot safely be scaled at runtime. The
 * catalog instead records the desired ground plane directly in final model
 * space, after sampling the exact public run clip and skin.
 */
async function deriveRunGroundY(path, spec, runTiming, runSize) {
  const bytes = await readFile(path);
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const gltf = await new GLTFLoader().parseAsync(arrayBuffer, '');
  const meshes = [];
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh) meshes.push(object);
  });
  if (meshes.length !== 1) {
    throw new Error(
      `${spec.unit}: runGround requires exactly one skinned mesh, found ${meshes.length}`,
    );
  }
  const mesh = meshes[0];
  const position = mesh.geometry.getAttribute('position');
  const skinIndex = mesh.geometry.getAttribute('skinIndex');
  const skinWeight = mesh.geometry.getAttribute('skinWeight');
  if (!position || !skinIndex || !skinWeight) {
    throw new Error(`${spec.unit}: runGround mesh has incomplete skin data`);
  }
  const configuredVertexCount = mesh.userData.boundsVertexCount;
  const vertexCount =
    Number.isInteger(configuredVertexCount) &&
    configuredVertexCount > 0 &&
    configuredVertexCount <= position.count
      ? configuredVertexCount
      : position.count;
  const boneTokens = spec.runGround.bones.map((name) => name.toLowerCase());
  const groundBones = new Set();
  for (const [index, bone] of mesh.skeleton.bones.entries()) {
    const name = bone.name.toLowerCase();
    if (boneTokens.some((token) => name.includes(token))) {
      groundBones.add(index);
    }
  }
  if (groundBones.size === 0) {
    throw new Error(
      `${spec.unit}: no bones match runGround tokens ${spec.runGround.bones.join(', ')}`,
    );
  }
  const groundVertices = [];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    let weight = 0;
    for (let component = 0; component < 4; component++) {
      if (groundBones.has(skinIndex.getComponent(vertex, component))) {
        weight += skinWeight.getComponent(vertex, component);
      }
    }
    // Include vertices primarily controlled by a configured foot/ankle branch,
    // while excluding nearby body vertices with only incidental blended weight.
    if (weight >= 0.25) groundVertices.push(vertex);
  }
  if (groundVertices.length === 0) {
    throw new Error(`${spec.unit}: runGround matched no weighted vertices`);
  }

  const clip = gltf.animations.find((animation) => animation.name === 'run');
  if (!clip) throw new Error(`${spec.unit}: final GLB has no run clip`);
  const mixer = new THREE.AnimationMixer(gltf.scene);
  const action = mixer.clipAction(clip);
  action.play();
  const firstFrameBounds = new THREE.Box3();
  const point = new THREE.Vector3();
  let footMinY = Infinity;
  for (let frame = 0; frame < runTiming.frames; frame++) {
    // Matches AnimatedModel's gallery sampling: one pose at the start of every
    // baked frame interval, with the looping endpoint intentionally omitted.
    const time = (frame / runTiming.frames) * clip.duration;
    mixer.setTime(0);
    mixer.setTime(time);
    gltf.scene.updateMatrixWorld(true);
    mesh.skeleton.update();
    if (frame === 0) {
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        point.fromBufferAttribute(position, vertex);
        mesh.applyBoneTransform(vertex, point).applyMatrix4(mesh.matrixWorld);
        firstFrameBounds.expandByPoint(point);
      }
    }
    for (const vertex of groundVertices) {
      point.fromBufferAttribute(position, vertex);
      mesh.applyBoneTransform(vertex, point).applyMatrix4(mesh.matrixWorld);
      footMinY = Math.min(footMinY, point.y);
    }
  }
  mixer.stopAllAction();

  const modelSize = firstFrameBounds.getSize(new THREE.Vector3());
  const modelExtent = Math.max(modelSize.x, modelSize.y, modelSize.z);
  const authoredExtent = Math.max(...runSize);
  if (!(modelExtent > 0) || !(authoredExtent > 0) || !Number.isFinite(footMinY)) {
    throw new Error(`${spec.unit}: could not derive a finite run ground plane`);
  }
  const modelUnitsPerAthena2Unit = modelExtent / authoredExtent;
  const runGroundY = footMinY - spec.runGround.margin * modelUnitsPerAthena2Unit;
  const rounded = Number(runGroundY.toFixed(9));
  console.log(
    `  ${spec.unit}: run ground ${rounded} from ${groundVertices.length} weighted vertices`,
  );
  return rounded;
}

function parseArgs(values) {
  const parsed = {};
  for (let i = 0; i < values.length; i++) {
    const arg = values[i];
    if (!arg.startsWith('--')) throw new Error(`Unexpected argument: ${arg}`);
    const key = arg.slice(2);
    if (key === 'include-existing') {
      parsed[key] = 'true';
      continue;
    }
    const value = values[++i];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
    parsed[key] = value;
  }
  return parsed;
}

function required(values, key) {
  const value = values[key];
  if (!value) throw new Error(`Missing --${key}`);
  return value;
}

function installBrowserShims() {
  globalThis.window ??= globalThis;
  globalThis.ProgressEvent ??= class ProgressEvent {
    constructor(type, init = {}) {
      this.type = type;
      Object.assign(this, init);
    }
  };
  globalThis.document ??= {
    createElementNS() {
      const listeners = new Map();
      const image = {
        addEventListener(type, listener) {
          listeners.set(type, listener);
        },
        removeEventListener(type) {
          listeners.delete(type);
        },
      };
      Object.defineProperty(image, 'src', {
        set() {
          queueMicrotask(() => listeners.get('load')?.());
        },
      });
      return image;
    },
  };
  globalThis.FileReader ??= class FileReader {
    result = null;
    onloadend = null;

    readAsArrayBuffer(blob) {
      void blob.arrayBuffer().then((result) => {
        this.result = result;
        this.onloadend?.({ target: this });
      });
    }

    readAsDataURL(blob) {
      void blob.arrayBuffer().then((result) => {
        const type = blob.type || 'application/octet-stream';
        this.result = `data:${type};base64,${Buffer.from(result).toString('base64')}`;
        this.onloadend?.({ target: this });
      });
    }
  };
}
