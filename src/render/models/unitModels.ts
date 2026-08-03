/**
 * Which authored model stands in for which unit, and how it is fitted.
 *
 * Everything asset-specific lives here — the file, the team skins, and the one
 * dimension each model is sized by — so the renderer stays generic and adding a
 * unit's model is adding a row to this table.
 */

import * as THREE from 'three';
import { EntityType } from '../../sim/types.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { loadAnimatedModel, type AnimatedModel } from './animated.js';

/**
 * How a model is scaled into world units.
 *
 * Which axis matters depends on the shape. A walker reads by how tall it stands,
 * so it is fitted on `y`; an aircraft reads by its wingspan, and fitting one on
 * height would size it by whatever fin happens to stick up.
 */
interface Fit {
  axis: 'x' | 'y' | 'z';
  /** Target size on that axis, in world units. One unit is one map tile. */
  target: number;
}

interface UnitModelSpec {
  type: EntityType;
  file: string;
  /** Team skin per player index. */
  skins: [string, string];
  fit: Fit;
}

/**
 * A tile is one world unit. The values are chosen against each other rather than
 * in isolation: the brawler is the heaviest thing on the field and reads as the
 * threat, the rifleman is a little smaller, and the gunship is fitted on its
 * wingspan so it stays a wide silhouette nothing on the ground shares.
 *
 * All three were then taken down a quarter. Sized against each other they were
 * right; sized against the buildings they were not, and a unit that comes up to
 * the roof of the barracks that trains it reads as a scale error rather than as
 * a big unit.
 */
const MODELS: UnitModelSpec[] = [
  {
    type: EntityType.Brawler,
    file: 'sword-machine.glb',
    skins: ['sword-machine-blue.ktx2', 'sword-machine-red.ktx2'],
    fit: { axis: 'y', target: 1.16 },
  },
  {
    type: EntityType.Rifleman,
    file: 'revolver.glb',
    skins: ['revolver-blue.ktx2', 'revolver-red.ktx2'],
    fit: { axis: 'y', target: 1.01 },
  },
  {
    type: EntityType.Gunship,
    file: 'beam-ship.glb',
    skins: ['beam-ship-blue.ktx2', 'beam-ship-red.ktx2'],
    fit: { axis: 'x', target: 1.43 },
  },
];

export interface LoadedUnitModel {
  type: EntityType;
  model: AnimatedModel;
  /** Team texture per player index, or null where none is available. */
  textures: (THREE.Texture | null)[];
  /** Multiplier taking the asset's own units to world units. */
  scale: number;
}

function assetUrl(name: string): string {
  // Pages serves the game from a subdirectory, so asset URLs need the same
  // prefix Vite gives the bundle.
  return `${import.meta.env.BASE_URL}models/${name}`;
}

/**
 * Load a team skin.
 *
 * The skins are KTX2/ETC1S: GPU-compressed, transcoded on the fly to whatever
 * format the device actually supports, and a tenth the size of the PNGs they
 * were encoded from. That matters here because six 1024-square hand-painted
 * atlases were eight megabytes of the download.
 *
 * `flipY` is *not* set. These models' UVs need a vertical flip, and a compressed
 * texture cannot be flipped as it uploads the way a PNG can — so the flip is
 * baked in by the encoder instead. See `scripts/encode-textures.mjs`.
 */
async function loadSkin(loader: KTX2Loader, file: string): Promise<THREE.Texture | null> {
  try {
    const tex = await loader.loadAsync(assetUrl(file));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  } catch {
    return null;
  }
}

/**
 * Load every authored unit model.
 *
 * Each resolves independently: one missing file leaves that unit on its
 * procedural stand-in rather than taking the others down with it. Skins are
 * optional the same way — without one the renderer falls back to flat team
 * colour, which is worse-looking but never blocks the unit from appearing.
 */
export async function loadUnitModels(
  renderer: THREE.WebGLRenderer,
): Promise<LoadedUnitModel[]> {
  // The transcoder picks a target format from what the GPU reports, so it needs
  // the renderer before it can decode anything. Its own WASM is left at the
  // default path: three.js resolves that against `import.meta.url`, so the
  // bundler emits and versions it. Pointing at a hand-copied one in `public/`
  // shipped the same 580 KB twice.
  const skinLoader = new KTX2Loader().detectSupport(renderer);

  const loaded = await Promise.all(
    MODELS.map(async (spec): Promise<LoadedUnitModel | null> => {
      try {
        const model = await loadAnimatedModel(assetUrl(spec.file));
        const textures = await Promise.all(spec.skins.map((f) => loadSkin(skinLoader, f)));
        const measured = Math.max(0.001, model.bindSize[spec.fit.axis]);
        return { type: spec.type, model, textures, scale: spec.fit.target / measured };
      } catch (err) {
        console.warn(`model ${spec.file} unavailable, keeping the procedural one`, err);
        return null;
      }
    }),
  );
  skinLoader.dispose();
  return loaded.filter((m): m is LoadedUnitModel => m !== null);
}
