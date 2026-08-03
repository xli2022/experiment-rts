/**
 * Which authored model stands in for which unit, and how it is fitted.
 *
 * Everything asset-specific lives here — the file, the team skins, and the one
 * dimension each model is sized by — so the renderer stays generic and adding a
 * unit's model is adding a row to this table.
 */

import * as THREE from 'three';
import { EntityType } from '../../sim/types.js';
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
 */
const MODELS: UnitModelSpec[] = [
  {
    type: EntityType.Brawler,
    file: 'sword-machine.glb',
    skins: ['sword-machine-blue.png', 'sword-machine-red.png'],
    fit: { axis: 'y', target: 1.55 },
  },
  {
    type: EntityType.Rifleman,
    file: 'revolver.glb',
    skins: ['revolver-blue.png', 'revolver-red.png'],
    fit: { axis: 'y', target: 1.35 },
  },
  {
    type: EntityType.Gunship,
    file: 'beam-ship.glb',
    skins: ['beam-ship-blue.png', 'beam-ship-red.png'],
    fit: { axis: 'x', target: 1.9 },
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

async function loadSkin(file: string): Promise<THREE.Texture | null> {
  try {
    const tex = await new THREE.TextureLoader().loadAsync(assetUrl(file));
    tex.colorSpace = THREE.SRGBColorSpace;
    // `flipY` stays at three's default. A texture authored for glTF would want
    // it off, but these GLBs were exported from FBX and kept three's UV
    // convention — verified by painting a marker at a known pixel of the real
    // atlas and checking it landed on the model's eye.
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
export async function loadUnitModels(): Promise<LoadedUnitModel[]> {
  const loaded = await Promise.all(
    MODELS.map(async (spec): Promise<LoadedUnitModel | null> => {
      try {
        const model = await loadAnimatedModel(assetUrl(spec.file));
        const textures = await Promise.all(spec.skins.map(loadSkin));
        const measured = Math.max(0.001, model.bindSize[spec.fit.axis]);
        return { type: spec.type, model, textures, scale: spec.fit.target / measured };
      } catch (err) {
        console.warn(`model ${spec.file} unavailable, keeping the procedural one`, err);
        return null;
      }
    }),
  );
  return loaded.filter((m): m is LoadedUnitModel => m !== null);
}
