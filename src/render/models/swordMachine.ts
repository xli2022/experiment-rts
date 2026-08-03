/**
 * The Sword Machine: the melee unit's authored model.
 *
 * An FBX rig with run, attack and death clips, converted to one GLB and baked
 * into an instanced skinned draw at load. Everything asset-specific lives here —
 * where the file is, how big it is in world units, and which texture each team
 * wears — so the renderer stays generic and a different model is a different
 * constant rather than a different code path.
 */

import * as THREE from 'three';
import { EntityType } from '../../sim/types.js';
import { loadAnimatedModel, type AnimatedModel } from './animated.js';

/** Which unit this model stands in for. */
export const SWORD_MACHINE_TYPE = EntityType.Brawler;

/**
 * World-unit height to draw it at.
 *
 * A tile is 1 world unit and the unit it replaces stood 1.3 tall. A little
 * taller than that, because a melee bruiser reads as a threat mostly through
 * silhouette. The asset's own size is measured at load rather than written down,
 * so re-exporting it at a different scale changes nothing here.
 */
const TARGET_HEIGHT = 1.55;

function assetUrl(name: string): string {
  // Pages serves the game from a subdirectory, so asset URLs need the same
  // prefix Vite gives the bundle.
  return `${import.meta.env.BASE_URL}models/${name}`;
}

export interface SwordMachineAssets {
  model: AnimatedModel;
  /** Team texture per player index, or null where none is available. */
  textures: (THREE.Texture | null)[];
  /** Multiplier taking the asset's own units to world units. */
  scale: number;
}

/**
 * Load the model and its team textures.
 *
 * Textures are optional: if a team's skin is missing the renderer falls back to
 * flat team colour, which is worse-looking but never blocks the unit from
 * appearing at all.
 */
export async function loadSwordMachine(): Promise<SwordMachineAssets> {
  const model = await loadAnimatedModel(assetUrl('sword-machine.glb'));

  const loader = new THREE.TextureLoader();
  const textures = await Promise.all(
    ['sword-machine-blue.png', 'sword-machine-red.png'].map(async (file) => {
      try {
        const tex = await loader.loadAsync(assetUrl(file));
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.flipY = false; // glTF UVs, unlike three's image default
        tex.needsUpdate = true;
        return tex;
      } catch {
        return null;
      }
    }),
  );

  return { model, textures, scale: TARGET_HEIGHT / Math.max(0.001, model.height) };
}
