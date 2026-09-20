/**
 * Which authored model stands in for which unit, and how it is fitted.
 *
 * Everything asset-specific lives here — the file, the team skins, and the size
 * the artist drew it at — so the renderer stays generic and adding a unit's
 * model is adding a row to this table.
 */

import * as THREE from 'three';
import { EntityType } from '../../sim/types.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { loadAnimatedModel, type AnimatedModel } from './animated.js';

/**
 * One row per unit: the asset, its team skins, and how big the artist made it.
 *
 * `runSize` is copied from `public/units/all-units.json` — the size of the
 * model's first baked run frame in Athena2's **shared** world scale, which is
 * the one measurement that is comparable between two rigs.
 * `tests/modelAssets.test.ts` checks every row against the catalog, so a
 * mistyped number fails rather than silently resizing a unit.
 */
export interface UnitModelSpec {
  type: EntityType;
  file: string;
  /** Team skin per player index. */
  skins: [string, string];
  /** Authored size in Athena2 world units, from the catalog's `runSize`. */
  runSize: readonly [number, number, number];
  /**
   * Rare, deliberate departure from the shared scale. One means "as authored".
   *
   * Every value here needs a reason in a comment beside it, because each one is
   * a place where the roster stops being to the artists' scale.
   */
  emphasis?: number;
}

/**
 * World units per Athena2 world unit, for the whole robot line.
 *
 * ## Why one number and not a target per unit
 *
 * Each row used to name an axis and a size to fit it to, measured against
 * `AnimatedModel.bindSize`. That silently sized every unit by a different
 * constant. These rigs were exported at their own scales and their bind poses
 * disagree wildly with the pose they are actually seen in — the Dark Golem
 * stands with its arms straight out, the Ice Golem is a wide crouch, the
 * Piercebot's bind box is 231 times its authored size where the Slicebot's is
 * 32. Fitting one axis of that measures the pose and the exporter, not the
 * unit, so no two rows meant the same thing and the roster came out with a
 * Dark Golem nearly three tiles across and a Boomwalker narrower than its own
 * collision circle.
 *
 * `runSize` is the fix: the catalog carries every model measured in one shared
 * scale, which is exactly why the gallery renders its cards from it. Multiply
 * it by one constant and the ladder between units is the one the artists drew.
 *
 * 0.78 is that constant. Two things pin it. A unit's footprint should land near
 * its collision diameter, and across the twelve `2 * radius / max(runSize.x,
 * runSize.z)` runs from 0.54 to 1.13 with the middle around 0.71. It should
 * also keep the two sizes this game had already settled on by eye before the
 * roster grew — a Slicebot 1.16 tall wants 0.85, a Beamdrone with a 1.43-tile
 * wingspan wants 0.69 — and 0.78 is the middle of those. The tallest ground
 * unit ends up at 1.44, comfortably under the 1.9 roof of the Barracks that
 * trains it, which is the check that catches a scale error from across the map.
 */
const ROBOT_SCALE = 0.78;

export const UNIT_MODELS: readonly UnitModelSpec[] = [
  // Barracks.
  {
    type: EntityType.Burstbot,
    file: 'revolver.glb',
    skins: ['revolver-blue.ktx2', 'revolver-red.ktx2'],
    runSize: [0.94, 0.88, 1.12],
  },
  {
    type: EntityType.Slicebot,
    file: 'sword-machine.glb',
    skins: ['sword-machine-blue.ktx2', 'sword-machine-red.ktx2'],
    runSize: [1.1, 1.36, 1.75],
  },
  {
    type: EntityType.Boomwalker,
    file: 'bomb.glb',
    skins: ['bomb-blue.ktx2', 'bomb-red.ktx2'],
    runSize: [0.49, 1.18, 0.71],
    // Authored, it is by some way the narrowest thing in the line — 0.38 of a
    // tile across, half its own collision circle. A unit whose entire job is to
    // be seen coming and stepped away from cannot be the one you do not notice.
    emphasis: 1.25,
  },
  {
    type: EntityType.Beamdrone,
    file: 'beam-ship.glb',
    skins: ['beam-ship-blue.ktx2', 'beam-ship-red.ktx2'],
    runSize: [2.07, 1.35, 1.38],
  },
  {
    type: EntityType.Fixomatic,
    file: 'healing-machine.glb',
    skins: ['healing-machine-blue.ktx2', 'healing-machine-red.ktx2'],
    runSize: [0.95, 1.35, 1.25],
  },
  {
    type: EntityType.Firespout,
    file: 'flamethrower.glb',
    skins: ['flamethrower-blue.ktx2', 'flamethrower-red.ktx2'],
    runSize: [0.82, 1.32, 1.29],
  },
  // Foundry.
  {
    type: EntityType.Piercebot,
    file: 'ballista.glb',
    skins: ['ballista-blue.ktx2', 'ballista-red.ktx2'],
    runSize: [1.01, 1.01, 1.75],
  },
  {
    type: EntityType.Arclight,
    file: 'tesla-coil.glb',
    skins: ['tesla-coil-blue.ktx2', 'tesla-coil-red.ktx2'],
    runSize: [1.48, 1.37, 1.07],
  },
  {
    type: EntityType.Sentry,
    file: 'cannon.glb',
    skins: ['cannon-blue.ktx2', 'cannon-red.ktx2'],
    runSize: [1.12, 1.71, 1.19],
  },
  {
    type: EntityType.DarkGolem,
    file: 'dark-golem.glb',
    skins: ['dark-golem-blue.ktx2', 'dark-golem-red.ktx2'],
    runSize: [2.06, 1.84, 1.55],
  },
  {
    type: EntityType.IceGolem,
    file: 'ice-golem.glb',
    skins: ['ice-golem-blue.ktx2', 'ice-golem-red.ktx2'],
    runSize: [2.51, 1.68, 1.21],
  },
  {
    type: EntityType.Plasmodrone,
    file: 'flying-machine.glb',
    skins: ['flying-machine-blue.ktx2', 'flying-machine-red.ktx2'],
    runSize: [1.85, 1.69, 1.43],
    // The one place the art and the cost list disagree. Authored, the heavy
    // gunship has a narrower span than the Beamdrone it shares the sky with,
    // and two flyers are told apart by width in the one glance a fight allows —
    // so the 4-supply one is made to be the wider of them.
    emphasis: 1.2,
  },
];

export interface LoadedUnitModel {
  type: EntityType;
  model: AnimatedModel;
  /** Team texture per player index, or null where none is available. */
  textures: (THREE.Texture | null)[];
  /** Multiplier taking the asset's own units to world units. */
  scale: number;
  /**
   * How tall the unit stands once scaled, in world units.
   *
   * Measured from the authored run size rather than from the bind pose, because
   * the health bar hangs off the top of what is actually drawn and the bind
   * pose is not it — several of these rigs bind with their arms overhead.
   */
  height: number;
}

function assetUrl(name: string): string {
  // Pages serves the game from a subdirectory, so asset URLs need the same
  // prefix Vite gives the bundle.
  return `${import.meta.env.BASE_URL}units/${name}`;
}

/**
 * Load a team skin.
 *
 * The skins are KTX2/ETC1S: GPU-compressed, transcoded on the fly to whatever
 * format the device actually supports, and a tenth the size of the PNGs they
 * were encoded from. That matters here because the hand-painted source atlases
 * add several megabytes per unit before compression.
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
export async function loadUnitModels(renderer: THREE.WebGLRenderer): Promise<LoadedUnitModel[]> {
  // The transcoder picks a target format from what the GPU reports, so it needs
  // the renderer before it can decode anything. Its own WASM is left at the
  // default path: three.js resolves that against `import.meta.url`, so the
  // bundler emits and versions it. Pointing at a hand-copied one in `public/`
  // shipped the same 580 KB twice.
  const skinLoader = new KTX2Loader().detectSupport(renderer);

  const loaded = await Promise.all(
    UNIT_MODELS.map(async (spec): Promise<LoadedUnitModel | null> => {
      try {
        // Framed on the run, not on everything baked: a death sprawl and an
        // overhead swing both reach further than the unit ever looks, and
        // sizing by them would shrink the body to fit its own weapon. It is
        // also the pose `runSize` was measured in, which is what makes the two
        // comparable at all.
        const model = await loadAnimatedModel(assetUrl(spec.file), { boundsClip: 'run' });
        const textures = await Promise.all(spec.skins.map((f) => loadSkin(skinLoader, f)));

        // Asset units to Athena2 units, then Athena2 units to world units. The
        // first ratio is per rig and undoes whatever scale it was exported at;
        // the second is one constant for the whole line.
        const drawn = model.firstFrameBounds.getSize(new THREE.Vector3());
        const measured = Math.max(0.001, drawn.x, drawn.y, drawn.z);
        const authored = Math.max(...spec.runSize);
        const worldScale = ROBOT_SCALE * (spec.emphasis ?? 1);

        return {
          type: spec.type,
          model,
          textures,
          scale: (authored / measured) * worldScale,
          height: spec.runSize[1] * worldScale,
        };
      } catch (err) {
        console.warn(`model ${spec.file} unavailable, keeping the procedural one`, err);
        return null;
      }
    }),
  );
  skinLoader.dispose();
  return loaded.filter((m): m is LoadedUnitModel => m !== null);
}
