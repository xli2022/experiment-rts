import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { loadAnimatedModel, type AnimatedModel } from '../src/render/models/animated.js';
import {
  galleryAnimationAt,
  galleryTapAt,
  galleryTapClips,
  GALLERY_TEAM_COLOURS,
  parseUnitCatalog,
  previewGroundOffset,
  proportionalPreviewScale,
  UnitGallery,
  type GalleryTapClip,
} from '../src/render/unitGallery.js';

if (typeof ProgressEvent === 'undefined') {
  Object.defineProperty(globalThis, 'ProgressEvent', {
    value: class extends Event {
      readonly lengthComputable: boolean;
      readonly loaded: number;
      readonly total: number;

      constructor(type: string, init: ProgressEventInit = {}) {
        super(type);
        this.lengthComputable = init.lengthComputable ?? false;
        this.loaded = init.loaded ?? 0;
        this.total = init.total ?? 0;
      }
    },
  });
}

const MODEL_ROOT = fileURLToPath(new URL('../public/units/', import.meta.url));
const GROUNDED_UNITS = new Map<string, readonly string[]>([
  ['FireDragon', ['foot', 'toe']],
  ['GriffinRider', ['foot', 'toe']],
  ['IceDragon', ['foot', 'toe']],
  ['SkeletalDragon', ['ankle', 'toe']],
  ['Sphinx', ['foot', 'toe']],
  ['Treant', ['ankle']],
]);

interface CatalogEntry {
  unit: string;
  file: string;
  runSize: [number, number, number];
  runGroundY?: number;
}

interface Catalog {
  models: CatalogEntry[];
}

afterEach(() => vi.restoreAllMocks());

describe('unit gallery team colours', () => {
  it('offers all four team colours for all 54 catalog models in palette order', async () => {
    const catalog = JSON.parse(await readFile(join(MODEL_ROOT, 'all-units.json'), 'utf8'));
    const models = parseUnitCatalog(catalog);
    expect(GALLERY_TEAM_COLOURS).toEqual(['Blue', 'Teal', 'Red', 'Orange']);
    expect(models).toHaveLength(54);
    for (const model of models) {
      expect(model.skins).toEqual(
        GALLERY_TEAM_COLOURS.map(
          (colour) => `${model.file.slice(0, -4)}-${colour.toLowerCase()}.ktx2`,
        ),
      );
    }
    const malformed = structuredClone(catalog);
    malformed.models[0].skins = [models[0]!.skins[0], models[0]!.skins[2]];
    expect(() => parseUnitCatalog(malformed)).toThrow('malformed');
    malformed.models[0].skins = [...models[0]!.skins].reverse();
    expect(() => parseUnitCatalog(malformed)).toThrow('malformed');
  });

  interface SkinPreview {
    texture: THREE.Texture | null;
    material: THREE.MeshLambertMaterial;
    skins: [string, string, string, string];
  }

  interface SkinController {
    session: { cancelled: boolean; completed: number; failed: number; total: number };
    status: { textContent: string };
    previews: Map<number, SkinPreview>;
    selectColour(skin: number): Promise<void>;
    updateProgress(session: SkinController['session']): void;
    acquireSkinLoader(): KTX2Loader;
    releaseSkinLoader(): void;
  }

  function skinController() {
    const gallery = new UnitGallery(
      {} as HTMLElement,
      {} as THREE.WebGLRenderer,
    ) as unknown as SkinController;
    const texture = new THREE.CompressedTexture([], 4, 4);
    const material = new THREE.MeshLambertMaterial({ map: texture });
    const preview: SkinPreview = {
      texture,
      material,
      skins: ['unit-blue.ktx2', 'unit-teal.ktx2', 'unit-red.ktx2', 'unit-orange.ktx2'],
    };
    gallery.previews.set(0, preview);
    gallery.session = { cancelled: false, completed: 1, failed: 0, total: 1 };
    gallery.status = { textContent: '' };
    vi.spyOn(KTX2Loader.prototype, 'detectSupport').mockImplementation(function (this: KTX2Loader) {
      return this;
    });
    return { gallery, preview, texture };
  }

  it('keeps the newest selected skin when earlier texture loads finish later', async () => {
    const { gallery, preview, texture } = skinController();
    const oldDispose = vi.spyOn(texture, 'dispose');
    const loaderDispose = vi.spyOn(KTX2Loader.prototype, 'dispose');
    const requests = new Map<string, (texture: THREE.CompressedTexture) => void>();
    vi.spyOn(KTX2Loader.prototype, 'loadAsync').mockImplementation(
      (url) => new Promise((resolve) => requests.set(url, resolve)),
    );
    const teal = new THREE.CompressedTexture([], 4, 4);
    const orange = new THREE.CompressedTexture([], 4, 4);
    const staleDispose = vi.spyOn(teal, 'dispose');
    const first = gallery.selectColour(1);
    const latest = gallery.selectColour(3);
    expect(KTX2Loader.prototype.detectSupport).toHaveBeenCalledOnce();
    requests.get('/units/unit-orange.ktx2')!(orange);
    await latest;
    expect(preview.material.map).toBe(orange);
    expect(preview.material.color.getHex()).toBe(0xffffff);
    expect(orange.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(loaderDispose).not.toHaveBeenCalled();
    requests.get('/units/unit-teal.ktx2')!(teal);
    await first;
    expect(preview.material.map).toBe(orange);
    expect(preview.texture).toBe(orange);
    expect(oldDispose).toHaveBeenCalledOnce();
    expect(staleDispose).toHaveBeenCalledOnce();
    expect(loaderDispose).toHaveBeenCalledOnce();
    preview.material.dispose();
    orange.dispose();
  });

  it('shares the initial transcoder with a reopened session until cancelled loads finish', async () => {
    const { gallery, preview, texture } = skinController();
    const loaderDispose = vi.spyOn(KTX2Loader.prototype, 'dispose');
    const requests = new Map<string, (texture: THREE.CompressedTexture) => void>();
    vi.spyOn(KTX2Loader.prototype, 'loadAsync').mockImplementation(
      (url) => new Promise((resolve) => requests.set(url, resolve)),
    );
    // The initial model worker holds a loader lease while a colour switch and
    // close/reopen overlap it. Cancelling a session cannot kill its tasks.
    gallery.acquireSkinLoader();
    const cancelled = gallery.selectColour(1);
    gallery.session.cancelled = true;
    gallery.session = { cancelled: false, completed: 1, failed: 0, total: 1 };
    const reopened = gallery.selectColour(3);
    expect(KTX2Loader.prototype.detectSupport).toHaveBeenCalledOnce();
    const orange = new THREE.CompressedTexture([], 4, 4);
    requests.get('/units/unit-orange.ktx2')!(orange);
    await reopened;
    const teal = new THREE.CompressedTexture([], 4, 4);
    const cancelledDispose = vi.spyOn(teal, 'dispose');
    requests.get('/units/unit-teal.ktx2')!(teal);
    await cancelled;
    expect(cancelledDispose).toHaveBeenCalledOnce();
    expect(preview.material.map).toBe(orange);
    expect(loaderDispose).not.toHaveBeenCalled();
    gallery.releaseSkinLoader();
    expect(loaderDispose).toHaveBeenCalledOnce();
    // Once the last lease finishes, a later open gets one fresh loader.
    gallery.acquireSkinLoader();
    expect(KTX2Loader.prototype.detectSupport).toHaveBeenCalledTimes(2);
    gallery.releaseSkinLoader();
    expect(loaderDispose).toHaveBeenCalledTimes(2);
    preview.material.dispose();
    texture.dispose();
    orange.dispose();
  });

  it('disposes a texture that finishes after the gallery session closes', async () => {
    const { gallery, preview, texture } = skinController();
    let resolve!: (texture: THREE.CompressedTexture) => void;
    vi.spyOn(KTX2Loader.prototype, 'loadAsync').mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const replacement = new THREE.CompressedTexture([], 4, 4);
    const dispose = vi.spyOn(replacement, 'dispose');
    const pending = gallery.selectColour(1);
    gallery.session.cancelled = true;
    resolve(replacement);
    await pending;
    expect(dispose).toHaveBeenCalledOnce();
    expect(preview.material.map).toBe(texture);
    preview.material.dispose();
    texture.dispose();
  });

  it('keeps the skin loading message until all existing previews have the new colour', async () => {
    const { gallery, preview } = skinController();
    gallery.session.completed = 0;
    let resolve!: (texture: THREE.CompressedTexture) => void;
    vi.spyOn(KTX2Loader.prototype, 'loadAsync').mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const replacement = new THREE.CompressedTexture([], 4, 4);
    const pending = gallery.selectColour(3);
    gallery.session.completed = 1;
    gallery.updateProgress(gallery.session);
    expect(gallery.status.textContent).toBe('Loading Orange team skins…');
    resolve(replacement);
    await pending;
    expect(gallery.status.textContent).toBe('1 models ready · Orange skins');
    preview.material.dispose();
    replacement.dispose();
  });

  it('uses the selected team fallback colour if its authored skin cannot load', async () => {
    const { gallery, preview, texture } = skinController();
    const dispose = vi.spyOn(texture, 'dispose');
    vi.spyOn(KTX2Loader.prototype, 'loadAsync').mockRejectedValue(new Error('missing skin'));
    await gallery.selectColour(1);
    expect(preview.texture).toBeNull();
    expect(preview.material.map).toBeNull();
    expect(preview.material.color.getHex()).toBe(0x35d6bd);
    expect(dispose).toHaveBeenCalledOnce();
    preview.material.dispose();
  });
});

describe('unit gallery proportional scale', () => {
  it("preserves Athena2's size ratio across different GLB source units", () => {
    const worldScale = 0.42;
    const authoredSmall = 0.75;
    const authoredLarge = 3.0;
    const glbSmall = 28;
    const glbLarge = 415;

    const renderedSmall = glbSmall * proportionalPreviewScale(authoredSmall, glbSmall, worldScale);
    const renderedLarge = glbLarge * proportionalPreviewScale(authoredLarge, glbLarge, worldScale);

    expect(renderedLarge / renderedSmall).toBeCloseTo(authoredLarge / authoredSmall, 10);
    expect(renderedSmall).not.toBeCloseTo(renderedLarge, 5);
  });

  it('retains animated-envelope grounding when no explicit plane is authored', () => {
    expect(previewGroundOffset(undefined, -12.5, 0.08)).toBe(1);
  });

  it("places each corrected unit's lowest run foot 0.02 Athena2 units above the grid", async () => {
    const catalog = JSON.parse(
      await readFile(join(MODEL_ROOT, 'all-units.json'), 'utf8'),
    ) as Catalog;
    for (const [unit, tokens] of GROUNDED_UNITS) {
      const entry = catalog.models.find((candidate) => candidate.unit === unit);
      expect(entry, unit).toBeDefined();
      expect(Number.isFinite(entry?.runGroundY), unit).toBe(true);
      const bytes = await readFile(join(MODEL_ROOT, entry!.file));
      const url = `data:model/gltf-binary;base64,${bytes.toString('base64')}`;
      const [model, gltf] = await Promise.all([
        loadAnimatedModel(url, 'run'),
        new GLTFLoader().loadAsync(url),
      ]);
      try {
        const mesh = firstSkinnedMesh(gltf.scene);
        const position = model.geometry.getAttribute('position');
        const skinIndex = model.geometry.getAttribute('skinIndex');
        const skinWeight = model.geometry.getAttribute('skinWeight');
        const configuredVertexCount = mesh.userData.boundsVertexCount;
        const vertexCount =
          Number.isInteger(configuredVertexCount) &&
          configuredVertexCount > 0 &&
          configuredVertexCount <= position.count
            ? configuredVertexCount
            : position.count;
        const groundBones = new Set(
          mesh.skeleton.bones
            .map((bone, index) => ({ index, name: bone.name.toLowerCase() }))
            .filter(({ name }) => tokens.some((token) => name.includes(token)))
            .map(({ index }) => index),
        );
        expect(groundBones.size, `${unit} ground bones`).toBeGreaterThan(0);
        const groundVertices: number[] = [];
        for (let vertex = 0; vertex < vertexCount; vertex++) {
          let weight = 0;
          for (let component = 0; component < 4; component++) {
            if (groundBones.has(skinIndex.getComponent(vertex, component))) {
              weight += skinWeight.getComponent(vertex, component);
            }
          }
          if (weight >= 0.25) groundVertices.push(vertex);
        }
        expect(groundVertices.length, `${unit} ground vertices`).toBeGreaterThan(0);

        const clip = model.clips.get('run')!;
        let footMinY = Infinity;
        for (let frame = 0; frame < clip.frameCount; frame++) {
          const ys = sampleVertexY(model, clip.startFrame + frame);
          for (const vertex of groundVertices) {
            footMinY = Math.min(footMinY, ys[vertex]);
          }
        }
        const modelSize = model.firstFrameBounds.getSize(new THREE.Vector3());
        const modelExtent = Math.max(modelSize.x, modelSize.y, modelSize.z);
        const authoredExtent = Math.max(...entry!.runSize);
        const worldScale = 0.42;
        const scale = proportionalPreviewScale(authoredExtent, modelExtent, worldScale);
        const renderedFootY =
          footMinY * scale +
          previewGroundOffset(entry!.runGroundY, model.animatedBounds.min.y, scale);
        expect(renderedFootY / worldScale, unit).toBeCloseTo(0.02, 6);
      } finally {
        model.geometry.dispose();
        model.boneTexture.dispose();
      }
    }
  });
});

describe('unit gallery tap playback', () => {
  it('retains run in the tap cycle for a model that rests in idle', () => {
    const model = {
      clips: new Map(['run', 'attack', 'die'].map((clip) => [clip, {}])),
    } as AnimatedModel;
    expect(galleryTapClips(model)).toEqual(['attack', 'die']);
    model.clips.set('idle', { startFrame: 0, frameCount: 90, duration: 3 });
    expect(galleryTapClips(model)).toEqual(['attack', 'die', 'run']);
  });

  it('shows an available idle and returns to it after a one-shot', () => {
    expect(galleryAnimationAt(null, undefined, 3.4, 'idle')).toEqual({
      clip: 'idle',
      time: 3.4,
      loop: true,
      finished: false,
    });
    for (const clip of ['attack', 'die'] as const) {
      const playing = { clip, startedAt: 2 };
      expect(galleryAnimationAt(playing, 1, 2.5, 'idle')).toMatchObject({
        clip,
        time: 0.5,
        loop: false,
        finished: false,
      });
      expect(galleryAnimationAt(playing, 1, 3, 'idle')).toMatchObject({
        clip: 'idle',
        time: 3,
        loop: true,
        finished: true,
      });
    }
  });

  it('gives each tap the next animation in turn, then starts the list over', () => {
    // The whole of the interaction: one control per card, and every animation
    // the unit has behind it.
    const taps: readonly GalleryTapClip[] = ['attack', 'die'];
    let cursor = 0;
    const played: string[] = [];
    for (let tap = 0; tap < 5; tap++) {
      const next = galleryTapAt(taps, cursor)!;
      played.push(next.clip);
      cursor = next.nextTap;
    }
    expect(played).toEqual(['attack', 'die', 'attack', 'die', 'attack']);
  });

  it('leaves a model that baked no tappable clip idling', () => {
    expect(galleryTapAt([], 0)).toBeNull();
  });

  it('plays a tapped clip once from frame zero before returning to the idle loop', () => {
    const attack = { clip: 'attack', startedAt: 10 } as const;
    expect(galleryAnimationAt(attack, 1.2, 10)).toEqual({
      clip: 'attack',
      time: 0,
      loop: false,
      finished: false,
    });
    expect(galleryAnimationAt(attack, 1.2, 10.75)).toMatchObject({
      clip: 'attack',
      time: 0.75,
      loop: false,
    });
    expect(galleryAnimationAt(attack, 1.2, 11.2)).toMatchObject({
      clip: 'run',
      loop: true,
      finished: true,
    });
  });

  it('plays whichever clip the tap chose, not only the attack', () => {
    // The second tap on a card plays its death, and it is a one-shot on the
    // same terms: play it through once, then back to idling.
    const die = { clip: 'die', startedAt: 3 } as const;
    const midway = galleryAnimationAt(die, 0.9, 3.4);
    expect(midway).toMatchObject({ clip: 'die', loop: false, finished: false });
    expect(midway.time).toBeCloseTo(0.4);
    expect(galleryAnimationAt(die, 0.9, 3.9)).toMatchObject({ clip: 'run', finished: true });
  });

  it('restarts an in-progress clip on every tap timestamp', () => {
    expect(galleryAnimationAt({ clip: 'attack', startedAt: 4 }, 1.2, 4.5).time).toBeCloseTo(0.5);
    expect(galleryAnimationAt({ clip: 'attack', startedAt: 4.5 }, 1.2, 4.5)).toMatchObject({
      clip: 'attack',
      time: 0,
      loop: false,
    });
  });

  it('gracefully keeps idling when nothing is playing, or the clip was never baked', () => {
    expect(galleryAnimationAt(null, undefined, 3)).toEqual({
      clip: 'run',
      time: 3,
      loop: true,
      finished: false,
    });
    expect(galleryAnimationAt({ clip: 'die', startedAt: 2 }, undefined, 3)).toEqual({
      clip: 'run',
      time: 3,
      loop: true,
      finished: false,
    });
  });
});

function firstSkinnedMesh(root: THREE.Object3D): THREE.SkinnedMesh {
  let result: THREE.SkinnedMesh | null = null;
  root.traverse((object) => {
    if (!result && (object as THREE.SkinnedMesh).isSkinnedMesh) {
      result = object as THREE.SkinnedMesh;
    }
  });
  if (!result) throw new Error('model has no skinned mesh');
  return result;
}

function sampleVertexY(model: AnimatedModel, frame: number): Float64Array {
  const position = model.geometry.getAttribute('position');
  const skinIndex = model.geometry.getAttribute('skinIndex');
  const skinWeight = model.geometry.getAttribute('skinWeight');
  const matrices = model.boneTexture.image.data as Float32Array;
  const frameOffset = frame * model.boneCount * 16;
  const blended = new THREE.Matrix4();
  const bone = new THREE.Matrix4();
  const point = new THREE.Vector3();
  const ys = new Float64Array(position.count);
  for (let vertex = 0; vertex < position.count; vertex++) {
    blended.elements.fill(0);
    for (let component = 0; component < 4; component++) {
      const weight = skinWeight.getComponent(vertex, component);
      if (weight === 0) continue;
      bone.fromArray(matrices, frameOffset + skinIndex.getComponent(vertex, component) * 16);
      for (let element = 0; element < 16; element++) {
        blended.elements[element] += bone.elements[element] * weight;
      }
    }
    blended.multiply(model.bindMatrix).premultiply(model.bindMatrixInverse);
    point
      .fromBufferAttribute(position as THREE.BufferAttribute, vertex)
      .applyMatrix4(blended)
      .applyMatrix4(model.nodeMatrix);
    ys[vertex] = point.y;
  }
  return ys;
}
