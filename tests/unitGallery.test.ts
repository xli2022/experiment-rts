import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { loadAnimatedModel, type AnimatedModel } from '../src/render/models/animated.js';
import {
  galleryAnimationAt,
  previewGroundOffset,
  proportionalPreviewScale,
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

describe('unit gallery attack playback', () => {
  it('plays an attack once from frame zero before returning to the run loop', () => {
    expect(galleryAnimationAt(1.2, 10, 10)).toEqual({
      clip: 'attack',
      time: 0,
      loop: false,
      finished: false,
    });
    expect(galleryAnimationAt(1.2, 10, 10.75)).toMatchObject({
      clip: 'attack',
      time: 0.75,
      loop: false,
    });
    expect(galleryAnimationAt(1.2, 10, 11.2)).toMatchObject({
      clip: 'run',
      loop: true,
      finished: true,
    });
  });

  it('restarts an in-progress attack on every click timestamp', () => {
    expect(galleryAnimationAt(1.2, 4, 4.5).time).toBeCloseTo(0.5);
    expect(galleryAnimationAt(1.2, 4.5, 4.5)).toMatchObject({
      clip: 'attack',
      time: 0,
      loop: false,
    });
  });

  it('gracefully keeps running when a model has no attack clip', () => {
    expect(galleryAnimationAt(undefined, 2, 3)).toEqual({
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
