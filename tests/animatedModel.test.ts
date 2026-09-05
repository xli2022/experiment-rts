import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { AnimatedUnitPool } from '../src/render/animatedUnits.js';
import {
  animatedExtent,
  loadAnimatedModel,
  type AnimatedModel,
} from '../src/render/models/animated.js';

// Three's FileLoader reports fetch progress in browsers. Node supplies fetch
// and Event, but not ProgressEvent, so loading a data URL needs this small test
// environment shim.
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

const MODEL_PATH = fileURLToPath(new URL('../public/models/sword-machine.glb', import.meta.url));
const MOVING_MESH_PATH = fileURLToPath(
  new URL('../public/models/necromancer.glb', import.meta.url),
);
const FIRE_DRAGON_PATH = fileURLToPath(
  new URL('../public/models/fire-dragon.glb', import.meta.url),
);
const TREANT_PATH = fileURLToPath(new URL('../public/models/treant.glb', import.meta.url));
const STATIC_MESH_PATH = fileURLToPath(new URL('../public/models/slime.glb', import.meta.url));

let assetUrl: string;
let allClips: AnimatedModel;
let runOnly: AnimatedModel;
let runAndAttack: AnimatedModel;
let fireDragonAll: AnimatedModel;
let fireDragonRun: AnimatedModel;

beforeAll(async () => {
  const glb = await readFile(MODEL_PATH);
  assetUrl = `data:model/gltf-binary;base64,${glb.toString('base64')}`;
  [allClips, runOnly, runAndAttack] = await Promise.all([
    loadAnimatedModel(assetUrl),
    loadAnimatedModel(assetUrl, 'run'),
    loadAnimatedModel(assetUrl, {
      clips: ['run', 'attack'],
      boundsClip: 'run',
    }),
  ]);
  const fireDragon = await readFile(FIRE_DRAGON_PATH);
  const fireDragonUrl = `data:model/gltf-binary;base64,${fireDragon.toString('base64')}`;
  [fireDragonAll, fireDragonRun] = await Promise.all([
    loadAnimatedModel(fireDragonUrl),
    loadAnimatedModel(fireDragonUrl, 'run'),
  ]);
});

afterAll(() => {
  allClips?.boneTexture.dispose();
  runOnly?.boneTexture.dispose();
  runAndAttack?.geometry.dispose();
  runAndAttack?.boneTexture.dispose();
  fireDragonAll?.geometry.dispose();
  fireDragonAll?.boneTexture.dispose();
  fireDragonRun?.geometry.dispose();
  fireDragonRun?.boneTexture.dispose();
});

describe('animated model clip filtering and bounds', () => {
  it('bakes only the requested clip', () => {
    expect([...runOnly.clips.keys()]).toEqual(['run']);
    expect(runOnly.totalFrames).toBe(runOnly.clips.get('run')?.frameCount);
    expect(runOnly.totalFrames).toBeLessThan(allClips.totalFrames);
    expect([...allClips.clips.keys()].sort()).toEqual(['attack', 'die', 'run']);
  });

  it('bakes selected clips while retaining bounds from one framing clip', () => {
    expect([...runAndAttack.clips.keys()].sort()).toEqual(['attack', 'run']);
    expect(runAndAttack.totalFrames).toBe(
      runAndAttack.clips.get('run')!.frameCount + runAndAttack.clips.get('attack')!.frameCount,
    );
    expect(runAndAttack.totalFrames).toBeLessThan(allClips.totalFrames);

    for (const axis of ['x', 'y', 'z'] as const) {
      expect(runAndAttack.animatedBounds.min[axis]).toBeCloseTo(
        runOnly.animatedBounds.min[axis],
        9,
      );
      expect(runAndAttack.animatedBounds.max[axis]).toBeCloseTo(
        runOnly.animatedBounds.max[axis],
        9,
      );
      expect(runAndAttack.firstFrameBounds.min[axis]).toBeCloseTo(
        runOnly.firstFrameBounds.min[axis],
        9,
      );
      expect(runAndAttack.firstFrameBounds.max[axis]).toBeCloseTo(
        runOnly.firstFrameBounds.max[axis],
        9,
      );
    }
  });

  it('exposes the requested clip envelope with the node transform applied', () => {
    const size = runOnly.animatedBounds.getSize(runOnly.bindSize.clone());

    // SwordMachine carries a -90-degree X node transform. These are its
    // resulting Y-up run dimensions; sampling without nodeMatrix swaps the
    // vertical and depth axes.
    expect(size.x).toBeCloseTo(50.3, 0);
    expect(size.y).toBeCloseTo(58.9, 0);
    expect(size.z).toBeCloseTo(70.3, 0);
    expect(runOnly.lowestY).toBe(runOnly.animatedBounds.min.y);
  });

  it('exposes an exact first-frame scale reference inside the run envelope', () => {
    const size = runOnly.firstFrameBounds.getSize(runOnly.bindSize.clone());
    expect(Math.max(size.x, size.y, size.z)).toBeGreaterThan(0);
    for (const axis of ['x', 'y', 'z'] as const) {
      expect(runOnly.animatedBounds.min[axis]).toBeLessThanOrEqual(
        runOnly.firstFrameBounds.min[axis],
      );
      expect(runOnly.animatedBounds.max[axis]).toBeGreaterThanOrEqual(
        runOnly.firstFrameBounds.max[axis],
      );
    }
  });

  it('keeps the all-clip extent and lowest point for normal loads', () => {
    for (const axis of ['x', 'y', 'z'] as const) {
      expect(allClips.animatedBounds.min[axis]).toBeLessThanOrEqual(
        runOnly.animatedBounds.min[axis],
      );
      expect(allClips.animatedBounds.max[axis]).toBeGreaterThanOrEqual(
        runOnly.animatedBounds.max[axis],
      );
    }
    expect(allClips.lowestY).toBe(allClips.animatedBounds.min.y);
  });

  it('rejects a requested clip the asset does not contain', async () => {
    await expect(loadAnimatedModel(assetUrl, 'idle')).rejects.toThrow('contains no idle animation');
    await expect(
      loadAnimatedModel(assetUrl, {
        clips: ['run', 'idle'],
        boundsClip: 'run',
      }),
    ).rejects.toThrow('contains no idle animation');
  });
});

describe('animated bounds sampling order', () => {
  it('matches an indexed mesh to its draw-order-expanded equivalent', () => {
    const vertexCount = 900;
    const drawCount = vertexCount * 3;
    const positions = new Float32Array(vertexCount * 3);
    const skinIndices = new Uint16Array(vertexCount * 4);
    const skinWeights = new Float32Array(vertexCount * 4);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      positions[vertex * 3] = vertex;
      skinWeights[vertex * 4] = 1;
    }
    const indices = new Uint16Array(drawCount);
    for (let ordinal = 0; ordinal < drawCount; ordinal++) {
      indices[ordinal] = ordinal % vertexCount;
    }

    const indexed = new THREE.BufferGeometry();
    indexed.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    indexed.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
    indexed.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
    indexed.setIndex(new THREE.BufferAttribute(indices, 1));

    const expandedPositions = new Float32Array(drawCount * 3);
    const expandedSkinIndices = new Uint16Array(drawCount * 4);
    const expandedSkinWeights = new Float32Array(drawCount * 4);
    for (let ordinal = 0; ordinal < drawCount; ordinal++) {
      const vertex = indices[ordinal]!;
      expandedPositions[ordinal * 3] = positions[vertex * 3]!;
      expandedSkinIndices[ordinal * 4] = skinIndices[vertex * 4]!;
      expandedSkinWeights[ordinal * 4] = skinWeights[vertex * 4]!;
    }
    const expanded = new THREE.BufferGeometry();
    expanded.setAttribute('position', new THREE.BufferAttribute(expandedPositions, 3));
    expanded.setAttribute('skinIndex', new THREE.BufferAttribute(expandedSkinIndices, 4));
    expanded.setAttribute('skinWeight', new THREE.BufferAttribute(expandedSkinWeights, 4));

    const bones = new Float32Array(32);
    new THREE.Matrix4().identity().toArray(bones, 0);
    new THREE.Matrix4().makeTranslation(1000, 0, 0).toArray(bones, 16);
    const identity = new THREE.Matrix4();
    const indexedBounds = animatedExtent(
      indexed,
      bones,
      1,
      2,
      identity,
      identity,
      identity,
      vertexCount,
    );
    const expandedBounds = animatedExtent(
      expanded,
      bones,
      1,
      2,
      identity,
      identity,
      identity,
      drawCount,
    );

    expect(indexedBounds.firstFrameBounds.min.toArray()).toEqual(
      expandedBounds.firstFrameBounds.min.toArray(),
    );
    expect(indexedBounds.firstFrameBounds.max.toArray()).toEqual(
      expandedBounds.firstFrameBounds.max.toArray(),
    );
    expect(indexedBounds.bounds.min.toArray()).toEqual(expandedBounds.bounds.min.toArray());
    expect(indexedBounds.bounds.max.toArray()).toEqual(expandedBounds.bounds.max.toArray());
    // Sampling unique POSITION ordinal instead would include vertex 899 and
    // produce 1899, diverging from the expanded triangle-corner order.
    expect(indexedBounds.bounds.max.x).toBe(1896);
  });
});

describe('instanced animation shader', () => {
  it('splices baked skinning into the visible and shadow passes', () => {
    const material = new THREE.MeshLambertMaterial();
    const pool = new AnimatedUnitPool(runOnly, material, 1);
    try {
      const shader = {
        uniforms: {},
        vertexShader: [
          '#include <common>',
          '#include <beginnormal_vertex>',
          '#include <begin_vertex>',
        ].join('\n'),
      };
      material.onBeforeCompile(shader as never, {} as never);

      const normalStart = shader.vertexShader.indexOf('#include <beginnormal_vertex>');
      const skinNormal = shader.vertexShader.indexOf('mat4 bakedSkinMatrix');
      const positionStart = shader.vertexShader.indexOf('#include <begin_vertex>');
      const skinPosition = shader.vertexShader.indexOf('transformed = ( bakedSkinMatrix');
      expect(normalStart).toBeGreaterThanOrEqual(0);
      expect(skinNormal).toBeGreaterThan(normalStart);
      expect(positionStart).toBeGreaterThan(skinNormal);
      expect(skinPosition).toBeGreaterThan(positionStart);

      expect(pool.mesh.castShadow).toBe(true);
      expect(pool.mesh.receiveShadow).toBe(true);
      const depthMaterial = pool.mesh.customDepthMaterial;
      expect(depthMaterial).toBeInstanceOf(THREE.MeshDepthMaterial);
      if (!depthMaterial) throw new Error('animated pool is missing its shadow material');

      const depthShader = {
        uniforms: {},
        vertexShader: ['#include <common>', '#include <begin_vertex>'].join('\n'),
      };
      depthMaterial.onBeforeCompile(depthShader as never, {} as never);
      const depthStart = depthShader.vertexShader.indexOf('#include <begin_vertex>');
      const depthMatrix = depthShader.vertexShader.indexOf('mat4 bakedSkinMatrix');
      const depthPosition = depthShader.vertexShader.indexOf('transformed = ( bakedSkinMatrix');
      expect(depthStart).toBeGreaterThanOrEqual(0);
      expect(depthMatrix).toBeGreaterThan(depthStart);
      expect(depthPosition).toBeGreaterThan(depthMatrix);
      expect(depthShader.uniforms).toHaveProperty('boneTexture');
    } finally {
      pool.dispose();
    }
  });
});

describe('direct Unity-sampled FireDragon', () => {
  it('loads all 129 poses as a compact standard skeletal model', () => {
    expect(fireDragonAll.boneCount).toBe(66);
    expect(fireDragonAll.totalFrames).toBe(129);
    expect(fireDragonAll.geometry.getAttribute('position').count).toBe(2868);
    expect(fireDragonAll.geometry.getIndex()?.count).toBe(8751);
    expect(fireDragonAll.geometry.morphAttributes.position ?? []).toHaveLength(0);
    expect(fireDragonAll.geometry.morphAttributes.normal ?? []).toHaveLength(0);
    expect(
      Object.fromEntries([...fireDragonAll.clips].map(([name, clip]) => [name, clip.frameCount])),
    ).toEqual({ run: 40, attack: 60, die: 29 });
    expect(fireDragonAll.boneTexture.image.width).toBe(264);
    expect(fireDragonAll.boneTexture.image.height).toBe(129);
    expect(
      Array.from(fireDragonAll.boneTexture.image.data as Float32Array).every(Number.isFinite),
    ).toBe(true);
  });

  it('filters the gallery load to run while keeping exact skeletal bounds', () => {
    expect(fireDragonRun.totalFrames).toBe(40);
    expect([...fireDragonRun.clips.keys()]).toEqual(['run']);
    expect(fireDragonRun.boneTexture.image.width).toBe(264);
    expect(fireDragonRun.boneTexture.image.height).toBe(40);
    expect(fireDragonRun.nodeMatrix.equals(new THREE.Matrix4())).toBe(true);
    expect(fireDragonRun.bindMatrix.equals(new THREE.Matrix4())).toBe(true);
    expect(fireDragonRun.bindMatrixInverse.equals(new THREE.Matrix4())).toBe(true);
    expect(fireDragonRun.firstFrameBounds.min.toArray()).toEqual([
      -83.11652403185458, -183.02326535701494, -173.60866556299126,
    ]);
    expect(fireDragonRun.firstFrameBounds.max.toArray()).toEqual([
      82.13969509045329, 223.25840041882662, 85.15994120003276,
    ]);
    for (const axis of ['x', 'y', 'z'] as const) {
      expect(fireDragonRun.animatedBounds.min[axis]).toBeLessThanOrEqual(
        fireDragonRun.firstFrameBounds.min[axis],
      );
      expect(fireDragonRun.animatedBounds.max[axis]).toBeGreaterThanOrEqual(
        fireDragonRun.firstFrameBounds.max[axis],
      );
    }
    expect(fireDragonRun.geometry.getAttribute('position').getX(0)).toBeCloseTo(31.1666718, 6);
  });

  it('bakes the same run pose as stock Three skeletal rendering', async () => {
    const loaded = await loadStockAndBaked(FIRE_DRAGON_PATH, 'run', 28);
    try {
      expect(loaded.baked.boneCount).toBe(66);
      const bakedMatrices = loaded.baked.boneTexture.image.data as Float32Array;
      const bakedOffset = loaded.frame * loaded.baked.boneCount * 16;
      const stockMatrices = loaded.mesh.skeleton.boneMatrices!;
      for (
        let vertex = 0;
        vertex < loaded.mesh.geometry.getAttribute('position').count;
        vertex += 137
      ) {
        const stock = skinVertex(
          loaded.mesh.geometry,
          vertex,
          stockMatrices,
          0,
          loaded.mesh.bindMatrix,
          loaded.mesh.bindMatrixInverse,
          loaded.mesh.matrixWorld,
        );
        const baked = skinVertex(
          loaded.baked.geometry,
          vertex,
          bakedMatrices,
          bakedOffset,
          loaded.baked.bindMatrix,
          loaded.baked.bindMatrixInverse,
          loaded.baked.nodeMatrix,
        );
        expect(baked.distanceTo(stock)).toBeLessThan(5e-4);
      }
    } finally {
      loaded.baked.boneTexture.dispose();
    }
  });
});

describe('animated mesh-node transforms', () => {
  it('matches stock SkinnedMesh output when an animated joint owns the mesh', async () => {
    const loaded = await loadStockAndBaked(MOVING_MESH_PATH, 'die', 27);
    try {
      // Necromancer's renderer moves below its animated scene root, exercising
      // the correction needed when a mesh node itself changes during a clip.
      expect(matrixMaxDelta(loaded.restNode, loaded.mesh.matrixWorld)).toBeGreaterThan(1);

      const bakedMatrices = loaded.baked.boneTexture.image.data as Float32Array;
      const bakedOffset = loaded.frame * loaded.baked.boneCount * 16;
      const stockMatrices = loaded.mesh.skeleton.boneMatrices;
      expect(stockMatrices).toBeDefined();

      for (
        let vertex = 0;
        vertex < loaded.mesh.geometry.getAttribute('position').count;
        vertex += 433
      ) {
        const stock = skinVertex(
          loaded.mesh.geometry,
          vertex,
          stockMatrices!,
          0,
          loaded.mesh.bindMatrix,
          loaded.mesh.bindMatrixInverse,
          loaded.mesh.matrixWorld,
        );
        const baked = skinVertex(
          loaded.baked.geometry,
          vertex,
          bakedMatrices,
          bakedOffset,
          loaded.baked.bindMatrix,
          loaded.baked.bindMatrixInverse,
          loaded.baked.nodeMatrix,
        );
        expect(baked.distanceTo(stock)).toBeLessThan(2e-4);
      }
    } finally {
      loaded.baked.boneTexture.dispose();
    }
  });

  it("leaves a static mesh node's sampled bone matrices unchanged", async () => {
    const loaded = await loadStockAndBaked(STATIC_MESH_PATH, 'run', 7);
    try {
      expect(matrixMaxDelta(loaded.restNode, loaded.mesh.matrixWorld)).toBeLessThan(1e-8);
      const bakedMatrices = loaded.baked.boneTexture.image.data as Float32Array;
      const bakedOffset = loaded.frame * loaded.baked.boneCount * 16;
      const stockMatrices = loaded.mesh.skeleton.boneMatrices!;
      let maximum = 0;
      for (let index = 0; index < stockMatrices.length; index++) {
        maximum = Math.max(
          maximum,
          Math.abs(bakedMatrices[bakedOffset + index]! - stockMatrices[index]!),
        );
      }
      expect(maximum).toBeLessThan(1e-6);
    } finally {
      loaded.baked.boneTexture.dispose();
    }
  });
});

describe('authored bounds prefixes', () => {
  it('frames Treant by its body without removing the attack Log', async () => {
    const bytes = await readFile(TREANT_PATH);
    const url = `data:model/gltf-binary;base64,${bytes.toString('base64')}`;
    const [run, attack] = await Promise.all([
      loadAnimatedModel(url, 'run'),
      loadAnimatedModel(url, 'attack'),
    ]);
    try {
      const position = run.geometry.getAttribute('position');
      expect(position.count).toBe(6240);

      // These are Athena2's body-only first-run-frame dimensions in source FBX
      // units. The zero-scale Log collapses far away from the body during run;
      // including that invisible point would change X/Z and make the gallery
      // camera jump even though the rendered body is correct.
      const firstFrameSize = run.firstFrameBounds.getSize(new THREE.Vector3());
      expect(firstFrameSize.x).toBeCloseTo(182.56672, 3);
      expect(firstFrameSize.y).toBeCloseTo(175.2953, 3);
      expect(firstFrameSize.z).toBeCloseTo(185.27319, 3);
      expect(run.bindSize.x).toBeCloseTo(223.85519, 3);
      expect(run.bindSize.y).toBeCloseTo(168.18493, 3);
      expect(run.bindSize.z).toBeCloseTo(136.72067, 3);

      const runLogSize = bakedVertexBounds(run, 6060, 6240, 0).getSize(new THREE.Vector3());
      const attackLogSize = bakedVertexBounds(
        attack,
        6060,
        6240,
        Math.floor(attack.totalFrames / 2),
      ).getSize(new THREE.Vector3());
      expect(runLogSize.length()).toBeLessThan(1e-4);
      expect(Math.min(attackLogSize.x, attackLogSize.y, attackLogSize.z)).toBeGreaterThan(20);
    } finally {
      run.boneTexture.dispose();
      attack.boneTexture.dispose();
    }
  });
});

async function loadStockAndBaked(
  path: string,
  clipName: string,
  frame: number,
): Promise<{
  baked: AnimatedModel;
  mesh: THREE.SkinnedMesh;
  restNode: THREE.Matrix4;
  frame: number;
}> {
  const bytes = await readFile(path);
  const url = `data:model/gltf-binary;base64,${bytes.toString('base64')}`;
  const [baked, gltf] = await Promise.all([
    loadAnimatedModel(url, clipName),
    new GLTFLoader().loadAsync(url),
  ]);
  let mesh: THREE.SkinnedMesh | null = null;
  gltf.scene.traverse((object) => {
    if (!mesh && (object as THREE.SkinnedMesh).isSkinnedMesh) {
      mesh = object as THREE.SkinnedMesh;
    }
  });
  if (!mesh) throw new Error(`${path} contains no skinned mesh`);
  const skinned = mesh as THREE.SkinnedMesh;
  gltf.scene.updateMatrixWorld(true);
  const restNode = skinned.matrixWorld.clone();
  const clip = gltf.animations.find((candidate) => candidate.name === clipName);
  if (!clip) throw new Error(`${path} contains no ${clipName} animation`);
  const bakedClip = baked.clips.get(clipName);
  if (!bakedClip) throw new Error(`${path} did not bake ${clipName}`);
  if (frame >= bakedClip.frameCount) throw new Error('sample frame is out of range');

  const mixer = new THREE.AnimationMixer(gltf.scene);
  mixer.clipAction(clip).play();
  const time = (frame / bakedClip.frameCount) * clip.duration;
  mixer.setTime(0);
  mixer.setTime(time);
  gltf.scene.updateMatrixWorld(true);
  skinned.skeleton.update();
  return { baked, mesh: skinned, restNode, frame };
}

function skinVertex(
  geometry: THREE.BufferGeometry,
  vertex: number,
  matrices: Float32Array,
  matrixOffset: number,
  bindMatrix: THREE.Matrix4,
  bindMatrixInverse: THREE.Matrix4,
  nodeMatrix: THREE.Matrix4,
): THREE.Vector3 {
  const position = geometry.getAttribute('position');
  const skinIndex = geometry.getAttribute('skinIndex');
  const skinWeight = geometry.getAttribute('skinWeight');
  const blended = new THREE.Matrix4();
  blended.elements.fill(0);
  const bone = new THREE.Matrix4();
  for (let component = 0; component < 4; component++) {
    const weight = skinWeight.getComponent(vertex, component);
    if (weight === 0) continue;
    bone.fromArray(matrices, matrixOffset + skinIndex.getComponent(vertex, component) * 16);
    for (let element = 0; element < 16; element++) {
      blended.elements[element] += bone.elements[element]! * weight;
    }
  }
  blended.multiply(bindMatrix).premultiply(bindMatrixInverse);
  return new THREE.Vector3()
    .fromBufferAttribute(position as THREE.BufferAttribute, vertex)
    .applyMatrix4(blended)
    .applyMatrix4(nodeMatrix);
}

function bakedVertexBounds(
  model: AnimatedModel,
  startVertex: number,
  endVertex: number,
  frame: number,
): THREE.Box3 {
  const bounds = new THREE.Box3();
  const matrices = model.boneTexture.image.data as Float32Array;
  const matrixOffset = frame * model.boneCount * 16;
  for (let vertex = startVertex; vertex < endVertex; vertex++) {
    bounds.expandByPoint(
      skinVertex(
        model.geometry,
        vertex,
        matrices,
        matrixOffset,
        model.bindMatrix,
        model.bindMatrixInverse,
        model.nodeMatrix,
      ),
    );
  }
  return bounds;
}

function matrixMaxDelta(left: THREE.Matrix4, right: THREE.Matrix4): number {
  let maximum = 0;
  for (let index = 0; index < 16; index++) {
    maximum = Math.max(maximum, Math.abs(left.elements[index]! - right.elements[index]!));
  }
  return maximum;
}
