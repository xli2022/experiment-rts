import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { EntityRenderer } from '../src/render/entities.js';
import type { AnimatedModel } from '../src/render/models/animated.js';
import { PLAYER_COLOURS, ProceduralModelProvider } from '../src/render/models/procedural.js';
import { coopMatch, duelMatch } from '../src/sim/match.js';
import { EntityType, type MatchConfig } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

function model(): AnimatedModel {
  return {
    geometry: new THREE.BoxGeometry(),
    boneTexture: new THREE.DataTexture(new Float32Array(16), 4, 1),
    boneCount: 1,
    totalFrames: 1,
    bindMatrix: new THREE.Matrix4(),
    bindMatrixInverse: new THREE.Matrix4(),
    nodeMatrix: new THREE.Matrix4(),
    clips: new Map(),
    bounds: new THREE.Box3(),
    animatedBounds: new THREE.Box3(),
    firstFrameBounds: new THREE.Box3(),
    lowestY: 0,
    bindSize: new THREE.Vector3(1, 1, 1),
  };
}

function withMaterials(
  config: MatchConfig,
  textures: (THREE.Texture | null)[],
  check: (materials: THREE.MeshLambertMaterial[]) => void,
): void {
  const entities = new EntityRenderer(new ProceduralModelProvider(), new World(config));
  const unit = model();
  const existing = new Set(entities.group.children);
  try {
    entities.useAnimatedModel(EntityType.Burstbot, unit, 1, 1, textures);
    const meshes = entities.group.children.filter((child) => !existing.has(child));
    expect(meshes).toHaveLength(config.teams.length);
    check(meshes.map((mesh) => (mesh as THREE.Mesh).material as THREE.MeshLambertMaterial));
  } finally {
    entities.dispose();
    unit.geometry.dispose();
    unit.boneTexture.dispose();
    for (const texture of textures) texture?.dispose();
  }
}

describe('authored player skins', () => {
  it('keeps a duel blue versus red with neutral material multiplication', () => {
    const textures = Array.from({ length: 4 }, () => new THREE.Texture());
    withMaterials(duelMatch(1), textures, (materials) => {
      expect(materials.map((material) => material.map)).toEqual([textures[0], textures[2]]);
      expect(materials.map((material) => material.color.getHex())).toEqual([0xffffff, 0xffffff]);
    });
  });

  it('gives co-op partners their own blue, teal, red and orange textures', () => {
    const textures = Array.from({ length: 4 }, () => new THREE.Texture());
    withMaterials(coopMatch(1), textures, (materials) => {
      expect(materials.map((material) => material.map)).toEqual(textures);
      expect(materials.every((material) => material.color.getHex() === 0xffffff)).toBe(true);
    });
  });

  it('uses flat player colours only for missing skins', () => {
    const textures = [new THREE.Texture(), null, new THREE.Texture(), null];
    withMaterials(coopMatch(1), textures, (materials) => {
      expect(materials.map((material) => material.map)).toEqual(textures);
      expect(materials.map((material) => material.color.getHex())).toEqual([
        0xffffff,
        PLAYER_COLOURS[1],
        0xffffff,
        PLAYER_COLOURS[3],
      ]);
    });
  });
});
