import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { EntityRenderer } from '../src/render/entities.js';
import { ProceduralModelProvider } from '../src/render/models/procedural.js';
import { fromFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType } from '../src/sim/types.js';

describe('production building models', () => {
  it.each([EntityType.Barracks, EntityType.Factory])(
    'shows the second-level structure only on upgraded building %s',
    (type) => {
      const provider = new ProceduralModelProvider();
      const { world } = new Simulation(1);
      const renderer = new EntityRenderer(provider, world);
      const spec = provider.get(type);
      const slot = world.pool.spawn(type, 0, fromFloat(40), fromFloat(40)) & 0xffff;
      world.pool.buildState[slot] = BuildState.Complete;
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      camera.position.set(40, 20, 50);
      camera.lookAt(40, 0, 40);
      const countFor = (geometry: THREE.BufferGeometry): number =>
        renderer.group.children.reduce((count, child) => {
          const mesh = child as THREE.InstancedMesh;
          return count + (mesh.geometry === geometry ? mesh.count : 0);
        }, 0);
      try {
        expect(spec.parts.some((part) => part.minLevel === 2)).toBe(true);
        renderer.captureSnapshot(world);
        renderer.update(world, 1, new Set(), camera);
        for (const part of spec.parts) expect(countFor(part.geometry)).toBe(part.minLevel ? 0 : 1);
        world.pool.upgrading[slot] = 1;
        renderer.update(world, 1, new Set(), camera);
        for (const part of spec.parts.filter((part) => part.minLevel))
          expect(countFor(part.geometry)).toBe(0);
        world.pool.upgrading[slot] = 0;
        world.pool.buildingLevel[slot] = 2;
        renderer.update(world, 1, new Set(), camera);
        for (const part of spec.parts) expect(countFor(part.geometry)).toBe(1);
      } finally {
        renderer.dispose();
        provider.dispose();
      }
    },
  );

  it('keeps every production silhouette inside its build footprint and health bar', () => {
    const provider = new ProceduralModelProvider();
    try {
      for (const type of [EntityType.Barracks, EntityType.Factory, EntityType.Airport]) {
        const spec = provider.get(type);
        const half = defOf(type).footprint / 2;
        for (const part of spec.parts) {
          part.geometry.computeBoundingBox();
          const matrix = new THREE.Matrix4().compose(
            new THREE.Vector3(...part.offset),
            new THREE.Quaternion().setFromEuler(new THREE.Euler(...(part.rotation ?? [0, 0, 0]))),
            new THREE.Vector3(1, 1, 1),
          );
          const bounds = part.geometry.boundingBox!.clone().applyMatrix4(matrix);
          expect(bounds.min.x).toBeGreaterThanOrEqual(-half);
          expect(bounds.max.x).toBeLessThanOrEqual(half);
          expect(bounds.min.z).toBeGreaterThanOrEqual(-half);
          expect(bounds.max.z).toBeLessThanOrEqual(half);
          expect(bounds.min.y).toBeGreaterThanOrEqual(-0.001);
          expect(bounds.max.y).toBeLessThanOrEqual(
            part.minLevel ? spec.level2Height! : spec.height,
          );
        }
      }
      expect(provider.get(EntityType.Airport).parts.every((part) => !part.minLevel)).toBe(true);
    } finally {
      provider.dispose();
    }
  });
});
