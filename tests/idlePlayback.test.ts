import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { EntityRenderer } from '../src/render/entities.js';
import type { AnimatedModel } from '../src/render/models/animated.js';
import { ProceduralModelProvider } from '../src/render/models/procedural.js';
import { fromInt } from '../src/sim/fixed.js';
import { EntityType } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

function fixture() {
  const world = new World(1);
  const id = world.pool.spawn(EntityType.Slicebot, 0, fromInt(40), fromInt(40));
  const slot = id & 0xffff;
  const renderer = new EntityRenderer(new ProceduralModelProvider(), world);
  // Distinct frame ranges expose the actual clip sent to the instanced shader.
  const model: AnimatedModel = {
    geometry: new THREE.BoxGeometry(),
    boneTexture: new THREE.DataTexture(new Float32Array(16 * 210), 4, 210),
    boneCount: 1,
    totalFrames: 210,
    bindMatrix: new THREE.Matrix4(),
    bindMatrixInverse: new THREE.Matrix4(),
    nodeMatrix: new THREE.Matrix4(),
    clips: new Map([
      ['run', { startFrame: 0, frameCount: 60, duration: 2 }],
      ['attack', { startFrame: 60, frameCount: 30, duration: 1 }],
      ['idle', { startFrame: 90, frameCount: 120, duration: 4 }],
    ]),
    bounds: new THREE.Box3(),
    animatedBounds: new THREE.Box3(),
    firstFrameBounds: new THREE.Box3(),
    lowestY: 0,
    bindSize: new THREE.Vector3(1, 1, 1),
  };
  const existing = new Set(renderer.group.children);
  renderer.useAnimatedModel(EntityType.Slicebot, model, 1, 1, []);
  const mesh = renderer.group.children.find((child) => !existing.has(child)) as THREE.InstancedMesh;
  const camera = new THREE.PerspectiveCamera();

  return {
    world,
    renderer,
    id,
    slot,
    draw(tick: number, alpha = 0) {
      world.tick = tick;
      renderer.update(world, alpha, new Set(), camera);
      expect(mesh.count).toBe(1);
      return {
        from: mesh.geometry.getAttribute('aFrame').getX(0),
        to: mesh.geometry.getAttribute('aFrameTo').getX(0),
        blend: mesh.geometry.getAttribute('aBlend').getX(0),
      };
    },
    move() {
      world.pool.posX[slot]! += fromInt(1);
      renderer.captureSnapshot(world);
    },
    attack(tick: number) {
      world.tick = tick;
      world.pool.attackWindup[slot] = 1;
      world.events.attackStarts.push(slot, slot);
      renderer.noteEvents(world, tick / 20);
      world.events.attackStarts.length = 0;
    },
    dispose() {
      renderer.dispose();
      model.geometry.dispose();
      model.boneTexture.dispose();
    },
  };
}

describe('Slicebot renderer idle playback', () => {
  it('advances idle frames while stationary and switches to run only while moving', () => {
    const f = fixture();
    try {
      f.draw(0);
      const idle = f.draw(10);
      expect(idle.from).toBeGreaterThanOrEqual(90);
      expect(idle.to).toBeLessThan(210);
      expect(f.draw(15).from).not.toBe(idle.from);

      f.move();
      f.draw(16);
      const running = f.draw(20);
      expect(running.from).toBeLessThan(60);
      expect(running.to).toBeLessThan(60);

      f.renderer.captureSnapshot(f.world);
      f.draw(21);
      expect(f.draw(25).from).toBeGreaterThanOrEqual(90);
    } finally {
      f.dispose();
    }
  });

  it('prioritizes attacks over movement and fades from the outgoing clip time', () => {
    const f = fixture();
    try {
      f.draw(0);
      const idle = f.draw(20);
      f.move();
      f.attack(24);
      expect(f.draw(24)).toEqual({ from: idle.from, to: 60, blend: 0 });
      const midway = f.draw(25, 0.2);
      expect(midway.from).toBe(idle.from);
      expect(midway.to).toBe(61);
      expect(midway.blend).toBeCloseTo(0.5);
      const attack = f.draw(28);
      expect(attack.from).toBeGreaterThanOrEqual(60);
      expect(attack.to).toBeLessThan(90);

      f.renderer.captureSnapshot(f.world);
      const followThrough = f.draw(43);
      expect(followThrough.from).toBe(88);
      const backToIdle = f.draw(44);
      expect(backToIdle.from).toBe(followThrough.from);
      expect(backToIdle.to).toBeGreaterThanOrEqual(90);
      expect(backToIdle.blend).toBe(0);
      expect(f.draw(48).from).toBeGreaterThanOrEqual(90);
    } finally {
      f.dispose();
    }
  });

  it('does not inherit attack playback or transition history when a slot is reused', () => {
    const f = fixture();
    try {
      f.draw(0);
      f.draw(20);
      f.attack(24);
      f.draw(24);
      expect(f.draw(30).from).toBeGreaterThanOrEqual(60);

      f.world.pool.destroy(f.id);
      f.renderer.captureSnapshot(f.world);
      const replacement = f.world.pool.spawn(EntityType.Slicebot, 0, fromInt(50), fromInt(50));
      expect(replacement & 0xffff).toBe(f.slot);
      f.renderer.captureSnapshot(f.world);
      const first = f.draw(31);
      expect(first.from).toBe(0);
      expect(first.to).toBeGreaterThanOrEqual(90);
      expect(first.blend).toBe(0);
      expect(f.draw(35).from).toBeGreaterThanOrEqual(90);
    } finally {
      f.dispose();
    }
  });
});
