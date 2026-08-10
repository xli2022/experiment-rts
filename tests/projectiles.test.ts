import { describe, expect, it } from 'vitest';
import { EntityType } from '../src/sim/types.js';
import { FLIGHT_ALTITUDE, FLYER_BOB, flyerAltitudeAt } from '../src/render/entities.js';
import {
  interpolateProjectileTransform,
  projectileAgeBeforeFrameUpdate,
  projectileImpactPoint,
  projectileLaunchPoint,
} from '../src/render/projectiles.js';

describe('projectile hardpoints', () => {
  it('launches a Beamdrone bolt from its elevated underbody emitter', () => {
    const point = projectileLaunchPoint(EntityType.Beamdrone, 10, 20, 0, 1);

    expect(point.x).toBeCloseTo(9.97, 6);
    expect(point.y).toBeGreaterThan(FLIGHT_ALTITUDE);
    expect(point.z).toBeCloseTo(20.22, 6);
  });

  it('keeps the Beamdrone emitter attached throughout its hover bob', () => {
    const elapsedS = Math.PI / (2 * 2.2);
    const visualYOffset = flyerAltitudeAt(elapsedS, 0) - FLIGHT_ALTITUDE;
    const launch = projectileLaunchPoint(EntityType.Beamdrone, 10, 20, 0, 1, visualYOffset);
    const impact = projectileImpactPoint(EntityType.Beamdrone, 12, 20, visualYOffset);

    expect(visualYOffset).toBeCloseTo(FLYER_BOB, 8);
    expect(launch.y).toBeCloseTo(FLIGHT_ALTITUDE + 0.3 + FLYER_BOB, 8);
    expect(impact.y).toBeCloseTo(launch.y, 8);
  });

  it('launches a Burstbot bolt from its lower-right authored muzzle', () => {
    const north = projectileLaunchPoint(EntityType.Burstbot, 4, 7, 0, 1);
    const east = projectileLaunchPoint(EntityType.Burstbot, 4, 7, 1, 0);

    expect(north).toEqual({ x: 4.24, y: 0.63, z: 7.67 });
    expect(east).toEqual({ x: 4.67, y: 0.63, z: 6.76 });
  });

  it('normalises fixed-point facing before applying the offset', () => {
    const unit = projectileLaunchPoint(EntityType.Turret, 3, 5, 0, 1);
    const scaled = projectileLaunchPoint(EntityType.Turret, 3, 5, 0, 65_535 / 65_536);

    expect(scaled).toEqual(unit);
  });

  it('meets flying targets at flight height without lifting ground impacts', () => {
    expect(projectileImpactPoint(EntityType.Beamdrone, 8, 9)).toEqual({
      x: 8,
      y: FLIGHT_ALTITUDE + 0.3,
      z: 9,
    });
    expect(projectileImpactPoint(EntityType.Slicebot, 8, 9)).toEqual({
      x: 8,
      y: 0.5,
      z: 9,
    });
  });
});

describe('projectile frame timing', () => {
  it('uses the same interpolated position and facing as the visible entity', () => {
    const snapshot = {
      slot: 7,
      prevX: 10,
      prevZ: 20,
      prevFaceX: 0,
      prevFaceZ: 1,
      currX: 10.22,
      currZ: 20.1,
      currFaceX: 0.64,
      currFaceZ: 0.76,
    };

    expect(interpolateProjectileTransform(snapshot, 0)).toEqual({
      x: 10,
      z: 20,
      faceX: 0,
      faceZ: 1,
    });
    expect(interpolateProjectileTransform(snapshot, 0.5)).toEqual({
      x: 10.11,
      z: 20.05,
      faceX: 0.32,
      faceZ: 0.88,
    });
  });

  it('keeps the newest catch-up bolt alive through one long frame update', () => {
    const beforeUpdate = projectileAgeBeforeFrameUpdate(120, 120, 0, 250);
    expect(beforeUpdate + 0.25).toBeCloseTo(0, 8);

    const olderShot = projectileAgeBeforeFrameUpdate(118, 120, 0.5, 250);
    expect(olderShot + 0.25).toBeCloseTo(0.125, 8);
  });
});
