import { describe, expect, it } from 'vitest';
import { EntityType } from '../src/sim/types.js';
import { Simulation } from '../src/sim/tick.js';
import { FogRenderer } from '../src/render/fog.js';
import { ProceduralModelProvider } from '../src/render/models/procedural.js';
import {
  EntityRenderer,
  FLIGHT_ALTITUDE,
  FLYER_BOB,
  flyerAltitudeAt,
} from '../src/render/entities.js';
import {
  interpolateProjectileTransform,
  primaryVictimIndex,
  projectileAgeBeforeFrameUpdate,
  projectileImpactPoint,
  projectileLaunchPoint,
  ProjectileRenderer,
} from '../src/render/projectiles.js';
import { fromFloat } from '../src/sim/fixed.js';
import { weaponProfileFor } from '../src/render/vfx/weapons.js';

describe('effects obey fog of war', () => {
  it('hides unseen enemy battles and deaths, but retains fire at a friendly unit', () => {
    const { world } = new Simulation(1);
    const pool = world.pool;
    const own = Array.from({ length: pool.count }, (_, i) => i).find(
      (i) => pool.owner[i] === 0 && pool.type[i] === EntityType.Worker,
    )!;
    const enemy = Array.from({ length: pool.count }, (_, i) => i).find(
      (i) => pool.owner[i] === 1 && pool.type[i] === EntityType.Worker,
    )!;
    const fog = new FogRenderer(world.map);
    const entities = new EntityRenderer(new ProceduralModelProvider(), world);
    const effects = new ProjectileRenderer();
    try {
      fog.update(world, 0);
      entities.captureSnapshot(world);
      const canSee = (index: number): boolean => fog.shouldDraw(world, index, 0);
      expect(canSee(enemy)).toBe(false);
      world.events.shots.push(enemy, enemy);
      world.events.deaths.push(enemy);
      effects.captureFromEvents(world, entities, canSee);
      effects.spawnDeaths(world, canSee);
      effects.flushPending(world.tick, 0, 0, 16);
      effects.update(16);
      expect(effects.group.children.every((mesh) => (mesh as { count?: number }).count === 0)).toBe(
        true,
      );

      world.events.shots.length = 0;
      world.events.shots.push(enemy, own);
      effects.captureFromEvents(world, entities, canSee);
      effects.flushPending(world.tick, 0, 0, 16);
      effects.update(16);
      expect(effects.group.children.some((mesh) => (mesh as { count?: number }).count! > 0)).toBe(
        true,
      );
    } finally {
      effects.dispose();
      entities.dispose();
      fog.dispose();
    }
  });
});

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

describe('one attack is one shot', () => {
  /**
   * Stand an attacker of `type` at the origin facing +Z, put `victims` targets
   * in front of it, and raise the (attacker, victim) pairs the simulation would
   * raise for one attack that reached all of them.
   */
  function fireOnce(
    type: EntityType,
    victims: readonly { x: number; z: number }[],
  ): ProjectileRenderer {
    const { world } = new Simulation(1);
    const pool = world.pool;
    const attacker = pool.spawn(type, 0, fromFloat(20), fromFloat(20)) & 0xffff;
    pool.faceX[attacker] = 0;
    pool.faceY[attacker] = 65_536;

    const targets = victims.map(
      (at) => pool.spawn(EntityType.Burstbot, 1, fromFloat(at.x), fromFloat(at.z)) & 0xffff,
    );

    const entities = new EntityRenderer(new ProceduralModelProvider(), world);
    const effects = new ProjectileRenderer();
    entities.captureSnapshot(world);
    entities.captureSnapshot(world);
    for (const target of targets) world.events.shots.push(attacker, target);
    effects.captureFromEvents(world, entities);
    effects.flushPending(world.tick, 0, 0, 16);
    entities.dispose();
    return effects;
  }

  it("sends one shell at a splash weapon's victims, not one each", () => {
    // What the simulation reports for a Sentry shell that caught five units:
    // five pairs, all from the same attacker, in the same tick.
    const effects = fireOnce(EntityType.Sentry, [
      { x: 20, z: 27 },
      { x: 21.4, z: 27.6 },
      { x: 18.7, z: 26.4 },
      { x: 20.6, z: 28.1 },
      { x: 19.2, z: 28 },
    ]);
    try {
      expect(effects.liveEffectCounts().ordnance).toBe(1);
    } finally {
      effects.dispose();
    }
  });

  it('gives the Arclight a bolt per enemy, because its coils really do', () => {
    const effects = fireOnce(EntityType.Arclight, [
      { x: 20, z: 24 },
      { x: 22, z: 23 },
      { x: 18, z: 23.5 },
    ]);
    try {
      const counts = effects.liveEffectCounts();
      // Lightning is drawn as a chain of short ribbons, so three enemies is
      // three chains of several segments each — and no travelling ordnance.
      expect(counts.ordnance).toBe(0);
      expect(counts.beams).toBeGreaterThanOrEqual(3 * 4);
    } finally {
      effects.dispose();
    }
  });

  it("puts nothing in the air for a weapon swung at arm's length", () => {
    const effects = fireOnce(EntityType.DarkGolem, [{ x: 20, z: 21.1 }]);
    try {
      expect(weaponProfileFor(EntityType.DarkGolem).style).toBe('melee');
      expect(effects.liveEffectCounts().ordnance).toBe(0);
    } finally {
      effects.dispose();
    }
  });
});

describe('the victim an attack was aimed at', () => {
  it('is the one lined up with where the attacker is pointing', () => {
    // Facing +Z. The second entry is dead ahead; the others are splash.
    const victims = [
      { x: 3, z: 6 },
      { x: 0, z: 7 },
      { x: -2.5, z: 6.5 },
    ];

    expect(primaryVictimIndex(0, 0, 0, 1, victims)).toBe(1);
    // Turn the shooter toward the first, and the answer turns with it.
    expect(primaryVictimIndex(0, 0, 0.45, 0.89, victims)).toBe(0);
  });

  it('is unchanged by a facing that fixed point left off unit length', () => {
    const victims = [
      { x: 4, z: 1 },
      { x: 0.5, z: 5 },
    ];

    expect(primaryVictimIndex(0, 0, 0, 65_535 / 65_536, victims)).toBe(
      primaryVictimIndex(0, 0, 0, 1, victims),
    );
  });

  it('answers with the first candidate when the geometry has no answer', () => {
    expect(primaryVictimIndex(5, 5, 0, 1, [{ x: 5, z: 5 }])).toBe(0);
  });
});

describe('effects survive a frame longer than they are', () => {
  it('spends a burst from an older tick rather than replaying it', () => {
    const { world } = new Simulation(1);
    const pool = world.pool;
    const attacker = pool.spawn(EntityType.Burstbot, 0, fromFloat(20), fromFloat(20)) & 0xffff;
    pool.faceX[attacker] = 0;
    pool.faceY[attacker] = 65_536;
    const target = pool.spawn(EntityType.Burstbot, 1, fromFloat(20), fromFloat(24)) & 0xffff;

    const entities = new EntityRenderer(new ProceduralModelProvider(), world);
    const effects = new ProjectileRenderer();
    try {
      entities.captureSnapshot(world);
      entities.captureSnapshot(world);
      world.events.shots.push(attacker, target);
      effects.captureFromEvents(world, entities);
      effects.flushPending(world.tick, 0, 0, 16);
      effects.update(16);
      expect(effects.liveEffectCounts().ordnance).toBe(1);

      // A single frame long enough to cover the whole flight lands the shot
      // rather than leaving it hanging in the air.
      effects.update(250);
      const after = effects.liveEffectCounts();
      expect(after.ordnance).toBe(0);
      expect(after.particles).toBeGreaterThan(0);
    } finally {
      effects.dispose();
      entities.dispose();
    }
  });
});
