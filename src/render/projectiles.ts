/**
 * Transient effects: weapon tracers, impact flashes, death debris and the marker
 * that confirms a click landed.
 *
 * Purely cosmetic. The simulation already resolves a shot the instant it fires —
 * damage is applied immediately, and `world.events.shots` records who hit whom
 * that tick. These visuals are read off that list *after* the fact, so a peer
 * that never draws them plays an identical game.
 *
 * That separation is why this file is free to use wall-clock time, floating
 * point and `Math.random`, all of which are banned inside `src/sim/**`. Nothing
 * here can feed back into the simulation.
 *
 * Ranged attacks get a bolt that travels from shooter to target; melee gets a
 * flash at the point of contact, because a tracer across half a metre reads as a
 * glitch rather than a hit.
 */

import * as THREE from 'three';
import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import { EntityType } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { PLAYER_COLOURS } from './models/procedural.js';

/** Simultaneous effects of each kind. Beyond this, new ones replace the oldest. */
const CAPACITY = 512;

/**
 * Seconds a bolt takes to cross from shooter to target.
 *
 * Long enough to actually register at RTS camera distance. Much below this and
 * the bolt exists for a single frame, which reads as a flicker rather than a
 * shot.
 */
const BOLT_TRAVEL_S = 0.13;
/** Seconds an impact flash lives. */
const FLASH_LIFE_S = 0.22;

/** Below this attack range a shot is melee and gets no travelling bolt. */
const MELEE_RANGE = 1.5;

/** Seconds a death burst lives. Longer than a hit, so a kill reads as a kill. */
const DEBRIS_LIFE_S = 0.75;
/** Chunks thrown out per destroyed entity. */
const DEBRIS_PER_DEATH = 7;
/** Seconds the click marker lives. */
const MARKER_LIFE_S = 0.45;

interface Bolt {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  y: number;
  age: number;
  colour: number;
  active: boolean;
}

interface Flash {
  x: number;
  y: number;
  z: number;
  age: number;
  colour: number;
  active: boolean;
}

/** A tumbling chunk thrown out when something dies. */
interface Debris {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  spin: number;
  age: number;
  life: number;
  size: number;
  colour: number;
  active: boolean;
}

/** The expanding ring drawn where the player clicked. */
interface Marker {
  x: number;
  z: number;
  age: number;
  colour: number;
  active: boolean;
}

export class ProjectileRenderer {
  readonly group = new THREE.Group();

  private readonly bolts: Bolt[] = [];
  private readonly flashes: Flash[] = [];
  private readonly debris: Debris[] = [];
  private readonly markers: Marker[] = [];
  private boltCursor = 0;
  private flashCursor = 0;
  private debrisCursor = 0;
  private markerCursor = 0;

  private readonly boltMesh: THREE.InstancedMesh;
  private readonly flashMesh: THREE.InstancedMesh;
  private readonly debrisMesh: THREE.InstancedMesh;
  private readonly markerMesh: THREE.InstancedMesh;
  private readonly disposables: { dispose(): void }[] = [];

  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3();
  private readonly colour = new THREE.Color();
  private readonly dir = new THREE.Vector3();

  constructor() {
    // A thin box stretched along its travel direction reads as a bolt without
    // needing a texture or a shader.
    const boltGeo = new THREE.BoxGeometry(0.13, 0.13, 1);
    const boltMat = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false });
    this.boltMesh = new THREE.InstancedMesh(boltGeo, boltMat, CAPACITY);

    const flashGeo = new THREE.SphereGeometry(0.18, 8, 6);
    const flashMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.85,
      depthTest: false,
    });
    this.flashMesh = new THREE.InstancedMesh(flashGeo, flashMat, CAPACITY);

    // Small tumbling cubes, thrown out on death.
    const debrisGeo = new THREE.BoxGeometry(1, 1, 1);
    const debrisMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.debrisMesh = new THREE.InstancedMesh(debrisGeo, debrisMat, CAPACITY);

    // A flat ring on the ground confirming where an order was given.
    const markerGeo = new THREE.RingGeometry(0.55, 0.78, 24);
    markerGeo.rotateX(-Math.PI / 2);
    const markerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
      side: THREE.DoubleSide,
    });
    this.markerMesh = new THREE.InstancedMesh(markerGeo, markerMat, 64);

    for (const mesh of [this.boltMesh, this.flashMesh, this.debrisMesh, this.markerMesh]) {
      mesh.frustumCulled = false;
      mesh.count = 0;
      mesh.renderOrder = 9;
      this.group.add(mesh);
    }
    this.disposables.push(
      boltGeo, boltMat, flashGeo, flashMat, debrisGeo, debrisMat, markerGeo, markerMat,
    );

    for (let i = 0; i < CAPACITY; i++) {
      this.bolts.push({ x0: 0, z0: 0, x1: 0, z1: 0, y: 0, age: 0, colour: 0, active: false });
      this.flashes.push({ x: 0, y: 0, z: 0, age: 0, colour: 0, active: false });
      this.debris.push({
        x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
        spin: 0, age: 0, life: 0, size: 0.2, colour: 0, active: false,
      });
    }
    for (let i = 0; i < 64; i++) {
      this.markers.push({ x: 0, z: 0, age: 0, colour: 0, active: false });
    }
  }

  /**
   * Throw debris where something died.
   *
   * Buildings get more, bigger, slower chunks than infantry — a Command Post
   * falling should not look like a rifleman being shot.
   */
  spawnDeaths(world: World): void {
    const pool = world.pool;
    const deaths = world.events.deaths;

    for (let k = 0; k < deaths.length; k++) {
      const i = deaths[k]!;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      // Exhausted mineral patches vanish rather than exploding.
      if (type === EntityType.MineralPatch) continue;

      const owner = pool.owner[i]!;
      const colour = PLAYER_COLOURS[owner] ?? 0x9aa4b2;
      const x = toFloat(pool.posX[i]!);
      const z = toFloat(pool.posY[i]!);
      const big = def.isBuilding;
      const count = big ? DEBRIS_PER_DEATH * 2 : DEBRIS_PER_DEATH;

      for (let n = 0; n < count; n++) {
        const d = this.debris[this.debrisCursor]!;
        this.debrisCursor = (this.debrisCursor + 1) % CAPACITY;
        const angle = Math.random() * Math.PI * 2;
        const speed = (big ? 2.2 : 3.0) * (0.4 + Math.random() * 0.8);
        d.x = x;
        d.y = big ? 0.8 : 0.4;
        d.z = z;
        d.vx = Math.cos(angle) * speed;
        d.vz = Math.sin(angle) * speed;
        d.vy = (big ? 3.4 : 2.8) * (0.5 + Math.random() * 0.8);
        d.spin = (Math.random() - 0.5) * 14;
        d.age = 0;
        d.life = DEBRIS_LIFE_S * (0.7 + Math.random() * 0.6);
        d.size = (big ? 0.34 : 0.17) * (0.6 + Math.random() * 0.8);
        d.colour = colour;
        d.active = true;
      }

      // A flash at the same spot sells the impact.
      const flash = this.flashes[this.flashCursor]!;
      this.flashCursor = (this.flashCursor + 1) % CAPACITY;
      flash.x = x;
      flash.y = big ? 0.9 : 0.5;
      flash.z = z;
      flash.age = 0;
      flash.colour = 0xffcc66;
      flash.active = true;
    }
  }

  /**
   * Drop a ring where the player clicked.
   *
   * Order feedback is easy to skip and disproportionately important: without it
   * there is no way to tell a missed click from a unit that simply has not
   * started moving yet.
   */
  spawnClickMarker(x: number, z: number, colour: number): void {
    const m = this.markers[this.markerCursor]!;
    this.markerCursor = (this.markerCursor + 1) % this.markers.length;
    m.x = x;
    m.z = z;
    m.age = 0;
    m.colour = colour;
    m.active = true;
  }

  /**
   * Turn this tick's shots into effects. Call once per simulation tick.
   *
   * `events.shots` holds flat (attacker, target) slot-index pairs and is cleared
   * at the start of the next step, so this has to run while it is still valid.
   */
  spawnFromEvents(world: World): void {
    const shots = world.events.shots;
    const pool = world.pool;

    for (let k = 0; k + 1 < shots.length; k += 2) {
      const attacker = shots[k]!;
      const target = shots[k + 1]!;
      if (pool.alive[attacker] !== 1) continue;

      const def = defOf(pool.type[attacker]! as EntityType);
      const owner = pool.owner[attacker]!;
      const colour = PLAYER_COLOURS[owner] ?? 0xffffff;

      const ax = toFloat(pool.posX[attacker]!);
      const az = toFloat(pool.posY[attacker]!);
      const tx = toFloat(pool.posX[target]!);
      const tz = toFloat(pool.posY[target]!);

      // Fire from roughly weapon height rather than the ground.
      const ay = pool.type[attacker] === EntityType.Turret ? 0.75 : 0.55;

      if (toFloat(def.attackRange) >= MELEE_RANGE) {
        const bolt = this.bolts[this.boltCursor]!;
        this.boltCursor = (this.boltCursor + 1) % CAPACITY;
        bolt.x0 = ax;
        bolt.z0 = az;
        bolt.x1 = tx;
        bolt.z1 = tz;
        bolt.y = ay;
        bolt.age = 0;
        bolt.colour = colour;
        bolt.active = true;
      }

      const flash = this.flashes[this.flashCursor]!;
      this.flashCursor = (this.flashCursor + 1) % CAPACITY;
      flash.x = tx;
      flash.y = 0.5;
      flash.z = tz;
      // Delay the flash until the bolt would arrive, so it lands on impact
      // rather than at the moment of firing.
      flash.age = toFloat(def.attackRange) >= MELEE_RANGE ? -BOLT_TRAVEL_S : 0;
      flash.colour = colour;
      flash.active = true;
    }
  }

  /** Advance and rebuild the instance buffers. Call once per rendered frame. */
  update(dtMs: number): void {
    const dt = Math.min(dtMs, 250) / 1000;
    let boltCount = 0;
    let flashCount = 0;

    for (const bolt of this.bolts) {
      if (!bolt.active) continue;
      bolt.age += dt;
      if (bolt.age >= BOLT_TRAVEL_S) {
        bolt.active = false;
        continue;
      }
      if (boltCount >= CAPACITY) continue;

      // Draw a short segment travelling along the path rather than a line
      // spanning the whole distance — it reads as a projectile, not a laser.
      const t = bolt.age / BOLT_TRAVEL_S;
      const dx = bolt.x1 - bolt.x0;
      const dz = bolt.z1 - bolt.z0;
      const dist = Math.hypot(dx, dz);
      if (dist < 1e-4) continue;

      const headT = Math.min(1, t + 0.18);
      const cx = bolt.x0 + dx * ((t + headT) / 2);
      const cz = bolt.z0 + dz * ((t + headT) / 2);
      const length = Math.max(0.25, dist * (headT - t));

      this.dir.set(dx / dist, 0, dz / dist);
      this.quat.setFromUnitVectors(FORWARD, this.dir);
      this.position.set(cx, bolt.y, cz);
      this.scale.set(1, 1, length);
      this.matrix.compose(this.position, this.quat, this.scale);
      this.boltMesh.setMatrixAt(boltCount, this.matrix);
      this.colour.setHex(bolt.colour).lerp(WHITE, 0.45);
      this.boltMesh.setColorAt(boltCount, this.colour);
      boltCount++;
    }

    for (const flash of this.flashes) {
      if (!flash.active) continue;
      flash.age += dt;
      if (flash.age < 0) continue; // still waiting for the bolt to land
      if (flash.age >= FLASH_LIFE_S) {
        flash.active = false;
        continue;
      }
      if (flashCount >= CAPACITY) continue;

      // Expand and fade: the scale carries the animation, since a shared
      // material cannot fade instances independently.
      const t = flash.age / FLASH_LIFE_S;
      const size = 0.5 + t * 1.3;
      this.position.set(flash.x, flash.y, flash.z);
      this.scale.setScalar(size * (1 - t * 0.75));
      this.matrix.compose(this.position, IDENTITY, this.scale);
      this.flashMesh.setMatrixAt(flashCount, this.matrix);
      this.colour.setHex(flash.colour).lerp(WHITE, 0.6);
      this.flashMesh.setColorAt(flashCount, this.colour);
      flashCount++;
    }

    let debrisCount = 0;
    for (const d of this.debris) {
      if (!d.active) continue;
      d.age += dt;
      if (d.age >= d.life) {
        d.active = false;
        continue;
      }
      if (debrisCount >= CAPACITY) continue;

      // Ballistic arc with a floor. Cheap, and reads correctly at this scale.
      d.vy -= GRAVITY * dt;
      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.z += d.vz * dt;
      if (d.y < 0.08) {
        d.y = 0.08;
        d.vy = -d.vy * 0.35;
        d.vx *= 0.6;
        d.vz *= 0.6;
      }

      const fade = 1 - d.age / d.life;
      this.position.set(d.x, d.y, d.z);
      this.quat.setFromAxisAngle(TUMBLE_AXIS, d.spin * d.age);
      this.scale.setScalar(d.size * (0.4 + fade * 0.6));
      this.matrix.compose(this.position, this.quat, this.scale);
      this.debrisMesh.setMatrixAt(debrisCount, this.matrix);
      this.colour.setHex(d.colour).multiplyScalar(0.5 + fade * 0.5);
      this.debrisMesh.setColorAt(debrisCount, this.colour);
      debrisCount++;
    }

    let markerCount = 0;
    for (const m of this.markers) {
      if (!m.active) continue;
      m.age += dt;
      if (m.age >= MARKER_LIFE_S) {
        m.active = false;
        continue;
      }
      const t = m.age / MARKER_LIFE_S;
      // Expand and thin out — a ripple, so it is unmistakably a click and not a
      // selection ring.
      this.position.set(m.x, 0.08, m.z);
      this.scale.setScalar(0.35 + t * 1.15);
      this.matrix.compose(this.position, IDENTITY, this.scale);
      this.markerMesh.setMatrixAt(markerCount, this.matrix);
      this.colour.setHex(m.colour).multiplyScalar(1 - t * 0.7);
      this.markerMesh.setColorAt(markerCount, this.colour);
      markerCount++;
    }

    this.boltMesh.count = boltCount;
    this.boltMesh.instanceMatrix.needsUpdate = true;
    if (this.boltMesh.instanceColor) this.boltMesh.instanceColor.needsUpdate = true;
    this.flashMesh.count = flashCount;
    this.flashMesh.instanceMatrix.needsUpdate = true;
    if (this.flashMesh.instanceColor) this.flashMesh.instanceColor.needsUpdate = true;
    this.debrisMesh.count = debrisCount;
    this.debrisMesh.instanceMatrix.needsUpdate = true;
    if (this.debrisMesh.instanceColor) this.debrisMesh.instanceColor.needsUpdate = true;
    this.markerMesh.count = markerCount;
    this.markerMesh.instanceMatrix.needsUpdate = true;
    if (this.markerMesh.instanceColor) this.markerMesh.instanceColor.needsUpdate = true;
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.group.clear();
  }
}

const FORWARD = new THREE.Vector3(0, 0, 1);
const TUMBLE_AXIS = new THREE.Vector3(0.4, 1, 0.3).normalize();
/** World units per second squared. Tuned by eye, not by physics. */
const GRAVITY = 11;
const IDENTITY = new THREE.Quaternion();
const WHITE = new THREE.Color(0xffffff);
