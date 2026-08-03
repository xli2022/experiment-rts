/**
 * Entity rendering: instancing, interpolation, selection rings, health bars.
 *
 * ## Instancing
 *
 * One `InstancedMesh` per (entity type, model part, owner). A hundred riflemen
 * cost one draw call per part instead of a hundred, and team colour is baked
 * into the pool key so no per-instance material switching is needed.
 *
 * ## Interpolation
 *
 * The simulation ticks at 20Hz but the screen refreshes at 60Hz or higher, so
 * drawing raw simulation positions would look like a slideshow. Each tick is
 * snapshotted, and every frame draws between the last two snapshots by `alpha`.
 *
 * `alpha` is clamped to [0,1] by the lockstep runner. That clamp is what makes a
 * network stall look like the game pausing rather than units sliding through
 * walls: without it, a stalled peer keeps extrapolating past the last known
 * position for as long as the stall lasts.
 */

import * as THREE from 'three';
import { defOf } from '../config/rules.js';
import { ENTITY_CAPACITY } from '../sim/entities.js';
import { toFloat } from '../sim/fixed.js';
import { BuildState, EntityType, NEUTRAL } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { colourFor, PLAYER_COLOURS } from './models/procedural.js';
import type { ModelProvider } from './models/provider.js';

/** Instances allocated per pool. Comfortably above a 200-supply army. */
const POOL_CAPACITY = 512;

interface Pool {
  mesh: THREE.InstancedMesh;
  count: number;
}

export class EntityRenderer {
  readonly group = new THREE.Group();

  /** Snapshot of the previous and current simulation ticks. */
  private readonly prevX = new Float32Array(ENTITY_CAPACITY);
  private readonly prevZ = new Float32Array(ENTITY_CAPACITY);
  private readonly prevFx = new Float32Array(ENTITY_CAPACITY);
  private readonly prevFz = new Float32Array(ENTITY_CAPACITY);
  private readonly currX = new Float32Array(ENTITY_CAPACITY);
  private readonly currZ = new Float32Array(ENTITY_CAPACITY);
  private readonly currFx = new Float32Array(ENTITY_CAPACITY);
  private readonly currFz = new Float32Array(ENTITY_CAPACITY);
  private readonly wasAlive = new Uint8Array(ENTITY_CAPACITY);

  private readonly pools = new Map<string, Pool>();
  private readonly disposables: { dispose(): void }[] = [];

  private readonly selectionRings: THREE.InstancedMesh;
  private readonly healthBg: THREE.InstancedMesh;
  private readonly healthFill: THREE.InstancedMesh;

  private readonly matrix = new THREE.Matrix4();
  private readonly quat = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3(1, 1, 1);
  private readonly position = new THREE.Vector3();
  private readonly colour = new THREE.Color();

  constructor(
    private readonly provider: ModelProvider,
    world: World,
  ) {
    this.buildPools(world);

    // Selection rings lie flat on the ground with depth testing off, so they
    // stay visible even when a unit is standing behind a building.
    const ringGeo = new THREE.RingGeometry(0.86, 1.0, 20);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x7dff9b,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    this.selectionRings = new THREE.InstancedMesh(ringGeo, ringMat, POOL_CAPACITY);
    this.selectionRings.frustumCulled = false;
    this.selectionRings.renderOrder = 10;
    this.group.add(this.selectionRings);
    this.disposables.push(ringGeo, ringMat);

    // The bar quad is anchored at its LEFT edge, not its centre. Scaling a
    // centre-anchored quad shrinks it toward the middle, so a damaged bar would
    // pull away from both ends instead of draining from the right.
    const barGeo = new THREE.PlaneGeometry(1, 1);
    barGeo.translate(0.5, 0, 0);

    const bgMat = new THREE.MeshBasicMaterial({ color: 0x11151c, depthTest: false });
    // No `vertexColors` here. That flag makes three.js look for a per-vertex
    // colour attribute which a PlaneGeometry does not have; per-*instance*
    // colours from setColorAt are a different code path that needs no flag. With
    // it set, the fill rendered a meaningless colour and the bar never appeared
    // to track health at all.
    const fillMat = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false });

    this.healthBg = new THREE.InstancedMesh(barGeo, bgMat, POOL_CAPACITY);
    this.healthFill = new THREE.InstancedMesh(barGeo, fillMat, POOL_CAPACITY);
    this.healthBg.frustumCulled = false;
    this.healthFill.frustumCulled = false;
    // Distinct render orders: with depth testing off, equal orders leave it to
    // scene order whether the fill draws over the background or under it.
    this.healthBg.renderOrder = 11;
    this.healthFill.renderOrder = 12;
    this.group.add(this.healthBg, this.healthFill);
    this.disposables.push(barGeo, bgMat, fillMat);

    this.captureSnapshot(world);
    this.captureSnapshot(world); // prime both buffers so nothing lerps from origin
  }

  /** Create an instanced mesh for every (type, part, owner) combination. */
  private buildPools(world: World): void {
    const types: EntityType[] = [
      EntityType.Worker,
      EntityType.Rifleman,
      EntityType.Brawler,
      EntityType.CommandPost,
      EntityType.Depot,
      EntityType.Barracks,
      EntityType.Turret,
      EntityType.MineralPatch,
    ];
    const owners = [0, 1, NEUTRAL];

    for (const type of types) {
      const spec = this.provider.get(type);
      for (const owner of owners) {
        // Only mineral patches are neutral; skip the pointless combinations.
        if (type === EntityType.MineralPatch ? owner !== NEUTRAL : owner === NEUTRAL) continue;

        for (let p = 0; p < spec.parts.length; p++) {
          const part = spec.parts[p]!;
          const material = new THREE.MeshLambertMaterial({
            color: colourFor(part.role, owner),
          });
          this.disposables.push(material);
          const mesh = new THREE.InstancedMesh(part.geometry, material, POOL_CAPACITY);
          mesh.frustumCulled = false;
          mesh.castShadow = true;
          mesh.count = 0;
          this.group.add(mesh);
          this.pools.set(poolKey(type, p, owner), { mesh, count: 0 });
        }
      }
    }
    void world;
  }

  /**
   * Record the current simulation state as the newest snapshot.
   *
   * Call once per simulation tick, never per frame. Newly spawned entities get
   * both buffers set to the same value so they appear where they were built
   * rather than sliding in from wherever the recycled slot last was.
   */
  captureSnapshot(world: World): void {
    const pool = world.pool;
    for (let i = 0; i < pool.count; i++) {
      const alive = pool.alive[i] === 1;
      const x = toFloat(pool.posX[i]!);
      const z = toFloat(pool.posY[i]!);
      const fx = toFloat(pool.faceX[i]!);
      const fz = toFloat(pool.faceY[i]!);

      if (alive && this.wasAlive[i] === 0) {
        // Fresh spawn, or a reused slot: no history to interpolate from.
        this.prevX[i] = x;
        this.prevZ[i] = z;
        this.prevFx[i] = fx;
        this.prevFz[i] = fz;
      } else {
        this.prevX[i] = this.currX[i]!;
        this.prevZ[i] = this.currZ[i]!;
        this.prevFx[i] = this.currFx[i]!;
        this.prevFz[i] = this.currFz[i]!;
      }

      this.currX[i] = x;
      this.currZ[i] = z;
      this.currFx[i] = fx;
      this.currFz[i] = fz;
      this.wasAlive[i] = alive ? 1 : 0;
    }
  }

  /**
   * Write this frame's transforms.
   *
   * @param alpha interpolation factor in [0,1] between the last two snapshots.
   * @param selected entity slot indices to draw selection rings for.
   */
  update(
    world: World,
    alpha: number,
    selected: ReadonlySet<number>,
    camera: THREE.Camera,
  ): void {
    for (const pool of this.pools.values()) pool.count = 0;
    let ringCount = 0;
    let barCount = 0;

    const pool = world.pool;
    const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;

    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;

      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      const owner = pool.owner[i]!;
      const spec = this.provider.get(type);

      const x = this.prevX[i]! + (this.currX[i]! - this.prevX[i]!) * a;
      const z = this.prevZ[i]! + (this.currZ[i]! - this.prevZ[i]!) * a;

      // Interpolate the facing *vector* and derive the angle from it. Lerping
      // angles directly wraps badly at the +/-PI seam, producing a spin.
      const fx = this.prevFx[i]! + (this.currFx[i]! - this.prevFx[i]!) * a;
      const fz = this.prevFz[i]! + (this.currFz[i]! - this.prevFz[i]!) * a;
      const yaw = Math.atan2(fx, fz);
      this.quat.setFromAxisAngle(UP, yaw);

      // Buildings under construction sink into the ground, rising as they
      // finish — cheap, readable progress with no extra geometry.
      let sink = 0;
      if (def.isBuilding && pool.buildState[i] !== BuildState.Complete) {
        const progress =
          def.buildTicks > 0 ? Math.min(1, pool.buildProgress[i]! / def.buildTicks) : 0;
        sink = -(1 - progress) * spec.height * 0.8;
      }

      for (let p = 0; p < spec.parts.length; p++) {
        const entry = this.pools.get(poolKey(type, p, owner));
        if (!entry || entry.count >= POOL_CAPACITY) continue;
        const part = spec.parts[p]!;

        this.position.set(part.offset[0], part.offset[1] + sink, part.offset[2]);
        this.position.applyQuaternion(this.quat);
        this.position.x += x;
        this.position.z += z;

        if (part.rotation) {
          const local = new THREE.Quaternion().setFromEuler(
            new THREE.Euler(part.rotation[0], part.rotation[1], part.rotation[2]),
          );
          this.matrix.compose(this.position, this.quat.clone().multiply(local), this.scale);
        } else {
          this.matrix.compose(this.position, this.quat, this.scale);
        }

        entry.mesh.setMatrixAt(entry.count, this.matrix);
        entry.count++;
      }

      if (selected.has(i) && ringCount < POOL_CAPACITY) {
        this.position.set(x, 0.06, z);
        this.scale.set(spec.radius * 1.6, 1, spec.radius * 1.6);
        this.matrix.compose(this.position, IDENTITY, this.scale);
        this.selectionRings.setMatrixAt(ringCount++, this.matrix);
        this.scale.set(1, 1, 1);
      }

      // Health bars only when damaged or selected — a screen full of full bars
      // is noise, and the information a player needs is "what is hurt".
      const damaged = pool.hp[i]! < def.maxHp;
      if ((damaged || selected.has(i)) && barCount < POOL_CAPACITY && def.maxHp > 1) {
        const frac = Math.max(0, Math.min(1, pool.hp[i]! / def.maxHp));
        const width = def.isBuilding ? 2.0 : 0.9;
        const y = spec.height + 0.35;

        // Billboard the bar, then step half its width along the camera's own
        // right vector to place the left edge. Offsetting in world X instead
        // only happens to look right while the camera faces one direction.
        barRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
        const leftEdgeX = x - barRight.x * (width / 2);
        const leftEdgeY = y - barRight.y * (width / 2);
        const leftEdgeZ = z - barRight.z * (width / 2);

        this.position.set(leftEdgeX, leftEdgeY, leftEdgeZ);
        this.scale.set(width, 0.14, 1);
        this.matrix.compose(this.position, camera.quaternion, this.scale);
        this.healthBg.setMatrixAt(barCount, this.matrix);

        // Same left edge; only the width changes, so the bar drains rightward.
        this.scale.set(Math.max(0.0001, width * frac), 0.14, 1);
        this.matrix.compose(this.position, camera.quaternion, this.scale);
        this.healthFill.setMatrixAt(barCount, this.matrix);

        // Green through amber to red, so damage is readable at a glance.
        this.colour.setHSL(frac * 0.33, 0.85, 0.5);
        this.healthFill.setColorAt(barCount, this.colour);
        barCount++;
        this.scale.set(1, 1, 1);
      }
    }

    for (const entry of this.pools.values()) {
      entry.mesh.count = entry.count;
      entry.mesh.instanceMatrix.needsUpdate = true;
    }
    this.selectionRings.count = ringCount;
    this.selectionRings.instanceMatrix.needsUpdate = true;
    this.healthBg.count = barCount;
    this.healthBg.instanceMatrix.needsUpdate = true;
    this.healthFill.count = barCount;
    this.healthFill.instanceMatrix.needsUpdate = true;
    if (this.healthFill.instanceColor) this.healthFill.instanceColor.needsUpdate = true;
  }

  /** Team colour as a CSS string, for HUD elements. */
  static playerColourCss(player: number): string {
    return `#${(PLAYER_COLOURS[player] ?? 0x9aa4b2).toString(16).padStart(6, '0')}`;
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.pools.clear();
    this.group.clear();
  }
}

const UP = new THREE.Vector3(0, 1, 0);
/** Scratch for the camera-space right vector used to anchor health bars. */
const barRight = new THREE.Vector3();
const IDENTITY = new THREE.Quaternion();

function poolKey(type: EntityType, part: number, owner: number): string {
  return `${type}:${part}:${owner}`;
}
