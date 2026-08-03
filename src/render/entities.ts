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
import type { FogRenderer } from './fog.js';
import { MAX_CLIFF_HEIGHT } from './terrain.js';

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
  /** Drives the hover bob for air units. Cosmetic, so wall-clock is fine. */
  private bobPhase = 0;

  constructor(
    private readonly provider: ModelProvider,
    world: World,
  ) {
    this.buildPools(world);

    // Selection rings are a mark on the ground, not an overlay: they depth-test
    // like everything else, so a ring goes behind the cliff or the building that
    // is in front of it instead of floating over the top. `depthWrite` stays off
    // because they are translucent and drawn in one instanced call — writing
    // depth would let whichever overlapping ring happened to come first in the
    // buffer punch a hole in its neighbour.
    const ringGeo = new THREE.RingGeometry(0.86, 1.0, 20);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: SELECTION_RING_COLOUR,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });
    this.selectionRings = new THREE.InstancedMesh(ringGeo, ringMat, POOL_CAPACITY);
    this.selectionRings.frustumCulled = false;
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
      EntityType.Gunship,
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
    elapsedS = 0,
    fog?: FogRenderer,
    localPlayer = 0,
  ): void {
    this.bobPhase = elapsedS * 2.2;
    for (const pool of this.pools.values()) pool.count = 0;
    let ringCount = 0;
    let barCount = 0;

    const pool = world.pool;
    const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;

    // Health bars are measured in screen space; these are what convert. The
    // camera's basis is constant across the frame, so it is resolved once.
    barForward.set(0, 0, -1).applyQuaternion(camera.quaternion);
    // Perspective term 1/tan(fov/2): NDC height = world height * this / depth.
    const focalScale = camera.projectionMatrix.elements[5] || 1;

    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      // Fog hides enemies by not drawing them. Covering them with a dark plane
      // would not work — units stand above the ground, not on the texture.
      if (fog && !fog.shouldDraw(world, i, localPlayer)) continue;

      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      const owner = pool.owner[i]!;
      const spec = this.provider.get(type);

      // Air units are drawn well above the ground, with a slow bob. Altitude is
      // purely visual — the simulation is 2D and treats them like anything else
      // — but without it a gunship parked over infantry is unreadable.
      const altitude = def.flying
        ? FLIGHT_ALTITUDE + Math.sin(this.bobPhase + i * 0.7) * FLYER_BOB
        : 0;

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

        this.position.set(part.offset[0], part.offset[1] + sink + altitude, part.offset[2]);
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

        // The bar is a piece of UI that happens to live in the scene, so it is
        // sized in *screen* units and converted to world units at this entity's
        // depth. Sized in world units it was a merged hairline when zoomed out
        // and a banner when zoomed in.
        barRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
        barUp.set(0, 1, 0).applyQuaternion(camera.quaternion);

        const headY = spec.height + altitude;
        // Depth along the view axis, which is what perspective divides by.
        const viewDepth =
          (x - camera.position.x) * barForward.x +
          (headY - camera.position.y) * barForward.y +
          (z - camera.position.z) * barForward.z;
        // World units per unit of normalised device height, at this depth.
        const perNdc = Math.max(0.001, viewDepth) / focalScale;

        const width = (def.isBuilding ? BAR_WIDTH_NDC * 1.8 : BAR_WIDTH_NDC) * perNdc;
        const thickness = BAR_HEIGHT_NDC * perNdc;
        const lift = BAR_LIFT_NDC * perNdc;

        // Step up along the camera's own up vector, not world Y. World Y
        // projects to a slanted screen direction for anything away from the
        // centre of the view, which left every bar drifting sideways off its
        // unit — worse the further out and the closer in the camera was.
        // Then step back half a width along camera-right to put the left edge
        // where the bar should start, since the quad grows rightward.
        this.position.set(
          x + barUp.x * lift - barRight.x * (width / 2),
          headY + barUp.y * lift - barRight.y * (width / 2),
          z + barUp.z * lift - barRight.z * (width / 2),
        );
        this.scale.set(width, thickness, 1);
        this.matrix.compose(this.position, camera.quaternion, this.scale);
        this.healthBg.setMatrixAt(barCount, this.matrix);

        // Same left edge; only the width changes, so the bar drains rightward.
        this.scale.set(Math.max(0.0001, width * frac), thickness, 1);
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

/** How far a flyer's hover carries it above and below its nominal altitude. */
export const FLYER_BOB = 0.12;
/** How far the lowest part of a flyer's model hangs below its origin. */
export const FLYER_UNDERHANG = 0.3;
/** Daylight left between a flyer's lowest point and the tallest terrain. */
const FLIGHT_CLEARANCE = 0.35;

/**
 * How high above the ground air units are drawn.
 *
 * Derived from the terrain rather than chosen, because air has to *look* like
 * air: with the two as independent numbers, cliffs grew when the map gained
 * elevation and gunships ended up flying through ridges. The margin covers the
 * model's underhang, the hover bob, and a visible gap on top.
 */
export const FLIGHT_ALTITUDE =
  MAX_CLIFF_HEIGHT + FLYER_UNDERHANG + FLYER_BOB + FLIGHT_CLEARANCE;
/**
 * Selection rings are neutral grey rather than a team colour.
 *
 * What is selected is the player's own business, so the ring should not compete
 * with the colours that carry game state — team ownership, health, damage.
 */
const SELECTION_RING_COLOUR = 0xd2d7de;
/** Scratch for the camera basis vectors used to place health bars. */
const barRight = new THREE.Vector3();
const barUp = new THREE.Vector3();
const barForward = new THREE.Vector3();
const IDENTITY = new THREE.Quaternion();

/**
 * Health bar geometry, in normalised device units.
 *
 * NDC height spans 2 across the viewport, so 0.1 is a twentieth of the screen
 * height — about 45 px at 900. Expressing the bar this way is what keeps it the
 * same size whether the camera is at its closest or fully pulled back.
 */
const BAR_WIDTH_NDC = 0.1;
const BAR_HEIGHT_NDC = 0.014;
/** Gap between the top of the model and the bar. */
const BAR_LIFT_NDC = 0.032;

function poolKey(type: EntityType, part: number, owner: number): string {
  return `${type}:${part}:${owner}`;
}
