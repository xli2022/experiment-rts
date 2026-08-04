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
import { AnimatedUnitPool } from './animatedUnits.js';
import type { AnimatedModel } from './models/animated.js';

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
  /**
   * When each unit last actually attacked, in elapsed seconds.
   *
   * The simulation reports every shot it fires, so the swing is driven by the
   * thing that really happened rather than inferred from the unit's order. That
   * distinction is the whole bug this replaced: a unit defending itself has no
   * attack *order* at all — it is idle and simply fighting what walked up to it
   * — so an order-based guess never played the animation for the most common
   * fight in the game.
   */
  private readonly attackedAt = new Float32Array(ENTITY_CAPACITY).fill(-1e9);

  /**
   * The clip each unit was last drawn on, and when it changed.
   *
   * A unit that starts running, or stops to swing, would otherwise jump between
   * poses on one frame. Holding the outgoing clip lets the shader ease out of
   * it. Stored as small integers rather than the clip names themselves because
   * this is per-entity state touched every frame.
   */
  private readonly poseClip = new Uint8Array(ENTITY_CAPACITY);
  private readonly poseFrom = new Uint8Array(ENTITY_CAPACITY);
  /** When the current transition started, in elapsed seconds. */
  private readonly poseChangedAt = new Float32Array(ENTITY_CAPACITY).fill(-1e9);
  /** Where the outgoing clip had got to when it was left behind. */
  private readonly poseFromTime = new Float32Array(ENTITY_CAPACITY);

  private readonly pools = new Map<string, Pool>();
  private readonly disposables: { dispose(): void }[] = [];

  /**
   * Types drawn from a baked skinned model instead of primitive parts.
   *
   * Registered after load, so the game starts on the procedural model and the
   * animated one takes over the moment its asset arrives — a unit type is never
   * missing while a 600 KB download is in flight.
   */
  private readonly animated = new Map<
    EntityType,
    { model: AnimatedModel; scale: number; groundOffset: number; height: number }
  >();
  private readonly animatedPools = new Map<string, AnimatedUnitPool>();

  /**
   * Units that have died and are still playing their death clip.
   *
   * The simulation frees an entity the tick it dies, so there is nothing left to
   * hang a death animation on — the corpse has to be the renderer's own. It
   * carries the position and facing captured at the moment of death, because the
   * entity slot behind it is reused within seconds.
   */
  private readonly corpses: Corpse[] = [];

  /** When the local player last issued an order, in elapsed seconds. */
  private orderedAt = -1e9;

  /** Pulsing disc under a building that is training something. */
  private readonly productionRings: THREE.InstancedMesh;

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

    // A filled disc rather than a ring, so it reads as a glow at the base of the
    // building rather than a second selection marker.
    const prodGeo = new THREE.CircleGeometry(1, 24);
    prodGeo.rotateX(-Math.PI / 2);
    const prodMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.productionRings = new THREE.InstancedMesh(prodGeo, prodMat, POOL_CAPACITY);
    this.productionRings.frustumCulled = false;
    this.productionRings.count = 0;
    this.group.add(this.productionRings);
    this.disposables.push(prodGeo, prodMat);

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

  /**
   * Note that the local player just gave an order, so the rings acknowledge it.
   *
   * Called from input, not from the simulation: the whole point is that it
   * happens on the frame of the click rather than turns later when the command
   * executes.
   */
  noteOrderIssued(elapsedS: number): void {
    this.orderedAt = elapsedS;
  }

  /**
   * Draw this entity type from a baked skinned model from now on.
   *
   * The procedural pools for the type are hidden rather than destroyed, so this
   * is reversible and costs nothing if the asset never loads.
   */
  useAnimatedModel(
    type: EntityType,
    model: AnimatedModel,
    scale: number,
    textures: (THREE.Texture | null)[],
  ): void {
    // Lift the model so the lowest point of its animation rests on the ground
    // rather than the lowest point of its bind pose, which a run cycle dips well
    // below.
    this.animated.set(type, {
      model,
      scale,
      groundOffset: -model.lowestY * scale,
      // The authored model is rarely the exact height of the primitive it
      // replaces, and the health bar hangs off the top of whatever is drawn.
      height: model.bindSize.y * scale,
    });

    for (const owner of [0, 1]) {
      const material = new THREE.MeshLambertMaterial({
        map: textures[owner] ?? null,
        // Without a texture the team colour has to carry the whole read, so the
        // model is tinted flat rather than left white.
        color: textures[owner] ? 0xffffff : (PLAYER_COLOURS[owner] ?? 0x9aa4b2),
      });
      const anim = new AnimatedUnitPool(model, material, POOL_CAPACITY);
      this.animatedPools.set(animatedKey(type, owner), anim);
      this.group.add(anim.mesh);
    }

    const spec = this.provider.get(type);
    for (const owner of [0, 1]) {
      for (let p = 0; p < spec.parts.length; p++) {
        const entry = this.pools.get(poolKey(type, p, owner));
        if (entry) entry.mesh.visible = false;
      }
    }
  }

  /**
   * Record this tick's deaths so their death animation can play out.
   *
   * Call once per simulation tick, from the same place the other per-tick
   * effects are read — `world.events` is cleared at the start of the next step.
   */
  noteDeaths(world: World, elapsedS: number): void {
    const pool = world.pool;
    // Shots come in (attacker, target) pairs.
    const shots = world.events.shots;
    for (let k = 0; k < shots.length; k += 2) {
      this.attackedAt[shots[k]!] = elapsedS;
    }

    for (const i of world.events.deaths) {
      const type = pool.type[i]! as EntityType;
      if (!this.animated.has(type)) continue;
      if (this.corpses.length >= MAX_CORPSES) this.corpses.shift();
      this.corpses.push({
        type,
        owner: pool.owner[i]!,
        x: this.currX[i]!,
        z: this.currZ[i]!,
        yaw: Math.atan2(this.currFx[i]!, this.currFz[i]!),
        flying: defOf(type).flying,
        bornS: elapsedS,
      });
    }
  }

  /**
   * Emit every corpse still worth drawing, and retire the rest.
   *
   * A flyer's wreck falls the rest of the way to the ground over its death clip
   * rather than lying in the air; everything else sinks away once the clip has
   * played, which clears the battlefield without the bodies simply vanishing.
   */
  private drawCorpses(elapsedS: number, fog: FogRenderer | undefined, localPlayer: number): void {
    for (let k = this.corpses.length - 1; k >= 0; k--) {
      const corpse = this.corpses[k]!;
      const entry = this.animated.get(corpse.type);
      const age = elapsedS - corpse.bornS;
      const clip = entry?.model.clips.get('die');
      if (!entry || !clip || age > clip.duration + CORPSE_LINGER_S) {
        this.corpses.splice(k, 1);
        continue;
      }
      const anim = this.animatedPools.get(animatedKey(corpse.type, corpse.owner));
      if (!anim) continue;
      // A corpse is not an entity any more, so it obeys fog by position: an
      // enemy dying out of sight should not announce itself.
      if (corpse.owner !== localPlayer && fog && !fog.isVisibleAt(corpse.x, corpse.z)) continue;

      const fell = corpse.flying
        ? FLIGHT_ALTITUDE * Math.max(0, 1 - age / Math.max(0.001, clip.duration))
        : 0;
      const sink = Math.max(0, age - clip.duration) / CORPSE_LINGER_S;

      this.quat.setFromAxisAngle(UP, corpse.yaw);
      this.position.set(corpse.x, fell + entry.groundOffset - sink * entry.height, corpse.z);
      this.scale.setScalar(entry.scale);
      this.matrix.compose(this.position, this.quat, this.scale);
      this.scale.set(1, 1, 1);
      anim.add(this.matrix, AnimatedUnitPool.frameFor(entry.model, 'die', age, false));
    }
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
        // Nor any swing to finish. Slots are recycled within seconds, so a
        // brand-new unit would otherwise inherit the last occupant's attack and
        // walk out of the barracks mid-swing.
        this.attackedAt[i] = -1e9;
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
    for (const anim of this.animatedPools.values()) anim.begin();
    let ringCount = 0;
    let barCount = 0;
    let prodCount = 0;

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

      const animated = this.animated.get(type);
      if (animated) {
        const anim = this.animatedPools.get(animatedKey(type, owner));
        if (anim) {
          const swing = animated.model.clips.get('attack');
          const pose = poseFor(
            elapsedS - this.attackedAt[i]!,
            swing ? swing.duration : 0,
            Math.hypot(this.currX[i]! - this.prevX[i]!, this.currZ[i]! - this.prevZ[i]!),
            // Offset each unit's stride by its slot so an army does not march in
            // perfect lockstep, which reads as one object rather than many.
            elapsedS + i * 0.37,
          );

          const target = framesForPose(animated.model, pose);
          let from = target.from;
          let to = target.to;
          let blend = target.blend;

          // Note the switch the frame it happens, so the fade is timed from the
          // change rather than from whenever this unit was next drawn.
          const id = CLIP_IDS[pose.clip] ?? 0;
          if (this.poseClip[i] !== id) {
            this.poseFrom[i] = this.poseClip[i]!;
            this.poseFromTime[i] = pose.time;
            this.poseChangedAt[i] = elapsedS;
            this.poseClip[i] = id;
          }

          const fading = elapsedS - this.poseChangedAt[i]!;
          if (fading >= 0 && fading < CROSSFADE_S) {
            // Ease out of the old clip and into the new one. The outgoing pose
            // is frozen where it was left: continuing to advance it would need a
            // third and fourth texture read for no visible gain over an eighth
            // of a second.
            const previous = framesForPose(animated.model, {
              clip: CLIP_NAMES[this.poseFrom[i]!] ?? 'run',
              time: this.poseFromTime[i]!,
              loop: true,
            });
            from = previous.from;
            to = target.from;
            blend = fading / CROSSFADE_S;
          }

          this.position.set(x, altitude + animated.groundOffset, z);
          this.scale.setScalar(animated.scale);
          this.matrix.compose(this.position, this.quat, this.scale);
          this.scale.set(1, 1, 1);
          anim.add(this.matrix, from, to, blend);
        }
      }

      for (let p = 0; p < spec.parts.length && !animated; p++) {
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

      // A building mid-production gets a slow pulse of light at its base, so a
      // busy barracks is legible from across the map without selecting it.
      if (
        pool.prodCount[i]! > 0 &&
        pool.buildState[i] === BuildState.Complete &&
        ringCount < POOL_CAPACITY
      ) {
        const beat = 0.72 + 0.28 * Math.sin(elapsedS * PRODUCTION_PULSE_RATE + i);
        const r = spec.radius * 1.35 * beat;
        this.position.set(x, 0.05, z);
        this.scale.set(r, 1, r);
        this.matrix.compose(this.position, IDENTITY, this.scale);
        this.productionRings.setMatrixAt(prodCount, this.matrix);
        this.productionRings.setColorAt(prodCount, this.colour.setHex(PLAYER_COLOURS[owner] ?? 0xffffff));
        prodCount++;
        this.scale.set(1, 1, 1);
      }

      if (selected.has(i) && ringCount < POOL_CAPACITY) {
        // A ring that kicks outward the instant an order is given.
        //
        // Lockstep executes a command turns after it is issued, so nothing the
        // unit itself does can happen on the frame you clicked. The ring can:
        // it is drawn from the local selection, never from simulation state, so
        // it costs nothing and cannot desync. This is the same trick that makes
        // an RTS with real latency feel immediate.
        const sinceOrder = elapsedS - this.orderedAt;
        const kick =
          sinceOrder >= 0 && sinceOrder < ACK_PULSE_S
            ? 1 + ACK_PULSE_SIZE * (1 - sinceOrder / ACK_PULSE_S)
            : 1;
        // Sized from the simulation's collision radius, not the art. The ring
        // is what tells a player how much room a unit takes up, so reading it
        // off the model would make it lie whenever the two disagree.
        const ringR = toFloat(def.radius) * RING_OVERSIZE * kick;
        this.position.set(x, 0.06, z);
        this.scale.set(ringR, 1, ringR);
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

        const headY = (animated ? animated.height : spec.height) + altitude;
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
    this.productionRings.count = prodCount;
    this.productionRings.instanceMatrix.needsUpdate = true;
    if (this.productionRings.instanceColor) this.productionRings.instanceColor.needsUpdate = true;
    this.selectionRings.count = ringCount;
    this.selectionRings.instanceMatrix.needsUpdate = true;
    this.healthBg.count = barCount;
    this.healthBg.instanceMatrix.needsUpdate = true;
    this.drawCorpses(elapsedS, fog, localPlayer);
    for (const anim of this.animatedPools.values()) anim.commit();

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
    for (const anim of this.animatedPools.values()) anim.dispose();
    this.animatedPools.clear();
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

/**
 * Selection ring size relative to the collision radius.
 *
 * A shade over 1 so the ring reads as sitting around the unit rather than
 * cutting through its feet.
 */
const RING_OVERSIZE = 1.15;

/** Radians per second of the production glow's pulse. About one beat a second. */
const PRODUCTION_PULSE_RATE = 5.4;

/** How far a unit must move between ticks to count as running. */
const MOVING_EPSILON = 0.004;

/** Which clip a unit is posed on this frame, and where in it. */
export interface Pose {
  clip: string;
  /** Seconds into the clip. */
  time: number;
  loop: boolean;
}

/**
 * Seconds spent easing from one clip into the next.
 *
 * Long enough to remove the pop, short enough that a swing still lands on the
 * frame the damage did — the attack clip is 1.5s, so an eighth of a second of
 * lead-in costs nothing that reads.
 */
const CROSSFADE_S = 0.12;

/** How long a selection ring stays kicked out after an order, and by how much. */
const ACK_PULSE_S = 0.22;
const ACK_PULSE_SIZE = 0.35;

/** Clip names as small integers, so per-entity pose state stays a typed array. */
const CLIP_NAMES = ['run', 'attack', 'die'] as const;
const CLIP_IDS: Record<string, number> = { run: 0, attack: 1, die: 2 };

/**
 * Choose a unit's clip. Pure, because getting it wrong is invisible in a diff.
 *
 * Two things about this rule were learned from a bug report of "the melee unit
 * never plays its attack animation":
 *
 * - **A swing is an event, not a state.** The old rule asked whether the unit
 *   held an attack *order*, which is only correlated with fighting: a unit
 *   defending itself has no order at all — it is idle, hitting whatever walked
 *   into range — so the most common fight in the game never animated. The
 *   simulation already publishes each shot in `world.events.shots`; `sinceAttack`
 *   is that, and it is the only honest signal.
 * - **The swing outranks the stride.** Units in contact are shoved around by
 *   separation every tick, so a brawler mid-melee is never quite stationary and
 *   a movement-first rule keeps it running on the spot. Order matters here, not
 *   just the conditions.
 *
 * The swing is timed from the shot rather than from wall-clock, so the blow
 * lands on the frame the damage did. It does not loop: a cooldown longer than
 * the clip should hold the follow-through, not restart the wind-up.
 */
export function poseFor(
  sinceAttack: number,
  swingDuration: number,
  movedPerTick: number,
  phase: number,
): Pose {
  if (swingDuration > 0 && sinceAttack >= 0 && sinceAttack < swingDuration) {
    return { clip: 'attack', time: sinceAttack, loop: false };
  }
  if (movedPerTick > MOVING_EPSILON) return { clip: 'run', time: phase, loop: true };
  // No authored idle clip; a frozen stride reads better than a T-pose.
  return { clip: 'run', time: 0, loop: true };
}

/**
 * Resolve a pose to the two bone-texture rows to draw it with.
 *
 * The pair is the clip's own two neighbouring frames, which is what turns the
 * 30Hz bake into motion smooth at display rate.
 */
export function framesForPose(
  model: AnimatedModel,
  pose: Pose,
): { from: number; to: number; blend: number } {
  return AnimatedUnitPool.framePairFor(model, pose.clip, pose.time, pose.loop);
}

/** Seconds a body lingers, sinking, once its death clip has played out. */
const CORPSE_LINGER_S = 1.4;
/** Bodies kept at once. A long battle would otherwise accumulate them forever. */
const MAX_CORPSES = 96;

interface Corpse {
  type: EntityType;
  owner: number;
  x: number;
  z: number;
  yaw: number;
  flying: boolean;
  bornS: number;
}

function animatedKey(type: EntityType, owner: number): string {
  return `${type}:${owner}`;
}

function poolKey(type: EntityType, part: number, owner: number): string {
  return `${type}:${part}:${owner}`;
}
