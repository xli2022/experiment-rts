/**
 * Transient effects: weapon fire, impacts, explosions, death debris and the
 * marker that confirms a click landed.
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
 * ## How a shot becomes an effect
 *
 * Every weapon runs through the same four stages, and each stage is optional:
 *
 *   muzzle  →  travel  →  impact  →  aftermath
 *
 * A Burstbot has all four in miniature. A Beamdrone has no travel, because a
 * beam is simply *there*. A Boomwalker has none of them: it is the payload, and
 * its death blast is the entire weapon. Which stages a unit gets, and what they
 * look like, is one row of `src/render/vfx/weapons.ts` — see there for why
 * ordnance carries the team colour and explosions do not.
 *
 * ## One attack, not one effect per victim
 *
 * `world.events.shots` holds an (attacker, victim) pair per *victim*, so a
 * Sentry shell that catches five units arrives here as five pairs. Drawing one
 * shell each would fire five mortars at one tile. Pairs from one attack are
 * consecutive, so they are regrouped into a single attack with a list of
 * victims, and the weapon decides what that means: one shell and one blast for
 * splash, one lance for pierce, and three separate bolts for the Arclight,
 * whose coils genuinely do pick their own enemies.
 *
 * Which victim the shell was aimed *at* is not in the events, but it does not
 * have to be: `resolveAttackImpact` points the attacker at its primary target
 * on the tick it fires, so the victim best lined up with the attacker's facing
 * is the one it shot at. Everything else was caught in the blast.
 *
 * ## Timing
 *
 * A frame may cover several simulation ticks, and effects from an older tick
 * must not all pile up at age zero. Every effect is therefore born with an age
 * — usually slightly negative, for a shot from the tick being rendered — and
 * anything with a negative age is simply not drawn yet. A projectile hands its
 * leftover age to the impact it triggers, so a burst from a catch-up tick
 * arrives already part-way through.
 */

import * as THREE from 'three';
import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import { EntityType, TICKS_PER_SECOND } from '../sim/types.js';
import type { World } from '../sim/world.js';
import {
  FLIGHT_ALTITUDE,
  flyerAltitudeAt,
  type EntityRenderer,
  type EntityTransformSnapshot,
} from './entities.js';
import { colourSlotFor, PLAYER_COLOURS } from './models/procedural.js';
import { BeamField, GroundField, SpriteField } from './vfx/fields.js';
import {
  beamTexture,
  flameTexture,
  glowTexture,
  ringTexture,
  scorchTexture,
  smokeTexture,
} from './vfx/textures.js';
import { tintedTeamColour, weaponProfileFor, type WeaponProfile } from './vfx/weapons.js';

/**
 * Pool sizes, per effect kind.
 *
 * Past these the oldest entry is recycled, which in practice means the oldest
 * frame of a battle already dense enough that nobody can count its sparks. They
 * are sized for the worst case a 128x128 map can produce — two full armies in
 * contact — rather than for the average, because the average costs nothing to
 * leave allocated and the worst case is exactly when the effects matter.
 */
const PARTICLE_CAPACITY = 1536;
const ORDNANCE_CAPACITY = 256;
const BEAM_CAPACITY = 384;
const DECAL_CAPACITY = 256;
const DEBRIS_CAPACITY = 512;
const MARKER_CAPACITY = 64;

/** Seconds a scorch mark stays on the ground before it has faded out. */
const SCORCH_LIFE_S = 4.5;
/** Seconds the click marker lives. */
const MARKER_LIFE_S = 0.45;
/** World units per second squared, for debris and sparks. By eye, not physics. */
const GRAVITY = 11;
/** Height a ground effect is drawn at, clear of the terrain without floating. */
const GROUND_Y = 0.04;

/**
 * The fire palette, shared by every explosion whoever caused it.
 *
 * Ordnance in flight is team-coloured; what it does on arrival is not. See the
 * colour section of `vfx/weapons.ts`.
 */
const WHITE = new THREE.Color(0xffffff);
const HOT = new THREE.Color(0xfff6e2);
const FIRE = new THREE.Color(0xffb347);
const EMBER = new THREE.Color(0xd8461f);
const SPARK = new THREE.Color(0xffd79a);
const SMOKE = new THREE.Color(0x8d8578);
const SMOKE_DARK = new THREE.Color(0x35302a);
const DUST = new THREE.Color(0x8e8375);
const FROST = new THREE.Color(0x8fe4ff);
/** The scorch texture carries its own colour, so its instances are untinted. */
const UNTINTED = new THREE.Color(0xffffff);

/** A presentation-space point. The simulation itself remains strictly 2D. */
export interface ProjectilePoint {
  x: number;
  y: number;
  z: number;
}

/** A weapon hardpoint in entity-local space: right, up, and forward. */
interface ProjectileOffset {
  right: number;
  up: number;
  forward: number;
}

/** One entity an attack landed on, resolved at the frame's final alpha. */
interface PendingVictim {
  type: EntityType;
  transform: EntityTransformSnapshot;
}

/** A copied attack waiting for the frame's final interpolation alpha. */
interface PendingShot {
  tick: number;
  attackerType: EntityType;
  attacker: EntityTransformSnapshot;
  victims: PendingVictim[];
  colour: number;
}

/** Transform resolved from the same previous/current pair as the entity mesh. */
export interface InterpolatedEntityTransform {
  x: number;
  z: number;
  faceX: number;
  faceZ: number;
}

/** Resolve a retained transform interval at the frame's final render alpha. */
export function interpolateProjectileTransform(
  snapshot: EntityTransformSnapshot,
  alpha: number,
): InterpolatedEntityTransform {
  const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
  return {
    x: snapshot.prevX + (snapshot.currX - snapshot.prevX) * a,
    z: snapshot.prevZ + (snapshot.currZ - snapshot.prevZ) * a,
    faceX: snapshot.prevFaceX + (snapshot.currFaceX - snapshot.prevFaceX) * a,
    faceZ: snapshot.prevFaceZ + (snapshot.currFaceZ - snapshot.prevFaceZ) * a,
  };
}

/**
 * Initial age which becomes simulation-timeline age after this frame's single
 * `update(dtMs)`. This keeps a shot from the newest catch-up tick alive even if
 * the wall-clock frame itself was longer than the bolt lifetime.
 */
export function projectileAgeBeforeFrameUpdate(
  shotTick: number,
  currentTick: number,
  alpha: number,
  dtMs: number,
): number {
  const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
  const frameDt = Math.min(Math.max(dtMs, 0), 250) / 1000;
  return (currentTick - shotTick + a) / TICKS_PER_SECOND - frameDt;
}

/**
 * Muzzle locations matched to the visible weapon on each unit.
 *
 * These deliberately live in the renderer rather than `EntityDef`: moving a
 * muzzle must not change a checksum or the instant at which damage is applied.
 * Local +Z is forward, the same convention `EntityRenderer` uses for facing.
 *
 * The first three were measured off the rig's attack-frame weapon node. The
 * rest are authored by eye against the drawn model — a robot 1.4 tiles tall
 * firing from its own centre looks like it is shooting out of its stomach, and
 * "roughly where the barrel is" is a great deal closer than that. Heights are
 * derived from `runSize` in `models/unitModels.ts` times `ROBOT_SCALE`.
 */
const PROJECTILE_OFFSETS: Partial<Record<EntityType, ProjectileOffset>> = {
  // Revolver's attack-frame lower-right `Dummy004` muzzle face, after its
  // 1.01-high fit and animation-ground lift.
  [EntityType.Burstbot]: { right: 0.24, up: 0.63, forward: 0.67 },
  // BeamShip's attack-frame `n7_Point001`, after its 1.43-wide fit and ground
  // lift: slightly left, 0.30 above flight origin and 0.22 forward.
  [EntityType.Beamdrone]: {
    right: -0.03,
    up: FLIGHT_ALTITUDE + 0.3,
    forward: 0.22,
  },
  // Tip of the long barrel in the procedural turret model.
  [EntityType.Turret]: { right: 0, up: 0.75, forward: 1.1 },
  // Blade, swung across the body, so the flourish starts off to one side.
  [EntityType.Slicebot]: { right: 0.3, up: 0.6, forward: 0.4 },
  // Nozzle, held low and forward: fire should come off the ground, not the head.
  [EntityType.Firespout]: { right: 0.1, up: 0.55, forward: 0.5 },
  // Shoulder coils, high and central — all three arcs leave from one place.
  [EntityType.Arclight]: { right: 0, up: 0.88, forward: 0.15 },
  // The longest barrel in the line; its model is 1.37 tiles deep.
  [EntityType.Piercebot]: { right: 0, up: 0.52, forward: 0.85 },
  // Mortar tube, near the top of a 1.33-tall chassis and pointing up.
  [EntityType.Sentry]: { right: 0, up: 1.05, forward: 0.1 },
  // Emitter on the repair rig's arm.
  [EntityType.Fixomatic]: { right: 0.18, up: 0.78, forward: 0.32 },
  // Fist, on the swing side of a 1.44-tall golem.
  [EntityType.DarkGolem]: { right: 0.34, up: 0.92, forward: 0.55 },
  // Open hand, thrown forward from the same height.
  [EntityType.IceGolem]: { right: 0.36, up: 0.86, forward: 0.5 },
  // Underbody hardpoint, forward of the hover origin.
  [EntityType.Plasmodrone]: {
    right: 0,
    up: FLIGHT_ALTITUDE + 0.12,
    forward: 0.4,
  },
  // A wrench at arm's length. Worth having so a worker fight is not two units
  // exchanging sparks from inside each other.
  [EntityType.Worker]: { right: 0.16, up: 0.45, forward: 0.3 },
};

const DEFAULT_PROJECTILE_OFFSET: ProjectileOffset = {
  right: 0,
  up: 0.55,
  forward: 0,
};

/**
 * Resolve a type-specific local muzzle into world space.
 *
 * Exported as a pure helper so the visual contract can be tested without a
 * WebGL context. Facing is normalised defensively because fixed-point rounding
 * can leave it a fraction either side of unit length.
 */
export function projectileLaunchPoint(
  type: EntityType,
  x: number,
  z: number,
  faceX: number,
  faceZ: number,
  visualYOffset = 0,
): ProjectilePoint {
  const offset = PROJECTILE_OFFSETS[type] ?? DEFAULT_PROJECTILE_OFFSET;
  const length = Math.hypot(faceX, faceZ);
  const fx = length > 1e-6 ? faceX / length : 0;
  const fz = length > 1e-6 ? faceZ / length : 1;

  // The local right vector is forward rotated 90 degrees around world up.
  const rx = fz;
  const rz = -fx;
  return {
    x: x + fx * offset.forward + rx * offset.right,
    y: offset.up + visualYOffset,
    z: z + fz * offset.forward + rz * offset.right,
  };
}

/**
 * Which of an attack's victims it was actually aimed at.
 *
 * The events say who was hurt, not who was shot at, and for a splash weapon
 * those are different lists: four of the five units a Sentry shell caught were
 * simply standing near the fifth. Drawing a shell at each of them fires five
 * mortars at one tile.
 *
 * The answer is in the attacker's facing. `resolveAttackImpact` turns the
 * attacker onto its primary target on the tick it fires, before it gathers
 * anything the blast also reached, so the victim best lined up with where the
 * attacker is now pointing is the one it shot at. Ties and a victim standing on
 * the attacker both resolve to the first candidate, which is as good an answer
 * as exists when the geometry has none.
 *
 * Facing arrives from fixed point and may be a fraction off unit length; that
 * scales every candidate's score equally, so it cannot change the winner.
 */
export function primaryVictimIndex(
  originX: number,
  originZ: number,
  faceX: number,
  faceZ: number,
  victims: readonly { x: number; z: number }[],
): number {
  let best = -Infinity;
  let primary = 0;
  for (let i = 0; i < victims.length; i++) {
    const dx = victims[i]!.x - originX;
    const dz = victims[i]!.z - originZ;
    const len = Math.hypot(dx, dz);
    const aim = len < 1e-4 ? Infinity : (dx * faceX + dz * faceZ) / len;
    if (aim > best) {
      best = aim;
      primary = i;
    }
  }
  return primary;
}

/** Height at which a shot and its impact effect meet the target. */
export function projectileImpactPoint(
  type: EntityType,
  x: number,
  z: number,
  visualYOffset = 0,
): ProjectilePoint {
  const def = defOf(type);
  return {
    x,
    // Ground combatants keep the established chest-height hit. Aircraft need
    // an elevated endpoint or incoming fire still appears beneath the model;
    // buildings are tall enough that chest height is their doorstep.
    y: (def.flying ? FLIGHT_ALTITUDE + 0.3 : def.isBuilding ? 1 : 0.5) + visualYOffset,
    z,
  };
}

/** Which buffer a particle is drawn into, and so how it reads. */
enum Layer {
  /** Soft additive light: flashes, embers, motes, blooms. */
  Glow,
  /** Additive fire, with a noisy eaten-away edge. */
  Flame,
  /** Blended smoke and dust, which darkens rather than lights. */
  Smoke,
  /** A glow stretched along its own velocity: sparks and shards. */
  Streak,
}

interface Particle {
  active: boolean;
  layer: Layer;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  /** Fraction of speed shed per second, as an exponential decay rate. */
  drag: number;
  gravity: number;
  age: number;
  life: number;
  /** World-unit diameter at birth and at death. */
  size0: number;
  size1: number;
  rot: number;
  spin: number;
  r0: number;
  g0: number;
  b0: number;
  r1: number;
  g1: number;
  b1: number;
  alpha: number;
  /** Fraction of life spent fading in. Zero is lit at full brightness. */
  flare: number;
  /** Streak length per unit of speed. Only read by `Layer.Streak`. */
  stretch: number;
}

/** Something in flight between a muzzle and whatever it is about to hit. */
interface Ordnance {
  active: boolean;
  profile: WeaponProfile;
  x0: number;
  y0: number;
  z0: number;
  x1: number;
  y1: number;
  z1: number;
  /** Apex height above the straight line, for a lobbed shot. Zero is flat. */
  arc: number;
  age: number;
  flight: number;
  r: number;
  g: number;
  b: number;
  /** Blast radius to hand the impact, in world units. Zero is a point hit. */
  splash: number;
  /** Seconds owed to the trail emitter, carried between frames. */
  emitDebt: number;
}

/** A ribbon on screen for a moment: a beam, a lightning segment, a slash. */
interface Beam {
  active: boolean;
  x0: number;
  y0: number;
  z0: number;
  x1: number;
  y1: number;
  z1: number;
  age: number;
  life: number;
  width: number;
  r: number;
  g: number;
  b: number;
  /** Width multiplier of the soft second pass. Zero draws the core alone. */
  halo: number;
  /** Depth of the brightness wobble, 0 to 1. */
  flicker: number;
  /** Phase, so two beams born together do not pulse in lockstep. */
  seed: number;
  headFade: number;
  tailFade: number;
}

enum DecalKind {
  /** Additive shockwave, occluded by whatever is standing on it. */
  Ring,
  /** Blended burn, likewise occluded. */
  Scorch,
  /** Order feedback, drawn through everything because it is UI. */
  Marker,
}

interface Decal {
  active: boolean;
  kind: DecalKind;
  x: number;
  z: number;
  age: number;
  life: number;
  r0: number;
  r1: number;
  rot: number;
  r: number;
  g: number;
  b: number;
  alpha: number;
  flare: number;
}

/** A tumbling chunk thrown out when something dies. */
interface Debris {
  active: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  spin: number;
  axisX: number;
  axisY: number;
  axisZ: number;
  age: number;
  life: number;
  size: number;
  aspect: number;
  colour: number;
  /** Chunks off a burning building trail smoke; infantry shrapnel does not. */
  smoking: boolean;
  emitDebt: number;
}

/**
 * Parameters for the next particle, as one reused object.
 *
 * An explosion emits dozens of particles in a frame and a battle emits
 * hundreds. A fresh options literal each time is garbage the frame loop does
 * not need to make, and seventeen positional arguments is not code anyone can
 * read. So: take the defaults from `emitter`, overwrite what differs, and call
 * `emit`. It is only ever live between those two calls.
 */
const EMIT = {
  layer: Layer.Glow,
  x: 0,
  y: 0,
  z: 0,
  vx: 0,
  vy: 0,
  vz: 0,
  drag: 0,
  gravity: 0,
  age: 0,
  life: 0.3,
  size0: 0.3,
  size1: 0.3,
  rot: 0,
  spin: 0,
  from: HOT,
  to: HOT,
  alpha: 1,
  flare: 0,
  stretch: 0,
};

/** Reset `EMIT` to its defaults for one particle. */
function emitter(layer: Layer, age: number): typeof EMIT {
  EMIT.layer = layer;
  EMIT.x = 0;
  EMIT.y = 0;
  EMIT.z = 0;
  EMIT.vx = 0;
  EMIT.vy = 0;
  EMIT.vz = 0;
  EMIT.drag = 0;
  EMIT.gravity = 0;
  EMIT.age = age;
  EMIT.life = 0.3;
  EMIT.size0 = 0.3;
  EMIT.size1 = 0.3;
  EMIT.rot = Math.random() * Math.PI * 2;
  EMIT.spin = 0;
  EMIT.from = HOT;
  EMIT.to = HOT;
  EMIT.alpha = 1;
  EMIT.flare = 0;
  EMIT.stretch = 0;
  return EMIT;
}

/** Uniform sample in `[min, max)`. */
function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/** Uniform sample in `[-span, span)`. */
function spread(span: number): number {
  return (Math.random() * 2 - 1) * span;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function ease(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Advance a particle's ballistics. Shared by the frame step and catch-up. */
function advance(p: Particle, dt: number): void {
  if (dt <= 0) return;
  if (p.gravity !== 0) p.vy -= p.gravity * dt;
  if (p.drag > 0) {
    const keep = Math.exp(-p.drag * dt);
    p.vx *= keep;
    p.vy *= keep;
    p.vz *= keep;
  }
  p.x += p.vx * dt;
  p.y += p.vy * dt;
  p.z += p.vz * dt;
  p.rot += p.spin * dt;
}

export class ProjectileRenderer {
  readonly group = new THREE.Group();

  private readonly pendingShots: PendingShot[] = [];

  private readonly particles: Particle[] = [];
  private readonly ordnance: Ordnance[] = [];
  private readonly beams: Beam[] = [];
  private readonly decals: Decal[] = [];
  private readonly debris: Debris[] = [];
  private readonly markers: Decal[] = [];

  private particleCursor = 0;
  private ordnanceCursor = 0;
  private beamCursor = 0;
  private decalCursor = 0;
  private debrisCursor = 0;
  private markerCursor = 0;

  private readonly glowField: SpriteField;
  private readonly flameField: SpriteField;
  private readonly smokeField: SpriteField;
  private readonly beamField: BeamField;
  private readonly ringField: GroundField;
  private readonly scorchField: GroundField;
  private readonly markerField: GroundField;
  private readonly debrisMesh: THREE.InstancedMesh;

  private readonly disposables: { dispose(): void }[] = [];

  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3();
  private readonly axis = new THREE.Vector3();
  private readonly colour = new THREE.Color();
  private readonly tint = new THREE.Color();
  private readonly hotScratch = new THREE.Color();

  constructor() {
    const glow = glowTexture();
    const flame = flameTexture();
    const smoke = smokeTexture();
    const beam = beamTexture();
    const ring = ringTexture();
    const scorch = scorchTexture();

    // Render order within the transparent pass, low to high: burns on the
    // ground, then shockwaves over them, then the fog plane at 5, then smoke,
    // fire and light. Anything at or below the fog's order is covered by the
    // shroud the way terrain is; anything above it is a muzzle flash, which
    // has to stay visible over the unit that made it.
    this.scorchField = new GroundField(scorch, DECAL_CAPACITY, 'normal', true, 2);
    this.ringField = new GroundField(ring, DECAL_CAPACITY, 'additive', true, 3);
    this.smokeField = new SpriteField(smoke, PARTICLE_CAPACITY, 'normal', 8);
    this.flameField = new SpriteField(flame, PARTICLE_CAPACITY, 'additive', 9);
    this.beamField = new BeamField(beam, PARTICLE_CAPACITY, 10);
    this.glowField = new SpriteField(glow, PARTICLE_CAPACITY, 'additive', 10);
    this.markerField = new GroundField(ring, MARKER_CAPACITY, 'additive', false, 11);

    // Debris is solid matter, so it is lit and opaque like the units it came
    // off — the one thing here that is not a light or a stain.
    const debrisGeo = new THREE.BoxGeometry(1, 1, 1);
    const debrisMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.debrisMesh = new THREE.InstancedMesh(debrisGeo, debrisMat, DEBRIS_CAPACITY);
    this.debrisMesh.frustumCulled = false;
    this.debrisMesh.count = 0;

    for (const field of [
      this.scorchField,
      this.ringField,
      this.smokeField,
      this.flameField,
      this.beamField,
      this.glowField,
      this.markerField,
    ]) {
      this.group.add(field.mesh);
      this.disposables.push(field);
    }
    this.group.add(this.debrisMesh);
    this.disposables.push(glow, flame, smoke, beam, ring, scorch, debrisGeo, debrisMat);

    for (let i = 0; i < PARTICLE_CAPACITY; i++) this.particles.push(blankParticle());
    for (let i = 0; i < ORDNANCE_CAPACITY; i++) this.ordnance.push(blankOrdnance());
    for (let i = 0; i < BEAM_CAPACITY; i++) this.beams.push(blankBeam());
    for (let i = 0; i < DECAL_CAPACITY; i++) this.decals.push(blankDecal());
    for (let i = 0; i < DEBRIS_CAPACITY; i++) this.debris.push(blankDebris());
    for (let i = 0; i < MARKER_CAPACITY; i++) this.markers.push(blankDecal());
  }

  /**
   * `colour` pushed `amount` of the way to white, in the shared hot scratch.
   *
   * What a bright core is: the same colour as the glow around it, blown out.
   * Returned rather than stored, and only valid until the next call — every
   * caller hands it straight to a field, which copies it.
   */
  private hotter(colour: THREE.Color, amount: number): THREE.Color {
    return this.hotScratch.copy(colour).lerp(WHITE, amount);
  }

  // -------------------------------------------------------------------------
  // Emitters
  // -------------------------------------------------------------------------

  /** Commit `EMIT` to the pool. */
  private emit(): void {
    const p = this.particles[this.particleCursor]!;
    this.particleCursor = (this.particleCursor + 1) % PARTICLE_CAPACITY;
    p.active = true;
    p.layer = EMIT.layer;
    p.x = EMIT.x;
    p.y = EMIT.y;
    p.z = EMIT.z;
    p.vx = EMIT.vx;
    p.vy = EMIT.vy;
    p.vz = EMIT.vz;
    p.drag = EMIT.drag;
    p.gravity = EMIT.gravity;
    p.age = EMIT.age;
    p.life = EMIT.life;
    p.size0 = EMIT.size0;
    p.size1 = EMIT.size1;
    p.rot = EMIT.rot;
    p.spin = EMIT.spin;
    p.r0 = EMIT.from.r;
    p.g0 = EMIT.from.g;
    p.b0 = EMIT.from.b;
    p.r1 = EMIT.to.r;
    p.g1 = EMIT.to.g;
    p.b1 = EMIT.to.b;
    p.alpha = EMIT.alpha;
    p.flare = EMIT.flare;
    p.stretch = EMIT.stretch;
    // An effect inherited from an older tick has already been alive for a
    // while, so catch its ballistics up rather than starting it where it was
    // born. One step is plenty over the quarter-second this can span.
    if (p.age > 0) advance(p, p.age);
  }

  private addBeam(
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    width: number,
    colour: THREE.Color,
    life: number,
    age: number,
    halo = 1.8,
    flicker = 0,
    headFade = 0.02,
    tailFade = 0.02,
  ): void {
    const b = this.beams[this.beamCursor]!;
    this.beamCursor = (this.beamCursor + 1) % BEAM_CAPACITY;
    b.active = true;
    b.x0 = x0;
    b.y0 = y0;
    b.z0 = z0;
    b.x1 = x1;
    b.y1 = y1;
    b.z1 = z1;
    b.age = age;
    b.life = life;
    b.width = width;
    b.r = colour.r;
    b.g = colour.g;
    b.b = colour.b;
    b.halo = halo;
    b.flicker = flicker;
    b.seed = Math.random() * 100;
    b.headFade = headFade;
    b.tailFade = tailFade;
  }

  private addDecal(
    kind: DecalKind,
    x: number,
    z: number,
    r0: number,
    r1: number,
    colour: THREE.Color,
    alpha: number,
    life: number,
    age: number,
    flare = 0,
  ): void {
    const pool = kind === DecalKind.Marker ? this.markers : this.decals;
    const cursor = kind === DecalKind.Marker ? this.markerCursor : this.decalCursor;
    const d = pool[cursor]!;
    if (kind === DecalKind.Marker) this.markerCursor = (cursor + 1) % MARKER_CAPACITY;
    else this.decalCursor = (cursor + 1) % DECAL_CAPACITY;
    d.active = true;
    d.kind = kind;
    d.x = x;
    d.z = z;
    d.age = age;
    d.life = life;
    d.r0 = r0;
    d.r1 = r1;
    d.rot = Math.random() * Math.PI * 2;
    d.r = colour.r;
    d.g = colour.g;
    d.b = colour.b;
    d.alpha = alpha;
    d.flare = flare;
  }

  // -------------------------------------------------------------------------
  // Effect vocabulary
  // -------------------------------------------------------------------------

  /** Struck metal: fast, short-lived, falling, and always warm. */
  private sparks(
    x: number,
    y: number,
    z: number,
    count: number,
    speed: number,
    age: number,
    dirX = 0,
    dirY = 0,
    dirZ = 0,
    cone = 1,
  ): void {
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(rand(-1, 1));
      const sx = Math.sin(phi) * Math.cos(theta);
      const sy = Math.cos(phi);
      const sz = Math.sin(phi) * Math.sin(theta);
      const v = speed * rand(0.45, 1);
      const e = emitter(Layer.Streak, age);
      e.x = x;
      e.y = y;
      e.z = z;
      // `cone` bends the scatter back along a direction — the spray off a
      // ricochet, rather than a firework.
      e.vx = (sx * (1 - cone) + dirX * cone) * v;
      e.vy = (sy * (1 - cone) + dirY * cone) * v + rand(0.5, 2.2);
      e.vz = (sz * (1 - cone) + dirZ * cone) * v;
      e.gravity = GRAVITY * 1.4;
      e.drag = 1.6;
      e.life = rand(0.2, 0.42);
      e.size0 = rand(0.09, 0.15);
      e.size1 = 0.03;
      e.from = HOT;
      e.to = EMBER;
      e.stretch = 0.075;
      e.alpha = 1;
      this.emit();
    }
  }

  /** Ground thrown up around a hit: slow, wide, and the only cold grey here. */
  private dust(x: number, z: number, radius: number, count: number, age: number): void {
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const v = radius * rand(1.1, 2.4);
      const e = emitter(Layer.Smoke, age);
      e.x = x + Math.cos(theta) * radius * 0.25;
      e.y = rand(0.06, 0.3);
      e.z = z + Math.sin(theta) * radius * 0.25;
      e.vx = Math.cos(theta) * v;
      e.vy = rand(0.3, 1.1);
      e.vz = Math.sin(theta) * v;
      e.drag = 2.6;
      e.life = rand(0.45, 0.85);
      e.size0 = radius * rand(0.4, 0.7);
      e.size1 = radius * rand(1.3, 2.1);
      e.from = DUST;
      e.to = SMOKE_DARK;
      e.alpha = 0.55;
      e.flare = 0.18;
      e.spin = spread(1.4);
      this.emit();
    }
  }

  /**
   * The shared explosion.
   *
   * Every blast in the game is this with a different radius and energy: the
   * splash off a Sentry shell, a Boomwalker going off, a Command Post falling.
   * Five layers, in the order they become visible — flash, fireball, sparks,
   * shockwave, smoke — plus a burn left behind.
   */
  private blast(
    x: number,
    y: number,
    z: number,
    radius: number,
    energy: number,
    age: number,
  ): void {
    // The white is a flashbulb, not the explosion: it is gone in a tenth of a
    // second and what is left is fire. Held any longer, every death on the
    // field is the same white disc.
    const flash = emitter(Layer.Glow, age);
    flash.x = x;
    flash.y = y;
    flash.z = z;
    flash.life = 0.1 + energy * 0.03;
    flash.size0 = radius * 0.7;
    flash.size1 = radius * 1.7;
    flash.from = HOT;
    flash.to = FIRE;
    flash.alpha = 1;
    this.emit();

    const fireballs = Math.round(4 + energy * 4);
    for (let i = 0; i < fireballs; i++) {
      const theta = Math.random() * Math.PI * 2;
      const reach = radius * rand(0, 0.45);
      const e = emitter(Layer.Flame, age + rand(0, 0.05));
      e.x = x + Math.cos(theta) * reach;
      e.y = y + rand(-0.15, 0.4) * radius;
      e.z = z + Math.sin(theta) * reach;
      e.vx = Math.cos(theta) * radius * rand(0.5, 1.4);
      e.vy = rand(0.5, 1.9);
      e.vz = Math.sin(theta) * radius * rand(0.5, 1.4);
      e.drag = 3.4;
      e.life = rand(0.32, 0.58);
      e.size0 = radius * rand(0.4, 0.7);
      e.size1 = radius * rand(0.85, 1.3);
      // One in five starts white-hot. Any more and a fireball is a white ball
      // with an orange rim rather than fire.
      e.from = i % 5 === 0 ? HOT : FIRE;
      e.to = EMBER;
      e.alpha = 0.95;
      e.spin = spread(2.2);
      this.emit();
    }

    this.sparks(x, y, z, Math.round(6 + energy * 7), 3 + energy * 2.5 + radius * 2, age);

    // The ring runs just outside the fire, which on a splash weapon makes it
    // an honest picture of how far the damage reached.
    this.addDecal(
      DecalKind.Ring,
      x,
      z,
      radius * 0.3,
      radius * 1.2,
      HOT,
      0.6,
      0.26 + energy * 0.04,
      age,
    );

    const smokes = Math.round(3 + energy * 3);
    for (let i = 0; i < smokes; i++) {
      const theta = Math.random() * Math.PI * 2;
      const e = emitter(Layer.Smoke, age + rand(0, 0.08));
      e.x = x + Math.cos(theta) * radius * rand(0, 0.45);
      e.y = y + rand(0, 0.45) * radius;
      e.z = z + Math.sin(theta) * radius * rand(0, 0.45);
      e.vx = Math.cos(theta) * radius * rand(0.2, 0.7);
      e.vy = rand(0.8, 1.9);
      e.vz = Math.sin(theta) * radius * rand(0.2, 0.7);
      e.drag = 1.6;
      e.life = rand(0.8, 1.4);
      e.size0 = radius * rand(0.6, 0.9);
      e.size1 = radius * rand(1.5, 2.3);
      e.from = SMOKE;
      e.to = SMOKE_DARK;
      e.alpha = 0.6;
      e.flare = 0.3;
      e.spin = spread(1);
      this.emit();
    }

    this.dust(x, z, radius * 0.8, Math.round(2 + energy * 2), age);
    this.addDecal(
      DecalKind.Scorch,
      x,
      z,
      radius * 0.65,
      radius * 0.85,
      UNTINTED,
      Math.min(1, 0.6 + energy * 0.2),
      SCORCH_LIFE_S,
      age,
    );
  }

  /** Jagged lightning between two points, as a chain of short ribbons. */
  private lightning(
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    width: number,
    colour: THREE.Color,
    life: number,
    age: number,
    jitter: number,
  ): void {
    const segments = 6;
    let px = x0;
    let py = y0;
    let pz = z0;
    for (let i = 1; i <= segments; i++) {
      const t = i / segments;
      // The ends are pinned — a bolt that misses the muzzle or the target is
      // not a near miss, it is a bug the player can see.
      const wander = i === segments ? 0 : jitter * Math.sin(t * Math.PI) * rand(0.5, 1.5);
      const nx = x0 + (x1 - x0) * t + spread(wander);
      const ny = y0 + (y1 - y0) * t + spread(wander * 0.7);
      const nz = z0 + (z1 - z0) * t + spread(wander);
      // Thinning toward the far end stops the chain reading as bent pipe.
      this.addBeam(px, py, pz, nx, ny, nz, width * (1.2 - 0.55 * t), colour, life, age, 2.2, 0.55);
      px = nx;
      py = ny;
      pz = nz;
    }
  }

  // -------------------------------------------------------------------------
  // Weapon stages
  // -------------------------------------------------------------------------

  /** Stage one: the flash, smoke and kick at the barrel. */
  private muzzle(
    profile: WeaponProfile,
    colour: THREE.Color,
    x: number,
    y: number,
    z: number,
    dx: number,
    dy: number,
    dz: number,
    age: number,
  ): void {
    const scale = 0.28 + profile.impact * 0.22;

    const flash = emitter(Layer.Glow, age);
    flash.x = x + dx * 0.08;
    flash.y = y + dy * 0.08;
    flash.z = z + dz * 0.08;
    flash.life = 0.075;
    flash.size0 = scale * 1.5;
    flash.size1 = scale * 0.5;
    flash.from = HOT;
    flash.to = colour;
    flash.alpha = 1;
    this.emit();

    // A stub of flame out of the barrel, which is what gives a shot a
    // direction at the shooter's end as well as the target's.
    this.addBeam(
      x,
      y,
      z,
      x + dx * scale * 1.6,
      y + dy * scale * 1.6,
      z + dz * scale * 1.6,
      profile.width * 2.4 + 0.06,
      colour,
      0.07,
      age,
      1.6,
      0,
      0.02,
      0.6,
    );

    this.sparks(x, y, z, 2, 3.5 * scale, age, dx, dy, dz, 0.75);

    if (profile.impact >= 1.1) {
      const e = emitter(Layer.Smoke, age);
      e.x = x + dx * 0.3;
      e.y = y + dy * 0.3;
      e.z = z + dz * 0.3;
      e.vx = dx * 1.6;
      e.vy = dy * 1.6 + 0.5;
      e.vz = dz * 1.6;
      e.drag = 3;
      e.life = 0.45;
      e.size0 = scale * 0.5;
      e.size1 = scale * 1.9;
      e.from = SMOKE;
      e.to = SMOKE_DARK;
      e.alpha = 0.3;
      e.flare = 0.2;
      this.emit();
    }
  }

  /** Stage two: put something in the air between muzzle and target. */
  private launch(
    profile: WeaponProfile,
    colour: THREE.Color,
    from: ProjectilePoint,
    to: ProjectilePoint,
    splash: number,
    age: number,
  ): void {
    const dist = Math.hypot(to.x - from.x, to.y - from.y, to.z - from.z);
    const lobbed = profile.style === 'mortar' || profile.style === 'plasma';
    const o = this.ordnance[this.ordnanceCursor]!;
    this.ordnanceCursor = (this.ordnanceCursor + 1) % ORDNANCE_CAPACITY;
    o.active = true;
    o.profile = profile;
    o.x0 = from.x;
    o.y0 = from.y;
    o.z0 = from.z;
    o.x1 = to.x;
    o.y1 = to.y;
    o.z1 = to.z;
    // A mortar has to clear whatever the Sentry is hiding behind, and a shell
    // that leaves the tube at the angle it lands at is not artillery.
    o.arc = lobbed ? (profile.style === 'mortar' ? dist * 0.3 + 1.2 : dist * 0.12 + 0.35) : 0;
    o.age = age;
    o.flight = Math.max(0.05, dist / Math.max(1, profile.speed));
    o.r = colour.r;
    o.g = colour.g;
    o.b = colour.b;
    o.splash = splash;
    o.emitDebt = 0;
  }

  /** Stage three and four: what arrival looks like, and what it leaves. */
  private impact(
    profile: WeaponProfile,
    colour: THREE.Color,
    x: number,
    y: number,
    z: number,
    dx: number,
    dy: number,
    dz: number,
    splash: number,
    age: number,
  ): void {
    const energy = profile.impact;

    if (splash > 0) {
      this.blast(x, Math.max(y, 0.35), z, splash, energy, age);
      if (profile.style === 'plasma') {
        // An energy burst throws its own colour about, and crackles instead of
        // smouldering. The blast above gives it weight; this gives it a hue.
        this.addDecal(DecalKind.Ring, x, z, splash * 0.3, splash * 1.35, colour, 0.7, 0.3, age);
        for (let i = 0; i < 6; i++) {
          const theta = Math.random() * Math.PI * 2;
          this.lightning(
            x,
            y,
            z,
            x + Math.cos(theta) * splash * rand(0.45, 0.85),
            y + rand(-0.3, 0.1),
            z + Math.sin(theta) * splash * rand(0.45, 0.85),
            0.06,
            colour,
            0.18,
            age,
            0.22,
          );
        }
      }
      return;
    }

    const flash = emitter(Layer.Glow, age);
    flash.x = x;
    flash.y = y;
    flash.z = z;
    flash.life = 0.16 + energy * 0.05;
    flash.size0 = 0.45 + energy * 0.5;
    flash.size1 = 1.15 + energy * 1.4;
    flash.from = HOT;
    flash.to = profile.style === 'frost' ? FROST : colour;
    flash.alpha = 1;
    this.emit();

    // Sparks come back off the surface, so they scatter against the shot.
    this.sparks(
      x,
      y,
      z,
      Math.round(3 + energy * 5),
      3 + energy * 3,
      age,
      -dx,
      -dy,
      -dz,
      profile.style === 'lance' ? 0.75 : 0.45,
    );

    switch (profile.style) {
      case 'frost': {
        this.addDecal(DecalKind.Ring, x, z, 0.3, 1.25, FROST, 0.55, 0.45, age);
        for (let i = 0; i < 7; i++) {
          const theta = Math.random() * Math.PI * 2;
          const e = emitter(Layer.Streak, age);
          e.x = x;
          e.y = y;
          e.z = z;
          e.vx = Math.cos(theta) * rand(1.5, 4.5);
          e.vy = rand(1, 3.5);
          e.vz = Math.sin(theta) * rand(1.5, 4.5);
          e.gravity = GRAVITY * 0.8;
          e.drag = 1.2;
          e.life = rand(0.3, 0.55);
          e.size0 = rand(0.05, 0.1);
          e.size1 = 0.015;
          e.from = HOT;
          e.to = FROST;
          e.stretch = 0.04;
          this.emit();
        }
        break;
      }
      case 'flame': {
        for (let i = 0; i < 3; i++) {
          const e = emitter(Layer.Flame, age + rand(0, 0.05));
          e.x = x + spread(0.25);
          e.y = y + rand(-0.1, 0.35);
          e.z = z + spread(0.25);
          e.vy = rand(1.2, 2.6);
          e.drag = 2.4;
          e.life = rand(0.25, 0.45);
          e.size0 = 0.4;
          e.size1 = 1.05;
          e.from = FIRE;
          e.to = EMBER;
          e.alpha = 0.8;
          e.spin = spread(2);
          this.emit();
        }
        break;
      }
      case 'beam':
      case 'lance':
      case 'arc': {
        // Energy weapons boil what they hit rather than shattering it.
        const e = emitter(Layer.Smoke, age);
        e.x = x;
        e.y = y + 0.1;
        e.z = z;
        e.vy = rand(1.4, 2.4);
        e.drag = 2;
        e.life = rand(0.35, 0.6);
        e.size0 = 0.22;
        e.size1 = 0.85;
        e.from = SMOKE;
        e.to = SMOKE_DARK;
        e.alpha = 0.3;
        e.flare = 0.25;
        this.emit();
        break;
      }
      default:
        this.dust(x, z, 0.35 + energy * 0.2, 1, age);
        break;
    }
  }

  /** A weapon with no travel time: the blow lands where the target stands. */
  private melee(
    profile: WeaponProfile,
    colour: THREE.Color,
    from: ProjectilePoint,
    to: ProjectilePoint,
    age: number,
  ): void {
    const dx = to.x - from.x;
    const dz = to.z - from.z;
    const len = Math.hypot(dx, dz) || 1;
    const nx = dx / len;
    const nz = dz / len;
    // Across the blow, not along it: a swing reads by the arc it sweeps.
    const ax = -nz;
    const az = nx;
    const reach = 0.45 + profile.impact * 0.3;

    for (let i = 0; i < 3; i++) {
      const offset = (i - 1) * 0.16;
      const lift = 0.26 - Math.abs(offset) * 0.5;
      this.addBeam(
        to.x - ax * reach + nx * offset,
        to.y + lift + offset * 0.4,
        to.z - az * reach + nz * offset,
        to.x + ax * reach + nx * offset,
        to.y - lift * 0.5 + offset * 0.4,
        to.z + az * reach + nz * offset,
        profile.width * (i === 1 ? 1 : 0.55),
        colour,
        profile.life,
        age + i * 0.012,
        2,
        0,
        0.3,
        0.3,
      );
    }

    this.impact(profile, colour, to.x, to.y, to.z, nx, 0, nz, 0, age + 0.02);
    if (profile.impact >= 1.5) this.dust(to.x, to.z, 0.55, 3, age);
  }

  /** A cone of fire from the nozzle, rather than anything that travels. */
  private flame(
    profile: WeaponProfile,
    colour: THREE.Color,
    from: ProjectilePoint,
    to: ProjectilePoint,
    age: number,
  ): void {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = to.z - from.z;
    const reach = Math.hypot(dx, dy, dz) || 1;
    const nx = dx / reach;
    const ny = dy / reach;
    const nz = dz / reach;
    const sideX = -nz;
    const sideZ = nx;

    for (let i = 0; i < 22; i++) {
      const t = i / 22;
      const fan = spread(0.22);
      const v = profile.speed * rand(0.75, 1.25);
      const e = emitter(Layer.Flame, age + t * 0.05);
      e.x = from.x + nx * 0.15;
      e.y = from.y + ny * 0.15;
      e.z = from.z + nz * 0.15;
      e.vx = (nx + sideX * fan) * v;
      e.vy = ny * v + rand(0.2, 1.1);
      e.vz = (nz + sideZ * fan) * v;
      e.drag = rand(3.4, 5);
      e.life = profile.life * rand(0.7, 1.25);
      e.size0 = rand(0.24, 0.42);
      e.size1 = rand(1.2, 1.9);
      // The near end of a flame is white, the far end is soot. Alternating the
      // start colour keeps the cone from banding into one flat sheet.
      e.from = i % 4 === 0 ? HOT : FIRE;
      e.to = i % 3 === 0 ? SMOKE_DARK : EMBER;
      e.alpha = 0.9;
      e.spin = spread(3);
      e.flare = 0.1;
      this.emit();
    }

    // Unburnt fuel: a few embers that outlive the flame and fall.
    for (let i = 0; i < 4; i++) {
      const e = emitter(Layer.Streak, age);
      e.x = from.x;
      e.y = from.y;
      e.z = from.z;
      e.vx = nx * profile.speed * rand(0.5, 1) + spread(1.5);
      e.vy = ny * profile.speed * 0.5 + rand(1, 2.5);
      e.vz = nz * profile.speed * rand(0.5, 1) + spread(1.5);
      e.gravity = GRAVITY * 0.7;
      e.drag = 1.2;
      e.life = rand(0.4, 0.8);
      e.size0 = 0.07;
      e.size1 = 0.02;
      e.from = FIRE;
      e.to = EMBER;
      e.stretch = 0.05;
      this.emit();
    }

    const glow = emitter(Layer.Glow, age);
    glow.x = from.x + nx * 0.2;
    glow.y = from.y + ny * 0.2;
    glow.z = from.z + nz * 0.2;
    glow.life = 0.12;
    glow.size0 = 0.8;
    glow.size1 = 0.3;
    glow.from = HOT;
    glow.to = colour;
    this.emit();
  }

  /** The Fixomatic's beam: help, drawn so it cannot be mistaken for harm. */
  private repair(
    profile: WeaponProfile,
    colour: THREE.Color,
    from: ProjectilePoint,
    to: ProjectilePoint,
    age: number,
  ): void {
    this.addBeam(
      from.x,
      from.y,
      from.z,
      to.x,
      to.y,
      to.z,
      profile.width,
      colour,
      profile.life,
      age,
      2.2,
      0.4,
      0.08,
      0.08,
    );

    // Motes running up the beam, then a soft bloom where the mending lands.
    for (let i = 0; i < 4; i++) {
      const t = i / 4;
      const e = emitter(Layer.Glow, age - t * profile.life * 0.5);
      e.x = from.x + (to.x - from.x) * t;
      e.y = from.y + (to.y - from.y) * t;
      e.z = from.z + (to.z - from.z) * t;
      e.vx = (to.x - from.x) / profile.life;
      e.vy = (to.y - from.y) / profile.life + 0.6;
      e.vz = (to.z - from.z) / profile.life;
      e.drag = 1.5;
      e.life = profile.life * 0.8;
      e.size0 = 0.16;
      e.size1 = 0.05;
      e.from = HOT;
      e.to = colour;
      e.alpha = 0.9;
      this.emit();
    }

    const bloom = emitter(Layer.Glow, age);
    bloom.x = to.x;
    bloom.y = to.y;
    bloom.z = to.z;
    bloom.life = profile.life;
    bloom.size0 = 0.35;
    bloom.size1 = 1.1;
    bloom.from = colour;
    bloom.to = colour;
    bloom.alpha = 0.55;
    bloom.flare = 0.25;
    this.emit();

    for (let i = 0; i < 3; i++) {
      const e = emitter(Layer.Glow, age + rand(0, 0.08));
      e.x = to.x + spread(0.35);
      e.y = to.y - 0.2;
      e.z = to.z + spread(0.35);
      e.vy = rand(1.2, 2.2);
      e.life = rand(0.35, 0.6);
      e.size0 = 0.14;
      e.size1 = 0.04;
      e.from = HOT;
      e.to = colour;
      e.alpha = 0.8;
      this.emit();
    }
  }

  // -------------------------------------------------------------------------
  // Simulation events in
  // -------------------------------------------------------------------------

  /**
   * Blow up everything that died this tick.
   *
   * Buildings get a bigger, slower, dirtier version of the same explosion than
   * infantry — a Command Post falling should not look like a Burstbot being
   * shot — and a Boomwalker gets one sized to the blast the simulation actually
   * applied, because that radius is the unit's whole reason for existing.
   */
  spawnDeaths(world: World, canSee?: (index: number) => boolean): void {
    const pool = world.pool;
    const deaths = world.events.deaths;

    for (let k = 0; k < deaths.length; k++) {
      const i = deaths[k]!;
      if (canSee && !canSee(i)) continue;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      // Exhausted mineral patches vanish rather than exploding.
      if (type === EntityType.MineralPatch) continue;

      const owner = pool.owner[i]!;
      const teamColour = PLAYER_COLOURS[colourSlotFor(owner, world.players.length)] ?? 0x9aa4b2;
      const x = toFloat(pool.posX[i]!);
      const z = toFloat(pool.posY[i]!);
      const big = def.isBuilding;
      const splash = toFloat(def.splashRadius);
      const footprint = big ? def.footprint * 0.5 : toFloat(def.radius);

      if (def.detonates && splash > 0) {
        // Not a death so much as the weapon going off, so it is sized by the
        // blast the player has to learn to step out of.
        this.blast(x, 0.5, z, splash, 2.6, 0);
      } else if (big) {
        // A building comes apart in stages across its own footprint rather
        // than in one ball, which is what makes it read as large.
        const spread3 = footprint * 0.8;
        for (let n = 0; n < 3; n++) {
          this.blast(
            x + spread(spread3),
            rand(0.4, 1.6),
            z + spread(spread3),
            footprint * rand(0.7, 1.1),
            2.2,
            -n * 0.09,
          );
        }
      } else {
        this.blast(x, 0.55, z, Math.max(0.7, footprint * 1.8), 1, 0);
      }

      const chunks = big ? 22 : 9;
      for (let n = 0; n < chunks; n++) {
        const d = this.debris[this.debrisCursor]!;
        this.debrisCursor = (this.debrisCursor + 1) % DEBRIS_CAPACITY;
        const angle = Math.random() * Math.PI * 2;
        const speed = (big ? 2.1 : 2.6) * rand(0.35, 1);
        d.active = true;
        d.x = x + spread(big ? footprint * 0.6 : 0.15);
        d.y = big ? rand(0.5, 1.4) : 0.45;
        d.z = z + spread(big ? footprint * 0.6 : 0.15);
        d.vx = Math.cos(angle) * speed;
        d.vz = Math.sin(angle) * speed;
        d.vy = (big ? 4 : 3.3) * rand(0.5, 1.2);
        d.spin = spread(9);
        this.axis.set(rand(-1, 1), rand(-1, 1), rand(-1, 1));
        if (this.axis.lengthSq() < 1e-6) this.axis.set(0, 1, 0);
        this.axis.normalize();
        d.axisX = this.axis.x;
        d.axisY = this.axis.y;
        d.axisZ = this.axis.z;
        d.age = 0;
        d.life = rand(0.7, 1.3);
        d.size = (big ? 0.3 : 0.15) * rand(0.6, 1.5);
        // Shards rather than dice: a cube tumbling is a cube, but a slab
        // tumbling is wreckage.
        d.aspect = rand(0.3, 1);
        d.colour = teamColour;
        d.smoking = big || n % 4 === 0;
        d.emitDebt = 0;
      }
    }
  }

  /**
   * Drop a ring where the player clicked.
   *
   * Order feedback is easy to skip and disproportionately important: without it
   * there is no way to tell a missed click from a unit that simply has not
   * started moving yet. Two rings, one opening outward and one closing in on
   * the point, so it reads as an order landing on a spot rather than as another
   * selection circle.
   */
  spawnClickMarker(x: number, z: number, colour: number): void {
    this.colour.setHex(colour);
    this.addDecal(DecalKind.Marker, x, z, 0.3, 1.5, this.colour, 0.85, MARKER_LIFE_S, 0);
    this.addDecal(DecalKind.Marker, x, z, 1.15, 0.6, this.colour, 0.6, MARKER_LIFE_S * 0.7, 0);
  }

  /**
   * Retain this tick's attacks until lockstep has produced the frame's final
   * alpha.
   *
   * Both transforms and entity metadata are copied immediately: a later
   * catch-up tick can move the entities or recycle any slot before rendering.
   * Pairs from one attack are consecutive in `shots`, so they are gathered here
   * into one attack with every victim it reached.
   */
  captureFromEvents(
    world: World,
    entities: EntityRenderer,
    canSee?: (index: number) => boolean,
  ): void {
    const shots = world.events.shots;
    const pool = world.pool;

    let k = 0;
    while (k + 1 < shots.length) {
      const attacker = shots[k]!;
      let end = k;
      while (end + 1 < shots.length && shots[end] === attacker) end += 2;

      // A battle between unseen enemies must not reveal itself through effects
      // drawn above the fog plane. Keep incoming fire at a visible unit.
      let visible = canSee === undefined || canSee(attacker);
      if (!visible) {
        for (let j = k + 1; j < end; j += 2) {
          if (canSee!(shots[j]!)) {
            visible = true;
            break;
          }
        }
      }

      if (visible) {
        const victims: PendingVictim[] = [];
        for (let j = k + 1; j < end; j += 2) {
          const target = shots[j]!;
          victims.push({
            type: pool.type[target]! as EntityType,
            transform: entities.transformSnapshot(target),
          });
        }
        const owner = pool.owner[attacker]!;
        this.pendingShots.push({
          tick: world.tick,
          attackerType: pool.type[attacker]! as EntityType,
          attacker: entities.transformSnapshot(attacker),
          victims,
          colour: PLAYER_COLOURS[colourSlotFor(owner, world.players.length)] ?? 0xffffff,
        });
      }

      k = end;
    }
  }

  /** Resolve retained attacks and spawn their effects once alpha is known. */
  flushPending(currentTick: number, alpha: number, elapsedS: number, dtMs: number): void {
    for (const shot of this.pendingShots) {
      const profile = weaponProfileFor(shot.attackerType);
      const colour = tintedTeamColour(profile, shot.colour, this.tint);
      const attacker = interpolateProjectileTransform(shot.attacker, alpha);
      const age = projectileAgeBeforeFrameUpdate(shot.tick, currentTick, alpha, dtMs);
      const attackerDef = defOf(shot.attackerType);
      const launch = projectileLaunchPoint(
        shot.attackerType,
        attacker.x,
        attacker.z,
        attacker.faceX,
        attacker.faceZ,
        attackerDef.flying ? flyerAltitudeAt(elapsedS, shot.attacker.slot) - FLIGHT_ALTITUDE : 0,
      );

      const impacts: ProjectilePoint[] = [];
      for (let i = 0; i < shot.victims.length; i++) {
        const victim = shot.victims[i]!;
        const at = interpolateProjectileTransform(victim.transform, alpha);
        const def = defOf(victim.type);
        impacts.push(
          projectileImpactPoint(
            victim.type,
            at.x,
            at.z,
            def.flying ? flyerAltitudeAt(elapsedS, victim.transform.slot) - FLIGHT_ALTITUDE : 0,
          ),
        );
      }
      if (impacts.length === 0) continue;
      const primary = primaryVictimIndex(
        attacker.x,
        attacker.z,
        shot.attacker.currFaceX,
        shot.attacker.currFaceZ,
        impacts,
      );

      this.dispatch(profile, colour, attackerDef.splashRadius, launch, impacts, primary, age);
    }
    this.pendingShots.length = 0;
  }

  /** Turn one resolved attack into the stages its weapon actually has. */
  private dispatch(
    profile: WeaponProfile,
    colour: THREE.Color,
    splashRadiusFixed: number,
    launch: ProjectilePoint,
    impacts: ProjectilePoint[],
    primary: number,
    age: number,
  ): void {
    const splash = toFloat(splashRadiusFixed);
    const target = impacts[primary]!;
    const dx = target.x - launch.x;
    const dy = target.y - launch.y;
    const dz = target.z - launch.z;
    const dist = Math.hypot(dx, dy, dz) || 1;
    const nx = dx / dist;
    const ny = dy / dist;
    const nz = dz / dist;

    switch (profile.style) {
      // The payload is the unit. Its death blast, this same tick, is the shot.
      case 'detonate':
        return;

      case 'melee':
        this.melee(profile, colour, launch, target, age);
        return;

      case 'flame':
        this.flame(profile, colour, launch, target, age);
        for (const point of impacts) {
          this.impact(profile, colour, point.x, point.y, point.z, nx, ny, nz, 0, age + 0.1);
        }
        return;

      case 'repair':
        this.repair(profile, colour, launch, target, age);
        return;

      case 'beam': {
        this.muzzle(profile, colour, launch.x, launch.y, launch.z, nx, ny, nz, age);
        this.addBeam(
          launch.x,
          launch.y,
          launch.z,
          target.x,
          target.y,
          target.z,
          profile.width,
          colour,
          profile.life,
          age,
          2.4,
          0.35,
          0.06,
          0.02,
        );
        this.impact(profile, colour, target.x, target.y, target.z, nx, ny, nz, splash, age);
        return;
      }

      case 'lance': {
        // The shot does not stop at what it hit — that is the whole point of
        // the weapon — so the beam carries on past the furthest victim.
        let reach = dist;
        for (const point of impacts) {
          reach = Math.max(
            reach,
            Math.hypot(point.x - launch.x, point.y - launch.y, point.z - launch.z),
          );
        }
        reach += 1.4;
        this.muzzle(profile, colour, launch.x, launch.y, launch.z, nx, ny, nz, age);
        this.addBeam(
          launch.x,
          launch.y,
          launch.z,
          launch.x + nx * reach,
          launch.y + ny * reach,
          launch.z + nz * reach,
          profile.width,
          colour,
          profile.life,
          age,
          2.6,
          0.2,
          0.04,
          0.35,
        );
        for (const point of impacts) {
          this.impact(profile, colour, point.x, point.y, point.z, nx, ny, nz, 0, age);
        }
        return;
      }

      case 'arc': {
        // Three coils, three enemies, three bolts. The one weapon here where a
        // victim each is what actually happened.
        this.muzzle(profile, colour, launch.x, launch.y, launch.z, nx, ny, nz, age);
        for (const point of impacts) {
          const span = Math.hypot(point.x - launch.x, point.y - launch.y, point.z - launch.z);
          this.lightning(
            launch.x,
            launch.y,
            launch.z,
            point.x,
            point.y,
            point.z,
            profile.width,
            colour,
            profile.life,
            age,
            Math.min(0.55, span * 0.12),
          );
          const towardX = (point.x - launch.x) / (span || 1);
          const towardZ = (point.z - launch.z) / (span || 1);
          this.impact(profile, colour, point.x, point.y, point.z, towardX, 0, towardZ, 0, age);
        }
        return;
      }

      default: {
        // Something leaves the barrel and arrives. One round, whatever its
        // blast then caught.
        this.muzzle(profile, colour, launch.x, launch.y, launch.z, nx, ny, nz, age);
        this.launch(profile, colour, launch, target, splash, age);
        if (splash > 0) {
          for (let i = 0; i < impacts.length; i++) {
            if (i === primary) continue;
            const point = impacts[i]!;
            this.sparks(point.x, point.y, point.z, 2, 2.5, age + dist / Math.max(1, profile.speed));
          }
        }
        return;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  /** Advance every pool and rebuild the instance buffers. Once per frame. */
  update(dtMs: number): void {
    const dt = Math.min(dtMs, 250) / 1000;

    this.glowField.begin();
    this.flameField.begin();
    this.smokeField.begin();
    this.beamField.begin();
    this.ringField.begin();
    this.scorchField.begin();
    this.markerField.begin();

    // Ordnance and debris before particles, because both spawn particles that
    // should be visible on the frame they are created rather than the next one.
    this.stepOrdnance(dt);
    this.stepDebris(dt);
    this.stepBeams(dt);
    this.stepParticles(dt);
    this.stepDecals(dt);

    this.glowField.commit();
    this.flameField.commit();
    this.smokeField.commit();
    this.beamField.commit();
    this.ringField.commit();
    this.scorchField.commit();
    this.markerField.commit();
  }

  /** Position along a shot's path at `t` in 0..1, written into `position`. */
  private pathAt(o: Ordnance, t: number): THREE.Vector3 {
    const clamped = clamp01(t);
    return this.position.set(
      o.x0 + (o.x1 - o.x0) * clamped,
      o.y0 + (o.y1 - o.y0) * clamped + o.arc * 4 * clamped * (1 - clamped),
      o.z0 + (o.z1 - o.z0) * clamped,
    );
  }

  private stepOrdnance(dt: number): void {
    for (const o of this.ordnance) {
      if (!o.active) continue;
      o.age += dt;
      if (o.age >= o.flight) {
        const dx = o.x1 - o.x0;
        const dy = o.y1 - o.y0;
        const dz = o.z1 - o.z0;
        const len = Math.hypot(dx, dy, dz) || 1;
        this.colour.setRGB(o.r, o.g, o.b);
        this.impact(
          o.profile,
          this.colour,
          o.x1,
          o.y1,
          o.z1,
          dx / len,
          dy / len,
          dz / len,
          o.splash,
          // Hand on whatever of the flight overshot this frame, so a shot from
          // a catch-up tick does not restart its burst from zero.
          o.age - o.flight,
        );
        o.active = false;
        continue;
      }
      if (o.age < 0) continue;

      const t = o.age / o.flight;
      const x = this.pathAt(o, t).x;
      const y = this.position.y;
      const z = this.position.z;
      // Direction from the path itself, so a lobbed shell noses over at the
      // top of its arc instead of pointing at the target the whole way.
      const back = this.pathAt(o, t - 0.03);
      const dx = x - back.x;
      const dy = y - back.y;
      const dz = z - back.z;
      const len = Math.hypot(dx, dy, dz) || 1;
      const nx = dx / len;
      const ny = dy / len;
      const nz = dz / len;

      this.colour.setRGB(o.r, o.g, o.b);
      const style = o.profile.style;
      const trail = o.profile.trail;

      if (trail > 0) {
        // The tracer is the projectile: a streak that fades out behind a hot
        // head, which is what a round crossing open ground actually looks like
        // at this distance. Two passes, as a beam gets — a wide skirt in the
        // team colour under a near-white core — because one pass of a
        // saturating additive core is white whoever fired it.
        const spent = Math.min(trail, o.flight * o.profile.speed * t + 0.25);
        const hx = x + nx * o.profile.width * 0.5;
        const hy = y + ny * o.profile.width * 0.5;
        const hz = z + nz * o.profile.width * 0.5;
        const tx = x - nx * spent;
        const ty = y - ny * spent;
        const tz = z - nz * spent;
        const w = o.profile.width;
        this.beamField.push(hx, hy, hz, tx, ty, tz, w * 4.5, this.colour, 0.8, 0.02, 0.85);
        const core = this.hotter(this.colour, 0.5);
        this.beamField.push(hx, hy, hz, tx, ty, tz, w, core, 1, 0.02, 0.85);
      }

      const head = style === 'plasma' ? 0.5 + Math.sin(o.age * 42) * 0.07 : 0.2;
      const size = head + o.profile.width * 1.5;
      this.glowField.push(x, y, z, size * 2.4, 0, this.colour, 0.8);
      this.glowField.push(x, y, z, size, 0, this.hotter(this.colour, 0.38), 1);

      // Trails that need actual particles: smoke off a shell, plasma boiling
      // off a bolt, frost crystallising behind a shard.
      const interval = style === 'mortar' ? 0.02 : style === 'plasma' ? 0.022 : 0.05;
      if (style === 'mortar' || style === 'plasma' || style === 'frost' || style === 'cannon') {
        o.emitDebt += dt;
        while (o.emitDebt >= interval) {
          o.emitDebt -= interval;
          this.trailPuff(o, style, x, y, z, nx, ny, nz);
        }
      }
    }
  }

  /** One puff of whatever a shot leaves behind it. */
  private trailPuff(
    o: Ordnance,
    style: WeaponProfile['style'],
    x: number,
    y: number,
    z: number,
    nx: number,
    ny: number,
    nz: number,
  ): void {
    if (style === 'mortar') {
      const e = emitter(Layer.Smoke, 0);
      e.x = x - nx * 0.2 + spread(0.05);
      e.y = y - ny * 0.2;
      e.z = z - nz * 0.2 + spread(0.05);
      e.vy = rand(0.2, 0.7);
      e.drag = 1.4;
      e.life = rand(0.5, 0.9);
      e.size0 = 0.22;
      e.size1 = rand(0.8, 1.2);
      e.from = SMOKE;
      e.to = SMOKE_DARK;
      e.alpha = 0.65;
      e.flare = 0.1;
      e.spin = spread(1.5);
      this.emit();
      return;
    }

    const e = emitter(Layer.Glow, 0);
    e.x = x - nx * 0.12 + spread(0.06);
    e.y = y - ny * 0.12 + spread(0.06);
    e.z = z - nz * 0.12 + spread(0.06);
    e.vx = spread(0.6);
    e.vy = style === 'frost' ? rand(-0.8, -0.2) : rand(0.1, 0.8);
    e.vz = spread(0.6);
    e.drag = 2;
    e.life = style === 'plasma' ? rand(0.18, 0.32) : rand(0.2, 0.4);
    e.size0 = style === 'cannon' ? 0.12 : 0.18;
    e.size1 = 0.03;
    e.from = style === 'cannon' ? SPARK : HOT;
    this.colour.setRGB(o.r, o.g, o.b);
    e.to = style === 'frost' ? FROST : this.colour;
    e.alpha = 0.85;
    this.emit();
  }

  private stepBeams(dt: number): void {
    for (const b of this.beams) {
      if (!b.active) continue;
      b.age += dt;
      if (b.age >= b.life) {
        b.active = false;
        continue;
      }
      if (b.age < 0) continue;

      const u = 1 - b.age / b.life;
      // Light does not dim gradually over a sixth of a second: it is on, and
      // then it is out. Holding full brightness for most of the life and
      // easing off at the end is the difference between a beam and a wire.
      let alpha = ease(Math.min(1, u / 0.4));
      // A beam that holds perfectly steady for a sixth of a second looks
      // painted on; one that wobbles looks like it is carrying power.
      if (b.flicker > 0) {
        alpha *= 1 - b.flicker * 0.5 * (1 - Math.sin(b.age * 70 + b.seed));
      }

      // A beam that thins as it dies fades like light going out. One that
      // keeps its width and only loses opacity ends up a grey wire strung
      // between two units, which is the one way this can look cheap.
      const width = b.width * (0.5 + 0.5 * u);

      this.colour.setRGB(b.r, b.g, b.b);
      if (b.halo > 0) {
        this.beamField.push(
          b.x0,
          b.y0,
          b.z0,
          b.x1,
          b.y1,
          b.z1,
          width * b.halo,
          this.colour,
          alpha * 0.85,
          b.headFade,
          b.tailFade,
        );
      }
      // The core is the same colour blown out: hot at the centre, its own hue
      // at the edges.
      const core = this.hotter(this.colour, 0.55);
      this.beamField.push(
        b.x0,
        b.y0,
        b.z0,
        b.x1,
        b.y1,
        b.z1,
        width,
        core,
        alpha,
        b.headFade,
        b.tailFade,
      );
    }
  }

  private stepParticles(dt: number): void {
    for (const p of this.particles) {
      if (!p.active) continue;
      p.age += dt;
      if (p.age >= p.life) {
        p.active = false;
        continue;
      }
      if (p.age < 0) continue;
      advance(p, Math.min(dt, p.age));

      const t = p.age / p.life;
      const u = 1 - t;
      const rise = p.flare > 0 ? Math.min(1, t / p.flare) : 1;
      // Smoke holds its opacity and then goes; light falls away immediately.
      const fall = p.layer === Layer.Smoke ? ease(Math.min(1, u / 0.45)) : u * (0.35 + 0.65 * u);
      const alpha = p.alpha * rise * fall;
      if (alpha <= 0.004) continue;

      const size = p.size0 + (p.size1 - p.size0) * (1 - u * u);
      this.colour.setRGB(
        p.r0 + (p.r1 - p.r0) * t,
        p.g0 + (p.g1 - p.g0) * t,
        p.b0 + (p.b1 - p.b0) * t,
      );

      if (p.layer === Layer.Streak) {
        // A spark drawn as a dot is a dot; drawn as its own last few
        // centimetres of travel it is a spark. The streak is free motion blur.
        const speed = Math.hypot(p.vx, p.vy, p.vz);
        if (speed < 1e-4) continue;
        const length = Math.max(size, speed * p.stretch) / speed;
        const tx = p.x - p.vx * length;
        const ty = p.y - p.vy * length;
        const tz = p.z - p.vz * length;
        this.beamField.push(p.x, p.y, p.z, tx, ty, tz, size, this.colour, alpha, 0.02, 0.8);
        continue;
      }

      const field =
        p.layer === Layer.Flame
          ? this.flameField
          : p.layer === Layer.Smoke
            ? this.smokeField
            : this.glowField;
      field.push(p.x, p.y, p.z, size, p.rot, this.colour, alpha);
    }
  }

  private stepDecals(dt: number): void {
    for (const pool of [this.decals, this.markers]) {
      for (const d of pool) {
        if (!d.active) continue;
        d.age += dt;
        if (d.age >= d.life) {
          d.active = false;
          continue;
        }
        if (d.age < 0) continue;

        const t = d.age / d.life;
        const u = 1 - t;
        const rise = d.flare > 0 ? Math.min(1, t / d.flare) : 1;
        // A burn sits at full strength and then weathers away; a shockwave is
        // brightest the instant it appears.
        const fall = d.kind === DecalKind.Scorch ? ease(Math.min(1, u / 0.35)) : u * u;
        const alpha = d.alpha * rise * fall;
        if (alpha <= 0.004) continue;

        const radius = d.r0 + (d.r1 - d.r0) * (1 - u * u);
        const field =
          d.kind === DecalKind.Scorch
            ? this.scorchField
            : d.kind === DecalKind.Marker
              ? this.markerField
              : this.ringField;
        this.colour.setRGB(d.r, d.g, d.b);
        field.push(d.x, GROUND_Y, d.z, radius, d.rot, this.colour, alpha);
      }
    }
  }

  private stepDebris(dt: number): void {
    let count = 0;
    for (const d of this.debris) {
      if (!d.active) continue;
      d.age += dt;
      if (d.age >= d.life) {
        d.active = false;
        continue;
      }
      if (count >= DEBRIS_CAPACITY) continue;

      // Ballistic arc with a floor. Cheap, and reads correctly at this scale.
      d.vy -= GRAVITY * dt;
      // Air resistance, so a chunk arcs to a stop near the wreck instead of
      // sailing off across the battlefield.
      const keep = Math.exp(-1.2 * dt);
      d.vx *= keep;
      d.vz *= keep;
      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.z += d.vz * dt;
      let landed = false;
      if (d.y < 0.08) {
        d.y = 0.08;
        d.vy = -d.vy * 0.35;
        d.vx *= 0.6;
        d.vz *= 0.6;
        landed = true;
      }

      if (d.smoking) {
        d.emitDebt += dt;
        while (d.emitDebt >= 0.05) {
          d.emitDebt -= 0.05;
          const e = emitter(Layer.Smoke, 0);
          e.x = d.x;
          e.y = d.y;
          e.z = d.z;
          e.vy = rand(0.4, 1.1);
          e.drag = 1.8;
          e.life = rand(0.35, 0.7);
          e.size0 = d.size * 1.2;
          e.size1 = d.size * 4;
          e.from = SMOKE;
          e.to = SMOKE_DARK;
          e.alpha = 0.26;
          e.flare = 0.2;
          this.emit();
        }
      }
      // A chunk hitting the ground kicks up its own little puff.
      if (landed && Math.abs(d.vy) > 1.2) this.dust(d.x, d.z, d.size * 1.6, 1, 0);

      const fade = 1 - d.age / d.life;
      this.position.set(d.x, d.y, d.z);
      this.axis.set(d.axisX, d.axisY, d.axisZ);
      this.quat.setFromAxisAngle(this.axis, d.spin * d.age);
      this.scale.set(d.size, d.size * d.aspect, d.size * (2 - d.aspect));
      this.matrix.compose(this.position, this.quat, this.scale);
      this.debrisMesh.setMatrixAt(count, this.matrix);
      // Wreckage cools and darkens rather than turning translucent, since an
      // opaque lit mesh cannot fade.
      this.colour.setHex(d.colour).multiplyScalar(0.28 + fade * 0.72);
      this.debrisMesh.setColorAt(count, this.colour);
      count++;
    }

    this.debrisMesh.count = count;
    this.debrisMesh.instanceMatrix.needsUpdate = true;
    if (this.debrisMesh.instanceColor) this.debrisMesh.instanceColor.needsUpdate = true;
  }

  /**
   * How many of each pool are alive.
   *
   * Exposed because "one attack draws one shell" is a contract rather than a
   * detail — see `primaryVictimIndex` — and because it is the number to look
   * at when wondering whether a pool is running out during a real fight.
   */
  liveEffectCounts(): {
    ordnance: number;
    beams: number;
    particles: number;
    decals: number;
    debris: number;
  } {
    return {
      ordnance: countActive(this.ordnance),
      beams: countActive(this.beams),
      particles: countActive(this.particles),
      decals: countActive(this.decals) + countActive(this.markers),
      debris: countActive(this.debris),
    };
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.group.clear();
  }
}

function countActive(pool: readonly { active: boolean }[]): number {
  let n = 0;
  for (const entry of pool) if (entry.active) n++;
  return n;
}

function blankParticle(): Particle {
  return {
    active: false,
    layer: Layer.Glow,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    drag: 0,
    gravity: 0,
    age: 0,
    life: 0,
    size0: 0,
    size1: 0,
    rot: 0,
    spin: 0,
    r0: 1,
    g0: 1,
    b0: 1,
    r1: 1,
    g1: 1,
    b1: 1,
    alpha: 1,
    flare: 0,
    stretch: 0,
  };
}

function blankOrdnance(): Ordnance {
  return {
    active: false,
    profile: weaponProfileFor(EntityType.Burstbot),
    x0: 0,
    y0: 0,
    z0: 0,
    x1: 0,
    y1: 0,
    z1: 0,
    arc: 0,
    age: 0,
    flight: 0,
    r: 1,
    g: 1,
    b: 1,
    splash: 0,
    emitDebt: 0,
  };
}

function blankBeam(): Beam {
  return {
    active: false,
    x0: 0,
    y0: 0,
    z0: 0,
    x1: 0,
    y1: 0,
    z1: 0,
    age: 0,
    life: 0,
    width: 0,
    r: 1,
    g: 1,
    b: 1,
    halo: 0,
    flicker: 0,
    seed: 0,
    headFade: 0.02,
    tailFade: 0.02,
  };
}

function blankDecal(): Decal {
  return {
    active: false,
    kind: DecalKind.Ring,
    x: 0,
    z: 0,
    age: 0,
    life: 0,
    r0: 0,
    r1: 0,
    rot: 0,
    r: 1,
    g: 1,
    b: 1,
    alpha: 1,
    flare: 0,
  };
}

function blankDebris(): Debris {
  return {
    active: false,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    spin: 0,
    axisX: 0,
    axisY: 1,
    axisZ: 0,
    age: 0,
    life: 0,
    size: 0.2,
    aspect: 1,
    colour: 0xffffff,
    smoking: false,
    emitDebt: 0,
  };
}
