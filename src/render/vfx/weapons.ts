/**
 * What each weapon looks like when it goes off.
 *
 * One row per unit that can attack, read by `ProjectileRenderer`. The
 * simulation knows a weapon by four numbers — range, damage, splash and
 * cooldown — and nothing in those says whether the shot is a bullet, a beam or
 * a shell lobbed over a wall. That is what this table says, and it is the only
 * place that says it.
 *
 * It lives in the renderer on purpose. Moving a muzzle, retiming a tracer or
 * recolouring a blast must never change a checksum or the instant damage lands,
 * and nothing here is reachable from `src/sim/**`.
 *
 * ## Colour: the shot says whose it is, the blast says what happened
 *
 * The palette rule this game already lives by is that **hue answers "may I
 * shoot that"** — team colours are picked to stay apart at minimap scale, and
 * mineral purple is kept clear of all of them. Weapon effects have to hold that
 * line while still telling a Sentry shell from a Burstbot round.
 *
 * So the split is by *where* the effect is, not by what fired it:
 *
 * - **Ordnance in flight** — the tracer, the beam, the muzzle flash, the shell —
 *   is the team colour pulled a little toward the weapon's own hue, over a
 *   near-white hot core. This is the part that crosses open ground between two
 *   armies, and it is the part that has to be readable as *whose*.
 * - **What happens at the far end** — the fireball, the sparks, the smoke, the
 *   scorch — is drawn in fire colours for everyone. It is sitting on top of a
 *   unit whose team colour is already right there, so spending hue on it would
 *   only muddy the one signal that matters.
 *
 * Two deliberate exceptions. The Fixomatic's beam is green almost regardless of
 * team, because "is this helping or hurting" is a different question from
 * "whose is it" and green has always answered it. The Ice Golem's hit keeps a
 * pale blue shatter, because chill is an ability the player has to *notice* —
 * the panel prints it, and the hit should show it.
 */

import * as THREE from 'three';
import { defOf } from '../../config/rules.js';
import { toFloat } from '../../sim/fixed.js';
import { EntityType } from '../../sim/types.js';

/**
 * How a weapon delivers its damage on screen.
 *
 * Chosen by silhouette and motion rather than by damage, because motion is what
 * survives being read across a battle at RTS camera distance: a lobbed shell,
 * a straight tracer and a beam that is simply *there* are three different
 * events at a glance, where three differently-coloured bolts are one event
 * three times.
 */
export type WeaponStyle =
  /** A fast, thin round. Burstbot. */
  | 'bullet'
  /** A heavier, slower bolt with a bright head. Turret. */
  | 'cannon'
  /** An instant beam, on screen for a moment after the damage. Beamdrone. */
  | 'beam'
  /** A long rail beam that carries on past what it hit. Piercebot. */
  | 'lance'
  /** Jagged lightning, one per enemy the coils pick. Arclight. */
  | 'arc'
  /** A short cone of fire rather than a projectile. Firespout. */
  | 'flame'
  /** A shell lobbed on a high arc, trailing smoke. Sentry. */
  | 'mortar'
  /** A glowing bolt on a shallow arc that bursts. Plasmodrone. */
  | 'plasma'
  /** A shard that shatters into a cold bloom. Ice Golem. */
  | 'frost'
  /** A mending beam and motes running along it. Fixomatic. */
  | 'repair'
  /** No travel: a flourish and sparks where the blow lands. */
  | 'melee'
  /** The attacker is the payload; its death blast is the whole effect. */
  | 'detonate';

export interface WeaponProfile {
  readonly style: WeaponStyle;
  /**
   * Travel speed in world units per second, for styles that travel.
   *
   * Fast enough that the bolt is never far behind the damage the simulation
   * already applied, slow enough to be a moving object rather than a flicker.
   */
  readonly speed: number;
  /** Seconds an instant beam or a melee flourish stays on screen. */
  readonly life: number;
  /** World-unit width of the bolt, beam or shard. */
  readonly width: number;
  /** World-unit length of the streak drawn behind a travelling bolt. */
  readonly trail: number;
  /** The weapon's own hue, mixed into the team colour by `tintMix`. */
  readonly tint: number;
  /** How far the team colour is pulled toward `tint`, 0 to 1. */
  readonly tintMix: number;
  /**
   * Overall energy of the hit, as a multiplier on spark count, flash size and
   * how much dust it kicks up. Roughly tracks damage, but by eye: what matters
   * is that a Sentry shell reads as worse news than a Burstbot round.
   */
  readonly impact: number;
}

/**
 * Anything that shoots without a row here, and anything that swings.
 *
 * Every attacker in the roster has a row, so these are a safety net for a unit
 * added later rather than shapes anything is expected to use.
 */
export const DEFAULT_WEAPON: WeaponProfile = {
  style: 'bullet',
  speed: 40,
  life: 0.16,
  width: 0.1,
  trail: 1.4,
  tint: 0xfff0c8,
  tintMix: 0.3,
  impact: 0.8,
};

export const DEFAULT_MELEE: WeaponProfile = {
  style: 'melee',
  speed: 0,
  life: 0.18,
  width: 0.11,
  trail: 0,
  tint: 0xffd9a0,
  tintMix: 0.4,
  impact: 0.9,
};

const PROFILES: Partial<Record<EntityType, WeaponProfile>> = {
  // A wrench, not a weapon. Small, warm, over almost before it starts.
  [EntityType.Worker]: {
    style: 'melee',
    speed: 0,
    life: 0.16,
    width: 0.07,
    trail: 0,
    tint: 0xffd9a0,
    tintMix: 0.35,
    impact: 0.4,
  },
  [EntityType.Burstbot]: {
    style: 'bullet',
    speed: 46,
    life: 0.14,
    width: 0.1,
    trail: 1.5,
    tint: 0xfff0c0,
    tintMix: 0.3,
    impact: 0.7,
  },
  // A blade, so the flourish is cool white and the sparks are struck metal.
  [EntityType.Slicebot]: {
    style: 'melee',
    speed: 0,
    life: 0.18,
    width: 0.11,
    trail: 0,
    tint: 0xe8f2ff,
    tintMix: 0.45,
    impact: 0.95,
  },
  [EntityType.Turret]: {
    style: 'cannon',
    speed: 38,
    life: 0.18,
    width: 0.2,
    trail: 2.2,
    tint: 0xffdca0,
    tintMix: 0.3,
    impact: 1.2,
  },
  [EntityType.Beamdrone]: {
    style: 'beam',
    speed: 0,
    life: 0.17,
    width: 0.1,
    trail: 0,
    tint: 0xbfe8ff,
    tintMix: 0.4,
    impact: 0.8,
  },
  // It never fires anything: the walker arrives and stops existing, and the
  // death blast in `spawnDeaths` is the entire weapon.
  [EntityType.Boomwalker]: {
    style: 'detonate',
    speed: 0,
    life: 0,
    width: 0,
    trail: 0,
    tint: 0xffb14a,
    tintMix: 0.7,
    impact: 2.6,
  },
  [EntityType.Fixomatic]: {
    style: 'repair',
    speed: 0,
    life: 0.3,
    width: 0.13,
    trail: 0,
    tint: 0x63ffa8,
    tintMix: 0.85,
    impact: 0.5,
  },
  [EntityType.Firespout]: {
    style: 'flame',
    speed: 13,
    life: 0.34,
    width: 0,
    trail: 0,
    tint: 0xffa63a,
    tintMix: 0.55,
    impact: 1,
  },
  [EntityType.Arclight]: {
    style: 'arc',
    speed: 0,
    life: 0.2,
    width: 0.13,
    trail: 0,
    tint: 0xcfe6ff,
    tintMix: 0.35,
    impact: 0.85,
  },
  // Warm white rather than the violet a railgun wants: violet is the mineral
  // colour and belongs to nobody, which is a promise a weapon must not break.
  [EntityType.Piercebot]: {
    style: 'lance',
    speed: 0,
    life: 0.28,
    width: 0.17,
    trail: 0,
    tint: 0xfff2d0,
    tintMix: 0.3,
    impact: 1.6,
  },
  [EntityType.Sentry]: {
    style: 'mortar',
    speed: 26,
    life: 0,
    width: 0.2,
    trail: 0,
    tint: 0xffc46a,
    tintMix: 0.35,
    impact: 2.4,
  },
  [EntityType.DarkGolem]: {
    style: 'melee',
    speed: 0,
    life: 0.22,
    width: 0.17,
    trail: 0,
    tint: 0xffb98a,
    tintMix: 0.4,
    impact: 1.8,
  },
  [EntityType.IceGolem]: {
    style: 'frost',
    speed: 30,
    life: 0.18,
    width: 0.13,
    trail: 1.1,
    tint: 0x9fe8ff,
    tintMix: 0.7,
    impact: 1,
  },
  [EntityType.Plasmodrone]: {
    style: 'plasma',
    speed: 18,
    life: 0,
    width: 0.22,
    trail: 0,
    tint: 0x7df0ff,
    tintMix: 0.5,
    impact: 2,
  },
};

/**
 * Below this attack range, a weapon with no row of its own is a melee weapon.
 *
 * Only reached by a unit added to the roster without a row here. It matters
 * anyway: a tracer drawn across half a metre reads as a glitch rather than as
 * a hit, so the wrong default is visibly wrong rather than merely plain.
 */
const MELEE_RANGE = 1.5;

export function weaponProfileFor(type: EntityType): WeaponProfile {
  const profile = PROFILES[type];
  if (profile !== undefined) return profile;
  return toFloat(defOf(type).attackRange) < MELEE_RANGE ? DEFAULT_MELEE : DEFAULT_WEAPON;
}

/** Scratch for `tintedTeamColour`, so resolving a colour allocates nothing. */
const tintScratch = new THREE.Color();

/**
 * The team colour, pulled toward the weapon's own hue.
 *
 * Writes into `out` and returns it. This is the colour of everything that
 * travels; what happens where it lands is drawn in fire colours instead.
 */
export function tintedTeamColour(
  profile: WeaponProfile,
  teamColour: number,
  out: THREE.Color,
): THREE.Color {
  out.setHex(teamColour);
  tintScratch.setHex(profile.tint);
  return out.lerp(tintScratch, profile.tintMix);
}
