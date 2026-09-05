/**
 * Which animation a unit plays, and why the old rule never showed a swing.
 *
 * Reported as "the melee unit moves toward its target and attacks, but the
 * attack animation is not played". The clip was chosen from two things that
 * merely *correlate* with fighting:
 *
 *   if (moving) run; else if (order is Attack/AttackMove) attack;
 *
 * Both halves are wrong, and either alone is enough to hide every swing:
 *
 * - A unit defending itself carries no attack order. It is idle, hitting
 *   whatever walked into range — the most common fight in the game — so the
 *   `else if` never fired.
 * - Units in contact are shoved apart by separation every tick, so a Slicebot
 *   in melee is never quite stationary and the `if` won anyway.
 *
 * The fix is to stop inferring: the simulation publishes every authoritative
 * attack start, so the swing is driven by the wind-up. These tests pin the rule,
 * and pin the fact that the sim reports the start in the case that broke.
 */

import { describe, expect, it } from 'vitest';
import {
  framesForPose,
  poseFor,
  secondsSinceSimulationTick,
  shouldKeepAttackVisual,
} from '../src/render/entities.js';
import { defOf } from '../src/config/rules.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order } from '../src/sim/types.js';

const FIX = 65536;
/** Roughly a sword-machine's attack clip. */
const SWING = 1.5;
/** Comfortably above MOVING_EPSILON: a unit genuinely under way. */
const RUNNING = 0.05;
/** The jitter separation imparts to units standing in a scrum. */
const JOSTLE = 0.01;

describe('simulation-tick animation time', () => {
  it('puts alpha zero on the attack-start tick at clip time zero', () => {
    expect(secondsSinceSimulationTick(100, 100, 0)).toBe(0);
    expect(secondsSinceSimulationTick(100, 100, 0.5)).toBe(0.025);
  });

  it('is continuous across an interpolated tick boundary', () => {
    expect(secondsSinceSimulationTick(100, 104, 1)).toBe(secondsSinceSimulationTick(100, 105, 0));
  });

  it('freezes once lockstep interpolation clamps during a stall', () => {
    const atStall = secondsSinceSimulationTick(100, 104, 1);
    // The runner can accumulate arbitrarily more wall time, but keeps reporting
    // the same simulation tick and an alpha clamped to one.
    expect(secondsSinceSimulationTick(100, 104, 20)).toBe(atStall);
  });

  it('reaches Slicebot impact time on the authoritative impact tick', () => {
    const foreswing = defOf(EntityType.Slicebot).attackForeswing;
    const impactTime = secondsSinceSimulationTick(200, 200 + foreswing, 0);
    expect(impactTime).toBe(0.45);
    expect(poseFor(impactTime, SWING, 0, 0).time).toBe(impactTime);
  });
});

describe('attack presentation cancellation', () => {
  it('keeps an active wind-up before its impact tick', () => {
    expect(shouldKeepAttackVisual(true, false, 1, true)).toBe(true);
  });

  it('keeps follow-through at wind-up zero after an authoritative hit or whiff', () => {
    expect(shouldKeepAttackVisual(true, true, 0, true)).toBe(true);
  });

  it('cancels at the nominal impact boundary when no impact event arrived', () => {
    expect(shouldKeepAttackVisual(true, false, 0, true)).toBe(false);
  });

  it('drops playback when the attacker slot is dead or recycled', () => {
    expect(shouldKeepAttackVisual(true, true, 0, false)).toBe(false);
  });
});

describe('choosing a clip', () => {
  it('swings while a shot is still playing out, even while being jostled', () => {
    // This is the regression. The old rule saw movement and ran.
    const pose = poseFor(0.2, SWING, JOSTLE, 99);
    expect(pose.clip).toBe('attack');
  });

  it('swings even mid-stride, so a unit that fires while closing still shows it', () => {
    expect(poseFor(0.2, SWING, RUNNING, 99).clip).toBe('attack');
  });

  it('times the swing from attack start, so foreswing reaches the impact pose', () => {
    // Not from wall-clock: an animation free-running against the cooldown would
    // land its hit at whatever point in the cycle it happened to be at.
    for (const t of [0, 0.3, 1.4]) {
      expect(poseFor(t, SWING, 0, 99).time).toBeCloseTo(t, 6);
    }
  });

  it('holds the follow-through rather than restarting the wind-up', () => {
    // A cooldown longer than the clip must not loop the swing.
    expect(poseFor(0.5, SWING, 0, 99).loop).toBe(false);
  });

  it('goes back to running once the swing has played out', () => {
    const after = poseFor(SWING + 0.01, SWING, RUNNING, 99);
    expect(after.clip).toBe('run');
    expect(after.loop).toBe(true);
  });

  it('holds the first frame of the stride when still and not fighting', () => {
    // No rig ships an idle clip and none is synthesised, so standing still is
    // frame zero of the run.
    const idle = poseFor(SWING + 0.01, SWING, 0, 99);
    expect(`${idle.clip} @ ${idle.time}`).toBe('run @ 0');
  });

  it('never asks for a swing a model does not have', () => {
    // Duration 0 means the clip is missing; posing on it would read the wrong
    // rows of the bone texture entirely.
    expect(poseFor(0, 0, 0, 99).clip).toBe('run');
  });

  it('ignores a unit that has never attacked', () => {
    // Fresh slots are stamped far in the past rather than at zero.
    expect(poseFor(1e9, SWING, 0, 99).clip).toBe('run');
  });
});

/**
 * An idle Slicebot with an enemy standing next to it, on open ground.
 *
 * The spot is searched for rather than hard-coded: the map is more than half
 * cliff, and a unit standing in rock is ejected by `clampToMap` every tick.
 */
function stageDuel(): { sim: Simulation; slicebot: number; enemy: number } {
  const sim = new Simulation(0x51ce7a11);
  const pool = sim.world.pool;
  const map = sim.world.map;
  const start = map.starts[0]!;

  let spot: { x: number; y: number } | null = null;
  for (let r = 4; r < 30 && !spot; r++) {
    for (let dy = -r; dy <= r && !spot; dy++) {
      for (let dx = -r; dx <= r && !spot; dx++) {
        const x = start.tileX + dx;
        const y = start.tileY + dy;
        if (map.isWalkable(x, y) && map.isWalkable(x + 1, y)) spot = { x, y };
      }
    }
  }
  if (!spot) throw new Error('no open ground near the start');

  const ax = spot.x + 0.5;
  const ay = spot.y + 0.5;
  const slicebot =
    pool.spawn(EntityType.Slicebot, 0, Math.round(ax * FIX), Math.round(ay * FIX)) & 0xffff;
  const enemy =
    pool.spawn(EntityType.Burstbot, 1, Math.round((ax + 1) * FIX), Math.round(ay * FIX)) & 0xffff;
  return { sim, slicebot, enemy };
}

describe('the signal behind it', () => {
  it('reports an attack start for an idle Slicebot defending itself', () => {
    // The exact case the old order-based rule could not see: no order at all.
    const { sim, slicebot, enemy } = stageDuel();
    const pool = sim.world.pool;
    const hp0 = pool.hp[enemy]!;

    let started = false;
    for (let t = 0; t < 60 && !started; t++) {
      pool.hp[enemy] = hp0;
      pool.order[enemy] = Order.Hold;
      pool.order[slicebot] = Order.None;
      sim.step([]);
      const starts = sim.world.events.attackStarts;
      for (let k = 0; k < starts.length; k += 2) {
        if (starts[k] === slicebot) started = true;
      }
    }
    expect(`idle ${defOf(EntityType.Slicebot).name} reported a start: ${started}`).toBe(
      `idle ${defOf(EntityType.Slicebot).name} reported a start: true`,
    );
  });

  it('reports attack starts as (attacker, target) pairs', () => {
    // The renderer indexes starts[k] as the attacker; a flipped pair would
    // animate the victim instead, which looks like nothing happening at all.
    const { sim, enemy } = stageDuel();
    const pool = sim.world.pool;
    const hp0 = pool.hp[enemy]!;

    let seen = 0;
    for (let t = 0; t < 60; t++) {
      pool.hp[enemy] = hp0;
      sim.step([]);
      const starts = sim.world.events.attackStarts;
      for (let k = 0; k + 1 < starts.length; k += 2) {
        const attacker = starts[k]!;
        const target = starts[k + 1]!;
        expect(defOf(pool.type[attacker]! as EntityType).attackRange).toBeGreaterThan(0);
        expect(pool.owner[attacker]).not.toBe(pool.owner[target]);
        seen++;
      }
    }
    // Without a fight the assertions above are vacuous, so the fight is required.
    expect(seen).toBeGreaterThan(0);
  });
});

/**
 * A stand-in for a baked model. `framesForPose` only ever reads the clip table,
 * so the texture, geometry and skeleton are irrelevant here.
 */
function fakeModel(
  clips: Record<string, { startFrame: number; frameCount: number; duration: number }>,
) {
  return { clips: new Map(Object.entries(clips)) } as unknown as Parameters<
    typeof framesForPose
  >[0];
}

describe('resolving a pose to bone-texture rows', () => {
  const model = fakeModel({
    run: { startFrame: 0, frameCount: 20, duration: 0.667 },
    attack: { startFrame: 20, frameCount: 45, duration: 1.5 },
  });

  it('straddles the two frames either side of the moment asked for', () => {
    // The bake is 30Hz and the screen is not. Rounding to the nearest whole
    // frame shows each pose twice at 60fps, which reads as a judder.
    // Deliberately between two baked frames: 0.5s lands exactly on frame 15 at
    // 30Hz, which would give a blend of zero and prove nothing.
    const at = framesForPose(model, { clip: 'attack', time: 0.5 + 1 / 90, loop: false });
    expect(at.to).toBe(at.from + 1);
    expect(at.blend).toBeGreaterThan(0);
    expect(at.blend).toBeLessThan(1);
  });

  it('wraps a looping clip round to its own first frame', () => {
    // The last frame of a run must blend back into the first, not off the end
    // of the clip and into whatever was baked after it.
    const end = framesForPose(model, { clip: 'run', time: 0.667 - 0.001, loop: true });
    expect(end.from).toBe(19);
    expect(end.to).toBe(0);
  });

  it('holds the last frame of a clip that does not loop', () => {
    const done = framesForPose(model, { clip: 'attack', time: 99, loop: false });
    expect(`${done.from} -> ${done.to}`).toBe('64 -> 64');
  });

  it('parks on the clip’s own first frame when asked for time zero', () => {
    const still = framesForPose(model, { clip: 'run', time: 0, loop: true });
    expect(`${still.from} -> ${still.to} @ ${still.blend}`).toBe('0 -> 1 @ 0');
  });

  it('survives a model missing the clip entirely', () => {
    const bare = fakeModel({});
    expect(framesForPose(bare, { clip: 'run', time: 1, loop: true }).from).toBe(0);
    expect(framesForPose(bare, { clip: 'attack', time: 1, loop: false }).from).toBe(0);
  });
});
