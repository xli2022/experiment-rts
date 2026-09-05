/**
 * The click target is the selection ring.
 *
 * A screen-space tolerance was tried first and is worse in a way that is easy
 * to miss: it does not shrink when the camera pulls back, so a distant clump of
 * units becomes a lottery, and it never lines up with the ring the player can
 * actually see. A hit area you cannot see is one you cannot aim at.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { defOf } from '../src/config/rules.js';
import { toFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, type PlayerId } from '../src/sim/types.js';
import { pickAt, RING_OVERSIZE, Selection } from '../src/input/selection.js';

const FIX = 65536;

/** An overhead camera looking straight down at (x, z). */
function cameraOver(x: number, z: number): THREE.Camera {
  const cam = new THREE.PerspectiveCamera(50, 1.6, 0.1, 500);
  cam.position.set(x, 26, z + 0.001);
  cam.lookAt(x, 0, z);
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  return cam;
}

/** Where a world-space ground point lands in normalised device coordinates. */
function ndcOf(cam: THREE.Camera, x: number, z: number): { x: number; y: number } {
  const v = new THREE.Vector3(x, 0, z).project(cam);
  return { x: v.x, y: v.y };
}

/** Open ground clear of the starting base. */
function clearSpot(sim: Simulation): { x: number; y: number } {
  const { map } = sim.world;
  const start = map.starts[0]!;
  for (let r = 8; r < 30; r++) {
    for (let dx = -r; dx <= r; dx++) {
      if (map.isWalkable(start.tileX + dx, start.tileY + r)) {
        return { x: start.tileX + dx, y: start.tileY + r };
      }
    }
  }
  throw new Error('no clear ground');
}

describe('clicking a unit', () => {
  it('hits inside the ring and misses outside it', () => {
    for (const type of [EntityType.Burstbot, EntityType.Slicebot, EntityType.Beamdrone]) {
      const sim = new Simulation(0x51ce7a11);
      const spot = clearSpot(sim);
      const x = spot.x + 0.5;
      const z = spot.y + 0.5;
      const idx =
        sim.world.pool.spawn(type, 0 as PlayerId, Math.round(x * FIX), Math.round(z * FIX)) &
        0xffff;

      const ring = toFloat(defOf(type).radius) * RING_OVERSIZE;
      const cam = cameraOver(x, z);

      const inside = ndcOf(cam, x + ring * 0.8, z);
      const outside = ndcOf(cam, x + ring * 1.4, z);

      expect(
        `${defOf(type).name} inside: ${pickAt(sim.world, cam, inside.x, inside.y, 0) === idx}`,
      ).toBe(`${defOf(type).name} inside: true`);
      expect(
        `${defOf(type).name} outside: ${pickAt(sim.world, cam, outside.x, outside.y, 0) === idx}`,
      ).toBe(`${defOf(type).name} outside: false`);
    }
  });

  it('keeps the same target no matter how far the camera pulls back', () => {
    // The property a screen-space tolerance cannot have. Zooming out must not
    // silently widen or narrow what a click selects.
    const sim = new Simulation(0x51ce7a11);
    const spot = clearSpot(sim);
    const x = spot.x + 0.5;
    const z = spot.y + 0.5;
    const type = EntityType.Slicebot;
    const idx =
      sim.world.pool.spawn(type, 0 as PlayerId, Math.round(x * FIX), Math.round(z * FIX)) & 0xffff;
    const ring = toFloat(defOf(type).radius) * RING_OVERSIZE;

    for (const height of [14, 26, 50, 90]) {
      const cam = new THREE.PerspectiveCamera(50, 1.6, 0.1, 500);
      cam.position.set(x, height, z + 0.001);
      cam.lookAt(x, 0, z);
      cam.updateMatrixWorld(true);
      cam.updateProjectionMatrix();

      const inside = ndcOf(cam, x + ring * 0.8, z);
      const outside = ndcOf(cam, x + ring * 1.4, z);
      expect(`h${height} inside: ${pickAt(sim.world, cam, inside.x, inside.y, 0) === idx}`).toBe(
        `h${height} inside: true`,
      );
      expect(`h${height} outside: ${pickAt(sim.world, cam, outside.x, outside.y, 0) === idx}`).toBe(
        `h${height} outside: false`,
      );
    }
  });

  it('treats a building as its footprint, not a circle', () => {
    // Clicking the middle of a Command Post has to work, and so does clicking
    // near a corner — the same square-versus-circle problem that made workers
    // walk around one to deliver.
    const sim = new Simulation(0x51ce7a11);
    const pool = sim.world.pool;
    let cp = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === EntityType.CommandPost) {
        cp = i;
      }
    }
    const x = toFloat(pool.posX[cp]!);
    const z = toFloat(pool.posY[cp]!);
    const cam = cameraOver(x, z);

    for (const [dx, dz] of [
      [0, 0],
      [1.8, 0],
      [0, 1.8],
      [1.8, 1.8],
      [-1.8, -1.8],
    ]) {
      const p = ndcOf(cam, x + dx!, z + dz!);
      expect(
        `(${dx},${dz}) picks the Command Post: ${pickAt(sim.world, cam, p.x, p.y, 0) === cp}`,
      ).toBe(`(${dx},${dz}) picks the Command Post: true`);
    }

    // And well outside it does not.
    const far = ndcOf(cam, x + 4, z + 4);
    expect(pickAt(sim.world, cam, far.x, far.y, 0)).not.toBe(cp);
  });
});

/**
 * Control groups, and the second press.
 *
 * Pressing a group's key when that group is already selected jumps the camera
 * to it — the StarCraft II behaviour. The interesting case is the one that
 * must *not* jump: pressing the key after clicking elsewhere reselects the
 * group and leaves the view alone, because the player is looking for the units,
 * not for where they are.
 */
describe('control groups', () => {
  function twoUnits(): { sim: Simulation; sel: Selection; ids: number[] } {
    const sim = new Simulation(0x51ce7a11);
    const spot = clearSpot(sim);
    const pool = sim.world.pool;
    const ids = [0, 1].map(
      (k) =>
        pool.spawn(
          EntityType.Burstbot,
          0 as PlayerId,
          Math.round((spot.x + 0.5 + k * 2) * FIX),
          Math.round((spot.y + 0.5) * FIX),
        ) & 0xffff,
    );
    return { sim, sel: new Selection(0 as PlayerId), ids };
  }

  it('reports the second press of an already-selected group', () => {
    const { sim, sel, ids } = twoUnits();
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);

    expect(sel.recallGroup(1, sim.world)).toBe('again');
  });

  it('reports a plain reselect after the player clicks away', () => {
    const { sim, sel, ids } = twoUnits();
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);

    sel.clear();
    expect(sel.recallGroup(1, sim.world)).toBe('selected');
    // And now the group is live again, so the next press is the jump.
    expect(sel.recallGroup(1, sim.world)).toBe('again');
  });

  it('reports a partial selection as a reselect, not a jump', () => {
    // Half the group selected is not the group, so this press is the one that
    // gathers them — jumping here would move the view out from under a player
    // who was mid-click.
    const { sim, sel, ids } = twoUnits();
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);

    sel.set([ids[0]!], sim.world);
    expect(sel.recallGroup(1, sim.world)).toBe('selected');
  });

  it('has nothing to recall for an unassigned key, and does not jump', () => {
    const { sim, sel } = twoUnits();
    expect(sel.recallGroup(7, sim.world)).toBe('missing');
  });

  it('centres on the middle of the selection', () => {
    const { sim, sel, ids } = twoUnits();
    sel.set(ids, sim.world);
    const pool = sim.world.pool;
    const mid = sel.centroid(sim.world)!;
    const want = {
      x: (toFloat(pool.posX[ids[0]!]!) + toFloat(pool.posX[ids[1]!]!)) / 2,
      z: (toFloat(pool.posY[ids[0]!]!) + toFloat(pool.posY[ids[1]!]!)) / 2,
    };
    expect(`${mid.x.toFixed(3)},${mid.z.toFixed(3)}`).toBe(
      `${want.x.toFixed(3)},${want.z.toFixed(3)}`,
    );
  });

  it('ignores the dead when centring', () => {
    // A control group keeps its members until they die; centring on a corpse's
    // last position would drag the view off the survivors.
    const { sim, sel, ids } = twoUnits();
    sel.set(ids, sim.world);
    sim.world.pool.alive[ids[1]!] = 0;
    const mid = sel.centroid(sim.world)!;
    expect(mid.x.toFixed(3)).toBe(toFloat(sim.world.pool.posX[ids[0]!]!).toFixed(3));
  });
});

/**
 * A slot index is not an identity.
 *
 * The pool recycles slots and bumps a generation on every free so that a stale
 * reference can be spotted. Selection tracked bare indices, so a dead unit's
 * slot — refilled by the next unit trained — silently rejoined the selection.
 * A control group whose members had all died came back holding three Beamdrones
 * that were never put in it.
 */
describe('selection identity across slot reuse', () => {
  function stage(): { sim: Simulation; sel: Selection; spot: { x: number; y: number } } {
    const sim = new Simulation(0x51ce7a11);
    return { sim, sel: new Selection(0 as PlayerId), spot: clearSpot(sim) };
  }

  function spawn(
    sim: Simulation,
    spot: { x: number; y: number },
    type: EntityType,
    k: number,
  ): number {
    return (
      sim.world.pool.spawn(
        type,
        0 as PlayerId,
        Math.round((spot.x + 0.5 + k) * FIX),
        Math.round((spot.y + 0.5) * FIX),
      ) & 0xffff
    );
  }

  it('forgets a control group once every member is dead', () => {
    const { sim, sel, spot } = stage();
    const pool = sim.world.pool;
    const ids = [0, 1, 2].map((k) => spawn(sim, spot, EntityType.Burstbot, k));
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);

    for (const i of ids) pool.destroy(pool.idAt(i));
    sel.clear();
    expect(sel.recallGroup(1, sim.world)).toBe('missing');

    // The slots come back as something else entirely, and the group must not.
    const fresh = [0, 1, 2].map((k) => spawn(sim, spot, EntityType.Beamdrone, k + 4));
    expect(fresh.some((i) => ids.includes(i))).toBe(true); // the reuse really happened
    expect(sel.recallGroup(1, sim.world)).toBe('missing');
    expect(sel.indices.size).toBe(0);
  });

  it('does not hand back strangers when the group is first recalled after a reuse', () => {
    // The reported case, and the one the delete-on-empty above does not cover:
    // the player never presses the key while the slots are empty. They press it
    // once, later, by which time three new units are standing in those slots.
    const { sim, sel, spot } = stage();
    const pool = sim.world.pool;
    const ids = [0, 1, 2].map((k) => spawn(sim, spot, EntityType.Burstbot, k));
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);
    sel.clear();

    for (const i of ids) pool.destroy(pool.idAt(i));
    const fresh = [0, 1, 2].map((k) => spawn(sim, spot, EntityType.Beamdrone, k + 4));
    expect(fresh.some((i) => ids.includes(i))).toBe(true);

    expect(sel.recallGroup(1, sim.world)).toBe('missing');
    expect(sel.indices.size).toBe(0);
  });

  it('keeps the survivors of a group and drops only the dead', () => {
    const { sim, sel, spot } = stage();
    const pool = sim.world.pool;
    const ids = [0, 1, 2].map((k) => spawn(sim, spot, EntityType.Burstbot, k));
    sel.set(ids, sim.world);
    sel.assignGroup(1, sim.world);

    pool.destroy(pool.idAt(ids[1]!));
    // Something else takes the vacated slot before the group is next recalled.
    const intruder = spawn(sim, spot, EntityType.Beamdrone, 5);
    expect(intruder).toBe(ids[1]!);

    sel.clear();
    expect(sel.recallGroup(1, sim.world)).toBe('selected');
    expect([...sel.indices].sort((a, b) => a - b)).toEqual(
      [ids[0]!, ids[2]!].sort((a, b) => a - b),
    );
  });

  it('drops a selected unit whose slot was reused, rather than adopting the newcomer', () => {
    const { sim, sel, spot } = stage();
    const pool = sim.world.pool;
    const victim = spawn(sim, spot, EntityType.Burstbot, 0);
    sel.set([victim], sim.world);

    pool.destroy(pool.idAt(victim));
    const replacement = spawn(sim, spot, EntityType.Beamdrone, 1);
    expect(replacement).toBe(victim); // same slot, new entity

    sel.prune(sim.world);
    expect(sel.indices.size).toBe(0);
    expect(sel.ids(sim.world)).toEqual([]);
  });
});
