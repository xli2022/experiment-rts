/**
 * A scripted "player" that drives the simulation through a realistic command
 * stream, for determinism testing.
 *
 * The generator reads world state (so it issues *valid* commands against real
 * entity ids, exercising the systems rather than being rejected at the door) and
 * records everything it emits. The recorded log can then be replayed into a
 * fresh simulation, which is what makes the determinism test meaningful: the
 * second run receives exactly the bytes a peer would have received over the
 * network.
 */

import { CommandType, type Command } from '../../src/sim/commands.js';
import { defOf } from '../../src/config/rules.js';
import { idIndex } from '../../src/sim/entities.js';
import { fromInt } from '../../src/sim/fixed.js';
import { Rng } from '../../src/sim/rng.js';
import { BuildState, EntityType, NO_ENTITY, Order, type MatchConfig } from '../../src/sim/types.js';
import { Simulation } from '../../src/sim/tick.js';
import type { World } from '../../src/sim/world.js';

/** Commands issued on one tick. */
export type CommandLog = Command[][];

/**
 * The player's own mineral patches, nearest their Command Post first.
 *
 * Workers used to be handed patches by a fixed offset into every patch on the
 * map, which sent one player's opening across the map to mine at the other's
 * base. That player's income never recovered: it sat supply-blocked at 10 for
 * two thousand ticks, unable to afford even a Depot, so only one side of the
 * scripted match ever fielded an army.
 *
 * Ordered by tile distance and then by index — a strict total order, computed
 * in exact integer arithmetic, so record and replay agree.
 */
function homePatches(world: World, player: number): number[] {
  const pool = world.pool;
  const hqs = ownedIndices(world, player, EntityType.CommandPost);
  if (hqs.length === 0) return [];
  const hx = pool.tileX[hqs[0]!]!;
  const hy = pool.tileY[hqs[0]!]!;

  const out: number[] = [];
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] === 1 && pool.type[i] === EntityType.MineralPatch) out.push(i);
  }
  const distSq = (i: number) => {
    const dx = pool.tileX[i]! - hx;
    const dy = pool.tileY[i]! - hy;
    return dx * dx + dy * dy;
  };
  out.sort((a, b) => distSq(a) - distSq(b) || a - b);
  // Only the home cluster; anything past it is the enemy's line or an expansion
  // across open ground, and walking there is how the income vanished.
  return out.filter((i) => distSq(i) <= 20 * 20);
}

function ownedIndices(world: World, player: number, type?: EntityType): number[] {
  const pool = world.pool;
  const out: number[] = [];
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.owner[i] !== player) continue;
    if (type !== undefined && pool.type[i] !== type) continue;
    out.push(i);
  }
  return out;
}

/**
 * Produce the commands one player issues this tick.
 *
 * The behaviour is intentionally busy — moving, fighting, building and training
 * all at once — because the point is to exercise as many systems as possible in
 * as few ticks as possible.
 */
function generateFor(world: World, player: number, rng: Rng): Command[] {
  const cmds: Command[] = [];
  const pool = world.pool;
  const tick = world.tick;

  // Opening: put every worker on minerals.
  if (tick === 2) {
    const workers = ownedIndices(world, player, EntityType.Worker);
    const patches = homePatches(world, player);
    if (patches.length > 0) {
      for (let k = 0; k < workers.length; k++) {
        const patch = patches[k % patches.length]!;
        cmds.push({
          type: CommandType.Harvest,
          player,
          units: [pool.idAt(workers[k]!)],
          target: pool.idAt(patch),
        });
      }
    }
  }

  // Keep the Command Post producing workers, but stop short of the supply cap
  // so there is always room to train an army.
  const ps = world.player(player);
  const workerCount = ownedIndices(world, player, EntityType.Worker).length;
  if (tick % 60 === 10 + player && workerCount < 14) {
    const hqs = ownedIndices(world, player, EntityType.CommandPost);
    if (hqs.length > 0) {
      cmds.push({
        type: CommandType.Train,
        player,
        building: pool.idAt(hqs[0]!),
        unit: EntityType.Worker,
      });
    }
  }

  // --- construction -------------------------------------------------------
  //
  // This whole section exists to get an army onto the field, and for a long
  // time it did not. A Barracks costs 150 and the players start with 50, so a
  // single attempt at tick 120 was always rejected for affordability —
  // silently, because validation lives in the simulation and not here. The
  // result: no barracks, no army, and the determinism checks this helper feeds
  // covered combat to the tune of one shot in 6000 ticks, worker on worker.
  //
  // Three things are needed and each was missing:
  //   - retry, since the first attempt cannot be afforded;
  //   - staffing, or the builder wanders back to minerals and the shell sits;
  //   - one site at a time, or every retry lays another foundation nobody
  //     finishes and the player ends up with five simultaneous half-depots.
  const sites: number[] = [];
  for (const i of ownedIndices(world, player)) {
    if (!defOf(pool.type[i]! as EntityType).isBuilding) continue;
    if (pool.buildState[i] !== BuildState.Complete) sites.push(i);
  }

  if (sites.length > 0 && tick % 20 === player) {
    const staffed = new Set<number>();
    const free: number[] = [];
    for (const w of ownedIndices(world, player, EntityType.Worker)) {
      if (pool.order[w] !== Order.Build) {
        free.push(w);
        continue;
      }
      const target = pool.orderTarget[w]!;
      if (target !== NO_ENTITY && pool.isAlive(target)) staffed.add(idIndex(target));
    }
    const orphan = sites.find((i) => !staffed.has(i));
    if (orphan !== undefined && free.length > 0) {
      cmds.push({
        type: CommandType.Build,
        player,
        worker: pool.idAt(free[0]!),
        building: pool.type[orphan]! as EntityType,
        tileX: pool.tileX[orphan]!,
        tileY: pool.tileY[orphan]!,
      });
    }
  }

  // One depot, then the barracks, then depots as needed. Depot first because a
  // player pinned at the Command Post's 10 supply saves for a Barracks it could
  // not train out of anyway; barracks next because it takes 45 seconds to raise
  // and nothing fights until it is up.
  const hasRacks = ownedIndices(world, player, EntityType.Barracks).length > 0;
  const wantDepot = ps.supplyMax - ps.supplyUsed < 5 && ps.supplyMax < 60;
  const want =
    !hasRacks && ps.supplyMax > 10
      ? EntityType.Barracks
      : wantDepot
        ? EntityType.Depot
        : !hasRacks
          ? EntityType.Barracks
          : undefined;
  if (want !== undefined && sites.length === 0 && tick > 120 && tick % 50 === player * 7) {
    const workers = ownedIndices(world, player, EntityType.Worker);
    const hqs = ownedIndices(world, player, EntityType.CommandPost);
    if (workers.length > 0 && hqs.length > 0) {
      const hq = hqs[0]!;
      const def = defOf(want);
      // Search outward for a legal spot so the command is not silently dropped.
      let placed = false;
      for (let r = 5; r < 14 && !placed; r++) {
        for (let d = 0; d < 8 && !placed; d++) {
          const tx = pool.tileX[hq]! + (d % 3) * r - r;
          const ty = pool.tileY[hq]! + ((d / 3) | 0) * r - r;
          if (!world.map.canPlace(tx, ty, def.footprint)) continue;
          cmds.push({
            type: CommandType.Build,
            player,
            worker: pool.idAt(workers[workers.length - 1]!),
            building: want,
            tileX: tx,
            tileY: ty,
          });
          placed = true;
        }
      }
    }
  }

  if (tick > 200 && tick % 40 === 5 + player) {
    const racks = ownedIndices(world, player, EntityType.Barracks);
    for (const b of racks) {
      cmds.push({
        type: CommandType.Train,
        player,
        building: pool.idAt(b),
        unit: rng.nextInt(2) === 0 ? EntityType.Burstbot : EntityType.Slicebot,
      });
    }
  }

  // Throw the army about, which produces pathfinding load, unit collisions and
  // combat between the two players.
  //
  // Mostly at the enemy start rather than always at a random tile: two armies
  // sent to independent random points wander past each other, and combat is the
  // half of the simulation these checks most need to cover.
  if (tick > 100 && tick % 25 === 3 + player * 2) {
    const army: number[] = [];
    for (const i of ownedIndices(world, player)) {
      const t = pool.type[i]! as EntityType;
      if (t === EntityType.Burstbot || t === EntityType.Slicebot) army.push(pool.idAt(i));
    }
    if (army.length > 0) {
      // The player directly opposite. Starts are stored in mirrored halves, so
      // `p + n/2` is always the slot whose opening is this one's 180-degree
      // rotation — which on a four-start map is the opponent across the
      // diagonal rather than the ally next door.
      const starts = world.map.starts.length;
      const enemyStart = world.map.starts[(player + (starts >> 1)) % starts];
      const wander = rng.nextInt(4) === 0 || enemyStart === undefined;
      const tx = wander ? fromInt(rng.nextInt(world.map.width)) : fromInt(enemyStart.tileX);
      const ty = wander ? fromInt(rng.nextInt(world.map.height)) : fromInt(enemyStart.tileY);
      cmds.push({ type: CommandType.AttackMove, player, units: army, x: tx, y: ty });
    }
  }

  // Occasionally reassign an idle worker, to exercise order switching.
  if (tick % 33 === player) {
    const workers = ownedIndices(world, player, EntityType.Worker).filter(
      (i) => pool.order[i] === Order.None,
    );
    if (workers.length > 0) {
      const pick = workers[rng.nextInt(workers.length)]!;
      const patches: number[] = [];
      for (const i of homePatches(world, player)) {
        if (pool.resourceAmount[i]! > 0) patches.push(i);
      }
      if (patches.length > 0) {
        cmds.push({
          type: CommandType.Harvest,
          player,
          units: [pool.idAt(pick)],
          target: pool.idAt(patches[rng.nextInt(patches.length)]!),
        });
      }
    }
  }

  return cmds;
}

/**
 * Run a match, recording every command issued.
 *
 * Returns the per-tick command log plus a checksum after every tick, so a
 * failing comparison can name the exact tick where two runs diverged rather than
 * just reporting that they did.
 */
export function recordMatch(
  match: MatchConfig | number,
  ticks: number,
): { log: CommandLog; checksums: number[] } {
  const sim = new Simulation(match);
  const seed = typeof match === 'number' ? match : match.seed;
  const rng = new Rng(seed ^ 0xa5a5a5);
  const log: CommandLog = [];
  const checksums: number[] = [];

  for (let t = 0; t < ticks; t++) {
    const commands: Command[] = [];
    // Every slot in the match, not a constant: `MAX_PLAYERS` bounds what the
    // engine supports rather than who is playing, and reading it here scripted
    // two players that did not exist on the duel map.
    for (let p = 0; p < sim.world.players.length; p++) {
      const forPlayer = generateFor(sim.world, p, rng);
      for (const c of forPlayer) commands.push(c);
    }
    log.push(commands);
    sim.step(commands);
    checksums.push(sim.checksum());
  }

  return { log, checksums };
}

/** Replay a recorded log into a fresh simulation, checksumming every tick. */
export function replayMatch(match: MatchConfig | number, log: CommandLog): number[] {
  const sim = new Simulation(match);
  const checksums: number[] = [];
  for (let t = 0; t < log.length; t++) {
    // Deep-copy so the replay cannot observe mutations made by the first run,
    // which would mask a real ordering bug.
    sim.step(cloneCommands(log[t]!));
    checksums.push(sim.checksum());
  }
  return checksums;
}

export function cloneCommands(commands: Command[]): Command[] {
  return commands.map(
    (c) => ({ ...c, ...('units' in c ? { units: [...c.units] } : {}) }) as Command,
  );
}

/** Summarise a world, for failure messages that say something useful. */
export function describeWorld(world: World): string {
  const pool = world.pool;
  let units = 0;
  let buildings = 0;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1) continue;
    if (pool.owner[i]! < 0) continue;
    if (defOf(pool.type[i]! as EntityType).isBuilding) buildings++;
    else units++;
  }
  const perPlayer = world.players
    .map((p, i) => `p${i}(min=${p.minerals} sup=${p.supplyUsed}/${p.supplyMax})`)
    .join(' ');
  return `tick=${world.tick} units=${units} buildings=${buildings} ${perPlayer}`;
}

export { idIndex, NO_ENTITY };
