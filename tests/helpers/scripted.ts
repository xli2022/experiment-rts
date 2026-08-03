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
import { EntityType, MAX_PLAYERS, NO_ENTITY, Order } from '../../src/sim/types.js';
import { Simulation } from '../../src/sim/tick.js';
import type { World } from '../../src/sim/world.js';

/** Commands issued on one tick. */
export type CommandLog = Command[][];

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
    const patches: number[] = [];
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.type[i] === EntityType.MineralPatch) patches.push(i);
    }
    if (patches.length > 0) {
      for (let k = 0; k < workers.length; k++) {
        const patch = patches[(k + player * 3) % patches.length]!;
        cmds.push({
          type: CommandType.Harvest,
          player,
          units: [pool.idAt(workers[k]!)],
          target: pool.idAt(patch),
        });
      }
    }
  }

  // Keep the Command Post producing workers.
  if (tick % 60 === 10 + player) {
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

  // Put down a barracks early, then train army out of it.
  if (tick === 120 + player * 7) {
    const workers = ownedIndices(world, player, EntityType.Worker);
    const hqs = ownedIndices(world, player, EntityType.CommandPost);
    if (workers.length > 0 && hqs.length > 0) {
      const hq = hqs[0]!;
      const def = defOf(EntityType.Barracks);
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
            building: EntityType.Barracks,
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
        unit: rng.nextInt(2) === 0 ? EntityType.Rifleman : EntityType.Brawler,
      });
    }
  }

  // Periodically throw the army at a random point, which produces pathfinding
  // load, unit collisions, and eventually combat between the two players.
  if (tick > 100 && tick % 25 === 3 + player * 2) {
    const army: number[] = [];
    for (const i of ownedIndices(world, player)) {
      const t = pool.type[i]! as EntityType;
      if (t === EntityType.Rifleman || t === EntityType.Brawler) army.push(pool.idAt(i));
    }
    if (army.length > 0) {
      const tx = fromInt(rng.nextInt(world.map.width));
      const ty = fromInt(rng.nextInt(world.map.height));
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
      for (let i = 0; i < pool.count; i++) {
        if (
          pool.alive[i] === 1 &&
          pool.type[i] === EntityType.MineralPatch &&
          pool.resourceAmount[i]! > 0
        ) {
          patches.push(i);
        }
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
  seed: number,
  ticks: number,
  mapSize?: number,
): { log: CommandLog; checksums: number[] } {
  const sim = new Simulation(seed, mapSize);
  const rng = new Rng(seed ^ 0xa5a5a5);
  const log: CommandLog = [];
  const checksums: number[] = [];

  for (let t = 0; t < ticks; t++) {
    const commands: Command[] = [];
    for (let p = 0; p < MAX_PLAYERS; p++) {
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
export function replayMatch(seed: number, log: CommandLog, mapSize?: number): number[] {
  const sim = new Simulation(seed, mapSize);
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
  return commands.map((c) => ({ ...c, ...('units' in c ? { units: [...c.units] } : {}) }) as Command);
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
  const p0 = world.player(0);
  const p1 = world.player(1);
  return (
    `tick=${world.tick} units=${units} buildings=${buildings} ` +
    `p0(min=${p0.minerals} sup=${p0.supplyUsed}/${p0.supplyMax}) ` +
    `p1(min=${p1.minerals} sup=${p1.supplyUsed}/${p1.supplyMax})`
  );
}

export { idIndex, NO_ENTITY };
