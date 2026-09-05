/**
 * Mirror-symmetry probes.
 *
 * The map is an exact 180-degree rotation of itself and every player's opening
 * is the exact rotation of their opposite number's. So a match in which the
 * second half's commands are the exact rotations of the first half's should
 * stay an exact mirror for its whole length: every entity has a twin whose
 * position is the rotated position and whose every other field is identical.
 *
 * These helpers say whether that holds and, when it does not, name the first
 * tick, the first entity and the first field where it stopped holding — the
 * same shape of answer the determinism tests give, for the same reason: "it
 * diverged" is not actionable, "worker 11's path at tick 19" is.
 *
 * Twins are matched by owner and `serial`, the per-owner creation ordinal,
 * because slot indices are not mirrored: the first player's entities take the
 * low slots at setup and both halves recycle each other's slots after the
 * first death. Neutral entities never move, so they are matched by position.
 */

import { defOf } from '../../src/config/rules.js';
import { CommandType, type Command } from '../../src/sim/commands.js';
import { idIndex } from '../../src/sim/entities.js';
import { FIX_HALF, fromInt, vecLenSqRaw } from '../../src/sim/fixed.js';
import { mirrorTile } from '../../src/sim/map.js';
import { Simulation } from '../../src/sim/tick.js';
import {
  BuildState,
  EntityType,
  NEUTRAL,
  NO_ENTITY,
  Order,
  type EntityId,
  type MatchConfig,
  type PlayerId,
} from '../../src/sim/types.js';
import type { World } from '../../src/sim/world.js';

/** The player whose opening is the rotation of `player`'s. */
export function mirrorPlayer(world: World, player: PlayerId): PlayerId {
  const n = world.players.length;
  return (player + (n >> 1)) % n;
}

/** A world coordinate rotated 180 degrees about the map centre. */
export function mirrorX(world: World, x: number): number {
  return fromInt(world.map.width) - x;
}

export function mirrorY(world: World, y: number): number {
  return fromInt(world.map.height) - y;
}

/** A tile index rotated 180 degrees; -1 stays -1. */
export function mirrorTileIndex(world: World, tile: number): number {
  return tile < 0 ? tile : world.map.width * world.map.height - 1 - tile;
}

/**
 * The twin in `b` of every living entity in `a`, by slot index, or -1.
 *
 * `b` may be `a` itself (one mirrored world) or a second world whose roster is
 * the first's rotated (the same match with the seats swapped).
 */
export function twinMap(a: World, b: World): Int32Array {
  const pa = a.pool;
  const pb = b.pool;
  const twins = new Int32Array(pa.count).fill(-1);

  // Positions are below 2^24 on every map, so a pair packs into one exact
  // float64 integer; serials are far below 2^20.
  const bySerial = new Map<number, number>();
  const byPos = new Map<number, number>();
  for (let j = 0; j < pb.count; j++) {
    if (pb.alive[j] !== 1) continue;
    const o = pb.owner[j]!;
    if (o === NEUTRAL) byPos.set(pb.posX[j]! * 16777216 + pb.posY[j]!, j);
    else bySerial.set((o + 1) * 1048576 + pb.serial[j]!, j);
  }

  for (let i = 0; i < pa.count; i++) {
    if (pa.alive[i] !== 1) continue;
    const o = pa.owner[i]!;
    const j =
      o === NEUTRAL
        ? byPos.get(mirrorX(a, pa.posX[i]!) * 16777216 + mirrorY(a, pa.posY[i]!))
        : bySerial.get((mirrorPlayer(a, o) + 1) * 1048576 + pa.serial[i]!);
    if (j !== undefined && pb.type[j] === pa.type[i]) twins[i] = j;
  }
  return twins;
}

function usesPoint(order: number): boolean {
  return order !== Order.None && order !== Order.Hold;
}

/** Where and how two worlds (or one world and itself) first stop mirroring. */
export interface MirrorMismatch {
  tick: number;
  /** Slot in the first world, or -1 for a per-player field. */
  entity: number;
  /** Slot in the second world, or -1 when the twin is missing. */
  twin: number;
  field: string;
  value: number;
  expected: number;
}

export function describeMismatch(m: MirrorMismatch | null): string {
  if (m === null) return 'mirrored';
  const who = m.entity < 0 ? 'player state' : `entity ${m.entity} vs twin ${m.twin}`;
  return `tick ${m.tick}: ${who}: ${m.field} is ${m.value}, expected ${m.expected}`;
}

/**
 * The first way in which `b` fails to be the rotation of `a`, or null.
 *
 * Fields are checked in order of how much they explain: a position that has
 * drifted usually causes everything after it. Player banks come first of all,
 * because a mismatched bank at tick 3000 says "the economy diverged" before any
 * entity has to be looked at.
 */
export function mirrorMismatch(a: World, b: World): MirrorMismatch | null {
  const tick = a.tick;
  const pa = a.pool;
  const pb = b.pool;
  const n = a.players.length;
  const bad = (
    entity: number,
    twin: number,
    field: string,
    value: number,
    expected: number,
  ): MirrorMismatch => ({ tick, entity, twin, field, value, expected });

  for (let p = 0; p < n; p++) {
    const q = mirrorPlayer(a, p);
    const sa = a.players[p]!;
    const sb = b.players[q]!;
    if (sa.minerals !== sb.minerals)
      return bad(-1, -1, `minerals[p${p}]`, sa.minerals, sb.minerals);
    if (sa.supplyUsed !== sb.supplyUsed) {
      return bad(-1, -1, `supplyUsed[p${p}]`, sa.supplyUsed, sb.supplyUsed);
    }
    if (sa.supplyMax !== sb.supplyMax)
      return bad(-1, -1, `supplyMax[p${p}]`, sa.supplyMax, sb.supplyMax);
    if (sa.defeated !== sb.defeated)
      return bad(-1, -1, `defeated[p${p}]`, +sa.defeated, +sb.defeated);
  }

  const countA = new Int32Array(n + 1);
  const countB = new Int32Array(n + 1);
  for (let i = 0; i < pa.count; i++) if (pa.alive[i] === 1) countA[pa.owner[i]! + 1]!++;
  for (let j = 0; j < pb.count; j++) if (pb.alive[j] === 1) countB[pb.owner[j]! + 1]!++;
  for (let p = -1; p < n; p++) {
    const q = p < 0 ? -1 : mirrorPlayer(a, p);
    if (countA[p + 1] !== countB[q + 1]) {
      return bad(-1, -1, `aliveCount[p${p}]`, countA[p + 1]!, countB[q + 1]!);
    }
  }

  const twins = twinMap(a, b);
  const sameId = (idA: EntityId, idB: EntityId): boolean => {
    const liveA = pa.isAlive(idA);
    const liveB = pb.isAlive(idB);
    if (!liveA || !liveB) return liveA === liveB && (idA === NO_ENTITY) === (idB === NO_ENTITY);
    return twins[idIndex(idA)] === idIndex(idB);
  };

  for (let i = 0; i < pa.count; i++) {
    if (pa.alive[i] !== 1) continue;
    const j = twins[i]!;
    if (j < 0) return bad(i, -1, 'twin', pa.owner[i]!, NEUTRAL);
    const def = defOf(pa.type[i]! as EntityType);

    const eq = (field: string, va: number, vb: number): MirrorMismatch | null =>
      va === vb ? null : bad(i, j, field, va, vb);
    const checks: (() => MirrorMismatch | null)[] = [
      () => eq('posX', mirrorX(a, pa.posX[i]!), pb.posX[j]!),
      () => eq('posY', mirrorY(a, pa.posY[i]!), pb.posY[j]!),
      () => eq('hp', pa.hp[i]!, pb.hp[j]!),
      () => eq('order', pa.order[i]!, pb.order[j]!),
      () =>
        sameId(pa.orderTarget[i]!, pb.orderTarget[j]!)
          ? null
          : bad(i, j, 'orderTarget', pa.orderTarget[i]!, pb.orderTarget[j]!),
      () => eq('carrying', pa.carrying[i]!, pb.carrying[j]!),
      () => eq('harvestTimer', pa.harvestTimer[i]!, pb.harvestTimer[j]!),
      () =>
        sameId(pa.harvestPatch[i]!, pb.harvestPatch[j]!)
          ? null
          : bad(i, j, 'harvestPatch', pa.harvestPatch[i]!, pb.harvestPatch[j]!),
      () => eq('resourceAmount', pa.resourceAmount[i]!, pb.resourceAmount[j]!),
      () => eq('buildState', pa.buildState[i]!, pb.buildState[j]!),
      () => eq('buildProgress', pa.buildProgress[i]!, pb.buildProgress[j]!),
      () =>
        def.isBuilding
          ? eq('tileX', mirrorTile(a.map.width, pa.tileX[i]!, def.footprint), pb.tileX[j]!)
          : null,
      () =>
        def.isBuilding
          ? eq('tileY', mirrorTile(a.map.height, pa.tileY[i]!, def.footprint), pb.tileY[j]!)
          : null,
      () => eq('prodCount', pa.prodCount[i]!, pb.prodCount[j]!),
      () => {
        for (let s = 0; s < pa.prodCount[i]!; s++) {
          const m = eq(`prodQueue[${s}]`, pa.prodAt(i, s), pb.prodAt(j, s));
          if (m) return m;
        }
        return null;
      },
      () => eq('prodProgress', pa.prodProgress[i]!, pb.prodProgress[j]!),
      () => eq('hasRally', pa.hasRally[i]!, pb.hasRally[j]!),
      () => (pa.hasRally[i] === 1 ? eq('rallyX', mirrorX(a, pa.rallyX[i]!), pb.rallyX[j]!) : null),
      () => (pa.hasRally[i] === 1 ? eq('rallyY', mirrorY(a, pa.rallyY[i]!), pb.rallyY[j]!) : null),
      () => eq('speed', pa.speed[i]!, pb.speed[j]!),
      () => eq('attackCooldown', pa.attackCooldown[i]!, pb.attackCooldown[j]!),
      () => eq('attackWindup', pa.attackWindup[i]!, pb.attackWindup[j]!),
      () =>
        sameId(pa.attackTarget[i]!, pb.attackTarget[j]!)
          ? null
          : bad(i, j, 'attackTarget', pa.attackTarget[i]!, pb.attackTarget[j]!),
      () =>
        sameId(pa.combatTarget[i]!, pb.combatTarget[j]!)
          ? null
          : bad(i, j, 'combatTarget', pa.combatTarget[i]!, pb.combatTarget[j]!),
      () => eq('pathLen', pa.pathLen[i]!, pb.pathLen[j]!),
      () => eq('pathCursor', pa.pathCursor[i]!, pb.pathCursor[j]!),
      () => eq('pathPending', pa.pathPending[i]!, pb.pathPending[j]!),
      () => {
        for (let s = 0; s < pa.pathLen[i]!; s++) {
          const m = eq(`pathNode[${s}]`, mirrorTileIndex(a, pa.pathNode(i, s)), pb.pathNode(j, s));
          if (m) return m;
        }
        return null;
      },
      () => eq('flowGoal', mirrorTileIndex(a, pa.flowGoal[i]!), pb.flowGoal[j]!),
      () => eq('navGoal', mirrorTileIndex(a, pa.navGoal[i]!), pb.navGoal[j]!),
      () => eq('pathCooldown', pa.pathCooldown[i]!, pb.pathCooldown[j]!),
      () => eq('pursuing', pa.pursuing[i]!, pb.pursuing[j]!),
      () =>
        pa.pursuing[i] === 1 ? eq('pursueX', mirrorX(a, pa.pursueX[i]!), pb.pursueX[j]!) : null,
      () =>
        pa.pursuing[i] === 1 ? eq('pursueY', mirrorY(a, pa.pursueY[i]!), pb.pursueY[j]!) : null,
      () => eq('arriveBest', pa.arriveBest[i]!, pb.arriveBest[j]!),
      () => eq('arriveStall', pa.arriveStall[i]!, pb.arriveStall[j]!),
      // The order point is only meaningful while an order reads it; before the
      // first order it is an unset zero, which is not its own rotation.
      () =>
        usesPoint(pa.order[i]!) ? eq('orderX', mirrorX(a, pa.orderX[i]!), pb.orderX[j]!) : null,
      () =>
        usesPoint(pa.order[i]!) ? eq('orderY', mirrorY(a, pa.orderY[i]!), pb.orderY[j]!) : null,
      () => eq('faceX', -pa.faceX[i]!, pb.faceX[j]!),
      () => eq('faceY', -pa.faceY[i]!, pb.faceY[j]!),
    ];
    for (const check of checks) {
      const m = check();
      if (m) return m;
    }
  }
  return null;
}

/**
 * The command the opposite number would issue: same intent, every entity
 * replaced by its twin and every coordinate rotated.
 */
export function mirrorCommand(world: World, cmd: Command, twins: Int32Array): Command {
  const pool = world.pool;
  const player = mirrorPlayer(world, cmd.player);
  const twinId = (id: EntityId): EntityId => {
    const t = twins[idIndex(id)];
    return t === undefined || t < 0 ? NO_ENTITY : pool.idAt(t);
  };
  switch (cmd.type) {
    case CommandType.Move:
    case CommandType.AttackMove:
      return {
        type: cmd.type,
        player,
        units: cmd.units.map(twinId),
        x: mirrorX(world, cmd.x),
        y: mirrorY(world, cmd.y),
      };
    case CommandType.Attack:
    case CommandType.Harvest:
      return { type: cmd.type, player, units: cmd.units.map(twinId), target: twinId(cmd.target) };
    case CommandType.Build: {
      const footprint = defOf(cmd.building).footprint;
      return {
        type: cmd.type,
        player,
        worker: twinId(cmd.worker),
        building: cmd.building,
        tileX: mirrorTile(world.map.width, cmd.tileX, footprint),
        tileY: mirrorTile(world.map.height, cmd.tileY, footprint),
      };
    }
    case CommandType.Stop:
    case CommandType.Hold:
      return { type: cmd.type, player, units: cmd.units.map(twinId) };
    case CommandType.Train:
      return { type: cmd.type, player, building: twinId(cmd.building), unit: cmd.unit };
    case CommandType.CancelTrain:
      return { type: cmd.type, player, building: twinId(cmd.building), slot: cmd.slot };
    case CommandType.SetRally:
      return {
        type: cmd.type,
        player,
        building: twinId(cmd.building),
        x: mirrorX(world, cmd.x),
        y: mirrorY(world, cmd.y),
      };
    case CommandType.Surrender:
      return { type: cmd.type, player };
  }
}

/** Commands for the first half of the roster this tick; the rest are mirrored. */
export type MirrorScript = (world: World, tick: number) => Command[];

export interface MirrorReport {
  name: string;
  ticks: number;
  first: MirrorMismatch | null;
  /** How many ticks failed to mirror, for a sense of scale. */
  brokenTicks: number;
  winner: number;
  matchOver: boolean;
}

/**
 * Run a scripted match with the second half mirroring the first, checking the
 * world against its own rotation after every tick.
 */
export function probeScript(
  name: string,
  config: MatchConfig,
  script: MirrorScript,
  ticks: number,
): MirrorReport {
  const sim = new Simulation(config);
  const world = sim.world;
  let first: MirrorMismatch | null = null;
  let brokenTicks = 0;
  for (let t = 0; t < ticks && !world.matchOver; t++) {
    const own = script(world, t);
    const twins = twinMap(world, world);
    const all = own.slice();
    for (const c of own) all.push(mirrorCommand(world, c, twins));
    sim.step(all);
    const m = mirrorMismatch(world, world);
    if (m) {
      brokenTicks++;
      first ??= m;
    }
  }
  return {
    name,
    ticks: world.tick,
    first,
    brokenTicks,
    winner: world.winner,
    matchOver: world.matchOver,
  };
}

/** Run a bot-driven match and check the world against its own rotation. */
export function probeBots(name: string, config: MatchConfig, ticks: number): MirrorReport {
  const sim = new Simulation(config);
  const world = sim.world;
  let first: MirrorMismatch | null = null;
  let brokenTicks = 0;
  for (let t = 0; t < ticks && !world.matchOver; t++) {
    sim.step([]);
    const m = mirrorMismatch(world, world);
    if (m) {
      brokenTicks++;
      first ??= m;
    }
  }
  return {
    name,
    ticks: world.tick,
    first,
    brokenTicks,
    winner: world.winner,
    matchOver: world.matchOver,
  };
}

/**
 * Run two bot-driven matches whose rosters are each other's rotation and check
 * that the second is the rotation of the first, tick for tick.
 */
export function probePair(
  name: string,
  configA: MatchConfig,
  configB: MatchConfig,
  ticks: number,
): MirrorReport & { winnerB: number } {
  const simA = new Simulation(configA);
  const simB = new Simulation(configB);
  let first: MirrorMismatch | null = null;
  let brokenTicks = 0;
  let t = 0;
  for (; t < ticks && !(simA.world.matchOver && simB.world.matchOver); t++) {
    if (!simA.world.matchOver) simA.step([]);
    if (!simB.world.matchOver) simB.step([]);
    const m = mirrorMismatch(simA.world, simB.world);
    if (m) {
      brokenTicks++;
      first ??= m;
    }
  }
  return {
    name,
    ticks: t,
    first,
    brokenTicks,
    winner: simA.world.winner,
    winnerB: simB.world.winner,
    matchOver: simA.world.matchOver,
  };
}

// ---------------------------------------------------------------------------
// Scripts
// ---------------------------------------------------------------------------

/** Living entities of one owner, oldest first. */
function owned(world: World, player: PlayerId, type?: EntityType): number[] {
  const pool = world.pool;
  const out: number[] = [];
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.owner[i] !== player) continue;
    if (type !== undefined && pool.type[i] !== type) continue;
    out.push(i);
  }
  out.sort((x, y) => pool.serial[x]! - pool.serial[y]!);
  return out;
}

/**
 * The live patch nearest an entity, ties to the lowest tile index.
 *
 * Only ever asked for a first-half player, whose canonical frame is the map's
 * own, so the plain index is the canonical one.
 */
function nearestPatch(world: World, i: number): number {
  const pool = world.pool;
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  let bestTile = -1;
  for (let j = 0; j < pool.count; j++) {
    if (pool.alive[j] !== 1 || pool.type[j] !== EntityType.MineralPatch) continue;
    if (pool.resourceAmount[j]! <= 0) continue;
    const d = vecLenSqRaw(pool.posX[j]! - pool.posX[i]!, pool.posY[j]! - pool.posY[i]!);
    const tile = world.map.tileOfPos(pool.posX[j]!, pool.posY[j]!);
    if (d < bestDist || (d === bestDist && tile < bestTile)) {
      bestDist = d;
      best = j;
      bestTile = tile;
    }
  }
  return best;
}

function harvestIdle(world: World, player: PlayerId, all: boolean, cmds: Command[]): void {
  const pool = world.pool;
  for (const w of owned(world, player, EntityType.Worker)) {
    if (!all && pool.order[w] !== Order.None) continue;
    const patch = nearestPatch(world, w);
    if (patch < 0) continue;
    cmds.push({
      type: CommandType.Harvest,
      player,
      units: [pool.idAt(w)],
      target: pool.idAt(patch),
    });
  }
}

/** Every worker mines; idle ones are sent back every few seconds. */
export const harvestScript: MirrorScript = (world, tick) => {
  const cmds: Command[] = [];
  if (tick !== 2 && tick % 100 !== 50) return cmds;
  const half = world.players.length >> 1;
  for (let p = 0; p < half; p++) harvestIdle(world, p, tick === 2, cmds);
  return cmds;
};

/**
 * A clear spot for a footprint near a base, searched in the map's own frame.
 *
 * Requires a one-tile moat of open ground around it, like the bot does, so
 * units can always get past. Only called for first-half players; the twin's
 * spot is the mirror of this one.
 */
function placeable(world: World, cp: number, footprint: number): { x: number; y: number } | null {
  const map = world.map;
  const ox = world.pool.tileX[cp]!;
  const oy = world.pool.tileY[cp]!;
  for (let ring = 4; ring <= 20; ring += 2) {
    for (let dy = -ring; dy <= ring; dy++) {
      for (let dx = -ring; dx <= ring; dx++) {
        if (dx > -ring && dx < ring && dy > -ring && dy < ring) continue;
        const x = ox + dx;
        const y = oy + dy;
        if (map.canPlace(x - 1, y - 1, footprint + 2)) return { x, y };
      }
    }
  }
  return null;
}

/**
 * A busy match: workers, buildings, a rally, a long A* errand, a group
 * attack-move that meets its mirror in the middle, and an expansion.
 */
export const fullScript: MirrorScript = (world, tick) => {
  const cmds: Command[] = [];
  const pool = world.pool;
  const map = world.map;
  const half = world.players.length >> 1;

  for (let p = 0; p < half; p++) {
    if (tick === 2) harvestIdle(world, p, true, cmds);
    if (tick % 100 === 50) harvestIdle(world, p, false, cmds);

    const posts = owned(world, p, EntityType.CommandPost).filter(
      (i) => pool.buildState[i] === BuildState.Complete,
    );
    const workers = owned(world, p, EntityType.Worker);
    const home = posts[0];
    if (home === undefined) continue;

    if (tick === 100 || tick === 400) {
      cmds.push({
        type: CommandType.Train,
        player: p,
        building: pool.idAt(home),
        unit: EntityType.Worker,
      });
    }

    const builder = workers[workers.length - 1];
    if ((tick === 300 || tick === 900) && builder !== undefined) {
      const building = tick === 300 ? EntityType.Depot : EntityType.Barracks;
      const spot = placeable(world, home, defOf(building).footprint);
      if (spot) {
        cmds.push({
          type: CommandType.Build,
          player: p,
          worker: pool.idAt(builder),
          building,
          tileX: spot.x,
          tileY: spot.y,
        });
      }
    }

    const barracks = owned(world, p, EntityType.Barracks).filter(
      (i) => pool.buildState[i] === BuildState.Complete,
    );
    if (tick === 2000) {
      for (const b of barracks) {
        cmds.push({
          type: CommandType.SetRally,
          player: p,
          building: pool.idAt(b),
          x: pool.posX[home]! + fromInt(9) + FIX_HALF,
          y: pool.posY[home]! + fromInt(9) + FIX_HALF,
        });
      }
    }
    if (tick >= 2000 && tick % 120 === 0) {
      for (const b of barracks) {
        if (pool.prodCount[b]! < 3) {
          cmds.push({
            type: CommandType.Train,
            player: p,
            building: pool.idAt(b),
            unit: EntityType.Burstbot,
          });
        }
      }
    }

    if (tick === 2200 && workers[0] !== undefined) {
      cmds.push({
        type: CommandType.Move,
        player: p,
        units: [pool.idAt(workers[0])],
        x: fromInt((map.width >> 1) - 6) + FIX_HALF,
        y: fromInt((map.height >> 1) - 6) + FIX_HALF,
      });
    }

    const army = owned(world, p, EntityType.Burstbot);
    if (tick === 3400 && army.length > 0) {
      cmds.push({
        type: CommandType.AttackMove,
        player: p,
        units: army.map((i) => pool.idAt(i)),
        x: fromInt(map.width >> 1) + FIX_HALF,
        y: fromInt(map.height >> 1) + FIX_HALF,
      });
    }
    if (tick > 3400 && tick % 200 === 0) {
      const idle = army.filter((i) => pool.order[i] === Order.None);
      const enemyPosts = owned(world, mirrorPlayer(world, p), EntityType.CommandPost);
      const target = enemyPosts[0];
      if (idle.length > 0 && target !== undefined) {
        cmds.push({
          type: CommandType.AttackMove,
          player: p,
          units: idle.map((i) => pool.idAt(i)),
          x: pool.posX[target]!,
          y: pool.posY[target]!,
        });
      }
    }

    if (tick === 4200 && builder !== undefined) {
      const site = map.expansions[p];
      if (site) {
        const hqHalf = defOf(EntityType.CommandPost).footprint >> 1;
        cmds.push({
          type: CommandType.Build,
          player: p,
          worker: pool.idAt(builder),
          building: EntityType.CommandPost,
          tileX: site.tileX - hqHalf,
          tileY: site.tileY - hqHalf,
        });
      }
    }
    if (tick > 4500 && tick % 150 === 0) {
      for (const post of posts) {
        if (pool.prodCount[post]! < 2) {
          cmds.push({
            type: CommandType.Train,
            player: p,
            building: pool.idAt(post),
            unit: EntityType.Worker,
          });
        }
      }
    }
  }
  return cmds;
};
