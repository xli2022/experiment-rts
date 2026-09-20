/** Headless pacing/composition telemetry: bun run scripts/gameplay-probe.ts [seeds=2] [seconds=600]. */
import { HeadlessMatch } from '../src/ai/headless.js';
import { ScriptedAgent } from '../src/ai/scripted.js';
import { defOf } from '../src/config/rules.js';
import { matchConfig } from '../src/sim/match.js';
import { EntityType, MapLayout, TICKS_PER_SECOND } from '../src/sim/types.js';

const seeds = Number(process.argv[2] ?? 2);
const duration = Number(process.argv[3] ?? 600);
if (!Number.isInteger(seeds) || seeds < 1 || !Number.isFinite(duration) || duration <= 0) {
  throw new Error('Usage: gameplay-probe.ts [positive integer seeds] [positive seconds]');
}

for (const layout of [MapLayout.Lanes, MapLayout.Quarters]) {
  for (let sample = 0; sample < seeds; sample++) {
    const seed = (0x51ce7a11 + sample * 7919) >>> 0;
    const count = layout === MapLayout.Quarters ? 4 : 2;
    const perSide = count / 2;
    const config = matchConfig(layout, seed, {
      botPlayers: Array.from({ length: count }, (_, p) => p),
    });
    // Unequal cadence lets games resolve; equal mirrored bots necessarily draw.
    const match = new HeadlessMatch(
      config,
      config.bots.map(
        ({ player }) =>
          [player, new ScriptedAgent({ thinkInterval: player < perSide ? 10 : 20 })] as const,
      ),
    );
    const players = Array.from({ length: count }, () => ({
      firstArmy: null as number | null,
      factoryStarted: null as number | null,
      airportStarted: null as number | null,
      barracksLevel2: null as number | null,
      factoryLevel2: null as number | null,
      firstFactoryUnit: null as number | null,
      firstHeavy: null as number | null,
      barracksUpgradeStarted: null as number | null,
      factoryUpgradeStarted: null as number | null,
      firstAir: null as number | null,
      expansionStarted: null as number | null,
      trained: {} as Record<string, number>,
      peakArmy: 0,
    }));
    const born = new Set<number>();
    const snapshots: unknown[] = [];
    let firstFight: number | null = null;
    let shots = 0;
    let repairs = 0;
    try {
      for (let tick = 0; tick < duration * TICKS_PER_SECOND && !match.world.matchOver; tick++) {
        match.step();
        const world = match.world;
        const time = world.tick / TICKS_PER_SECOND;
        const army = new Array<number>(count).fill(0);
        const posts = new Array<number>(count).fill(0);
        const pool = world.pool;
        for (let event = 0; event < world.events.shots.length; event += 2) {
          const attacker = world.events.shots[event]!;
          if (defOf(pool.type[attacker]! as EntityType).repairAmount > 0) {
            repairs++;
          } else {
            firstFight ??= time;
            shots++;
          }
        }
        for (let i = 0; i < pool.count; i++) {
          if (pool.alive[i] !== 1 || pool.owner[i]! < 0) continue;
          const owner = pool.owner[i]!;
          const stats = players[owner]!;
          const type = pool.type[i]! as EntityType;
          const def = defOf(type);
          const isArmy = !def.isBuilding && type !== EntityType.Worker;
          if (isArmy) {
            army[owner] = army[owner]! + 1;
            stats.firstArmy ??= time;
            if (defOf(EntityType.Factory).produces.includes(type)) stats.firstFactoryUnit ??= time;
            if (type === EntityType.DarkGolem || type === EntityType.IceGolem)
              stats.firstHeavy ??= time;
            if (def.flying) stats.firstAir ??= time;
          }
          if (type === EntityType.CommandPost) posts[owner] = posts[owner]! + 1;
          if (type === EntityType.Factory) {
            stats.factoryStarted ??= time;
            if (pool.buildingLevel[i]! >= 2) stats.factoryLevel2 ??= time;
          }
          if (type === EntityType.Barracks && pool.buildingLevel[i]! >= 2)
            stats.barracksLevel2 ??= time;
          if (type === EntityType.Airport) stats.airportStarted ??= time;
          if (pool.upgrading[i] === 1) {
            if (type === EntityType.Barracks) stats.barracksUpgradeStarted ??= time;
            if (type === EntityType.Factory) stats.factoryUpgradeStarted ??= time;
          }
          const id = pool.idAt(i);
          if (!born.has(id)) {
            born.add(id);
            if (isArmy) stats.trained[def.name] = (stats.trained[def.name] ?? 0) + 1;
          }
        }
        for (let p = 0; p < count; p++) {
          players[p]!.peakArmy = Math.max(players[p]!.peakArmy, army[p]!);
          if (posts[p]! > 1) players[p]!.expansionStarted ??= time;
        }
        if (world.tick % (120 * TICKS_PER_SECOND) === 0) {
          snapshots.push({ seconds: time, army, banks: world.players.map((p) => p.minerals) });
        }
      }
      console.log(
        JSON.stringify({
          layout: layout === MapLayout.Lanes ? 'lanes' : 'quarters',
          seed,
          seconds: match.world.tick / TICKS_PER_SECOND,
          finished: match.world.matchOver,
          winner: match.world.winner,
          firstFight,
          shots,
          repairs,
          players,
          snapshots,
        }),
      );
    } finally {
      match.dispose();
    }
  }
}
