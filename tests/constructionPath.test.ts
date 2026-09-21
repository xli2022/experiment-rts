import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BUILD_REACH, defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { vecDist } from '../src/sim/fixed.js';
import { AStar } from '../src/sim/pathing/astar.js';
import { ConstructionPaths } from '../src/sim/pathing/construction.js';
import { FlowField } from '../src/sim/pathing/flowfield.js';
import { lineOfSightClear, smoothPath, tileCentreX, tileCentreY } from '../src/sim/pathing/los.js';
import { inReach } from '../src/sim/systems/economy.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, Tile, type PlayerId } from '../src/sim/types.js';

const captured = JSON.parse(
  readFileSync(new URL('./fixtures/construction-pocket.json', import.meta.url), 'utf8'),
) as {
  seed: number;
  origin: number[];
  rows: string[];
  worker: number[];
  site: number[];
  originalGoal: number[];
};
const FIX = 65536;
afterEach(() => vi.restoreAllMocks());

function setup(exact: boolean, owner: PlayerId = 0) {
  const sim = new Simulation(captured.seed);
  const { world } = sim,
    { pool, map } = world;
  pool.alive.fill(0);
  map.occupied.fill(0);
  if (exact) {
    captured.rows.forEach((row, dy) =>
      [...row].forEach((value, dx) => {
        let x = captured.origin[0]! + dx,
          y = captured.origin[1]! + dy;
        if (owner === 1) {
          x = map.width - 1 - x;
          y = map.height - 1 - y;
        }
        const tile = map.index(x, y);
        map.tiles[tile] = value === '#' ? Tile.Cliff : Tile.Ground;
        map.occupied[tile] = value === 'X' || value === 'A' ? 1 : 0;
      }),
    );
  } else map.tiles.fill(Tile.Ground);
  const tx = exact ? captured.site[0]! : 54,
    ty = exact ? captured.site[1]! : 54;
  const siteX = owner === 0 ? tx : map.width - tx - 3;
  const siteY = owner === 0 ? ty : map.height - ty - 3;
  const site = world.placeBuilding(EntityType.Airport, owner, siteX, siteY);
  pool.buildState[site & 0xffff] = BuildState.Site;
  pool.buildProgress[site & 0xffff] = 0;
  const wx = exact ? captured.worker[0]! : 44.5,
    wy = exact ? captured.worker[1]! : 55.5;
  const worker = pool.spawn(
    EntityType.Worker,
    owner,
    Math.round((owner === 0 ? wx : map.width - wx) * FIX),
    Math.round((owner === 0 ? wy : map.height - wy) * FIX),
  );
  // Keep both seats viable so the regression measures a running match.
  const enemyTile = owner === 0 ? 90 : map.width - 90 - defOf(EntityType.CommandPost).footprint;
  const enemyPost = world.placeBuilding(
    EntityType.CommandPost,
    (1 - owner) as PlayerId,
    enemyTile,
    enemyTile,
  );
  pool.buildState[enemyPost & 0xffff] = BuildState.Complete;
  world.players[owner]!.minerals = 500;
  const build = {
    type: CommandType.Build as const,
    player: owner,
    worker,
    building: EntityType.Airport,
    tileX: siteX,
    tileY: siteY,
  };
  return { sim, site, worker, build };
}

function advance(context: ReturnType<typeof setup>, ticks: number): void {
  const { sim, worker } = context;
  const wi = worker & 0xffff;
  for (let tick = 0; tick < ticks; tick++) {
    const x = sim.world.pool.posX[wi]!,
      y = sim.world.pool.posY[wi]!;
    sim.step([]);
    const { pool, map } = sim.world;
    expect(pool.isAlive(worker)).toBe(true);
    const nx = pool.posX[wi]!,
      ny = pool.posY[wi]!;
    // A valid route walks there; clamping out of a blocked tile must not hide a teleport.
    expect(vecDist(x, y, nx, ny)).toBeLessThanOrEqual(defOf(EntityType.Worker).speedPerTick + 2);
    expect(lineOfSightClear(map, x, y, nx, ny, sim.world.flipOf(pool.owner[wi]!))).toBe(true);
    const tile = map.tileOfPos(nx, ny);
    expect(map.isWalkable(map.tileXOf(tile), map.tileYOf(tile))).toBe(true);
  }
}

describe('construction perimeter paths', () => {
  it('finishes the captured Airport despite a disconnected near edge and long alternate route', () => {
    const c = setup(true);
    const { map, pool } = c.sim.world;
    const start = map.tileOfPos(pool.posX[c.worker & 0xffff]!, pool.posY[c.worker & 0xffff]!);
    const originalGoal = map.index(captured.originalGoal[0]!, captured.originalGoal[1]!);
    expect(new AStar(map).find(map, start, originalGoal)).toEqual([]);
    const paths = new ConstructionPaths();
    const route = paths.find(c.sim.world, c.worker & 0xffff, c.site, []);
    expect(route.length).toBeGreaterThan(100); // Real detour, not a teleport through the surrounding walls.
    expect(route.at(-1)).not.toBe(originalGoal);
    c.sim.step([c.build]);
    advance(c, 3000);
    expect(pool.buildState[c.site & 0xffff]).toBe(BuildState.Complete);
    expect(pool.buildProgress[c.site & 0xffff]).toBe(defOf(EntityType.Airport).buildTicks);
  });

  it('finishes normal open-ground construction with one shared perimeter sweep', () => {
    const c = setup(false);
    const search = vi.spyOn(AStar.prototype, 'find');
    const sweep = vi.spyOn(FlowField.prototype, 'build');
    c.sim.step([c.build]);
    advance(c, 1000);
    expect(c.sim.world.pool.buildState[c.site & 0xffff]).toBe(BuildState.Complete);
    expect(search).not.toHaveBeenCalled();
    expect(sweep.mock.calls.filter((call) => call[2] !== undefined)).toHaveLength(1);
  });

  it('caches a completely inaccessible site and recovers when occupancy opens', () => {
    const c = setup(false);
    const { map, pool } = c.sim.world;
    map.setOccupied(51, 51, 9, 1);
    const search = vi.spyOn(AStar.prototype, 'find');
    const sweep = vi.spyOn(FlowField.prototype, 'build');
    c.sim.step([c.build]);
    advance(c, 400);
    expect(pool.buildProgress[c.site & 0xffff]).toBe(0);
    expect(search).not.toHaveBeenCalled();
    expect(sweep.mock.calls.filter((call) => call[2] !== undefined)).toHaveLength(1);
    map.setOccupied(51, 51, 9, 0);
    map.setOccupied(54, 54, 3, 1);
    advance(c, 1100);
    expect(pool.buildState[c.site & 0xffff]).toBe(BuildState.Complete);
    expect(sweep.mock.calls.filter((call) => call[2] !== undefined)).toHaveLength(2);
  });

  it('shares one sweep across builders and warmed caches return identical routes', () => {
    const c = setup(true),
      cache = new ConstructionPaths();
    const sweep = vi.spyOn(FlowField.prototype, 'build');
    const expected = cache.find(c.sim.world, c.worker & 0xffff, c.site, []);
    const pool = c.sim.world.pool;
    for (let n = 0; n < 20; n++) {
      const builder = pool.spawn(
        EntityType.Worker,
        0,
        pool.posX[c.worker & 0xffff]!,
        pool.posY[c.worker & 0xffff]!,
      );
      expect(cache.find(c.sim.world, builder & 0xffff, c.site, [])).toEqual(expected);
    }
    expect(sweep).toHaveBeenCalledTimes(1);
    expect(new ConstructionPaths().find(c.sim.world, c.worker & 0xffff, c.site, [])).toEqual(
      expected,
    );
  });

  it('returns identical routes with a warm, cold or evicted cache', () => {
    const c = setup(false),
      cache = new ConstructionPaths(1);
    const other = c.sim.world.placeBuilding(EntityType.Depot, 0, 60, 60);
    const route = cache.find(c.sim.world, c.worker & 0xffff, c.site, []);
    expect(route.length).toBeGreaterThan(0);
    expect(cache.find(c.sim.world, c.worker & 0xffff, c.site, [])).toEqual(route);
    cache.find(c.sim.world, c.worker & 0xffff, other, []);
    expect(cache.find(c.sim.world, c.worker & 0xffff, c.site, [])).toEqual(route);
    expect(new ConstructionPaths().find(c.sim.world, c.worker & 0xffff, c.site, [])).toEqual(route);
  });

  it('rejects a stale site handle and rebuilds for its reused slot', () => {
    const c = setup(false),
      cache = new ConstructionPaths();
    const { pool, map } = c.sim.world;
    const original = cache.find(c.sim.world, c.worker & 0xffff, c.site, []);
    pool.destroy(c.site);
    map.setOccupied(54, 54, 3, 0);
    const replacement = c.sim.world.placeBuilding(EntityType.Airport, 0, 70, 70);
    expect(replacement & 0xffff).toBe(c.site & 0xffff);
    expect(replacement).not.toBe(c.site);
    expect(cache.find(c.sim.world, c.worker & 0xffff, c.site, [])).toEqual([]);
    const route = cache.find(c.sim.world, c.worker & 0xffff, replacement, []);
    expect(route.length).toBeGreaterThan(0);
    expect(route.at(-1)).not.toBe(original.at(-1));
    expect(route).toEqual(
      new ConstructionPaths().find(c.sim.world, c.worker & 0xffff, replacement, []),
    );
  });

  it.each([0, 1] as const)(
    'approaches the seed center from an out-of-reach fractional position for player %s',
    (owner) => {
      const c = setup(false, owner);
      const { pool, map } = c.sim.world;
      const wi = c.worker & 0xffff;
      const position = Math.round(53.01 * FIX);
      pool.posX[wi] = pool.posY[wi] = owner === 0 ? position : map.width * FIX - position;
      expect(inReach(c.sim.world, wi, c.site & 0xffff, BUILD_REACH)).toBe(false);
      const tile = map.tileOfPosFor(pool.posX[wi]!, pool.posY[wi]!, owner === 1);
      expect(new ConstructionPaths().find(c.sim.world, wi, c.site, [])).toEqual([tile]);
      c.sim.step([c.build]);
      advance(c, 800);
      expect(pool.buildState[c.site & 0xffff]).toBe(BuildState.Complete);
    },
  );

  it('keeps the captured fractional first segment and every smoothed corner clear', () => {
    const c = setup(true);
    const { pool, map } = c.sim.world;
    let x = pool.posX[c.worker & 0xffff]!,
      y = pool.posY[c.worker & 0xffff]!;
    const path = new ConstructionPaths().find(c.sim.world, c.worker & 0xffff, c.site, []);
    const smoothed = smoothPath(map, path, x, y);
    expect(smoothed.length).toBeGreaterThan(1);
    for (const tile of smoothed) {
      const nx = tileCentreX(map, tile),
        ny = tileCentreY(map, tile);
      expect(lineOfSightClear(map, x, y, nx, ny)).toBe(true);
      x = nx;
      y = ny;
    }
  });

  it('rotates every tile of the captured fallback route for the opposite player', () => {
    const a = setup(true, 0),
      b = setup(true, 1);
    const routeA = new ConstructionPaths().find(a.sim.world, a.worker & 0xffff, a.site, []);
    const routeB = new ConstructionPaths().find(b.sim.world, b.worker & 0xffff, b.site, []);
    expect(routeB).toEqual(routeA.map((tile) => a.sim.world.map.mirrorIndex(tile)));
    a.sim.step([a.build]);
    b.sim.step([b.build]);
    for (let tick = 0; tick < 3000; tick++) {
      a.sim.step([]);
      b.sim.step([]);
      const pa = a.sim.world.pool,
        pb = b.sim.world.pool;
      expect(pa.isAlive(a.worker) && pb.isAlive(b.worker)).toBe(true);
      expect(pb.posX[b.worker & 0xffff]).toBe(
        a.sim.world.map.width * FIX - pa.posX[a.worker & 0xffff]!,
      );
      expect(pb.posY[b.worker & 0xffff]).toBe(
        a.sim.world.map.height * FIX - pa.posY[a.worker & 0xffff]!,
      );
      expect(pb.buildProgress[b.site & 0xffff]).toBe(pa.buildProgress[a.site & 0xffff]);
    }
    expect(a.sim.world.pool.buildState[a.site & 0xffff]).toBe(BuildState.Complete);
  });
});
