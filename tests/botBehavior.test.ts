import { describe, expect, it } from 'vitest';
import { botThink } from '../src/ai/bot.js';
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { fromFloat, toFloat } from '../src/sim/fixed.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { Simulation } from '../src/sim/tick.js';
import { BuildState, EntityType, NEUTRAL, Order, Tile } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';
import { Visibility } from '../src/vision/visibility.js';

function add(world: World, type: EntityType, count = 1, player = 0, x = 30.5): number[] {
  return Array.from({ length: count }, () => {
    const index = world.pool.spawn(type, player, fromFloat(x), fromFloat(30.5)) & 0xffff;
    world.pool.buildState[index] = BuildState.Complete;
    return index;
  });
}

function productionWorld(barracks = 1, factories = 0, airports = 0, level = 2): World {
  const world = new World(0x51ce7a11);
  world.player(0).minerals = 100000;
  world.player(0).supplyMax = 200;
  add(world, EntityType.Barracks, barracks);
  add(world, EntityType.Factory, factories);
  add(world, EntityType.Airport, airports);
  world.pool.buildingLevel.fill(level);
  return world;
}

function trained(world: World): EntityType[] {
  return botThink(world, 0)
    .filter((command) => command.type === CommandType.Train)
    .map((command) => command.unit);
}

describe('scripted bot army planning', () => {
  it('respects the first-tier rosters and keeps upgrading producers idle', () => {
    const world = productionWorld(1, 1, 1, 1);
    expect(trained(world)).toEqual([EntityType.Burstbot, EntityType.Sentry, EntityType.Beamdrone]);
    world.pool.upgrading[0] = 1;
    world.pool.upgrading[1] = 1;
    expect(trained(world)).toEqual([EntityType.Beamdrone]);
  });

  it('drains one queue for an upgrade while other producers keep training', () => {
    const world = productionWorld(2, 0, 0, 1);
    world.player(0).minerals = 300;
    add(world, EntityType.Burstbot, 6);
    world.pool.prodPush(0, EntityType.Burstbot);
    const waiting = botThink(world, 0);
    expect(waiting.some((command) => command.type === CommandType.UpgradeBuilding)).toBe(false);
    const training = waiting.filter((command) => command.type === CommandType.Train);
    expect(training).toHaveLength(1);
    expect(training[0]!.building).toBe(world.pool.idAt(1));
    world.pool.prodCount[0] = 0;
    const ready = botThink(world, 0);
    expect(ready).toContainEqual({
      type: CommandType.UpgradeBuilding,
      player: 0,
      building: world.pool.idAt(0),
    });
    expect(
      ready.some(
        (command) => command.type === CommandType.Train && command.building === world.pool.idAt(0),
      ),
    ).toBe(false);
    // Supply stalls production, but must not prevent investing in technology.
    world.player(0).supplyUsed = world.player(0).supplyMax;
    expect(botThink(world, 0)).toContainEqual({
      type: CommandType.UpgradeBuilding,
      player: 0,
      building: world.pool.idAt(0),
    });
  });

  it('builds an Airport after opening factory production', () => {
    const world = new Simulation(0x51ce7a11).world;
    world.player(0).minerals = 1000;
    world.player(0).supplyMax = 200;
    add(world, EntityType.Barracks, 2);
    add(world, EntityType.Factory);
    add(world, EntityType.Turret, 3);
    add(world, EntityType.Burstbot, 6);
    expect(
      botThink(world, 0).filter((command) => command.type === CommandType.Build)[0]?.building,
    ).toBe(EntityType.Airport);
  });

  it('opens with production while its starting supply still has room', () => {
    const world = new Simulation(0x51ce7a11).world;
    const construction = botThink(world, 0).filter((command) => command.type === CommandType.Build);
    expect(construction[0]?.building).toBe(EntityType.Barracks);
  });

  it('fields every role without depending on which clock phase production finishes in', () => {
    const world = productionWorld(1, 1, 1);
    const types = new Set<EntityType>();
    for (let batch = 0; batch < 40; batch++) {
      // Hold the clock fixed deliberately: composition should respond to the
      // force already bought, not alias a unit's train duration forever.
      for (const unit of trained(world)) {
        types.add(unit);
        add(world, unit);
      }
    }
    const roster = [
      ...defOf(EntityType.Barracks).produces,
      ...defOf(EntityType.Factory).produces,
      ...defOf(EntityType.Airport).produces,
    ];
    expect([...types].sort()).toEqual([...roster].sort());
  });

  it('adds a small repair contingent after a fighting core and counts simultaneous orders', () => {
    const world = productionWorld(4);
    expect(trained(world)).not.toContain(EntityType.Fixomatic);
    add(world, EntityType.Burstbot, 6);
    expect(trained(world).filter((type) => type === EntityType.Fixomatic)).toHaveLength(1);
    world.pool.prodPush(0, EntityType.Fixomatic);
    expect(trained(world)).not.toContain(EntityType.Fixomatic);
  });

  it('includes queued workers when deciding whether two bases are saturated', () => {
    const world = productionWorld(0);
    const posts = add(world, EntityType.CommandPost, 2);
    add(world, EntityType.Worker, 35);
    world.pool.prodPush(posts[0]!, EntityType.Worker);
    expect(trained(world)).not.toContain(EntityType.Worker);
  });

  it('reacts to observed air without countering hidden reinforcements', () => {
    const world = productionWorld();
    add(world, EntityType.Burstbot, 4);
    expect(trained(world)).toEqual([EntityType.Slicebot]);
    const air = add(world, EntityType.Beamdrone, 2, 1, 100.5);
    expect(trained(world)).toEqual([EntityType.Slicebot]);
    for (const index of air) world.pool.posX[index] = fromFloat(33.5);
    expect(trained(world)).toEqual([EntityType.Arclight]);
  });

  it('answers a visible melee swarm with splash while keeping a mixed force', () => {
    const world = productionWorld();
    add(world, EntityType.Burstbot, 2);
    expect(trained(world)).toEqual([EntityType.Slicebot]);
    add(world, EntityType.Slicebot, 4, 1, 33.5);
    expect(trained(world)).toEqual([EntityType.Firespout]);
  });

  it('uses heavier hits against observed armor instead of adding only small shots', () => {
    const world = productionWorld();
    expect(trained(world)).toEqual([EntityType.Burstbot]);
    add(world, EntityType.DarkGolem, 2, 1, 33.5);
    expect(trained(world)).toEqual([EntityType.Slicebot]);
  });

  it('scouts public map positions without reading hidden structures or resources', () => {
    const world = productionWorld();
    add(world, EntityType.Burstbot, 6);
    const before = botThink(world, 0);
    expect(before.some((c) => c.type === CommandType.AttackMove)).toBe(true);
    const hidden = add(world, EntityType.CommandPost, 1, 1, 100.5)[0]!;
    const patch = add(world, EntityType.MineralPatch, 1, NEUTRAL, 110.5)[0]!;
    world.pool.resourceAmount[patch] = 1000;
    expect(botThink(world, 0)).toEqual(before);
    world.pool.posX[hidden] = fromFloat(90.5);
    world.pool.resourceAmount[patch] = 0;
    expect(botThink(world, 0)).toEqual(before);
    world.pool.posX[hidden] = fromFloat(34.5);
    const observed = botThink(world, 0).find((c) => c.type === CommandType.AttackMove);
    expect(observed && 'x' in observed ? observed.x : null).toBe(fromFloat(34.5));
  });

  it('leaves an existing attack march alone while sending fresh reinforcements', () => {
    const world = productionWorld();
    add(world, EntityType.Burstbot, 6);
    const initial = botThink(world, 0).find((c) => c.type === CommandType.AttackMove)!;
    executeCommand(world, initial);
    expect(botThink(world, 0).filter((c) => c.type === CommandType.AttackMove)).toEqual([]);
    const fresh = add(world, EntityType.Burstbot)[0]!;
    const reinforce = botThink(world, 0).find((c) => c.type === CommandType.AttackMove);
    expect(reinforce && 'units' in reinforce ? reinforce.units : []).toEqual([
      world.pool.idAt(fresh),
    ]);
    world.pool.attackWindup[fresh] = 3;
    expect(botThink(world, 0).filter((c) => c.type === CommandType.AttackMove)).toEqual([]);
  });

  it('does not send a repair-only group on an offensive push', () => {
    const world = productionWorld();
    add(world, EntityType.Fixomatic, 8);
    expect(botThink(world, 0).filter((c) => c.type === CommandType.AttackMove)).toEqual([]);
  });

  it('keeps a producer working when its upgrade is unaffordable', () => {
    const world = productionWorld(1, 0, 0, 1);
    world.player(0).minerals = 99;
    add(world, EntityType.Burstbot, 6);
    expect(trained(world)).toHaveLength(1);
  });

  it('reserves an orphaned site builder instead of also assigning a second construction job', () => {
    const world = new Simulation(0x51ce7a11).world;
    world.player(0).minerals = 2000;
    world.player(0).supplyMax = 200;
    const start = world.map.starts[0]!;
    const site = world.placeBuilding(EntityType.Depot, 0, start.tileX + 8, start.tileY + 8);
    const commands = botThink(world, 0);
    const builds = commands.filter((c) => c.type === CommandType.Build);
    expect(builds).toHaveLength(1);
    expect(builds[0]!.tileX).toBe(world.pool.tileX[site & 0xffff]);
    expect(
      commands.some((c) => c.type === CommandType.Harvest && c.units.includes(builds[0]!.worker)),
    ).toBe(false);
  });

  it('rebuilds a lost Command Post with the last worker before spending its recovery bank', () => {
    const world = new Simulation(0x51ce7a11).world;
    const pool = world.pool;
    const post = 0;
    world.map.setOccupied(
      pool.tileX[post]!,
      pool.tileY[post]!,
      defOf(EntityType.CommandPost).footprint,
      0,
    );
    pool.destroy(pool.idAt(post));
    let kept = false;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1 || pool.owner[i] !== 0 || pool.type[i] !== EntityType.Worker)
        continue;
      if (kept) pool.destroy(pool.idAt(i));
      kept = true;
    }
    add(world, EntityType.Barracks);
    world.player(0).minerals = defOf(EntityType.CommandPost).mineralCost;
    const commands = botThink(world, 0);
    expect(commands.filter((c) => c.type === CommandType.Train)).toEqual([]);
    expect(commands.filter((c) => c.type === CommandType.Build).map((c) => c.building)).toEqual([
      EntityType.CommandPost,
    ]);
  });

  it('sends one expansion scout and uses that nearby worker to build after revealing the site', () => {
    const world = new Simulation(0x51ce7a11).world;
    world.player(0).minerals = 5000;
    world.player(0).supplyMax = 200;
    add(world, EntityType.Barracks, 8);
    add(world, EntityType.Factory, 2);
    add(world, EntityType.Airport);
    add(world, EntityType.Turret, 3);
    add(world, EntityType.Burstbot, 6);
    const move = botThink(world, 0).find((c) => c.type === CommandType.Move);
    expect(move).toBeDefined();
    if (!move || move.type !== CommandType.Move) throw new Error('expected expansion scout');
    executeCommand(world, move);
    expect(botThink(world, 0).filter((c) => c.type === CommandType.Move)).toEqual([]);
    const scout = move.units[0]! & 0xffff;
    world.pool.posX[scout] = move.x;
    world.pool.posY[scout] = move.y;
    world.pool.order[scout] = Order.None;
    const build = botThink(world, 0).find((c) => c.type === CommandType.Build);
    expect(build && 'worker' in build ? build.worker : null).toBe(move.units[0]);
    expect(build && 'building' in build ? build.building : null).toBe(EntityType.CommandPost);
  });

  it('avoids visible weapons when placing a foundation, without reacting to the same hidden weapon', () => {
    const world = new World(1);
    const { pool, map } = world;
    map.tiles.fill(Tile.Ground);
    map.occupied.fill(0);
    map.sealTerrain();
    const hq = world.placeBuilding(EntityType.CommandPost, 0, 20, 20) & 0xffff;
    pool.buildState[hq] = BuildState.Complete;
    add(world, EntityType.Worker, 2, 0, 22.5);
    world.player(0).minerals = 1000;
    world.player(0).supplyMax = 200;
    const planned = botThink(world, 0).find((c) => c.type === CommandType.Build);
    if (!planned || planned.type !== CommandType.Build) throw new Error('expected foundation');
    const enemy = add(world, EntityType.Sentry, 1, 1, planned.tileX + 1.5)[0]!;
    pool.posY[enemy] = fromFloat(planned.tileY - 8);
    const sight = new Visibility(map);
    sight.update(world, 0);
    expect(sight.canSee(world, enemy, 0)).toBe(false);
    expect(botThink(world, 0).find((c) => c.type === CommandType.Build)).toEqual(planned);

    const scout = add(world, EntityType.Worker, 1, 0, planned.tileX + 1.5)[0]!;
    pool.posY[scout] = pool.posY[enemy]! + fromFloat(3);
    sight.update(world, 0);
    expect(sight.canSee(world, enemy, 0)).toBe(true);
    const safer = botThink(world, 0).find((c) => c.type === CommandType.Build);
    if (!safer || safer.type !== CommandType.Build) throw new Error('expected safe alternative');
    expect([safer.tileX, safer.tileY]).not.toEqual([planned.tileX, planned.tileY]);
    const x = toFloat(pool.posX[enemy]!);
    const y = toFloat(pool.posY[enemy]!);
    const size = defOf(safer.building).footprint;
    const dx = Math.max(safer.tileX - x, 0, x - safer.tileX - size);
    const dy = Math.max(safer.tileY - y, 0, y - safer.tileY - size);
    const reach = toFloat(defOf(EntityType.Sentry).attackRange + defOf(EntityType.Sentry).radius);
    expect(dx * dx + dy * dy).toBeGreaterThan(reach * reach);

    // An existing foundation at the unsafe spot must not keep consuming fresh
    // builders after the previous one dies either.
    world.placeBuilding(EntityType.Airport, 0, planned.tileX, planned.tileY);
    expect(
      botThink(world, 0).some(
        (c) =>
          c.type === CommandType.Build && c.tileX === planned.tileX && c.tileY === planned.tileY,
      ),
    ).toBe(false);
  });
});
