import { describe, expect, it } from 'vitest';
import { defOf } from '../src/config/rules.js';
import { fromFloat } from '../src/sim/fixed.js';
import { applyDamage, combatSystem } from '../src/sim/systems/combat.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType, Order, Tile, type PlayerId } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';

function spawn(world: World, type: EntityType, player: PlayerId, x: number, y = 40): number {
  return world.pool.spawn(type, player, fromFloat(x), fromFloat(y)) & 0xffff;
}

function combat(world: World): void {
  world.grid.rebuild(world.pool);
  combatSystem(world);
}

describe('combat range edges', () => {
  it('acquires a target moved into sight by a newly occupied building footprint', () => {
    const sim = new Simulation(1);
    const { world } = sim;
    for (let y = 28; y <= 43; y++) {
      for (let x = 38; x <= 43; x++) world.map.tiles[world.map.index(x, y)] = Tile.Ground;
    }
    const sentry = spawn(world, EntityType.Sentry, 1, 40.5, 29.9);
    const target = spawn(world, EntityType.Worker, 0, 40.5, 40.5);
    world.placeBuilding(EntityType.CommandPost, 1, 39, 39);

    sim.step([]);

    expect(world.pool.posY[target]).toBe(fromFloat(38.5));
    expect(world.pool.combatTarget[sentry]).toBe(world.pool.idAt(target));
  });

  it('lets a Sentry acquire a reachable target while another enemy is in its dead zone', () => {
    const world = new World(1);
    const sentry = spawn(world, EntityType.Sentry, 0, 40);
    spawn(world, EntityType.Worker, 1, 41);
    const distant = spawn(world, EntityType.Worker, 1, 46);

    combat(world);

    expect(world.pool.attackTarget[sentry]).toBe(world.pool.idAt(distant));
    expect(world.pool.attackWindup[sentry]).toBe(defOf(EntityType.Sentry).attackForeswing);
  });

  it('whiffs if a Sentry target enters minimum range during the wind-up', () => {
    const world = new World(1);
    const sentry = spawn(world, EntityType.Sentry, 0, 40);
    const target = spawn(world, EntityType.Worker, 1, 44);
    combat(world);
    expect(world.pool.attackWindup[sentry]).toBeGreaterThan(0);

    world.pool.posX[target] = fromFloat(41);
    for (let t = 0; t < defOf(EntityType.Sentry).attackForeswing; t++) combat(world);

    expect(world.events.attackImpacts).toContain(sentry);
    expect(world.pool.hp[target]).toBe(defOf(EntityType.Worker).maxHp);
    expect(world.events.shots.filter((_, k) => k % 2 === 0)).not.toContain(sentry);
  });

  it('includes a building whose edge overlaps splash across a spatial bucket boundary', () => {
    const world = new World(1);
    const sentry = spawn(world, EntityType.Sentry, 0, 18);
    const primary = spawn(world, EntityType.Worker, 1, 11.9);
    const edge = spawn(world, EntityType.CommandPost, 1, 7.9);
    world.pool.order[sentry] = Order.Attack;
    world.pool.orderTarget[sentry] = world.pool.idAt(primary);

    for (let t = 0; t <= defOf(EntityType.Sentry).attackForeswing; t++) combat(world);

    expect(world.pool.hp[edge]).toBe(
      defOf(EntityType.CommandPost).maxHp - defOf(EntityType.Sentry).damage,
    );
  });

  it('includes a building in an extra coil reach across a spatial bucket boundary', () => {
    const world = new World(1);
    const arc = spawn(world, EntityType.Arclight, 0, 13.9);
    spawn(world, EntityType.Worker, 1, 15.9);
    const edge = spawn(world, EntityType.CommandPost, 1, 7.9);

    for (let t = 0; t <= defOf(EntityType.Arclight).attackForeswing; t++) combat(world);

    expect(world.pool.hp[edge]).toBe(
      defOf(EntityType.CommandPost).maxHp - defOf(EntityType.Arclight).damage,
    );
    expect(world.events.shots).toContain(arc);
  });
});

describe('repair timing', () => {
  it('treats mirrored repairers identically when both armies take damage this tick', () => {
    const world = new World(1);
    const a = spawn(world, EntityType.Burstbot, 0, 63);
    spawn(world, EntityType.Fixomatic, 0, 60);
    const b = spawn(world, EntityType.Burstbot, 1, 65);
    spawn(world, EntityType.Fixomatic, 1, 68);

    combat(world);

    expect(world.pool.hp[a]).toBe(world.pool.hp[b]);
    expect(world.pool.hp[a]).toBeLessThan(defOf(EntityType.Burstbot).maxHp);
  });
});

describe('simultaneous deaths', () => {
  it('reports one death when several attacks kill the same target in a tick', () => {
    const world = new World(1);
    const target = spawn(world, EntityType.Worker, 0, 40);
    applyDamage(world, target, defOf(EntityType.Worker).maxHp);
    applyDamage(world, target, 6);
    expect(world.events.deaths).toEqual([target]);
  });

  it('reports a detonating unit only once when it also takes lethal damage that tick', () => {
    const world = new World(1);
    const bomb = spawn(world, EntityType.Boomwalker, 0, 40);
    const victim = spawn(world, EntityType.Worker, 1, 41);
    world.pool.attackWindup[bomb] = 1;
    world.pool.attackTarget[bomb] = world.pool.idAt(victim);
    applyDamage(world, bomb, defOf(EntityType.Boomwalker).maxHp);

    combat(world);

    expect(world.events.deaths.filter((i) => i === bomb)).toHaveLength(1);
    expect(world.pool.hp[victim]).toBe(0);
  });
});
