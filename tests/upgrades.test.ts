import { describe, expect, it } from 'vitest';
import { buildingUpgrade, defOf, productionOptions } from '../src/config/rules.js';
import { isPacket } from '../src/net/transport.js';
import { CommandType } from '../src/sim/commands.js';
import { idIndex } from '../src/sim/entities.js';
import { fromInt } from '../src/sim/fixed.js';
import { economySystem } from '../src/sim/systems/economy.js';
import { executeCommand } from '../src/sim/systems/orders.js';
import { victorySystem } from '../src/sim/systems/victory.js';
import { BuildState, ENTITY_TYPE_COUNT, EntityType, seconds } from '../src/sim/types.js';
import { World } from '../src/sim/world.js';
import { mirrorCommand, mirrorMismatch, twinMap } from './helpers/mirror.js';

function setup(type = EntityType.Barracks) {
  const world = new World(0x1234);
  const building = world.placeBuilding(type, 0, 10, 10);
  const bi = idIndex(building);
  world.pool.buildState[bi] = BuildState.Complete;
  world.player(0).minerals = 5000;
  world.player(0).supplyMax = 200;
  return { world, pool: world.pool, building, bi };
}

const upgradeTypes = [EntityType.Barracks, EntityType.Factory] as const;

describe('building production levels', () => {
  it('keeps stable type ids and adds a single-level Airport with both aircraft', () => {
    expect(EntityType.Factory).toBe(11);
    expect(EntityType.Airport).toBe(19);
    expect(ENTITY_TYPE_COUNT).toBe(20);
    const airport = defOf(EntityType.Airport);
    expect([airport.mineralCost, airport.buildTicks, airport.footprint]).toEqual([
      200,
      seconds(35),
      3,
    ]);
    expect(productionOptions(EntityType.Airport)).toEqual([
      EntityType.Beamdrone,
      EntityType.Plasmodrone,
    ]);
    expect(buildingUpgrade(EntityType.Airport)).toBeUndefined();
    expect(buildingUpgrade(EntityType.Barracks)).toEqual({
      mineralCost: 100,
      buildTicks: seconds(20),
    });
    expect(buildingUpgrade(EntityType.Factory)).toEqual({
      mineralCost: 150,
      buildTicks: seconds(30),
    });
  });

  for (const type of upgradeTypes) {
    it(`unlocks only the upgraded ${defOf(type).name} after its full duration without repairing it`, () => {
      const { world, pool, building, bi } = setup(type);
      const other = idIndex(world.placeBuilding(type, 0, 20, 20));
      pool.buildState[other] = BuildState.Complete;
      const upgrade = buildingUpgrade(type)!;
      const locked = productionOptions(type, 2).slice(3);
      const train = (unit: EntityType, target = building) =>
        executeCommand(world, {
          type: CommandType.Train,
          player: 0,
          building: target,
          unit,
        });
      for (const unit of locked) train(unit);
      expect(pool.prodCount[bi]).toBe(0);
      expect(world.player(0).minerals).toBe(5000);
      pool.hp[bi] = 251;
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
      expect(world.player(0).minerals).toBe(5000 - upgrade.mineralCost);
      expect(pool.upgrading[bi]).toBe(1);
      for (let t = 0; t < upgrade.buildTicks - 1; t++) economySystem(world);
      for (const unit of productionOptions(type, 2)) train(unit);
      expect(pool.prodCount[bi]).toBe(0);
      expect(pool.buildingLevel[bi]).toBe(1);
      expect(pool.upgradeProgress[bi]).toBe(upgrade.buildTicks - 1);
      economySystem(world);
      expect(pool.buildingLevel[bi]).toBe(2);
      expect(pool.buildingLevel[other]).toBe(1);
      expect(pool.upgrading[bi]).toBe(0);
      expect(pool.upgradeProgress[bi]).toBe(0);
      expect(pool.hp[bi]).toBe(251);
      expect(world.events.completed).toContain(bi);
      for (const unit of locked) train(unit);
      expect(pool.prodCount[bi]).toBe(2);
      train(locked[0]!, pool.idAt(other));
      expect(pool.prodCount[other]).toBe(0);
      const bank = world.player(0).minerals;
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
      executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
      expect(world.player(0).minerals).toBe(bank);
    });
  }

  it('requires an owned, completed, idle, affordable, eligible level-one building', () => {
    for (const reason of [
      'enemy',
      'unfinished',
      'queued',
      'poor',
      'level2',
      'defeated',
      'airport',
      'worker',
    ]) {
      const type = reason === 'airport' ? EntityType.Airport : EntityType.Barracks;
      const { world, pool, building, bi } = setup(type);
      if (reason === 'enemy') pool.owner[bi] = 1;
      if (reason === 'unfinished') pool.buildState[bi] = BuildState.UnderConstruction;
      if (reason === 'queued') pool.prodPush(bi, EntityType.Burstbot);
      if (reason === 'poor') world.player(0).minerals = 99;
      if (reason === 'level2') pool.buildingLevel[bi] = 2;
      if (reason === 'defeated') world.player(0).defeated = true;
      if (reason === 'worker') pool.type[bi] = EntityType.Worker;
      const bank = world.player(0).minerals;
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
      expect(pool.upgrading[bi], reason).toBe(0);
      expect(world.player(0).minerals, reason).toBe(bank);
    }
  });

  it('refunds a cancellation once and restarts from zero while retaining the rally and damage', () => {
    const { world, pool, building, bi } = setup(EntityType.Factory);
    pool.hasRally[bi] = 1;
    pool.rallyX[bi] = fromInt(25);
    pool.rallyY[bi] = fromInt(26);
    pool.hp[bi] = 400;
    executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
    for (let t = 0; t < 20; t++) economySystem(world);
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 1, building });
    expect(pool.upgrading[bi]).toBe(1);
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
    expect(world.player(0).minerals).toBe(5000);
    expect([pool.buildingLevel[bi], pool.upgrading[bi], pool.upgradeProgress[bi]]).toEqual([
      1, 0, 0,
    ]);
    expect([pool.hp[bi], pool.hasRally[bi], pool.rallyX[bi], pool.rallyY[bi]]).toEqual([
      400,
      1,
      fromInt(25),
      fromInt(26),
    ]);
    executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
    economySystem(world);
    expect(pool.upgradeProgress[bi]).toBe(1);
  });

  it('rejects stale and malformed handles and clears upgrade state when a slot is reused', () => {
    const { world, pool, building, bi } = setup();
    for (const bad of [NaN, Infinity, building + 0.5, building + 2 ** 31, building + 2 ** 32]) {
      executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building: bad });
      expect(pool.upgrading[bi]).toBe(0);
    }
    executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
    economySystem(world);
    pool.destroy(building);
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
    expect(world.player(0).minerals).toBe(4900);
    const fresh = pool.spawn(EntityType.Factory, 0, fromInt(40), fromInt(40));
    expect(idIndex(fresh)).toBe(bi);
    expect([pool.buildingLevel[bi], pool.upgrading[bi], pool.upgradeProgress[bi]]).toEqual([
      1, 0, 0,
    ]);
    pool.buildState[bi] = BuildState.Complete;
    executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building: fresh });
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
    expect(pool.upgrading[bi]).toBe(1);
    expect(world.player(0).minerals).toBe(4750);
  });

  it('keeps a player alive when cancelling upgrades can fund a new unit', () => {
    const { world, pool, building, bi } = setup();
    const enemy = idIndex(world.placeBuilding(EntityType.CommandPost, 1, 100, 100));
    pool.buildState[enemy] = BuildState.Complete;
    world.player(1).minerals = 500;
    world.player(0).minerals = 100;
    executeCommand(world, { type: CommandType.UpgradeBuilding, player: 0, building });
    victorySystem(world);
    expect(world.player(0).defeated).toBe(false);
    expect(world.matchOver).toBe(false);
    executeCommand(world, { type: CommandType.CancelUpgrade, player: 0, building });
    executeCommand(world, {
      type: CommandType.Train,
      player: 0,
      building,
      unit: EntityType.Burstbot,
    });
    expect(pool.prodCount[bi]).toBe(1);
    victorySystem(world);
    expect(world.player(0).defeated).toBe(false);
  });

  it('trains both Airport units at level one and rejects aircraft elsewhere', () => {
    const { world, pool, building, bi } = setup(EntityType.Airport);
    const depot = idIndex(world.placeBuilding(EntityType.Depot, 0, 20, 20));
    pool.buildState[depot] = BuildState.Complete;
    for (const unit of productionOptions(EntityType.Airport)) {
      executeCommand(world, { type: CommandType.Train, player: 0, building, unit });
    }
    expect(pool.prodCount[bi]).toBe(2);
    for (const type of upgradeTypes) {
      const groundBuilding = world.placeBuilding(
        type,
        0,
        30,
        type === EntityType.Barracks ? 20 : 30,
      );
      const gi = idIndex(groundBuilding);
      pool.buildState[gi] = BuildState.Complete;
      pool.buildingLevel[gi] = 2;
      for (const unit of productionOptions(EntityType.Airport)) {
        executeCommand(world, {
          type: CommandType.Train,
          player: 0,
          building: groundBuilding,
          unit,
        });
      }
      expect(pool.prodCount[gi]).toBe(0);
    }
    for (let t = 0; t < seconds(60); t++) economySystem(world);
    const spawned: number[] = [];
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && !defOf(pool.type[i]! as EntityType).isBuilding)
        spawned.push(pool.type[i]!);
    }
    expect(spawned).toEqual([EntityType.Beamdrone, EntityType.Plasmodrone]);
    expect(pool.buildingLevel[bi]).toBe(1);
  });

  it('keeps upgrades, cancellation, and unlocked production rotationally mirrored', () => {
    const { world, pool, building, bi } = setup();
    const other = idIndex(
      world.placeBuilding(EntityType.Barracks, 1, world.map.width - 13, world.map.height - 13),
    );
    pool.buildState[other] = BuildState.Complete;
    for (const [player, x, y] of [
      [0, 20, 20],
      [1, world.map.width - 22, world.map.height - 22],
    ]) {
      const depot = idIndex(world.placeBuilding(EntityType.Depot, player!, x!, y!));
      pool.buildState[depot] = BuildState.Complete;
    }
    world.player(1).minerals = 5000;
    world.player(1).supplyMax = 200;
    const issue = (type: CommandType.UpgradeBuilding | CommandType.CancelUpgrade) => {
      const command = { type, player: 0, building };
      executeCommand(world, command);
      executeCommand(world, mirrorCommand(world, command, twinMap(world, world)));
    };
    issue(CommandType.UpgradeBuilding);
    for (let t = 0; t < 10; t++) economySystem(world);
    issue(CommandType.CancelUpgrade);
    issue(CommandType.UpgradeBuilding);
    for (let t = 0; t < buildingUpgrade(EntityType.Barracks)!.buildTicks; t++) economySystem(world);
    expect(pool.buildingLevel[bi]).toBe(2);
    expect(pool.buildingLevel[other]).toBe(2);
    const train = {
      type: CommandType.Train as const,
      player: 0,
      building,
      unit: EntityType.Arclight,
    };
    executeCommand(world, train);
    executeCommand(world, mirrorCommand(world, train, twinMap(world, world)));
    for (let t = 0; t < defOf(EntityType.Arclight).buildTicks; t++) economySystem(world);
    expect(mirrorMismatch(world, world)).toBeNull();
  });

  for (const field of ['buildingLevel', 'upgrading', 'upgradeProgress'] as const) {
    it(`includes ${field} in peer checksums`, () => {
      const { world, pool, bi } = setup();
      const before = world.checksum();
      pool[field][bi]! += 1;
      expect(world.checksum()).not.toBe(before);
    });
  }

  it('validates both upgrade commands on the wire', () => {
    for (const type of [CommandType.UpgradeBuilding, CommandType.CancelUpgrade]) {
      const accepts = (building: unknown) =>
        isPacket({
          player: 0,
          turns: [{ turn: 1, player: 0, commands: [{ type, player: 0, building }] }],
        });
      expect(accepts(131081)).toBe(true);
      for (const value of [undefined, '12', null, NaN, Infinity, 1.5])
        expect(accepts(value)).toBe(false);
    }
  });
});
