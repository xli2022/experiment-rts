import * as THREE from 'three';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import type { Game as GameType } from '../src/main.js';
import { Selection } from '../src/input/selection.js';
import { buildingUpgrade, defOf, unitRole } from '../src/config/rules.js';
import { idIndex } from '../src/sim/entities.js';
import { CommandType, MAX_COMMAND_UNITS, type Command } from '../src/sim/commands.js';
import { LocalNetwork } from '../src/net/localTransport.js';
import { LockstepRunner } from '../src/net/lockstep.js';
import { TRANSPORT_CHUNK_BYTES } from '../src/net/trysteroTransport.js';
import { coopMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import {
  EntityType,
  BuildState,
  MS_PER_TICK,
  Order,
  TICKS_PER_SECOND,
  type MatchConfig,
} from '../src/sim/types.js';
import type { CommandButton } from '../src/ui/hud.js';

let Game: typeof GameType;

beforeAll(async () => {
  // Import the real event/controller code without starting a WebGL application.
  // The entry point reports its missing canvas through its normal error path.
  vi.stubGlobal('document', { getElementById: () => null });
  const error = vi.spyOn(console, 'error').mockImplementation(() => {});
  ({ Game } = await import('../src/main.js'));
  await Promise.resolve();
  expect(error).toHaveBeenCalledWith(
    expect.objectContaining({ message: 'missing #viewport or #ui-root' }),
  );
  error.mockRestore();
  vi.unstubAllGlobals();
});

afterEach(() => vi.unstubAllGlobals());

interface Controller {
  sim: Simulation;
  selection: Selection;
  placing: EntityType | null;
  attackMovePending: boolean;
  pointerNdc: THREE.Vector2;
  conceded: boolean;
  finished: boolean;
  commandButtons: CommandButton[];
  handleKey(event: KeyboardEvent): void;
  issue(command: Command): void;
  issueGroundOrder(x: number, z: number, attackMove: boolean): void;
  buildCommandCard(): CommandButton[];
  attachInput(canvas: HTMLCanvasElement): void;
  checkResult(): void;
}

function controller(config: MatchConfig = coopMatch(1, { botPlayers: [1, 2, 3] })) {
  const sim = new Simulation(config);
  const issue = vi.fn<(command: Command) => void>();
  const showDialog = vi.fn();
  const stopAgents = vi.fn();
  const camera = { lookAt: vi.fn() };
  const hud = {
    pointerOverUi: false,
    dialogOpen: false,
    marquee: { style: { display: 'none' } },
    showDialog,
    setSurrenderAvailable: vi.fn(),
    toggleFullscreen: vi.fn(),
  };
  const gallery = { isOpen: false, close: vi.fn() };
  const game: Controller = Object.assign(Object.create(Game.prototype) as Controller, {
    sim,
    localPlayer: 0,
    localTeam: 0,
    selection: new Selection(0),
    pointerNdc: new THREE.Vector2(),
    placing: null,
    attackMovePending: false,
    finished: false,
    knockedOut: false,
    conceded: false,
    elapsedS: 0,
    ghost: { visible: false },
    updateGhost: vi.fn(),
    camera,
    hud,
    gallery,
    projectiles: { spawnClickMarker: vi.fn() },
    entities: { noteOrderIssued: vi.fn() },
    issue,
    stopAgents,
  });
  const own = (type: EntityType): number => {
    const pool = sim.world.pool;
    return Array.from({ length: pool.count }, (_, i) => i).find(
      (i) => pool.owner[i] === 0 && pool.type[i] === type,
    )!;
  };
  return { game, issue, showDialog, stopAgents, own, camera, hud, gallery };
}

function key(code: string, extra: object = {}): KeyboardEvent {
  return {
    code,
    target: null,
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    shiftKey: false,
    repeat: false,
    preventDefault: vi.fn(),
    ...extra,
  } as unknown as KeyboardEvent;
}

describe('game controls', () => {
  it.each([
    {
      building: EntityType.Barracks,
      basic: [EntityType.Slicebot, EntityType.Burstbot, EntityType.Firespout],
      advanced: [EntityType.Arclight, EntityType.Fixomatic],
    },
    {
      building: EntityType.Factory,
      basic: [EntityType.Boomwalker, EntityType.Sentry, EntityType.Piercebot],
      advanced: [EntityType.DarkGolem, EntityType.IceGolem],
    },
  ])(
    'shows level locks and upgrade controls for production building $building',
    ({ building, basic, advanced }) => {
      const { game, issue } = controller();
      const world = game.sim.world;
      world.player(0).minerals = 5000;
      const id = world.pool.spawn(building, 0, 30 * 65536, 30 * 65536);
      const index = idIndex(id);
      world.pool.buildState[index] = BuildState.Complete;
      game.selection.set([index], world);
      const card = (): CommandButton[] => (game.commandButtons = game.buildCommandCard());
      const buttonFor = (type: EntityType): CommandButton =>
        card().find((button) => button.label === defOf(type).name)!;
      for (const type of basic) expect(buttonFor(type).enabled).toBe(true);
      for (const type of advanced) {
        const button = buttonFor(type);
        expect(button.enabled).toBe(false);
        expect(button.requirement).toBe('Level 2');
        expect(button.description).toContain(`Requires ${defOf(building).name} level 2`);
        game.handleKey(key(`Key${button.key}`));
      }
      expect(issue).not.toHaveBeenCalled();
      expect(new Set(card().map((button) => button.key)).size).toBe(card().length);
      const upgrade = card().find((button) => button.key === 'U')!;
      expect(upgrade.enabled).toBe(true);
      expect(upgrade.cost).toBe(buildingUpgrade(building)!.mineralCost);
      expect(upgrade.description).toContain('idle training queue');
      game.handleKey(key('KeyU'));
      expect(issue).toHaveBeenLastCalledWith({
        type: CommandType.UpgradeBuilding,
        player: 0,
        building: id,
      });

      world.pool.upgrading[index] = 1;
      for (const type of [...basic, ...advanced]) expect(buttonFor(type).enabled).toBe(false);
      const cancel = card().find((button) => button.key === 'X')!;
      expect(cancel.label).toBe('Cancel upgrade');
      expect(cancel.description).toContain(
        `refund ${buildingUpgrade(building)!.mineralCost} minerals`,
      );
      game.handleKey(key('KeyX'));
      expect(issue).toHaveBeenLastCalledWith({
        type: CommandType.CancelUpgrade,
        player: 0,
        building: id,
      });

      world.pool.upgrading[index] = 0;
      world.pool.buildingLevel[index] = 2;
      for (const type of [...basic, ...advanced]) {
        const button = buttonFor(type);
        expect(button.enabled).toBe(true);
        expect(button.requirement).toBeUndefined();
      }
      expect(card().some((button) => button.key === 'U')).toBe(false);

      world.pool.buildingLevel[index] = 1;
      world.pool.prodPush(index, basic[0]!);
      const queuedUpgrade = card().find((button) => button.key === 'U')!;
      expect(queuedUpgrade.enabled).toBe(false);
      expect(queuedUpgrade.requirement).toBe('Idle queue');
      issue.mockClear();
      game.handleKey(key('KeyU'));
      expect(issue).not.toHaveBeenCalled();
      expect(card().find((button) => button.key === 'X')!.label).toBe('Cancel');
    },
  );

  it('places an Airport and trains its two flyers with distinct hotkeys', () => {
    const { game, issue, own } = controller();
    const world = game.sim.world;
    world.player(0).minerals = 5000;
    game.selection.set([own(EntityType.Worker)], world);
    game.commandButtons = game.buildCommandCard();
    const airport = game.commandButtons.find((button) => button.label === 'Airport')!;
    expect(airport.key).toBe('P');
    expect(airport.cost).toBe(defOf(EntityType.Airport).mineralCost);
    expect(game.commandButtons.find((button) => button.key === 'F')!.label).toBe('Factory');
    expect(new Set(game.commandButtons.map((button) => button.key)).size).toBe(
      game.commandButtons.length,
    );
    game.handleKey(key('KeyP'));
    expect(game.placing).toBe(EntityType.Airport);

    const id = world.pool.spawn(EntityType.Airport, 0, 30 * 65536, 30 * 65536);
    const index = idIndex(id);
    world.pool.buildState[index] = BuildState.Complete;
    game.selection.set([index], world);
    game.commandButtons = game.buildCommandCard();
    expect(game.commandButtons.map((button) => button.label)).toEqual(['Beamdrone', 'Plasmodrone']);
    expect(game.commandButtons.every((button) => button.enabled)).toBe(true);
    game.handleKey(key('KeyD'));
    game.handleKey(key('KeyP'));
    expect(issue.mock.calls.map(([command]) => command)).toEqual([
      { type: CommandType.Train, player: 0, building: id, unit: EntityType.Beamdrone },
      { type: CommandType.Train, player: 0, building: id, unit: EntityType.Plasmodrone },
    ]);
  });

  it('uses F for the current worker build menu and fullscreen outside it', () => {
    const { game, own, hud } = controller();
    const world = game.sim.world;
    game.selection.set([own(EntityType.CommandPost)], world);
    game.commandButtons = game.buildCommandCard();
    // A selection may change before the next frame refreshes the visible card.
    game.selection.set([own(EntityType.Worker)], world);
    world.player(0).minerals = defOf(EntityType.Factory).mineralCost;
    const build = key('KeyF');
    game.handleKey(build);
    expect(build.preventDefault).toHaveBeenCalledOnce();
    expect(game.placing).toBe(EntityType.Factory);
    expect(hud.toggleFullscreen).not.toHaveBeenCalled();
    game.handleKey(key('KeyF', { repeat: true }));
    expect(game.placing).toBe(EntityType.Factory);
    game.handleKey(key('KeyF'));
    expect(game.placing).toBeNull();

    world.player(0).minerals = 0;
    const denied = key('KeyF');
    game.handleKey(denied);
    expect(denied.preventDefault).toHaveBeenCalledOnce();
    expect(game.placing).toBeNull();
    expect(hud.toggleFullscreen).not.toHaveBeenCalled();

    // The reverse transition must discard the old Factory binding immediately.
    game.selection.set([own(EntityType.CommandPost)], world);
    game.handleKey(key('KeyF'));
    expect(hud.toggleFullscreen).toHaveBeenCalledOnce();
    expect(game.placing).toBeNull();
    game.handleKey(key('KeyF', { repeat: true }));
    expect(hud.toggleFullscreen).toHaveBeenCalledOnce();
    game.selection.clear();
    game.handleKey(key('KeyF'));
    expect(hud.toggleFullscreen).toHaveBeenCalledTimes(2);
  });

  it('leaves modified F shortcuts to the browser in either selection context', () => {
    const { game, own, hud } = controller();
    const world = game.sim.world;
    world.player(0).minerals = 5000;
    for (const type of [EntityType.Worker, EntityType.CommandPost]) {
      game.selection.set([own(type)], world);
      for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
        const event = key('KeyF', { [modifier]: true });
        game.handleKey(event);
        expect(event.preventDefault).not.toHaveBeenCalled();
        expect(game.placing).toBeNull();
        expect(hud.toggleFullscreen).not.toHaveBeenCalled();
      }
    }
  });

  it('consumes held quick-select keys without extra actions or claiming modified keys', () => {
    const { game, camera } = controller();
    const world = game.sim.world;
    world.pool.spawn(EntityType.Burstbot, 0, 30 * 65536, 30 * 65536);
    for (let i = 0; i < world.pool.count; i++) {
      if (world.pool.type[i] === EntityType.Worker) world.pool.order[i] = Order.None;
    }
    for (const code of ['F1', 'F2']) {
      game.handleKey(key(code));
      const selected = game.selection.ids(world);
      const cameraActions = camera.lookAt.mock.calls.length;
      for (const modifier of [null, 'ctrlKey', 'metaKey', 'altKey', 'shiftKey']) {
        const event = Object.assign(new Event('keydown', { cancelable: true }), {
          code,
          repeat: true,
          ...(modifier === null ? {} : { [modifier]: true }),
        }) as KeyboardEvent;
        game.handleKey(event);
        expect(event.defaultPrevented).toBe(modifier === null);
        expect(game.selection.ids(world)).toEqual(selected);
        expect(camera.lookAt).toHaveBeenCalledTimes(cameraActions);
      }
    }
    const other = Object.assign(new Event('keydown', { cancelable: true }), {
      code: 'KeyS',
      repeat: true,
    }) as KeyboardEvent;
    game.handleKey(other);
    expect(other.defaultPrevented).toBe(false);
  });

  it('issues Stop and Hold only once per keypress, including command-card shortcuts', () => {
    const { game, issue, own } = controller();
    game.selection.set([own(EntityType.Worker)], game.sim.world);
    for (const withCard of [false, true]) {
      game.commandButtons = withCard ? game.buildCommandCard() : [];
      for (const code of ['KeyS', 'KeyH']) {
        issue.mockClear();
        game.handleKey(key(code));
        for (let repeat = 0; repeat < 30; repeat++) game.handleKey(key(code, { repeat: true }));
        expect(issue).toHaveBeenCalledOnce();
        expect(issue.mock.calls[0]![0].type).toBe(
          code === 'KeyS' ? CommandType.Stop : CommandType.Hold,
        );
      }
    }
  });

  it('cycles idle workers with F1, skipping busy workers and centering the selected worker', () => {
    const { game, camera } = controller();
    const world = game.sim.world;
    const pool = world.pool;
    const workers = Array.from({ length: pool.count }, (_, i) => i).filter(
      (i) => pool.owner[i] === 0 && pool.type[i] === EntityType.Worker,
    );
    for (const i of workers) pool.order[i] = Order.None;
    pool.order[workers[0]!] = Order.Harvest;
    pool.order[workers[1]!] = Order.Build;
    const idle = workers.slice(2);
    expect(idle.length).toBeGreaterThan(1);
    game.placing = EntityType.Depot;
    game.attackMovePending = true;
    for (let press = 0; press <= idle.length; press++) {
      const event = key('F1');
      game.handleKey(event);
      expect(event.preventDefault).toHaveBeenCalledOnce();
      expect(game.selection.single()).toBe(idle[press % idle.length]);
      expect(camera.lookAt).toHaveBeenLastCalledWith(
        pool.posX[idle[press % idle.length]!]! / 65536,
        pool.posY[idle[press % idle.length]!]! / 65536,
      );
    }
    expect(game.placing).toBeNull();
    expect(game.attackMovePending).toBe(false);
  });

  it('selects a full army with F2 and sends every unit through bounded command packets', () => {
    const config = coopMatch(1);
    const { game, camera, own } = controller(config);
    const replica = new Simulation(config);
    const world = game.sim.world;
    const pool = world.pool;
    const expected: number[] = [];
    for (let unit = 0; unit < 130; unit++) {
      const type = unit % 2 === 0 ? EntityType.Burstbot : EntityType.Fixomatic;
      expected.push(pool.spawn(type, 0, 30 * 65536, 30 * 65536));
      replica.world.pool.spawn(type, 0, 30 * 65536, 30 * 65536);
    }
    pool.spawn(EntityType.Beamdrone, 1, 30 * 65536, 30 * 65536);
    pool.spawn(EntityType.Burstbot, 2, 30 * 65536, 30 * 65536);
    replica.world.pool.spawn(EntityType.Beamdrone, 1, 30 * 65536, 30 * 65536);
    replica.world.pool.spawn(EntityType.Burstbot, 2, 30 * 65536, 30 * 65536);
    game.selection.set([own(EntityType.Worker)], world);
    game.handleKey(key('F2'));
    expect(game.selection.ids(world)).toEqual(expected);
    expect(camera.lookAt).not.toHaveBeenCalled();
    game.handleKey(key('F2'));
    expect(camera.lookAt).toHaveBeenCalledOnce();

    const net = new LocalNetwork(2);
    const local = new LockstepRunner(game.sim, net.createTransport(0), {}, () => net.nowMs);
    const remote = new LockstepRunner(replica, net.createTransport(1), {}, () => net.nowMs);
    const turnCommands = new Map<number, Command[]>();
    const submit = net.submit.bind(net);
    net.submit = (from, packet) => {
      expect(new TextEncoder().encode(JSON.stringify(packet)).byteLength).toBeLessThanOrEqual(
        TRANSPORT_CHUNK_BYTES,
      );
      if (from === 0) {
        for (const turn of packet.turns) {
          if (turn.player === 0) turnCommands.set(turn.turn, turn.commands);
        }
      }
      submit(from, packet);
    };
    Object.assign(game, { runner: local, issue: (Game.prototype as unknown as Controller).issue });
    game.issueGroundOrder(35, 35, false);
    for (let frame = 0; frame < 20; frame++) {
      local.update(MS_PER_TICK);
      remote.update(MS_PER_TICK);
      net.advance(MS_PER_TICK);
    }
    const commands = [...turnCommands.values()].flat();
    expect(commands.flatMap((command) => ('units' in command ? command.units : []))).toEqual(
      expected,
    );
    for (const command of commands) {
      expect(command.type).toBe(CommandType.Move);
      expect('units' in command && command.units.length <= MAX_COMMAND_UNITS).toBe(true);
    }
    expect(local.currentTick).toBe(remote.currentTick);
    expect(game.sim.checksum()).toBe(replica.checksum());
  });

  it('describes unit roles and build times from the current balance table', () => {
    const { game, own } = controller();
    const world = game.sim.world;
    game.selection.set([own(EntityType.CommandPost)], world);
    const train = game
      .buildCommandCard()
      .find((button) => button.label === defOf(EntityType.Worker).name)!;
    expect(train.description).toContain(unitRole(EntityType.Worker));
    expect(train.description).toContain(
      `${defOf(EntityType.Worker).buildTicks / TICKS_PER_SECOND}s to train`,
    );
    expect(train.description).toContain(`${defOf(EntityType.Worker).supplyCost} supply`);
    game.selection.set([own(EntityType.Worker)], world);
    const build = game.buildCommandCard().find((button) => button.key === 'D')!;
    expect(build.description).toContain(unitRole(EntityType.Depot));
    expect(build.description).toContain(
      `${defOf(EntityType.Depot).buildTicks / TICKS_PER_SECOND}s to build`,
    );
    expect(build.description).toContain(`+${defOf(EntityType.Depot).supplyProvided} supply`);
  });

  it('leaves selection alone when quick-select shortcuts are blocked by input or match state', () => {
    const { game, own, hud, gallery } = controller();
    const world = game.sim.world;
    game.selection.set([own(EntityType.CommandPost)], world);
    const original = game.selection.ids(world);
    for (const code of ['F1', 'F2']) {
      for (const extra of [
        { target: { tagName: 'INPUT' } },
        { target: { tagName: 'TEXTAREA' } },
        { target: { isContentEditable: true } },
        { repeat: true },
        { ctrlKey: true },
        { altKey: true },
      ]) {
        game.handleKey(key(code, extra));
        expect(game.selection.ids(world)).toEqual(original);
      }
      hud.dialogOpen = true;
      game.handleKey(key(code));
      hud.dialogOpen = false;
      gallery.isOpen = true;
      game.handleKey(key(code));
      gallery.isOpen = false;
      game.finished = true;
      game.handleKey(key(code));
      game.finished = false;
      world.player(0).defeated = true;
      game.handleKey(key(code));
      world.player(0).defeated = false;
      expect(game.selection.ids(world)).toEqual(original);
    }
  });

  it('moves selected units and sets selected building rallies from the minimap', () => {
    const { game, own, issue } = controller();
    game.selection.set([own(EntityType.Worker), own(EntityType.CommandPost)], game.sim.world);
    game.issueGroundOrder(30, 30, false);
    expect(issue.mock.calls.map(([command]) => command.type)).toEqual([
      CommandType.SetRally,
      CommandType.Move,
    ]);
  });

  it('keeps attack placement and building placement mutually exclusive', () => {
    const { game, own } = controller();
    game.sim.world.player(0).minerals = 5000;
    game.selection.set([own(EntityType.Worker)], game.sim.world);
    const card = game.buildCommandCard();
    const depot = card.find((button) => button.key === 'D')!;
    const attack = card.find((button) => button.key === 'A')!;
    depot.onClick();
    expect(game.placing).toBe(EntityType.Depot);
    attack.onClick();
    expect(game.placing).toBeNull();
    expect(game.attackMovePending).toBe(true);
    depot.onClick();
    expect(game.placing).toBe(EntityType.Depot);
    expect(game.attackMovePending).toBe(false);
  });

  it('uses pointerdown coordinates even when no pointermove preceded it', () => {
    const { game } = controller();
    vi.stubGlobal('window', new EventTarget());
    const canvas = Object.assign(new EventTarget(), {
      getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }),
    });
    const contextOrder = vi.fn(() => expect(game.pointerNdc.toArray()).toEqual([0.5, -0.5]));
    Object.assign(game, { issueContextOrder: contextOrder });
    game.attachInput(canvas as unknown as HTMLCanvasElement);
    canvas.dispatchEvent(
      Object.assign(new Event('pointerdown'), { button: 2, clientX: 160, clientY: 95 }),
    );
    expect(contextOrder).toHaveBeenCalledOnce();
  });

  it('lets a naturally eliminated solo co-op player watch their surviving ally', () => {
    const { game, showDialog, stopAgents } = controller();
    game.sim.world.player(0).defeated = true;
    game.checkResult();
    expect(showDialog.mock.calls[0]?.[0]).toBe('You are out');
    expect(stopAgents).not.toHaveBeenCalled();
  });

  it('ends the local experience when the solo co-op player explicitly conceded', () => {
    const { game, showDialog, stopAgents } = controller();
    game.sim.world.player(0).defeated = true;
    game.conceded = true;
    game.checkResult();
    expect(showDialog.mock.calls[0]?.[0]).toBe('Defeat');
    expect(stopAgents).toHaveBeenCalledOnce();
  });
});
