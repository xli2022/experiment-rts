/**
 * Application entry point: scene setup, input, and the frame loop.
 *
 * The important structural point is that this file *drives* the simulation but
 * never reaches into it. Input produces `Command` objects, which go to the
 * lockstep runner, which decides when they execute. Rendering reads simulation
 * state and writes nothing back. A single-player skirmish and an online match
 * differ only in which `Transport` is constructed here.
 */

import * as THREE from 'three';
import { defOf } from './config/rules.js';
import { LockstepRunner } from './net/lockstep.js';
import { SoloTransport } from './net/localTransport.js';
import type { Transport } from './net/transport.js';
import { CommandType, type Command } from './sim/commands.js';
import { fromFloat } from './sim/fixed.js';
import { MAP_SIZE } from './sim/map.js';
import { Simulation } from './sim/tick.js';
import { BuildState, EntityType, NEUTRAL, NO_ENTITY, type PlayerId } from './sim/types.js';
import { EntityRenderer } from './render/entities.js';
import { RtsCamera } from './render/camera.js';
import { TerrainRenderer } from './render/terrain.js';
import { ProceduralModelProvider } from './render/models/procedural.js';
import { groundPointAt, pickAt, pickInBox, Selection } from './input/selection.js';
import { Hud, type CommandButton } from './ui/hud.js';
import { showLobby, type MatchSetup } from './ui/lobby.js';

/** Buildings the player can place, in command-card order. */
const BUILD_MENU: { type: EntityType; key: string }[] = [
  { type: EntityType.Depot, key: 'D' },
  { type: EntityType.Barracks, key: 'B' },
  { type: EntityType.Turret, key: 'T' },
];

class Game {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: RtsCamera;
  private readonly provider = new ProceduralModelProvider();
  private readonly entities: EntityRenderer;
  private readonly terrain: TerrainRenderer;
  private readonly selection: Selection;
  private readonly hud: Hud;
  private readonly runner: LockstepRunner;
  private readonly sim: Simulation;
  private readonly localPlayer: PlayerId;

  /** Building type awaiting placement, or null. */
  private placing: EntityType | null = null;
  private readonly ghost: THREE.Mesh;
  /** True when the next left click issues an attack-move. */
  private attackMovePending = false;

  private dragStart: { x: number; y: number } | null = null;
  private pointerNdc = new THREE.Vector2();
  private lastFrameMs = 0;
  private lastTick = -1;
  private finished = false;

  constructor(
    canvas: HTMLCanvasElement,
    uiRoot: HTMLElement,
    seed: number,
    transport: Transport,
    botPlayers: PlayerId[],
  ) {
    this.localPlayer = transport.localPlayer;
    this.sim = new Simulation(seed);

    // Slots played by the AI. Because the bot is deterministic it runs inside
    // the simulation on every peer, so every peer must agree on this set — it
    // comes from the lobby, which both sides agreed on before starting.
    for (const p of botPlayers) this.sim.botPlayers.add(p);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0b0e14);
    this.scene.fog = new THREE.Fog(0x0b0e14, 90, 190);

    this.camera = new RtsCamera(canvas, MAP_SIZE);
    this.terrain = new TerrainRenderer(this.sim.world.map);
    this.entities = new EntityRenderer(this.provider, this.sim.world);
    this.scene.add(this.terrain.group, this.entities.group);
    this.addLights();

    this.ghost = this.makeGhost();
    this.scene.add(this.ghost);

    this.selection = new Selection(this.localPlayer);
    this.hud = new Hud(uiRoot, MAP_SIZE, this.localPlayer, (x, z, secondary) => {
      if (secondary) this.issueGroundOrder(x, z, false);
      else this.camera.lookAt(x, z);
    });

    this.runner = new LockstepRunner(this.sim, transport, {
      onStall: (waiting) =>
        this.hud.showBanner(`Waiting for player ${waiting.join(', ')}…`, 'warn'),
      onResume: () => this.hud.hideBanner(),
      onDesync: (tick) =>
        this.hud.showDialog(
          'Desynchronised',
          `The two games diverged at tick ${tick}. In peer-to-peer play there is no ` +
            `authority to correct this, so the match cannot continue.`,
          [{ label: 'Reload', primary: true, onClick: () => location.reload() }],
        ),
    });

    // Open on the player's own base, which is where they will look first.
    const start = this.sim.world.map.starts[this.localPlayer]!;
    this.camera.lookAt(start.tileX, start.tileY);

    this.attachInput(canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  private addLights(): void {
    const sun = new THREE.DirectionalLight(0xfff2e0, 2.0);
    sun.position.set(50, 90, 30);
    this.scene.add(sun);
    this.scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x2a3020, 1.5));
  }

  private makeGhost(): THREE.Mesh {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 0.4, 1),
      new THREE.MeshBasicMaterial({ color: 0x7dff9b, transparent: true, opacity: 0.35 }),
    );
    mesh.visible = false;
    return mesh;
  }

  // -------------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------------

  private attachInput(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    canvas.addEventListener('pointermove', (e) => {
      const rect = canvas.getBoundingClientRect();
      this.pointerNdc.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.camera.overUi = this.hud.pointerOverUi;

      if (this.dragStart) this.updateMarquee(e.clientX, e.clientY);
      this.updateGhost();
    });

    canvas.addEventListener('pointerdown', (e) => {
      if (this.hud.pointerOverUi || this.finished) return;
      if (e.button === 0) {
        if (this.placing !== null) {
          this.placeBuilding();
          return;
        }
        if (this.attackMovePending) {
          const p = groundPointAt(this.camera.camera, this.pointerNdc.x, this.pointerNdc.y);
          this.attackMovePending = false;
          if (p) this.issueGroundOrder(p.x, p.z, true);
          return;
        }
        this.dragStart = { x: e.clientX, y: e.clientY };
      } else if (e.button === 2) {
        this.cancelModes();
        this.issueContextOrder();
      }
    });

    window.addEventListener('pointerup', (e) => {
      if (e.button !== 0 || !this.dragStart) return;
      const dx = Math.abs(e.clientX - this.dragStart.x);
      const dy = Math.abs(e.clientY - this.dragStart.y);
      // A short drag is a click, not a box — otherwise a slightly shaky click
      // selects nothing.
      if (dx < 5 && dy < 5) this.handleClickSelect(e.shiftKey);
      else this.handleBoxSelect(this.dragStart, { x: e.clientX, y: e.clientY }, e.shiftKey);
      this.dragStart = null;
      this.hud.marquee.style.display = 'none';
    });

    window.addEventListener('keydown', (e) => this.handleKey(e));
  }

  private handleKey(e: KeyboardEvent): void {
    if (e.target instanceof HTMLInputElement) return;

    if (e.code === 'Escape') {
      this.cancelModes();
      return;
    }

    // Control groups: Ctrl+N assigns, N recalls.
    const digit = e.code.startsWith('Digit') ? Number(e.code.slice(5)) : -1;
    if (digit >= 0 && digit <= 9) {
      if (e.ctrlKey || e.metaKey) {
        this.selection.assignGroup(digit);
        e.preventDefault();
      } else {
        this.selection.recallGroup(digit, this.sim.world);
      }
      return;
    }

    switch (e.code) {
      case 'KeyS':
        if (this.selection.hasOwnUnits(this.sim.world)) {
          this.issue({
            type: CommandType.Stop,
            player: this.localPlayer,
            units: this.selection.ids(this.sim.world),
          });
        }
        break;
      case 'KeyH':
        if (this.selection.hasOwnUnits(this.sim.world)) {
          this.issue({
            type: CommandType.Hold,
            player: this.localPlayer,
            units: this.selection.ids(this.sim.world),
          });
        }
        break;
      case 'KeyA':
        if (this.selection.hasOwnUnits(this.sim.world)) this.attackMovePending = true;
        break;
      default:
        break;
    }
  }

  private cancelModes(): void {
    this.placing = null;
    this.attackMovePending = false;
    this.ghost.visible = false;
  }

  private updateMarquee(x: number, y: number): void {
    if (!this.dragStart) return;
    const style = this.hud.marquee.style;
    style.display = 'block';
    style.left = `${Math.min(this.dragStart.x, x)}px`;
    style.top = `${Math.min(this.dragStart.y, y)}px`;
    style.width = `${Math.abs(x - this.dragStart.x)}px`;
    style.height = `${Math.abs(y - this.dragStart.y)}px`;
  }

  private handleClickSelect(additive: boolean): void {
    const hit = pickAt(
      this.sim.world,
      this.camera.camera,
      this.pointerNdc.x,
      this.pointerNdc.y,
      this.localPlayer,
    );
    if (hit < 0) {
      if (!additive) this.selection.clear();
      return;
    }
    if (additive) this.selection.toggle(hit);
    else this.selection.set([hit]);
  }

  private handleBoxSelect(
    a: { x: number; y: number },
    b: { x: number; y: number },
    additive: boolean,
  ): void {
    const canvas = this.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    const toNdc = (p: { x: number; y: number }): [number, number] => [
      ((p.x - rect.left) / rect.width) * 2 - 1,
      -((p.y - rect.top) / rect.height) * 2 + 1,
    ];
    const [ax, ay] = toNdc(a);
    const [bx, by] = toNdc(b);

    const found = pickInBox(
      this.sim.world,
      this.camera.camera,
      Math.min(ax, bx),
      Math.min(ay, by),
      Math.max(ax, bx),
      Math.max(ay, by),
      this.localPlayer,
    );
    if (additive) this.selection.add(found);
    else this.selection.set(found);
  }

  /**
   * Right-click: infer the order from what is under the cursor.
   *
   * Enemy → attack, mineral patch → harvest, ground → move. This single verb is
   * how RTS players issue most of their orders, so getting the inference right
   * matters more than any explicit button.
   */
  private issueContextOrder(): void {
    if (!this.selection.hasOwnUnits(this.sim.world)) return;
    const world = this.sim.world;
    const units = this.selection.ids(world);
    if (units.length === 0) return;

    const hit = pickAt(
      world,
      this.camera.camera,
      this.pointerNdc.x,
      this.pointerNdc.y,
      this.localPlayer,
    );

    if (hit >= 0) {
      const owner = world.pool.owner[hit]!;
      const type = world.pool.type[hit]! as EntityType;

      if (type === EntityType.MineralPatch) {
        this.issue({
          type: CommandType.Harvest,
          player: this.localPlayer,
          units,
          target: world.pool.idAt(hit),
        });
        return;
      }
      if (owner !== this.localPlayer && owner !== NEUTRAL) {
        this.issue({
          type: CommandType.Attack,
          player: this.localPlayer,
          units,
          target: world.pool.idAt(hit),
        });
        return;
      }
    }

    const point = groundPointAt(this.camera.camera, this.pointerNdc.x, this.pointerNdc.y);
    if (point) this.issueGroundOrder(point.x, point.z, false);
  }

  private issueGroundOrder(x: number, z: number, attackMove: boolean): void {
    const units = this.selection.ids(this.sim.world);
    if (units.length === 0) return;
    this.issue({
      type: attackMove ? CommandType.AttackMove : CommandType.Move,
      player: this.localPlayer,
      units,
      x: fromFloat(x),
      y: fromFloat(z),
    });
  }

  private updateGhost(): void {
    if (this.placing === null) {
      this.ghost.visible = false;
      return;
    }
    const point = groundPointAt(this.camera.camera, this.pointerNdc.x, this.pointerNdc.y);
    if (!point) return;

    const def = defOf(this.placing);
    const tileX = Math.floor(point.x) - (def.footprint >> 1);
    const tileY = Math.floor(point.z) - (def.footprint >> 1);
    const legal = this.sim.world.map.canPlace(tileX, tileY, def.footprint);

    this.ghost.visible = true;
    this.ghost.scale.set(def.footprint, 1, def.footprint);
    this.ghost.position.set(tileX + def.footprint / 2, 0.2, tileY + def.footprint / 2);
    (this.ghost.material as THREE.MeshBasicMaterial).color.setHex(legal ? 0x7dff9b : 0xff5a4a);
  }

  private placeBuilding(): void {
    if (this.placing === null) return;
    const point = groundPointAt(this.camera.camera, this.pointerNdc.x, this.pointerNdc.y);
    if (!point) return;

    const def = defOf(this.placing);
    const tileX = Math.floor(point.x) - (def.footprint >> 1);
    const tileY = Math.floor(point.z) - (def.footprint >> 1);
    if (!this.sim.world.map.canPlace(tileX, tileY, def.footprint)) {
      this.hud.showBanner('Cannot build there', 'warn');
      window.setTimeout(() => this.hud.hideBanner(), 1400);
      return;
    }

    const worker = this.findSelectedWorker();
    if (worker === NO_ENTITY) {
      this.hud.showBanner('Select a worker first', 'warn');
      window.setTimeout(() => this.hud.hideBanner(), 1400);
      return;
    }

    this.issue({
      type: CommandType.Build,
      player: this.localPlayer,
      worker,
      building: this.placing,
      tileX,
      tileY,
    });
    this.cancelModes();
  }

  private findSelectedWorker(): number {
    for (const i of this.selection.indices) {
      if (
        this.sim.world.pool.alive[i] === 1 &&
        this.sim.world.pool.type[i] === EntityType.Worker &&
        this.sim.world.pool.owner[i] === this.localPlayer
      ) {
        return this.sim.world.pool.idAt(i);
      }
    }
    return NO_ENTITY;
  }

  private issue(command: Command): void {
    this.runner.issue(command);
  }

  // -------------------------------------------------------------------------
  // Frame loop
  // -------------------------------------------------------------------------

  start(): void {
    this.lastFrameMs = performance.now();
    const frame = (nowMs: number): void => {
      const dtMs = Math.min(nowMs - this.lastFrameMs, 250);
      this.lastFrameMs = nowMs;
      this.tick(dtMs);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  private tick(dtMs: number): void {
    const alpha = this.runner.update(dtMs);

    // Snapshot exactly once per simulation tick, not per frame — interpolation
    // needs the last two *ticks*, and snapshotting per frame would collapse the
    // interval to nothing and make movement stutter.
    if (this.sim.world.tick !== this.lastTick) {
      this.entities.captureSnapshot(this.sim.world);
      this.lastTick = this.sim.world.tick;
    }

    this.camera.update(dtMs / 1000);
    this.selection.prune(this.sim.world);

    this.entities.update(this.sim.world, alpha, this.selection.indices, this.camera.camera);
    this.hud.updateResources(this.sim.world);
    this.hud.updateSelection(this.sim.world, this.selection.indices);
    this.hud.setCommands(this.buildCommandCard());
    this.hud.drawMinimap(
      this.sim.world,
      this.camera.focusPoint.x,
      this.camera.focusPoint.z,
      this.camera.zoomHeight * 0.42,
    );

    this.checkResult();
    this.renderer.render(this.scene, this.camera.camera);
  }

  /** Contextual buttons for the current selection. */
  private buildCommandCard(): CommandButton[] {
    const world = this.sim.world;
    const buttons: CommandButton[] = [];
    const minerals = world.player(this.localPlayer).minerals;

    // Production buildings offer their unit list.
    const single = this.selection.single();
    if (
      single >= 0 &&
      world.pool.owner[single] === this.localPlayer &&
      world.pool.buildState[single] === BuildState.Complete
    ) {
      const def = defOf(world.pool.type[single]! as EntityType);
      for (const unit of def.produces) {
        const unitDef = defOf(unit);
        buttons.push({
          key: unitDef.name[0]!,
          label: unitDef.name,
          cost: unitDef.mineralCost,
          enabled: minerals >= unitDef.mineralCost,
          onClick: () =>
            this.issue({
              type: CommandType.Train,
              player: this.localPlayer,
              building: world.pool.idAt(single),
              unit,
            }),
        });
      }
    }

    // Workers offer the build menu.
    if (this.findSelectedWorker() !== NO_ENTITY) {
      for (const entry of BUILD_MENU) {
        const def = defOf(entry.type);
        buttons.push({
          key: entry.key,
          label: def.name,
          cost: def.mineralCost,
          enabled: minerals >= def.mineralCost,
          active: this.placing === entry.type,
          onClick: () => {
            this.placing = this.placing === entry.type ? null : entry.type;
          },
        });
      }
    }

    if (this.selection.hasOwnUnits(world)) {
      buttons.push({
        key: 'A',
        label: 'Attack',
        enabled: true,
        active: this.attackMovePending,
        onClick: () => {
          this.attackMovePending = true;
        },
      });
      buttons.push({
        key: 'S',
        label: 'Stop',
        enabled: true,
        onClick: () =>
          this.issue({
            type: CommandType.Stop,
            player: this.localPlayer,
            units: this.selection.ids(world),
          }),
      });
    }

    return buttons;
  }

  private checkResult(): void {
    if (this.finished) return;
    const winner = this.sim.world.winner;
    if (winner === NO_ENTITY) return;

    this.finished = true;
    const won = winner === this.localPlayer;
    this.hud.showDialog(
      won ? 'Victory' : 'Defeat',
      won
        ? 'Every enemy structure has been destroyed.'
        : 'All of your structures were destroyed.',
      [{ label: 'Play again', primary: true, onClick: () => location.reload() }],
    );
  }

  private resize(): void {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.resize(width, height);
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  const canvas = document.getElementById('viewport') as HTMLCanvasElement | null;
  const uiRoot = document.getElementById('ui-root');
  if (!canvas || !uiRoot) throw new Error('missing #viewport or #ui-root');

  const params = new URLSearchParams(location.search);

  // `?skip=ai` boots straight into a skirmish, which is what the end-to-end
  // tests and screenshot tooling use. A seed in the URL makes any match
  // reproducible, which is how a desync report becomes debuggable.
  let setup: MatchSetup;
  if (params.get('skip') === 'ai') {
    setup = {
      transport: new SoloTransport(),
      seed: Number(params.get('seed') ?? 0) || 0x51ce7a11,
      localPlayer: 0,
      botPlayers: [1],
    };
  } else {
    setup = await showLobby(uiRoot);
    const override = Number(params.get('seed') ?? 0);
    if (override) setup.seed = override;
  }

  const game = new Game(canvas, uiRoot, setup.seed, setup.transport, setup.botPlayers);
  game.start();

  // Exposed for the end-to-end tests and for debugging a live match.
  (window as unknown as { __game: Game }).__game = game;
}

void boot();

export { Game };
