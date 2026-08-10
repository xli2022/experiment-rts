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
import { fromFloat, toFloat } from './sim/fixed.js';
import { MAP_SIZE } from './sim/map.js';
import { Simulation } from './sim/tick.js';
import { BuildState, EntityType, NEUTRAL, NO_ENTITY, type PlayerId } from './sim/types.js';
import { EntityRenderer } from './render/entities.js';
import { RtsCamera } from './render/camera.js';
import { TerrainRenderer } from './render/terrain.js';
import { ProceduralModelProvider } from './render/models/procedural.js';
import { ProjectileRenderer } from './render/projectiles.js';
import { FogRenderer } from './render/fog.js';
import { UnitGallery } from './render/unitGallery.js';
import { audio } from './audio/audio.js';
import { groundPointAt, pickAt, pickInBox, Selection } from './input/selection.js';
import { Hud, type CommandButton } from './ui/hud.js';
import { loadUnitModels } from './render/models/unitModels.js';
import { showLobby, type MatchSetup } from './ui/lobby.js';
import { onFullscreenChange } from './ui/fullscreen.js';

/** Buildings the player can place, in command-card order. */
const BUILD_MENU: { type: EntityType; key: string }[] = [
  { type: EntityType.Depot, key: 'D' },
  { type: EntityType.Barracks, key: 'B' },
  { type: EntityType.Turret, key: 'T' },
  // A Command Post can go anywhere, like any other structure — what makes the
  // expansion sites worth walking to is the mineral line already sitting there.
  { type: EntityType.CommandPost, key: 'C' },
];

/** Explicit production bindings; authored names do not necessarily have unique initials. */
const TRAIN_HOTKEYS: Readonly<Partial<Record<EntityType, string>>> = {
  [EntityType.Worker]: 'W',
  [EntityType.Burstbot]: 'B',
  [EntityType.Slicebot]: 'S',
  [EntityType.Beamdrone]: 'D',
};

function trainHotkey(type: EntityType): string {
  const key = TRAIN_HOTKEYS[type];
  if (key !== undefined) return key;
  throw new Error(`No production hotkey for entity type ${type}`);
}

class Game {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: RtsCamera;
  private readonly provider = new ProceduralModelProvider();
  private readonly entities: EntityRenderer;
  private readonly terrain: TerrainRenderer;
  private readonly projectiles = new ProjectileRenderer();
  private readonly fog: FogRenderer;
  private readonly selection: Selection;
  private readonly hud: Hud;
  private readonly gallery: UnitGallery;
  private readonly runner: LockstepRunner;
  private readonly sim: Simulation;
  private readonly localPlayer: PlayerId;

  /** Building type awaiting placement, or null. */
  private placing: EntityType | null = null;
  private readonly ghost: THREE.Mesh;
  /** True when the next left click issues an attack-move. */
  private attackMovePending = false;

  /**
   * The command card as last built, so its hotkeys can actually be pressed.
   *
   * The card has always drawn a letter in the corner of each button; nothing was
   * bound to those letters, so every one of them was a lie. Keeping the buttons
   * here lets one lookup drive both the click and the key.
   */
  private commandButtons: CommandButton[] = [];

  private dragStart: { x: number; y: number } | null = null;
  private pointerNdc = new THREE.Vector2();
  private lastFrameMs = 0;
  private finished = false;
  /** Wall-clock seconds since start, for purely cosmetic animation. */
  private elapsedS = 0;

  constructor(
    canvas: HTMLCanvasElement,
    uiRoot: HTMLElement,
    renderer: THREE.WebGLRenderer,
    gallery: UnitGallery,
    seed: number,
    transport: Transport,
    botPlayers: PlayerId[],
  ) {
    this.localPlayer = transport.localPlayer;
    this.sim = new Simulation(seed);
    this.renderer = renderer;
    this.gallery = gallery;

    // Slots played by the AI. Because the bot is deterministic it runs inside
    // the simulation on every peer, so every peer must agree on this set — it
    // comes from the lobby, which both sides agreed on before starting.
    for (const p of botPlayers) this.sim.botPlayers.add(p);

    this.scene.fog = new THREE.Fog(0x0b0e14, 90, 190);

    this.camera = new RtsCamera(canvas, MAP_SIZE);
    this.terrain = new TerrainRenderer(this.sim.world.map);
    this.entities = new EntityRenderer(this.provider, this.sim.world);
    this.fog = new FogRenderer(this.sim.world.map);
    this.fog.update(this.sim.world, this.localPlayer);
    this.scene.add(
      this.terrain.group,
      this.entities.group,
      this.projectiles.group,
      this.fog.mesh,
    );
    this.addLights();

    this.ghost = this.makeGhost();
    this.scene.add(this.ghost);

    // Authored models arrive asynchronously and swap themselves in. The match is
    // fully playable on the procedural stand-ins until they do, so a slow asset
    // delays nothing.
    void loadUnitModels(this.renderer).then((models) => {
      for (const { type, model, scale, textures } of models) {
        this.entities.useAnimatedModel(type, model, scale, textures);
      }
    });

    this.selection = new Selection(this.localPlayer);
    this.hud = new Hud(
      uiRoot,
      MAP_SIZE,
      this.localPlayer,
      (x, z, secondary) => {
        if (secondary) this.issueGroundOrder(x, z, false);
        else this.camera.lookAt(x, z);
      },
    );

    this.runner = new LockstepRunner(this.sim, transport, {
      onStep: () => this.consumeSimulationStep(),
      onStall: (waiting) =>
        this.hud.showBanner(`Waiting for player ${waiting.join(', ')}…`, 'warn'),
      onResume: () => this.hud.hideBanner(),
      onDesync: (tick) => {
        this.gallery.close();
        this.hud.showDialog(
          'Desynchronised',
          `The two games diverged at tick ${tick}. In peer-to-peer play there is no ` +
            `authority to correct this, so the match cannot continue.`,
          [{ label: 'Reload', primary: true, onClick: () => location.reload() }],
        );
      },
    });

    // Open on the player's own base, which is where they will look first.
    const start = this.sim.world.map.starts[this.localPlayer]!;
    this.camera.lookAt(start.tileX, start.tileY);

    this.attachInput(canvas);
    this.resize();
    window.addEventListener('resize', () => this.resize());
    // Entering or leaving fullscreen changes the viewport. Most browsers also
    // fire `resize`, but not all do so reliably, and a stale projection matrix
    // leaves the game letterboxed or stretched.
    onFullscreenChange(() => {
      // The window dimensions update a frame after the event.
      requestAnimationFrame(() => this.resize());
    });
  }

  private addLights(): void {
    const sun = new THREE.DirectionalLight(0xfff2e0, 2.0);
    const mapCentre = MAP_SIZE / 2;

    // Keep the original daylight direction, but aim it through the middle of
    // the battlefield. A directional light's target defaults to the origin,
    // which leaves most of a 128x128 map outside the default shadow camera.
    sun.position.set(mapCentre + 50, 90, mapCentre + 30);
    sun.target.position.set(mapCentre, 0, mapCentre);
    sun.castShadow = true;

    // One stable map covers the whole battlefield. Following the RTS camera
    // would buy a little more resolution, but it also makes shadows crawl across
    // the tile grid while panning. The half-diagonal needs roughly 91 units in
    // this light direction; 96 leaves room for cliffs, units, and map edges.
    const shadowExtent = MAP_SIZE * 0.75;
    const shadowCamera = sun.shadow.camera;
    shadowCamera.left = -shadowExtent;
    shadowCamera.right = shadowExtent;
    shadowCamera.top = shadowExtent;
    shadowCamera.bottom = -shadowExtent;
    shadowCamera.near = 30;
    shadowCamera.far = 200;
    shadowCamera.updateProjectionMatrix();

    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0002;
    sun.shadow.normalBias = 0.03;
    sun.shadow.radius = 2;

    // The target must be in the scene for three.js to update its world matrix.
    this.scene.add(sun, sun.target);
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
    if (this.gallery.isOpen) return;

    if (e.code === 'Escape') {
      this.cancelModes();
      return;
    }

    // Control groups: Ctrl+N assigns, N recalls.
    const digit = e.code.startsWith('Digit') ? Number(e.code.slice(5)) : -1;
    if (digit >= 0 && digit <= 9) {
      if (e.ctrlKey || e.metaKey) {
        this.selection.assignGroup(digit, this.sim.world);
        e.preventDefault();
      } else if (this.selection.recallGroup(digit, this.sim.world) === 'again') {
        // Second press of a key whose group is already selected jumps the view
        // to it, as StarCraft II does.
        const at = this.selection.centroid(this.sim.world);
        if (at) this.camera.lookAt(at.x, at.z);
      }
      return;
    }

    // Global toggles first, so they work regardless of what is selected.
    if (e.code === 'KeyM') {
      this.hud.toggleMute();
      return;
    }
    if (e.code === 'KeyF') {
      void this.hud.toggleFullscreen();
      return;
    }

    // Then whatever the command card currently offers. The card is contextual,
    // so the same key can mean a building with a worker selected and a unit
    // with a production building selected.
    if (!e.ctrlKey && !e.metaKey && !e.altKey && this.dispatchCommandKey(e.code)) {
      e.preventDefault();
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

  /**
   * Fire the command-card button bound to a key, if there is one.
   *
   * Disabled buttons deliberately still "consume" nothing but make a sound —
   * pressing B with no minerals should tell you why nothing happened rather than
   * silently doing nothing.
   */
  private dispatchCommandKey(code: string): boolean {
    if (!code.startsWith('Key')) return false;
    const letter = code.slice(3);

    for (const button of this.commandButtons) {
      if (button.key.toUpperCase() !== letter) continue;
      if (!button.enabled) {
        audio.play('denied', 0.7);
        return true;
      }
      button.onClick();
      audio.play('select', 0.6);
      return true;
    }
    return false;
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
    if (additive) this.selection.toggle(hit, this.sim.world);
    else this.selection.set([hit], this.sim.world);
    audio.play('select', 0.7);
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
    if (additive) this.selection.add(found, this.sim.world);
    else this.selection.set(found, this.sim.world);
  }

  /**
   * Right-click: infer the order from what is under the cursor.
   *
   * Enemy → attack, mineral patch → harvest, ground → move. This single verb is
   * how RTS players issue most of their orders, so getting the inference right
   * matters more than any explicit button.
   */
  private issueContextOrder(): void {
    const world = this.sim.world;

    // Rally points first, because the check below rejects the very case they
    // exist for. `hasOwnUnits` deliberately ignores buildings — they take no
    // movement orders — so a selection holding nothing but a Barracks bailed
    // out of this function before it could ever reach the rally code.
    const rallyPoint = groundPointAt(this.camera.camera, this.pointerNdc.x, this.pointerNdc.y);
    const rallied = rallyPoint !== null && this.setRallyPoints(rallyPoint.x, rallyPoint.z);

    if (!this.selection.hasOwnUnits(world)) return;
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
        this.projectiles.spawnClickMarker(
          toFloat(world.pool.posX[hit]!),
          toFloat(world.pool.posY[hit]!),
          0x54e0c8,
        );
        audio.play('order', 0.9);
        this.issue({
          type: CommandType.Harvest,
          player: this.localPlayer,
          units,
          target: world.pool.idAt(hit),
        });
        return;
      }
      // Own unfinished building: send every selected worker to finish it. Same
      // verb the player already uses for everything else.
      if (owner === this.localPlayer) {
        const def = defOf(type);
        const needsWork = def.isBuilding && world.pool.buildState[hit] !== BuildState.Complete;
        if (needsWork && this.orderWorkersTo(hit)) return;
      }

      if (owner !== this.localPlayer && owner !== NEUTRAL) {
        this.projectiles.spawnClickMarker(
          toFloat(world.pool.posX[hit]!),
          toFloat(world.pool.posY[hit]!),
          0xff5a4a,
        );
        audio.play('order', 0.9);
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
    if (point) this.issueGroundOrder(point.x, point.z, false, rallied);
  }

  /**
   * Send every selected worker to build or repair a structure.
   *
   * Returns false when nothing in the selection can do the job, so the caller
   * can fall through to its normal handling rather than swallowing the click.
   */
  private orderWorkersTo(buildingIndex: number): boolean {
    const world = this.sim.world;
    const workers: number[] = [];
    for (const i of this.selection.indices) {
      if (world.pool.alive[i] !== 1) continue;
      if (world.pool.owner[i] !== this.localPlayer) continue;
      if (world.pool.type[i] !== EntityType.Worker) continue;
      workers.push(i);
    }
    if (workers.length === 0) return false;

    for (const w of workers) {
      this.issue({
        type: CommandType.Build,
        player: this.localPlayer,
        worker: world.pool.idAt(w),
        building: world.pool.type[buildingIndex]! as EntityType,
        tileX: world.pool.tileX[buildingIndex]!,
        tileY: world.pool.tileY[buildingIndex]!,
      });
    }

    this.projectiles.spawnClickMarker(
      toFloat(world.pool.posX[buildingIndex]!),
      toFloat(world.pool.posY[buildingIndex]!),
      0x7dff9b,
    );
    this.entities.noteOrderIssued(this.elapsedS);
    audio.play('order', 0.9);
    return true;
  }

  /**
   * Right-clicking the ground with production buildings selected sets a rally.
   *
   * Returns false when the selection has no production building in it, so a
   * mixed selection still moves the units. Buildings cannot move, so there is
   * no ambiguity to resolve — the two orders never compete for the same click.
   */
  private setRallyPoints(x: number, z: number): boolean {
    const world = this.sim.world;
    let any = false;
    for (const i of this.selection.indices) {
      if (world.pool.alive[i] !== 1) continue;
      if (world.pool.owner[i] !== this.localPlayer) continue;
      if (world.pool.buildState[i] !== BuildState.Complete) continue;
      if (defOf(world.pool.type[i]! as EntityType).produces.length === 0) continue;
      this.issue({
        type: CommandType.SetRally,
        player: this.localPlayer,
        building: world.pool.idAt(i),
        x: fromFloat(x),
        y: fromFloat(z),
      });
      any = true;
    }
    if (any) {
      this.projectiles.spawnClickMarker(x, z, 0x8fd0ff);
      audio.play('order', 0.9);
    }
    return any;
  }

  private issueGroundOrder(x: number, z: number, attackMove: boolean, alreadyRallied = false): void {
    // A mixed selection does both: the buildings take the rally, the units take
    // the move. They never compete, since buildings cannot move anyway.
    if (!attackMove && !alreadyRallied && this.setRallyPoints(x, z)) return;
    const units = this.selection.ids(this.sim.world);
    if (units.length === 0) return;
    this.projectiles.spawnClickMarker(x, z, attackMove ? 0xff7a4a : 0x7dff9b);
    this.entities.noteOrderIssued(this.elapsedS);
    audio.play('order', 0.9);
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
      audio.play('denied', 0.8);
      this.hud.showBanner('Cannot build there', 'warn');
      window.setTimeout(() => this.hud.hideBanner(), 1400);
      return;
    }

    const worker = this.findSelectedWorker();
    if (worker === NO_ENTITY) {
      audio.play('denied', 0.8);
      this.hud.showBanner('Select a worker first', 'warn');
      window.setTimeout(() => this.hud.hideBanner(), 1400);
      return;
    }

    audio.play('build', 0.8);
    this.projectiles.spawnClickMarker(point.x, point.z, 0x7dff9b);
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
    this.elapsedS += dtMs / 1000;
    const alpha = this.runner.update(dtMs);
    // Shots are retained per simulation step, then resolved once the final
    // interpolation alpha is known. Flush before the single buffer rebuild so
    // effects emitted by this frame are visible immediately.
    this.projectiles.flushPending(this.sim.world.tick, alpha, this.elapsedS, dtMs);
    this.projectiles.update(dtMs);

    if (!this.gallery.isOpen) this.camera.update(dtMs / 1000);
    this.fog.refresh();
    // Cliffs have height and so stand through the fog plane; they are shaded to
    // match it instead of being covered by it.
    this.terrain.applyFog(this.fog);
    this.selection.prune(this.sim.world);

    this.entities.update(
      this.sim.world,
      alpha,
      this.selection.indices,
      this.camera.camera,
      this.elapsedS,
      this.fog,
      this.localPlayer,
    );
    this.hud.updateResources(this.sim.world);
    this.hud.updateSelection(this.sim.world, this.selection.indices);
    this.hud.updateProduction(this.sim.world, this.selection.indices);
    this.commandButtons = this.buildCommandCard();
    this.hud.setCommands(this.commandButtons);
    this.hud.drawMinimap(
      this.sim.world,
      this.camera.focusPoint.x,
      this.camera.focusPoint.z,
      this.camera.zoomHeight * 0.42,
      this.fog,
    );

    this.checkResult();
    if (this.gallery.isOpen) this.gallery.render(this.elapsedS);
    else this.renderer.render(this.scene, this.camera.camera);
  }

  /**
   * Drain one tick's transient presentation events before another tick can
   * clear them. Lockstep may execute several ticks during one render frame.
   */
  private consumeSimulationStep(): void {
    this.entities.captureSnapshot(this.sim.world);
    this.projectiles.captureFromEvents(this.sim.world, this.entities);
    this.projectiles.spawnDeaths(this.sim.world);
    this.entities.noteEvents(this.sim.world, this.elapsedS);
    this.playTickSounds();
    this.fog.update(this.sim.world, this.localPlayer);
  }

  /**
   * Turn this tick's events into sound.
   *
   * Volume falls off with distance from the camera so a battle across the map
   * does not drown out what is happening in front of the player, and only a few
   * of each kind play per tick — twenty simultaneous rifle shots is noise, not
   * information.
   */
  private playTickSounds(): void {
    const world = this.sim.world;
    const cx = this.camera.focusPoint.x;
    const cz = this.camera.focusPoint.z;

    const nearness = (x: number, z: number): number => {
      const d = Math.hypot(x - cx, z - cz);
      // Silent past roughly a screen's width away.
      return Math.max(0, 1 - d / 55);
    };

    let shotsPlayed = 0;
    const shots = world.events.shots;
    for (let k = 0; k + 1 < shots.length && shotsPlayed < 3; k += 2) {
      const i = shots[k]!;
      if (world.pool.alive[i] !== 1) continue;
      const near = nearness(toFloat(world.pool.posX[i]!), toFloat(world.pool.posY[i]!));
      if (near <= 0.05) continue;
      audio.play('shot', near * 0.8);
      shotsPlayed++;
    }

    let deathsPlayed = 0;
    for (const i of world.events.deaths) {
      if (deathsPlayed >= 2) break;
      if (world.pool.type[i] === EntityType.MineralPatch) continue;
      const near = nearness(toFloat(world.pool.posX[i]!), toFloat(world.pool.posY[i]!));
      if (near <= 0.05) continue;
      const big = defOf(world.pool.type[i]! as EntityType).isBuilding;
      audio.play(big ? 'explosion' : 'death', near);
      deathsPlayed++;
    }

    // A completed building is worth hearing wherever it is — it is something the
    // player deliberately started and is waiting on.
    for (const i of world.events.completed) {
      if (world.pool.owner[i] !== this.localPlayer) continue;
      audio.play('build', 0.9);
      break;
    }
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
          key: trainHotkey(unit),
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
      // Cancel the unit at the head of the queue. Without this a player whose
      // supply cannot fit the finished unit has no way out except building a
      // depot — the building simply stops producing, and nothing on screen
      // offers a way to change their mind.
      //
      // Last in the card, after the units it cancels: it appears and disappears
      // with the queue, and putting it first made every train button jump
      // sideways the moment production started.
      if (world.pool.prodCount[single]! > 0) {
        buttons.push({
          key: 'X',
          label: 'Cancel',
          enabled: true,
          onClick: () =>
            this.issue({
              type: CommandType.CancelTrain,
              player: this.localPlayer,
              building: world.pool.idAt(single),
              slot: 0,
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
    if (!this.sim.world.matchOver) return;

    this.finished = true;
    this.gallery.close();
    const winner = this.sim.world.winner;
    const replay = [{ label: 'Play again', primary: true, onClick: () => location.reload() }];

    // Both sides can be eliminated on the same tick, which is a draw rather
    // than a win for nobody.
    if (winner === NO_ENTITY) {
      this.hud.showDialog(
        'Draw',
        'Both sides were wiped out with nothing left to rebuild from.',
        replay,
      );
      return;
    }

    const won = winner === this.localPlayer;
    this.hud.showDialog(
      won ? 'Victory' : 'Defeat',
      won
        ? 'Every enemy structure has been destroyed.'
        : 'All of your structures were destroyed.',
      replay,
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

/**
 * Keep the shared renderer alive while the pre-match home screen owns it.
 * The match takes over the same renderer and gallery after a mode is chosen,
 * avoiding a second WebGL context or a second asset-loading path.
 */
async function showHome(
  root: HTMLElement,
  renderer: THREE.WebGLRenderer,
  gallery: UnitGallery,
): Promise<MatchSetup> {
  let frameId = 0;
  let galleryWasOpen = false;
  const resize = (): void => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  };
  const frame = (nowMs: number): void => {
    if (gallery.isOpen) {
      gallery.render(nowMs / 1000);
      galleryWasOpen = true;
    } else if (galleryWasOpen) {
      // The home page has no scene of its own to repaint the gallery's last
      // frame, so clear it once when the modal closes.
      renderer.clear(true, true, true);
      galleryWasOpen = false;
    }
    frameId = requestAnimationFrame(frame);
  };

  resize();
  renderer.clear(true, true, true);
  window.addEventListener('resize', resize);
  frameId = requestAnimationFrame(frame);

  try {
    return await showLobby(root, () => gallery.open());
  } finally {
    cancelAnimationFrame(frameId);
    window.removeEventListener('resize', resize);
    gallery.close();
    renderer.clear(true, true, true);
  }
}

async function boot(): Promise<void> {
  const canvas = document.getElementById('viewport') as HTMLCanvasElement | null;
  const uiRoot = document.getElementById('ui-root');
  if (!canvas || !uiRoot) throw new Error('missing #viewport or #ui-root');

  // The home gallery and the match deliberately share one WebGL context.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0b0e14);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const gallery = new UnitGallery(uiRoot, renderer);

  const params = new URLSearchParams(location.search);

  // `?skip=ai` boots straight into a skirmish, which is what the end-to-end
  // tests and screenshot tooling use. A seed in the URL makes any match
  // reproducible, which is how a desync report becomes debuggable.
  let setup: MatchSetup;
  if (params.get('skip') === 'ai') {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    setup = {
      transport: new SoloTransport(),
      seed: Number(params.get('seed') ?? 0) || 0x51ce7a11,
      localPlayer: 0,
      botPlayers: [1],
    };
  } else {
    setup = await showHome(uiRoot, renderer, gallery);
    const override = Number(params.get('seed') ?? 0);
    if (override) setup.seed = override;
  }

  const game = new Game(
    canvas,
    uiRoot,
    renderer,
    gallery,
    setup.seed,
    setup.transport,
    setup.botPlayers,
  );
  game.start();

  // Exposed for the end-to-end tests and for debugging a live match.
  (window as unknown as { __game: Game }).__game = game;
}

void boot();

export { Game };
