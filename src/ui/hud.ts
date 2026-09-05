/**
 * HUD: resources, selection, command card, minimap, banners.
 *
 * Plain DOM rather than canvas or a framework. Text layout, hit testing, focus
 * and accessibility all come free, and the cost is negligible because the HUD
 * touches only a handful of values per frame — cached element references and
 * `textContent` writes, never layout reads inside the loop.
 *
 * The minimap is the deliberate exception: it draws hundreds of dots, which is
 * exactly what a canvas is good at and exactly what DOM is bad at.
 */

import { defOf } from '../config/rules.js';
import { toFloat } from '../sim/fixed.js';
import { BuildState, EntityType, NEUTRAL, TICKS_PER_SECOND, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { colourSlotFor, PLAYER_COLOURS, RESOURCE_COLOUR } from '../render/models/procedural.js';
import {
  fullscreenSupported,
  isFullscreen,
  onFullscreenChange,
  toggleFullscreen,
} from './fullscreen.js';
import { audio } from '../audio/audio.js';
import { activityOf } from './status.js';

export interface CommandButton {
  key: string;
  label: string;
  cost?: number;
  enabled: boolean;
  active?: boolean;
  onClick: () => void;
}

const MINIMAP_PX = 168;

export class Hud {
  private readonly mineralValue: HTMLElement;
  private readonly supplyValue: HTMLElement;
  private readonly selectionTitle: HTMLElement;
  private readonly selectionDetail: HTMLElement;
  private readonly commandGrid: HTMLElement;
  private readonly production: HTMLElement;
  private readonly prodLabel: HTMLElement;
  private readonly prodEta: HTMLElement;
  private readonly prodFill: HTMLElement;
  private readonly prodQueue: HTMLElement;
  private readonly banner: HTMLElement;
  private readonly overlay: HTMLElement;
  private readonly minimap: HTMLCanvasElement;
  private readonly minimapCtx: CanvasRenderingContext2D;
  readonly marquee: HTMLElement;

  private readonly fullscreenBtn: HTMLButtonElement;
  private readonly muteBtn: HTMLButtonElement;
  private readonly surrenderBtn: HTMLButtonElement;

  /** True while the pointer is over a HUD panel, to suppress world clicks. */
  pointerOverUi = false;

  private lastButtonSignature = '';
  private minimapFrame = 0;

  /** One row per partner in the ally strip, in slot order. */
  private readonly allyRows: { player: PlayerId; minerals: HTMLElement; supply: HTMLElement }[] =
    [];

  constructor(
    root: HTMLElement,
    private readonly mapSize: number,
    private readonly localPlayer: PlayerId,
    private readonly onMinimapClick: (x: number, z: number, secondary: boolean) => void,
    /** Slots on the local player's side other than their own, ascending. */
    allies: readonly PlayerId[] = [],
    /**
     * How many slots the match has. The palette is laid out by side, not by
     * slot, so a colour cannot be looked up without it — see `colourSlotFor`.
     */
    private readonly playerCount: number = 2,
    /**
     * Called when the player confirms they are giving up.
     *
     * A callback rather than a command built here: the HUD does not know the
     * lockstep runner, and conceding is an ordinary command that executes on
     * its turn like any other — every peer applies it in the same order, so
     * there is nothing special about it but the button.
     */
    private readonly onSurrender: () => void = () => {},
  ) {
    const colour = (p: PlayerId): number =>
      PLAYER_COLOURS[colourSlotFor(p, playerCount)] ?? 0x888888;
    root.innerHTML = `
      <div class="panel" id="resources">
        <div class="stat">
          <span class="stat-dot" style="background:${hex(RESOURCE_COLOUR)}"></span>
          <span class="stat-value" id="mineral-value">0</span>
          <span class="stat-label">Minerals</span>
        </div>
        <div class="stat">
          <span class="stat-dot" style="background:${hex(colour(localPlayer))}"></span>
          <span class="stat-value" id="supply-value">0/0</span>
          <span class="stat-label">Supply</span>
        </div>
      </div>

      <div class="panel" id="allies"${allies.length === 0 ? ' hidden' : ''}>
        ${allies
          .map(
            (p, k) => `
          <div class="ally" data-ally="${p}">
            <span class="stat-dot" style="background:${hex(colour(p))}"></span>
            <span class="ally-name">${allies.length > 1 ? `Ally ${k + 1}` : 'Ally'}</span>
            <span class="ally-minerals">0</span>
            <span class="ally-supply">0/0</span>
          </div>`,
          )
          .join('')}
      </div>

      <div class="panel" id="minimap-panel">
        <canvas id="minimap" width="${MINIMAP_PX}" height="${MINIMAP_PX}"></canvas>
      </div>

      <div class="panel" id="command-panel">
        <div id="selection-title">Nothing selected</div>
        <div id="selection-detail"></div>
        <div id="production" hidden>
          <div id="prod-row">
            <span id="prod-label"></span>
            <span id="prod-eta"></span>
          </div>
          <div id="prod-track"><div id="prod-fill"></div></div>
          <div id="prod-queue"></div>
        </div>
        <div id="command-grid"></div>
      </div>

      <button class="panel" id="surrender-btn" type="button"
              title="Surrender" aria-label="Surrender">🏳</button>
      <button class="panel" id="mute-btn" type="button"
              title="Mute (M)" aria-label="Toggle sound"></button>
      <button class="panel" id="fullscreen-btn" type="button"
              title="Fullscreen (F)" aria-label="Toggle fullscreen"></button>

      <div class="panel" id="banner"></div>
      <div id="marquee"></div>
      <!--
        Only the things nothing else on screen tells you. Attack and Stop are
        printed on their own command-card buttons, and fullscreen and mute are
        buttons in the corner — repeating them here was four items of noise in
        the one strip a player has no reason to read twice.
      -->
      <div class="hint">Arrows / edge pan &nbsp;·&nbsp; wheel zoom &nbsp;·&nbsp; drag select &nbsp;·&nbsp; right-click order &nbsp;·&nbsp; Ctrl+1-9 groups</div>
      <div id="overlay" class="hidden"></div>
    `;

    this.mineralValue = must(root, '#mineral-value');
    this.supplyValue = must(root, '#supply-value');
    this.selectionTitle = must(root, '#selection-title');
    this.selectionDetail = must(root, '#selection-detail');
    this.commandGrid = must(root, '#command-grid');
    this.production = must(root, '#production');
    this.prodLabel = must(root, '#prod-label');
    this.prodEta = must(root, '#prod-eta');
    this.prodFill = must(root, '#prod-fill');
    this.prodQueue = must(root, '#prod-queue');
    this.banner = must(root, '#banner');
    this.overlay = must(root, '#overlay');
    this.marquee = must(root, '#marquee');
    this.minimap = must(root, '#minimap') as HTMLCanvasElement;
    this.minimapCtx = this.minimap.getContext('2d')!;

    for (const p of allies) {
      const row = must(root, `.ally[data-ally="${p}"]`);
      this.allyRows.push({
        player: p,
        minerals: row.querySelector('.ally-minerals') as HTMLElement,
        supply: row.querySelector('.ally-supply') as HTMLElement,
      });
    }

    this.muteBtn = must(root, '#mute-btn') as HTMLButtonElement;
    this.muteBtn.addEventListener('click', () => this.toggleMute());
    this.syncMuteLabel();

    // Confirmed, and deliberately given no hotkey. It is the one irreversible
    // thing on screen, and a stray keypress during a fight should not be able
    // to end a match.
    this.surrenderBtn = must(root, '#surrender-btn') as HTMLButtonElement;
    this.surrenderBtn.addEventListener('click', () => {
      this.showDialog(
        'Surrender?',
        'Everything you still own is lost and you are out of the match. ' +
          'This cannot be undone.',
        // The safe option is the primary one. Making the destructive button the
        // loud one is how a confirmation turns into a rubber stamp.
        [
          { label: 'Keep playing', primary: true, onClick: () => this.hideDialog() },
          {
            label: 'Surrender',
            onClick: () => {
              this.hideDialog();
              this.onSurrender();
            },
          },
        ],
      );
    });

    this.fullscreenBtn = must(root, '#fullscreen-btn') as HTMLButtonElement;
    if (fullscreenSupported()) {
      this.fullscreenBtn.addEventListener('click', () => void this.toggleFullscreen());
      // The browser can leave fullscreen without going through our button — Esc,
      // or the user switching apps — so track the real state rather than ours.
      onFullscreenChange(() => this.syncFullscreenLabel());
      this.syncFullscreenLabel();
    } else {
      this.fullscreenBtn.style.display = 'none';
    }

    // Panels swallow pointer events so a click on the command card never also
    // issues a world order behind it.
    for (const sel of [
      '#resources',
      '#allies',
      '#minimap-panel',
      '#command-panel',
      '#fullscreen-btn',
      '#mute-btn',
      '#surrender-btn',
    ]) {
      const panel = must(root, sel);
      panel.classList.add('interactive');
      panel.addEventListener('pointerenter', () => {
        this.pointerOverUi = true;
      });
      panel.addEventListener('pointerleave', () => {
        this.pointerOverUi = false;
      });
    }

    const handleMinimap = (e: PointerEvent): void => {
      const rect = this.minimap.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * this.mapSize;
      const z = ((e.clientY - rect.top) / rect.height) * this.mapSize;
      this.onMinimapClick(x, z, e.button === 2);
    };
    this.minimap.addEventListener('pointerdown', handleMinimap);
    this.minimap.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  /** Toggle all sound. */
  toggleMute(): void {
    audio.toggleMuted();
    this.syncMuteLabel();
  }

  private syncMuteLabel(): void {
    const muted = audio.muted;
    this.muteBtn.textContent = muted ? '🔇' : '🔊';
    this.muteBtn.title = muted ? 'Unmute (M)' : 'Mute (M)';
  }

  /** Enter or leave fullscreen. Safe to call from a click or a keypress. */
  async toggleFullscreen(): Promise<void> {
    await toggleFullscreen();
    this.syncFullscreenLabel();
  }

  private syncFullscreenLabel(): void {
    const active = isFullscreen();
    // Arrows pointing inward mean "shrink", outward mean "grow".
    this.fullscreenBtn.textContent = active ? '⤡' : '⛶';
    this.fullscreenBtn.title = active ? 'Exit fullscreen (F)' : 'Fullscreen (F)';
  }

  updateResources(world: World): void {
    const ps = world.player(this.localPlayer);
    this.mineralValue.textContent = String(ps.minerals);
    this.supplyValue.textContent = `${ps.supplyUsed}/${ps.supplyMax}`;
    // Being supply blocked is the most common reason a player's production
    // silently stops, so it gets a colour rather than needing to be noticed.
    this.supplyValue.classList.toggle(
      'supply-capped',
      (ps.supplyUsed >= ps.supplyMax && ps.supplyMax > 0) ||
        anySupplyBlocked(world, this.localPlayer),
    );

    // A partner's bank and supply, because in co-op the useful question is
    // often "can they afford this if I cannot" — and because a partner who has
    // stopped growing has usually stopped playing.
    for (const row of this.allyRows) {
      const ally = world.player(row.player);
      if (!ally) continue;
      row.minerals.textContent = String(ally.minerals);
      row.supply.textContent = ally.defeated ? 'out' : `${ally.supplyUsed}/${ally.supplyMax}`;
    }
  }

  updateSelection(world: World, selected: ReadonlySet<number>): void {
    if (selected.size === 0) {
      this.selectionTitle.textContent = 'Nothing selected';
      this.selectionDetail.textContent = '';
      return;
    }

    const counts = new Map<EntityType, number>();
    let totalHp = 0;
    let maxHp = 0;
    let idle = 0;
    for (const i of selected) {
      if (world.pool.alive[i] !== 1) continue;
      const type = world.pool.type[i]! as EntityType;
      counts.set(type, (counts.get(type) ?? 0) + 1);
      totalHp += world.pool.hp[i]!;
      maxHp += defOf(type).maxHp;
      if (activityOf(world, i) === 'idle') idle++;
    }

    if (selected.size === 1) {
      const i = [...selected][0]!;
      const type = world.pool.type[i]! as EntityType;
      const def = defOf(type);
      this.selectionTitle.textContent = def.name;
      const parts = [`${world.pool.hp[i]} / ${def.maxHp} HP`];
      // What it hits for. Worth showing because it is now the true figure —
      // nothing scales it per matchup — so comparing two units on the panel
      // tells a player what actually happens when they meet.
      if (def.damage > 0) parts.push(`ATK ${def.damage}`);
      if (type === EntityType.MineralPatch) {
        parts.push(`${world.pool.resourceAmount[i]} minerals left`);
      }
      // What it is doing, which for a worker is almost never visible from the
      // model alone — walking to a site, building, mining and repairing all
      // look like standing about.
      const activity = activityOf(world, i);
      if (activity) parts.push(activity);
      if (def.isBuilding && world.pool.buildState[i] !== BuildState.Complete) {
        const pct = Math.floor((world.pool.buildProgress[i]! / def.buildTicks) * 100);
        parts.push(`under construction ${pct}%`);
      }
      if (world.pool.prodCount[i]! > 0) {
        parts.push(`training ${world.pool.prodCount[i]} queued`);
      }
      this.selectionDetail.textContent = parts.join(' · ');
    } else {
      const summary = [...counts.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([type, n]) => `${n} ${defOf(type).name}`)
        .join(', ');
      this.selectionTitle.textContent = `${selected.size} selected`;
      // At squad scale the useful question is not what each one is doing but
      // whether any of them is doing nothing, so only the idle count survives.
      const detail = [summary, `${totalHp}/${maxHp} HP`];
      if (idle > 0) detail.push(`${idle} idle`);
      this.selectionDetail.textContent = detail.join(' · ');
    }
  }

  /**
   * Show what a selected building is training, and how far along it is.
   *
   * A production queue with no visible progress is the single most common thing
   * players ask about in an RTS — "is it building?" — so the bar reports the
   * unit by name, the fraction complete, and the remaining time.
   */
  updateProduction(world: World, selected: ReadonlySet<number>): void {
    const single = selected.size === 1 ? [...selected][0]! : -1;
    const pool = world.pool;

    if (
      single < 0 ||
      pool.alive[single] !== 1 ||
      pool.owner[single] !== this.localPlayer ||
      pool.prodCount[single]! === 0
    ) {
      this.production.hidden = true;
      return;
    }

    this.production.hidden = false;

    const current = pool.prodAt(single, 0);
    const def = defOf(current);
    const progress = Math.min(1, pool.prodProgress[single]! / Math.max(1, def.buildTicks));
    const remainingTicks = Math.max(0, def.buildTicks - pool.prodProgress[single]!);

    // A finished unit waits in the building until there is room for it. Without
    // saying so the panel just sits at 100% forever, which reads as the game
    // being broken — and it is easy to hit without the supply counter looking
    // full, because what matters is whether *this* unit fits, not whether
    // there is any headroom at all. One free supply trains a Burstbot and
    // stalls a Beamdrone.
    const blocked = supplyBlocked(world, single);
    this.production.classList.toggle('blocked', blocked);
    if (blocked) {
      this.prodLabel.textContent = `${def.name} needs ${def.supplyCost} supply`;
      this.prodEta.textContent = 'build a depot';
    } else {
      this.prodLabel.textContent = `Training ${def.name}`;
      this.prodEta.textContent = `${(remainingTicks / TICKS_PER_SECOND).toFixed(1)}s`;
    }
    this.prodFill.style.width = `${(progress * 100).toFixed(1)}%`;

    // The rest of the queue, so a player can see what they have committed to.
    const queued = pool.prodCount[single]! - 1;
    if (queued > 0) {
      const names: string[] = [];
      for (let slot = 1; slot < pool.prodCount[single]!; slot++) {
        names.push(defOf(pool.prodAt(single, slot)).name);
      }
      this.prodQueue.textContent = `Queued: ${names.join(', ')}`;
      this.prodQueue.hidden = false;
    } else {
      this.prodQueue.hidden = true;
    }
  }

  /**
   * Rebuild the command card.
   *
   * Only rebuilt when the buttons actually change — recreating DOM every frame
   * would drop clicks, because a button replaced between pointerdown and
   * pointerup never fires.
   */
  setCommands(buttons: CommandButton[]): void {
    const signature = buttons
      .map((b) => `${b.key}:${b.label}:${b.enabled ? 1 : 0}:${b.active ? 1 : 0}`)
      .join('|');
    if (signature === this.lastButtonSignature) return;
    this.lastButtonSignature = signature;

    this.commandGrid.innerHTML = '';
    for (const button of buttons) {
      const el = document.createElement('button');
      el.className = `cmd${button.active ? ' active' : ''}`;
      el.disabled = !button.enabled;
      el.innerHTML =
        `<span class="cmd-key">${button.key}</span>` +
        `<span>${button.label}</span>` +
        (button.cost !== undefined ? `<span class="cmd-cost">${button.cost}</span>` : '');
      el.addEventListener('click', button.onClick);
      this.commandGrid.append(el);
    }
  }

  showBanner(text: string, tone: 'info' | 'warn' | 'danger' = 'info'): void {
    this.banner.textContent = text;
    this.banner.className = `panel ${tone === 'info' ? '' : tone}`;
    this.banner.style.display = 'block';
  }

  hideBanner(): void {
    this.banner.style.display = 'none';
  }

  showDialog(
    title: string,
    body: string,
    actions: { label: string; primary?: boolean; onClick: () => void }[],
  ): void {
    this.overlay.classList.remove('hidden');
    this.overlay.innerHTML = `<div class="dialog"><h1></h1><p></p></div>`;
    const dialog = this.overlay.querySelector('.dialog')!;
    dialog.querySelector('h1')!.textContent = title;
    dialog.querySelector('p')!.textContent = body;
    for (const action of actions) {
      const button = document.createElement('button');
      button.textContent = action.label;
      if (action.primary) button.className = 'primary';
      button.addEventListener('click', action.onClick);
      dialog.append(button);
    }
  }

  hideDialog(): void {
    this.overlay.classList.add('hidden');
    this.overlay.innerHTML = '';
  }

  /** Hide the surrender button — there is nothing left to give up. */
  setSurrenderAvailable(available: boolean): void {
    this.surrenderBtn.hidden = !available;
  }

  /**
   * Redraw the minimap.
   *
   * Throttled to every few frames: it is a full canvas repaint and nothing on it
   * changes fast enough at 60Hz to be worth the cost.
   */
  drawMinimap(
    world: World,
    focusX: number,
    focusZ: number,
    viewRadius: number,
    fog?: {
      isExploredAt(tx: number, tz: number): boolean;
      isVisibleAt(x: number, z: number): boolean;
    },
  ): void {
    if (this.minimapFrame++ % 4 !== 0) return;

    const ctx = this.minimapCtx;
    const scale = MINIMAP_PX / this.mapSize;

    ctx.fillStyle = '#151b24';
    ctx.fillRect(0, 0, MINIMAP_PX, MINIMAP_PX);

    // Cliffs, sampled rather than drawn per tile — at this scale several tiles
    // share a pixel anyway.
    const step = 2;
    for (let y = 0; y < world.map.height; y += step) {
      for (let x = 0; x < world.map.width; x += step) {
        const explored = !fog || fog.isExploredAt(x, y);
        if (!explored) continue;
        if (world.map.tiles[world.map.index(x, y)] === 1) {
          // Cliff darker than ground, so the lanes read as the bright channels
          // they are. Drawn the other way round the eye follows the rock.
          ctx.fillStyle = '#232b38';
        } else {
          // Explored ground is drawn faintly so the shape of the map is
          // recoverable from memory without revealing what is on it.
          ctx.fillStyle = fog && !fog.isVisibleAt(x + 0.5, y + 0.5) ? '#3a4759' : '#4d6079';
        }
        ctx.fillRect(x * scale, y * scale, step * scale, step * scale);
      }
    }

    const pool = world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      const owner = pool.owner[i]!;

      const px0 = toFloat(pool.posX[i]!);
      const pz0 = toFloat(pool.posY[i]!);
      // The minimap obeys the same fog rules as the world view; showing enemy
      // positions here would defeat the entire point of having fog. A partner's
      // army is not an enemy — seeing where they are is the point of the strip
      // above, and of the map.
      if (fog && !world.areAllied(owner, this.localPlayer)) {
        const known =
          owner === NEUTRAL
            ? fog.isExploredAt(Math.floor(px0), Math.floor(pz0))
            : fog.isVisibleAt(px0, pz0);
        if (!known) continue;
      }

      ctx.fillStyle =
        owner === NEUTRAL
          ? hex(RESOURCE_COLOUR)
          : hex(PLAYER_COLOURS[colourSlotFor(owner, this.playerCount)] ?? 0x999999);

      const px = px0 * scale;
      const pz = pz0 * scale;
      // Buildings as squares, units as dots — shape carries information that
      // colour alone cannot at three pixels.
      if (def.isBuilding) {
        const s = Math.max(3, def.footprint * scale);
        ctx.fillRect(px - s / 2, pz - s / 2, s, s);
      } else {
        ctx.fillRect(px - 1.5, pz - 1.5, 3, 3);
      }
    }

    // Camera viewport indicator.
    ctx.strokeStyle = 'rgba(233, 240, 250, 0.85)';
    ctx.lineWidth = 1;
    ctx.strokeRect(
      (focusX - viewRadius) * scale,
      (focusZ - viewRadius) * scale,
      viewRadius * 2 * scale,
      viewRadius * 2 * scale,
    );
  }
}

function must(root: HTMLElement, selector: string): HTMLElement {
  const el = root.querySelector(selector);
  if (!el) throw new Error(`HUD element missing: ${selector}`);
  return el as HTMLElement;
}

function hex(colour: number): string {
  return `#${colour.toString(16).padStart(6, '0')}`;
}

/**
 * Is this building holding a unit it has finished but cannot release?
 *
 * The check the simulation makes, mirrored for display. Deliberately mirrored
 * rather than exported from the simulation: it is derived from state that is
 * already checksummed, and a second field for the renderer to read would be one
 * more thing that has to stay in step.
 */
function supplyBlocked(world: World, index: number): boolean {
  const pool = world.pool;
  if (pool.prodCount[index]! === 0) return false;
  const def = defOf(pool.prodAt(index, 0));
  if (pool.prodProgress[index]! < def.buildTicks) return false;
  const ps = world.player(pool.owner[index]! as PlayerId);
  return ps.supplyUsed + def.supplyCost > ps.supplyMax;
}

/** Does this player have anything finished and waiting on supply? */
function anySupplyBlocked(world: World, player: PlayerId): boolean {
  const pool = world.pool;
  for (let i = 0; i < pool.count; i++) {
    if (pool.alive[i] !== 1 || pool.owner[i] !== player) continue;
    if (supplyBlocked(world, i)) return true;
  }
  return false;
}
