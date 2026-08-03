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
import { BuildState, EntityType, NEUTRAL, type PlayerId } from '../sim/types.js';
import type { World } from '../sim/world.js';
import { PLAYER_COLOURS, RESOURCE_COLOUR } from '../render/models/procedural.js';

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
  private readonly banner: HTMLElement;
  private readonly overlay: HTMLElement;
  private readonly minimap: HTMLCanvasElement;
  private readonly minimapCtx: CanvasRenderingContext2D;
  readonly marquee: HTMLElement;

  /** True while the pointer is over a HUD panel, to suppress world clicks. */
  pointerOverUi = false;

  private lastButtonSignature = '';
  private minimapFrame = 0;

  constructor(
    root: HTMLElement,
    private readonly mapSize: number,
    private readonly localPlayer: PlayerId,
    private readonly onMinimapClick: (x: number, z: number, secondary: boolean) => void,
  ) {
    root.innerHTML = `
      <div class="panel" id="resources">
        <div class="stat">
          <span class="stat-dot" style="background:${hex(RESOURCE_COLOUR)}"></span>
          <span class="stat-value" id="mineral-value">0</span>
          <span class="stat-label">Minerals</span>
        </div>
        <div class="stat">
          <span class="stat-dot" style="background:${hex(PLAYER_COLOURS[localPlayer] ?? 0x888888)}"></span>
          <span class="stat-value" id="supply-value">0/0</span>
          <span class="stat-label">Supply</span>
        </div>
      </div>

      <div class="panel" id="minimap-panel">
        <canvas id="minimap" width="${MINIMAP_PX}" height="${MINIMAP_PX}"></canvas>
      </div>

      <div class="panel" id="command-panel">
        <div id="selection-title">Nothing selected</div>
        <div id="selection-detail"></div>
        <div id="command-grid"></div>
      </div>

      <div class="panel" id="banner"></div>
      <div id="marquee"></div>
      <div class="hint">Arrows / edge pan &nbsp;·&nbsp; wheel zoom &nbsp;·&nbsp; drag select &nbsp;·&nbsp; right-click order &nbsp;·&nbsp; A attack &nbsp;·&nbsp; S stop &nbsp;·&nbsp; Ctrl+1-9 groups</div>
      <div id="overlay" class="hidden"></div>
    `;

    this.mineralValue = must(root, '#mineral-value');
    this.supplyValue = must(root, '#supply-value');
    this.selectionTitle = must(root, '#selection-title');
    this.selectionDetail = must(root, '#selection-detail');
    this.commandGrid = must(root, '#command-grid');
    this.banner = must(root, '#banner');
    this.overlay = must(root, '#overlay');
    this.marquee = must(root, '#marquee');
    this.minimap = must(root, '#minimap') as HTMLCanvasElement;
    this.minimapCtx = this.minimap.getContext('2d')!;

    // Panels swallow pointer events so a click on the command card never also
    // issues a world order behind it.
    for (const sel of ['#resources', '#minimap-panel', '#command-panel']) {
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

  updateResources(world: World): void {
    const ps = world.player(this.localPlayer);
    this.mineralValue.textContent = String(ps.minerals);
    this.supplyValue.textContent = `${ps.supplyUsed}/${ps.supplyMax}`;
    // Being supply blocked is the most common reason a player's production
    // silently stops, so it gets a colour rather than needing to be noticed.
    this.supplyValue.classList.toggle(
      'supply-capped',
      ps.supplyUsed >= ps.supplyMax && ps.supplyMax > 0,
    );
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
    for (const i of selected) {
      if (world.pool.alive[i] !== 1) continue;
      const type = world.pool.type[i]! as EntityType;
      counts.set(type, (counts.get(type) ?? 0) + 1);
      totalHp += world.pool.hp[i]!;
      maxHp += defOf(type).maxHp;
    }

    if (selected.size === 1) {
      const i = [...selected][0]!;
      const type = world.pool.type[i]! as EntityType;
      const def = defOf(type);
      this.selectionTitle.textContent = def.name;
      const parts = [`${world.pool.hp[i]} / ${def.maxHp} HP`];
      if (type === EntityType.MineralPatch) {
        parts.push(`${world.pool.resourceAmount[i]} minerals left`);
      }
      if (world.pool.carrying[i]! > 0) parts.push(`carrying ${world.pool.carrying[i]}`);
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
      this.selectionDetail.textContent = `${summary} · ${totalHp}/${maxHp} HP`;
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

  showDialog(title: string, body: string, actions: { label: string; primary?: boolean; onClick: () => void }[]): void {
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

  /**
   * Redraw the minimap.
   *
   * Throttled to every few frames: it is a full canvas repaint and nothing on it
   * changes fast enough at 60Hz to be worth the cost.
   */
  drawMinimap(world: World, focusX: number, focusZ: number, viewRadius: number): void {
    if (this.minimapFrame++ % 4 !== 0) return;

    const ctx = this.minimapCtx;
    const scale = MINIMAP_PX / this.mapSize;

    ctx.fillStyle = '#151b24';
    ctx.fillRect(0, 0, MINIMAP_PX, MINIMAP_PX);

    // Cliffs, sampled rather than drawn per tile — at this scale several tiles
    // share a pixel anyway.
    ctx.fillStyle = '#39414f';
    const step = 2;
    for (let y = 0; y < world.map.height; y += step) {
      for (let x = 0; x < world.map.width; x += step) {
        if (world.map.tiles[world.map.index(x, y)] === 1) {
          ctx.fillRect(x * scale, y * scale, step * scale, step * scale);
        }
      }
    }

    const pool = world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] !== 1) continue;
      const type = pool.type[i]! as EntityType;
      const def = defOf(type);
      const owner = pool.owner[i]!;

      ctx.fillStyle =
        owner === NEUTRAL ? hex(RESOURCE_COLOUR) : hex(PLAYER_COLOURS[owner] ?? 0x999999);

      const px = toFloat(pool.posX[i]!) * scale;
      const pz = toFloat(pool.posY[i]!) * scale;
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
