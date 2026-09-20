import { afterEach, describe, expect, it, vi } from 'vitest';
import { Hud, type CommandButton } from '../src/ui/hud.js';
import { buildingUpgrade } from '../src/config/rules.js';
import { idIndex } from '../src/sim/entities.js';
import { World } from '../src/sim/world.js';
import { BuildState, EntityType, TICKS_PER_SECOND } from '../src/sim/types.js';

/** Only the command card's DOM surface; no WebGL or canvas is needed. */
class Button extends EventTarget {
  className = '';
  disabled = false;
  innerHTML = '';
  title = '';
}

function commandCard(): { hud: Hud; children: Button[] } {
  const children: Button[] = [];
  const grid = {
    set innerHTML(_html: string) {
      children.length = 0;
    },
    append(button: Button) {
      children.push(button);
    },
  };
  vi.stubGlobal('document', { createElement: () => new Button() });
  const hud = Object.assign(Object.create(Hud.prototype) as Hud, {
    commandGrid: grid,
    lastButtonSignature: '',
  });
  return { hud, children };
}

afterEach(() => vi.unstubAllGlobals());

describe('command card dispatch', () => {
  it('shows level requirements and clears them when a unit is unlocked', () => {
    const { hud, children } = commandCard();
    const onClick = vi.fn();
    const locked: CommandButton = {
      key: 'A',
      label: 'Arclight',
      requirement: 'Level 2',
      enabled: false,
      onClick,
    };
    hud.setCommands([locked]);
    expect(children[0]!.innerHTML).toContain('cmd-requirement');
    expect(children[0]!.innerHTML).toContain('Level 2');
    children[0]!.dispatchEvent(new Event('click'));
    expect(onClick).not.toHaveBeenCalled();
    hud.setCommands([{ ...locked, requirement: undefined, enabled: true }]);
    expect(children[0]!.innerHTML).not.toContain('Level 2');
    children[0]!.dispatchEvent(new Event('click'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('shows tactical descriptions and refreshes a changed tooltip', () => {
    const { hud, children } = commandCard();
    const button: CommandButton = {
      key: 'B',
      label: 'Burstbot',
      description: 'Ranged line unit · 12s to train · 1 supply',
      enabled: true,
      onClick: vi.fn(),
    };
    hud.setCommands([button]);
    expect(children[0]!.title).toBe(button.description);
    hud.setCommands([{ ...button, description: 'Ranged line unit · 10s to train · 1 supply' }]);
    expect(children[0]!.title).toBe('Ranged line unit · 10s to train · 1 supply');
    hud.setCommands([{ ...button, description: undefined }]);
    expect(children[0]!.title).toBe('');
  });

  it('uses the newly selected building without replacing identical buttons', () => {
    const { hud, children } = commandCard();
    const firstBuilding = vi.fn();
    const secondBuilding = vi.fn();
    const card: CommandButton = {
      key: 'W',
      label: 'Worker',
      cost: 50,
      enabled: true,
      onClick: firstBuilding,
    };
    hud.setCommands([card]);
    const button = children[0]!;
    hud.setCommands([{ ...card, onClick: secondBuilding }]);
    expect(children[0]).toBe(button);
    button.dispatchEvent(new Event('click'));
    expect(firstBuilding).not.toHaveBeenCalled();
    expect(secondBuilding).toHaveBeenCalledOnce();
  });

  it('refreshes a changed displayed cost and never dispatches disabled commands', () => {
    const { hud, children } = commandCard();
    const onClick = vi.fn();
    hud.setCommands([{ key: 'W', label: 'Worker', cost: 50, enabled: true, onClick }]);
    hud.setCommands([{ key: 'W', label: 'Worker', cost: 75, enabled: false, onClick }]);
    expect(children[0]!.innerHTML).toContain('75');
    children[0]!.dispatchEvent(new Event('click'));
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe('building upgrade presentation', () => {
  it('shows the building level, upgrade progress and remaining time', () => {
    const world = new World(1);
    const index = idIndex(world.pool.spawn(EntityType.Factory, 0, 30 * 65536, 30 * 65536));
    world.pool.buildState[index] = BuildState.Complete;
    const selectionTitle = { textContent: '', title: '' };
    const selectionDetail = { textContent: '' };
    const production = { hidden: true, classList: { toggle: vi.fn() } };
    const prodLabel = { textContent: '' };
    const prodEta = { textContent: '' };
    const prodFill = { style: { width: '' } };
    const prodQueue = { textContent: '', hidden: true };
    const hud: Hud = Object.assign(Object.create(Hud.prototype) as Hud, {
      localPlayer: 0,
      selectionTitle,
      selectionDetail,
      production,
      prodLabel,
      prodEta,
      prodFill,
      prodQueue,
    });
    const selected = new Set([index]);
    hud.updateSelection(world, selected);
    expect(selectionTitle.textContent).toBe('Factory · Level 1');
    hud.updateProduction(world, selected);
    expect(production.hidden).toBe(true);

    const ticks = buildingUpgrade(EntityType.Factory)!.buildTicks;
    world.pool.upgrading[index] = 1;
    world.pool.upgradeProgress[index] = ticks / 2;
    hud.updateSelection(world, selected);
    hud.updateProduction(world, selected);
    expect(selectionDetail.textContent).toContain('upgrading to level 2');
    expect(production.hidden).toBe(false);
    expect(production.classList.toggle).toHaveBeenLastCalledWith('blocked', false);
    expect(prodLabel.textContent).toBe('Upgrading to level 2');
    expect(prodEta.textContent).toBe(`${(ticks / 2 / TICKS_PER_SECOND).toFixed(1)}s`);
    expect(prodFill.style.width).toBe('50.0%');
    expect(prodQueue.textContent).toBe('Training paused during upgrade');

    world.pool.upgrading[index] = 0;
    world.pool.buildingLevel[index] = 2;
    hud.updateSelection(world, selected);
    hud.updateProduction(world, selected);
    expect(selectionTitle.textContent).toBe('Factory · Level 2');
    expect(selectionDetail.textContent).not.toContain('upgrading');
    expect(production.hidden).toBe(true);

    world.pool.upgrading[index] = 1;
    world.pool.owner[index] = 1;
    hud.updateProduction(world, selected);
    expect(production.hidden).toBe(true);
  });
});
