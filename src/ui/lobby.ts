/**
 * The opening menu: pick how to play, then hand back a transport.
 *
 * All three modes produce the same thing — a `Transport` and a seed — and the
 * game itself cannot tell them apart. That is the payoff of routing single
 * player through the lockstep scheduler too: there is no separate "offline"
 * code path that can rot.
 */

import { joinLocalRoom } from '../net/broadcastChannelTransport.js';
import { SoloTransport } from '../net/localTransport.js';
import {
  generateRoomCode,
  joinOnlineRoom,
  seedFromRoomCode,
} from '../net/trysteroTransport.js';
import type { Transport } from '../net/transport.js';
import type { PlayerId } from '../sim/types.js';
import { audio } from '../audio/audio.js';

export interface MatchSetup {
  transport: Transport;
  seed: number;
  localPlayer: PlayerId;
  /** Slots the AI should play. Empty in a two-human match. */
  botPlayers: PlayerId[];
}

export function showLobby(root: HTMLElement, onShowAllUnits: () => void): Promise<MatchSetup> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    root.append(overlay);

    const render = (html: string): HTMLElement => {
      overlay.innerHTML = `<div class="dialog">${html}</div>`;
      return overlay.querySelector('.dialog') as HTMLElement;
    };

    const showError = (message: string, retry: () => void): void => {
      const dialog = render(`
        <h1>Could not connect</h1>
        <p></p>
        <button class="primary" data-act="retry">Back to menu</button>
      `);
      dialog.querySelector('p')!.textContent = message;
      dialog.querySelector('[data-act="retry"]')!.addEventListener('click', retry);
    };

    const menu = (): void => {
      // Every route out of this menu is a click, which is exactly the user
      // gesture browsers require before an AudioContext may start.
      overlay.addEventListener('pointerdown', () => void audio.resume(), { once: true });

      const dialog = render(`
        <h1>Experiment RTS</h1>
        <p>Gather minerals, build a base, and destroy every enemy structure.</p>
        <button class="primary" data-act="ai">Skirmish vs AI</button>
        <button data-act="online">Play online</button>
        <button data-act="local">Two tabs on this computer</button>
        <button data-act="units" aria-haspopup="dialog"
                aria-controls="unit-gallery-dialog">All units</button>
      `);

      dialog.querySelector('[data-act="ai"]')!.addEventListener('click', () => {
        overlay.remove();
        resolve({
          transport: new SoloTransport(),
          seed: (Math.random() * 0xffffffff) >>> 0,
          localPlayer: 0,
          botPlayers: [1],
        });
      });

      dialog.querySelector('[data-act="online"]')!.addEventListener('click', online);
      dialog.querySelector('[data-act="local"]')!.addEventListener('click', local);
      dialog.querySelector('[data-act="units"]')!.addEventListener('click', onShowAllUnits);
    };

    const online = (): void => {
      const suggested = generateRoomCode();
      const dialog = render(`
        <h1>Play online</h1>
        <p>Share a code with the other player. You both enter the same code —
           whoever arrives first hosts. No account, and no server of ours
           involved: once connected, the game runs directly between the two
           browsers.</p>
        <input class="interactive" id="room-code" value="${suggested}"
               maxlength="8" autocomplete="off" spellcheck="false"
               style="width:100%;padding:11px;border-radius:9px;border:1px solid var(--panel-edge);
                      background:rgba(10,14,20,0.7);color:var(--text);font:inherit;
                      font-size:20px;text-align:center;letter-spacing:0.22em;text-transform:uppercase" />
        <button class="primary" data-act="go">Connect</button>
        <button data-act="back">Back</button>
      `);

      const input = dialog.querySelector('#room-code') as HTMLInputElement;
      input.focus();
      input.select();

      dialog.querySelector('[data-act="back"]')!.addEventListener('click', menu);
      dialog.querySelector('[data-act="go"]')!.addEventListener('click', () => {
        const code = input.value.trim().toUpperCase();
        if (code.length < 3) return;
        connect(code);
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const code = input.value.trim().toUpperCase();
          if (code.length >= 3) connect(code);
        }
      });
    };

    const connect = (code: string): void => {
      const dialog = render(`
        <h1>Connecting…</h1>
        <p id="status">Looking for the other player.</p>
        <button data-act="cancel">Cancel</button>
      `);
      const status = dialog.querySelector('#status') as HTMLElement;
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', menu);

      // Both peers derive the same seed from the room code, so the map is
      // agreed before either side sends anything.
      joinOnlineRoom({
        roomCode: code,
        seed: seedFromRoomCode(code),
        onStatus: (message) => {
          status.textContent = message;
        },
      })
        .then(({ transport, seed, localPlayer }) => {
          overlay.remove();
          resolve({ transport, seed, localPlayer, botPlayers: [] });
        })
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), menu);
        });
    };

    const local = (): void => {
      const dialog = render(`
        <h1>Two tabs</h1>
        <p>Open this page in a second tab and choose the same option there.
           The two tabs talk directly, with no network involved — handy for
           trying multiplayer without a second machine.</p>
        <p id="status" style="color:var(--warn)">Waiting for the second tab…</p>
        <button data-act="cancel">Cancel</button>
      `);
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', menu);

      joinLocalRoom('lan', (Math.random() * 0xffffffff) >>> 0)
        .then(({ transport, seed, localPlayer }) => {
          overlay.remove();
          resolve({ transport, seed, localPlayer, botPlayers: [] });
        })
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), menu);
        });
    };

    menu();
  });
}
