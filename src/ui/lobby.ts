/**
 * The opening menu: pick how to play, then hand back a transport and a match.
 *
 * Every mode produces the same pair — a `Transport` and a `MatchConfig` — and
 * the game itself cannot tell them apart. That is the payoff of routing single
 * player through the lockstep scheduler too: there is no separate "offline" code
 * path that can rot, and co-op is not a separate mode of the *game*, only a
 * different roster handed to the same one.
 *
 * ## Both peers must choose the same thing
 *
 * The map, the number of players and the AI's difficulty all change what the
 * simulation computes, so a lobby that let two peers disagree would produce a
 * desync on the first tick rather than a game. The choice is therefore reduced
 * to one opaque string (`modeId`) that the transports compare during their
 * handshake, and a mismatch is refused with a message a person can act on.
 */

import { joinLocalRoom } from '../net/broadcastChannelTransport.js';
import { SoloTransport } from '../net/localTransport.js';
import {
  generateRoomCode,
  joinOnlineRoom,
  seedFromRoomCode,
} from '../net/trysteroTransport.js';
import type { Transport } from '../net/transport.js';
import { coopMatch, duelMatch } from '../sim/match.js';
import { BotDifficulty, type MatchConfig } from '../sim/types.js';
import { audio } from '../audio/audio.js';

export interface MatchSetup {
  transport: Transport;
  /** The whole agreed description of the match. */
  config: MatchConfig;
}

/** What the player picked, before a seed is known. */
type LobbyMode =
  /** One human against one AI, on the duel map. */
  | { kind: 'skirmish'; difficulty: BotDifficulty }
  /** Two humans on the duel map. */
  | { kind: 'duel' }
  /** Two players a side on the four-corner map, against the AI. */
  | { kind: 'coop'; difficulty: BotDifficulty; withAiPartner: boolean };

const DIFFICULTY_LABELS: Readonly<Record<BotDifficulty, string>> = {
  [BotDifficulty.Easy]: 'Easy',
  [BotDifficulty.Normal]: 'Normal',
  [BotDifficulty.Hard]: 'Hard',
};

/**
 * The string two peers compare to check they picked the same thing.
 *
 * Everything that feeds `configFor` has to appear in it, or a difference that
 * changes the simulation could slip past the handshake — which is precisely the
 * class of bug the handshake exists to catch.
 */
function modeId(mode: LobbyMode): string {
  if (mode.kind === 'duel') return 'duel';
  if (mode.kind === 'skirmish') return `skirmish:${mode.difficulty}`;
  return `coop:${mode.difficulty}:${mode.withAiPartner ? 'trio' : 'pair'}`;
}

/** Turn a choice plus an agreed seed into the match every peer will run. */
function configFor(mode: LobbyMode, seed: number): MatchConfig {
  if (mode.kind === 'duel') return duelMatch(seed, { botPlayers: [] });
  if (mode.kind === 'skirmish') {
    return duelMatch(seed, { botPlayers: [1], difficulty: mode.difficulty });
  }
  return coopMatch(seed, {
    difficulty: mode.difficulty,
    // Slot 1 is the second human seat. Filling it with a bot is what lets one
    // person try the co-op map without waiting for a partner.
    botPlayers: mode.withAiPartner ? [1, 2, 3] : [2, 3],
  });
}

export function showLobby(root: HTMLElement, onShowAllUnits: () => void): Promise<MatchSetup> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    root.append(overlay);

    /** Difficulty the co-op screen currently has selected. */
    let coopDifficulty = BotDifficulty.Normal;

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

    // The slot is not carried here: `Transport.localPlayer` is the same value
    // and is what the game actually reads, so a second copy could only ever
    // disagree with it.
    const start = (transport: Transport, mode: LobbyMode, seed: number): void => {
      overlay.remove();
      resolve({ transport, config: configFor(mode, seed) });
    };

    const menu = (): void => {
      // Every route out of this menu is a click, which is exactly the user
      // gesture browsers require before an AudioContext may start.
      overlay.addEventListener('pointerdown', () => void audio.resume(), { once: true });

      const dialog = render(`
        <h1>Experiment RTS</h1>
        <p>Gather minerals, build a base, and destroy every enemy structure.</p>
        <button class="primary" data-act="ai">Skirmish vs AI</button>
        <button data-act="coop">Co-op vs AI</button>
        <button data-act="online">Play online</button>
        <button data-act="local">Two tabs on this computer</button>
        <button data-act="units" aria-haspopup="dialog"
                aria-controls="unit-gallery-dialog">All units</button>
      `);

      dialog.querySelector('[data-act="ai"]')!.addEventListener('click', () => {
        start(new SoloTransport(), { kind: 'skirmish', difficulty: BotDifficulty.Normal }, randomSeed());
      });

      dialog.querySelector('[data-act="coop"]')!.addEventListener('click', coop);
      dialog.querySelector('[data-act="online"]')!.addEventListener('click', () =>
        online({ kind: 'duel' }),
      );
      dialog.querySelector('[data-act="local"]')!.addEventListener('click', () =>
        local({ kind: 'duel' }),
      );
      dialog.querySelector('[data-act="units"]')!.addEventListener('click', onShowAllUnits);
    };

    const coop = (): void => {
      const dialog = render(`
        <h1>Co-op vs AI</h1>
        <p>You and a partner share one side of a four-corner map against two AI
           opponents. Allies never damage each other, see through each other's
           scouting, and win or lose together — a side is only beaten once every
           structure on it is gone.</p>
        <div id="difficulty-row">
          ${[BotDifficulty.Easy, BotDifficulty.Normal, BotDifficulty.Hard]
            .map(
              (d) =>
                `<button data-difficulty="${d}" class="chip">${DIFFICULTY_LABELS[d]}</button>`,
            )
            .join('')}
        </div>
        <button class="primary" data-act="online">Play online with a friend</button>
        <button data-act="local">Two tabs on this computer</button>
        <button data-act="solo">Play solo with an AI partner</button>
        <button data-act="back">Back</button>
      `);

      const syncChips = (): void => {
        for (const chip of dialog.querySelectorAll<HTMLElement>('[data-difficulty]')) {
          chip.classList.toggle('active', Number(chip.dataset.difficulty) === coopDifficulty);
        }
      };
      for (const chip of dialog.querySelectorAll<HTMLElement>('[data-difficulty]')) {
        chip.addEventListener('click', () => {
          coopDifficulty = Number(chip.dataset.difficulty) as BotDifficulty;
          syncChips();
        });
      }
      syncChips();

      const modeWith = (withAiPartner: boolean): LobbyMode => ({
        kind: 'coop',
        difficulty: coopDifficulty,
        withAiPartner,
      });

      dialog.querySelector('[data-act="online"]')!.addEventListener('click', () =>
        online(modeWith(false)),
      );
      dialog.querySelector('[data-act="local"]')!.addEventListener('click', () =>
        local(modeWith(false)),
      );
      dialog.querySelector('[data-act="solo"]')!.addEventListener('click', () => {
        start(new SoloTransport(), modeWith(true), randomSeed());
      });
      dialog.querySelector('[data-act="back"]')!.addEventListener('click', menu);
    };

    const online = (mode: LobbyMode): void => {
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

      const back = mode.kind === 'coop' ? coop : menu;
      dialog.querySelector('[data-act="back"]')!.addEventListener('click', back);
      dialog.querySelector('[data-act="go"]')!.addEventListener('click', () => {
        const code = input.value.trim().toUpperCase();
        if (code.length < 3) return;
        connect(code, mode);
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const code = input.value.trim().toUpperCase();
          if (code.length >= 3) connect(code, mode);
        }
      });
    };

    const connect = (code: string, mode: LobbyMode): void => {
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
        mode: modeId(mode),
        onStatus: (message) => {
          status.textContent = message;
        },
      })
        .then(({ transport, seed }) => start(transport, mode, seed))
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), menu);
        });
    };

    const local = (mode: LobbyMode): void => {
      const dialog = render(`
        <h1>Two tabs</h1>
        <p>Open this page in a second tab and choose the same option there.
           The two tabs talk directly, with no network involved — handy for
           trying multiplayer without a second machine.</p>
        <p id="status" style="color:var(--warn)">Waiting for the second tab…</p>
        <button data-act="cancel">Cancel</button>
      `);
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', menu);

      joinLocalRoom('lan', randomSeed(), modeId(mode))
        .then(({ transport, seed }) => start(transport, mode, seed))
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), menu);
        });
    };

    menu();
  });
}

/** A fresh match seed. Only ever called before a match exists. */
function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}
