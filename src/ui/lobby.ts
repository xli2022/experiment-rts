/**
 * The opening menu: pick how to play, then hand back a transport and a match.
 *
 * Every mode produces the same pair — a `Transport` and a `MatchConfig` — and
 * the game itself cannot tell them apart. That is the payoff of routing single
 * player through the lockstep scheduler too: there is no separate "offline" code
 * path that can rot, and neither co-op nor versus is a separate mode of the
 * *game*, only a different roster handed to the same one.
 *
 * ## One shape per screen
 *
 * The menu offers three modes and each opens the same kind of screen: what it
 * is, how hard the AI plays if there is any AI in it, and the ways to start it.
 * Grouping them that way is not only tidiness — the two versus modes are the
 * same match over two different transports, and listing them at the top level
 * beside the modes made "who am I playing" and "how do we connect" look like
 * one question when they are two.
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
  | { kind: 'versus' }
  /** Two players a side on the four-corner map, against the AI. */
  | { kind: 'coop'; difficulty: BotDifficulty; withAiPartner: boolean };

const DIFFICULTY_LABELS: Readonly<Record<BotDifficulty, string>> = {
  [BotDifficulty.Easy]: 'Easy',
  [BotDifficulty.Normal]: 'Normal',
  [BotDifficulty.Hard]: 'Hard',
};

const DIFFICULTIES = [BotDifficulty.Easy, BotDifficulty.Normal, BotDifficulty.Hard];

/** Turn a choice plus an agreed seed into the match every peer will run. */
function configFor(mode: LobbyMode, seed: number): MatchConfig {
  if (mode.kind === 'versus') return duelMatch(seed, { botPlayers: [] });
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

/**
 * The string two peers compare to check they picked the same thing.
 *
 * Derived from the config rather than from the choice that produced it, and
 * that is the whole point: a hand-written summary of `LobbyMode` is a second
 * list of everything that matters, and the day someone adds a field to one and
 * not the other, two peers pass the handshake and desync on the first tick —
 * exactly the failure the handshake exists to prevent. Serialising the artefact
 * cannot drift from the artefact.
 *
 * The seed is zeroed because it is agreed separately, from the room code. Key
 * order is stable because both peers run the same build — the protocol version
 * check guarantees it — and the string is only ever compared for equality,
 * never parsed.
 */
function modeId(mode: LobbyMode): string {
  return JSON.stringify(configFor(mode, 0));
}

/**
 * One action button on a mode screen.
 *
 * `run` takes no argument and builds its own mode when clicked, rather than
 * being handed one when the screen is drawn: the difficulty chips can change
 * the answer after that, so a mode captured up front would start the match on
 * whatever was selected when the screen opened.
 */
interface ModeAction {
  label: string;
  primary?: boolean;
  run: () => void;
}

export function showLobby(root: HTMLElement, onShowAllUnits: () => void): Promise<MatchSetup> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    root.append(overlay);

    /**
     * How hard the AI plays, shared by every screen that offers the choice.
     *
     * One value rather than one per screen, so a player who sets Hard for a
     * skirmish and then opens co-op finds Hard still selected. It also survives
     * the Back button out of the connect screens, which route back to the mode
     * they came from.
     */
    let difficulty = BotDifficulty.Normal;

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

    // ---------------------------------------------------------------------
    // Screens
    // ---------------------------------------------------------------------

    const menu = (): void => {
      // Every route out of this menu is a click, which is exactly the user
      // gesture browsers require before an AudioContext may start.
      overlay.addEventListener('pointerdown', () => void audio.resume(), { once: true });

      const dialog = render(`
        <h1>Experiment RTS</h1>
        <p>Gather minerals, build a base, and destroy every enemy structure.</p>
        <button class="primary" data-act="skirmish">Skirmish vs AI</button>
        <button data-act="coop">Co-op vs AI</button>
        <button data-act="versus">Versus another player</button>
        <button data-act="units" aria-haspopup="dialog"
                aria-controls="unit-gallery-dialog">All units</button>
      `);

      dialog.querySelector('[data-act="skirmish"]')!.addEventListener('click', skirmish);
      dialog.querySelector('[data-act="coop"]')!.addEventListener('click', coop);
      dialog.querySelector('[data-act="versus"]')!.addEventListener('click', versus);
      dialog.querySelector('[data-act="units"]')!.addEventListener('click', onShowAllUnits);
    };

    /** Render a mode: what it is, how hard the AI plays, and how to start it. */
    const modeScreen = (spec: {
      title: string;
      blurb: string;
      /** Show the difficulty picker. True exactly when the mode has an AI in it. */
      hasBots: boolean;
      actions: ModeAction[];
      back: () => void;
    }): void => {
      const chips = spec.hasBots
        ? `<div id="difficulty-row" role="group" aria-label="AI difficulty">
             ${DIFFICULTIES.map(
               (d) =>
                 `<button class="chip" data-difficulty="${d}">${DIFFICULTY_LABELS[d]}</button>`,
             ).join('')}
           </div>`
        : '';

      const dialog = render(`
        <h1>${spec.title}</h1>
        <p>${spec.blurb}</p>
        ${chips}
        ${spec.actions
          .map(
            (a, i) =>
              `<button data-run="${i}"${a.primary ? ' class="primary"' : ''}>${a.label}</button>`,
          )
          .join('')}
        <button data-act="back">Back</button>
      `);

      if (spec.hasBots) {
        const syncChips = (): void => {
          for (const chip of dialog.querySelectorAll<HTMLElement>('[data-difficulty]')) {
            const value = Number(chip.dataset.difficulty);
            chip.classList.toggle('active', value === difficulty);
            chip.setAttribute('aria-pressed', String(value === difficulty));
          }
        };
        for (const chip of dialog.querySelectorAll<HTMLElement>('[data-difficulty]')) {
          chip.addEventListener('click', () => {
            difficulty = Number(chip.dataset.difficulty) as BotDifficulty;
            syncChips();
          });
        }
        syncChips();
      }

      spec.actions.forEach((action, i) => {
        dialog.querySelector(`[data-run="${i}"]`)!.addEventListener('click', action.run);
      });
      dialog.querySelector('[data-act="back"]')!.addEventListener('click', spec.back);
    };

    const skirmish = (): void => {
      modeScreen({
        title: 'Skirmish vs AI',
        blurb: `One base each, on the three-lane map, against a single AI opponent.
                Difficulty changes only what it does — how hard it works its
                economy, how soon it commits, whether it comes home when its base
                is attacked. It gets no bonus income and no extra units.`,
        hasBots: true,
        actions: [
          {
            label: 'Start match',
            primary: true,
            run: () =>
              start(new SoloTransport(), { kind: 'skirmish', difficulty }, randomSeed()),
          },
        ],
        back: menu,
      });
    };

    const coop = (): void => {
      const withPartner = (withAiPartner: boolean): LobbyMode => ({
        kind: 'coop',
        difficulty,
        withAiPartner,
      });
      modeScreen({
        title: 'Co-op vs AI',
        blurb: `You and a partner share one side of a four-corner map against two
                AI opponents. Allies never damage each other, see through each
                other's scouting, and win or lose together — a side is only
                beaten once every structure on it is gone.`,
        hasBots: true,
        actions: [
          {
            label: 'Play online with a friend',
            primary: true,
            run: () => online(withPartner(false), coop),
          },
          { label: 'Two tabs on this computer', run: () => local(withPartner(false), coop) },
          {
            // The only route that fills the second human seat with an AI.
            label: 'Play solo with an AI partner',
            run: () => start(new SoloTransport(), withPartner(true), randomSeed()),
          },
        ],
        back: menu,
      });
    };

    const versus = (): void => {
      modeScreen({
        title: 'Versus another player',
        blurb: `One base each on the three-lane map, no AI involved — the same
                match either way, over whichever connection suits you. Play
                online with a room code, or against a second tab on this
                computer with no network at all.`,
        hasBots: false,
        actions: [
          { label: 'Play online', primary: true, run: () => online({ kind: 'versus' }, versus) },
          {
            label: 'Two tabs on this computer',
            run: () => local({ kind: 'versus' }, versus),
          },
        ],
        back: menu,
      });
    };

    // ---------------------------------------------------------------------
    // Connecting
    // ---------------------------------------------------------------------

    const online = (mode: LobbyMode, back: () => void): void => {
      const suggested = generateRoomCode();
      const dialog = render(`
        <h1>Play online</h1>
        <p>Share a code with the other player. You both enter the same code —
           whoever arrives first hosts. No account, and no server of ours
           involved: once connected, the game runs directly between the two
           browsers.</p>
        <input class="interactive" id="room-code" value="${suggested}"
               maxlength="8" autocomplete="off" spellcheck="false" />
        <button class="primary" data-act="go">Connect</button>
        <button data-act="back">Back</button>
      `);

      const input = dialog.querySelector('#room-code') as HTMLInputElement;
      input.focus();
      input.select();

      const go = (): void => {
        const code = input.value.trim().toUpperCase();
        if (code.length >= 3) connect(code, mode, back);
      };
      dialog.querySelector('[data-act="back"]')!.addEventListener('click', back);
      dialog.querySelector('[data-act="go"]')!.addEventListener('click', go);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') go();
      });
    };

    const connect = (code: string, mode: LobbyMode, back: () => void): void => {
      const dialog = render(`
        <h1>Connecting…</h1>
        <p id="status">Looking for the other player.</p>
        <button data-act="cancel">Cancel</button>
      `);
      const status = dialog.querySelector('#status') as HTMLElement;
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', back);

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
          showError(error instanceof Error ? error.message : String(error), back);
        });
    };

    const local = (mode: LobbyMode, back: () => void): void => {
      const dialog = render(`
        <h1>Two tabs</h1>
        <p>Open this page in a second tab and choose the same option there.
           The two tabs talk directly, with no network involved — handy for
           trying multiplayer without a second machine.</p>
        <p id="status" style="color:var(--warn)">Waiting for the second tab…</p>
        <button data-act="cancel">Cancel</button>
      `);
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', back);

      joinLocalRoom('lan', randomSeed(), modeId(mode))
        .then(({ transport, seed }) => start(transport, mode, seed))
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), back);
        });
    };

    menu();
  });
}

/** A fresh match seed. Only ever called before a match exists. */
function randomSeed(): number {
  return (Math.random() * 0xffffffff) >>> 0;
}
