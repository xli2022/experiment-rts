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
import { generateRoomCode, joinOnlineRoom, seedFromRoomCode } from '../net/trysteroTransport.js';
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

/**
 * The chips to offer, derived from the labels rather than listed again.
 *
 * `DIFFICULTY_LABELS` is a `Record<BotDifficulty, string>`, so the compiler
 * makes it exhaustive; a hand-written array is not checked against anything, and
 * a fourth difficulty would simply never get a chip with nothing failing to say
 * so. Integer-like keys enumerate in ascending numeric order, which is the order
 * the row should read in anyway.
 */
const DIFFICULTIES = Object.keys(DIFFICULTY_LABELS).map(Number) as BotDifficulty[];

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
 * The seed is zeroed because it is agreed separately, from the room code, and
 * the string is only ever compared for equality, never parsed.
 *
 * What makes it stable is narrower than "both peers run the same build" —
 * `PROTOCOL_VERSION` is what rules that out, and it only rules it out if
 * somebody bumps it. It is stable because `matchConfig` returns one
 * fixed-order object literal whose every value is a finite number or an array
 * of them: no key is integer-like (which would enumerate first regardless of
 * insertion order), no field can be `undefined` (silently dropped, so two
 * different choices would serialise the same), and nothing can reach
 * `NaN`/`Infinity` (both become `null`). Adding an optional field to
 * `MatchConfig` breaks that, and the symptom is two players who picked the same
 * mode being told they did not.
 */
function modeId(mode: LobbyMode): string {
  return JSON.stringify(configFor(mode, 0));
}

/**
 * What the two peers have to agree on, in words.
 *
 * The handshake compares `modeId`, and since difficulty follows the player
 * between screens the likeliest mismatch is now one neither of them chose
 * deliberately: pick Hard for a skirmish, back out, open co-op, and Hard is
 * already selected. The transport can only report "you chose different modes"
 * — it never interprets the string — so the connect screens have to show what
 * is being compared, or a mismatch is unactionable from anything on screen.
 */
function modeSummary(mode: LobbyMode): string {
  if (mode.kind === 'versus') return 'Versus another player';
  const label = DIFFICULTY_LABELS[mode.difficulty];
  return mode.kind === 'skirmish' ? `Skirmish vs AI · ${label}` : `Co-op vs AI · ${label}`;
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

    // Every route into a match is a click, which is exactly the user gesture
    // browsers require before an AudioContext may start. Registered once, on
    // the overlay itself, rather than on entering the menu: the menu is now the
    // Back destination of three mode screens, and a `once` listener that has
    // not fired yet is not replaced by registering another — a player
    // navigating by keyboard would stack one per visit.
    overlay.addEventListener('pointerdown', () => void audio.resume(), { once: true });

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
      // "Back", not "Back to menu": `retry` is the screen the attempt started
      // from, which since the modes were grouped is the mode screen rather than
      // the main menu.
      const dialog = render(`
        <h1>Could not connect</h1>
        <p></p>
        <button class="primary" data-act="retry">Back</button>
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
            run: () => start(new SoloTransport(), { kind: 'skirmish', difficulty }, randomSeed()),
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

    /**
     * Wire a connect screen's Cancel button to actually abandon the join.
     *
     * Navigating away is not enough. Both joins hold a live channel or room for
     * the whole of their timeout, so a cancelled attempt keeps listening: a
     * peer (or, for two tabs, this tab's own next attempt) that arrives inside
     * that window still completes a handshake, and the abandoned promise then
     * starts a match the player backed out of — or refuses one they never made,
     * with a message about "the other tab" when there is no other tab. The
     * signal closes the channel at the source, which is the only place it can
     * be closed from.
     */
    const abandonOn = (dialog: HTMLElement, back: () => void): AbortController => {
      const attempt = new AbortController();
      dialog.querySelector('[data-act="cancel"]')!.addEventListener('click', () => {
        attempt.abort();
        back();
      });
      return attempt;
    };

    const online = (mode: LobbyMode, back: () => void): void => {
      const suggested = generateRoomCode();
      const dialog = render(`
        <h1>Play online</h1>
        <p>Share a code with the other player. You both enter the same code —
           whoever arrives first hosts. No account, and no server of ours
           involved: once connected, the game runs directly between the two
           browsers.</p>
        <p class="mode-summary">You are starting: <strong>${modeSummary(mode)}</strong>.
           The other player has to pick the same, difficulty included.</p>
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
      const attempt = abandonOn(dialog, back);

      // Both peers derive the same seed from the room code, so the map is
      // agreed before either side sends anything.
      joinOnlineRoom({
        roomCode: code,
        seed: seedFromRoomCode(code),
        mode: modeId(mode),
        signal: attempt.signal,
        onStatus: (message) => {
          status.textContent = message;
        },
      })
        .then(({ transport, seed }) => start(transport, mode, seed))
        .catch((error: unknown) => {
          if (attempt.signal.aborted) return;
          showError(error instanceof Error ? error.message : String(error), back);
        });
    };

    const local = (mode: LobbyMode, back: () => void): void => {
      const dialog = render(`
        <h1>Two tabs</h1>
        <p>Open this page in a second tab and choose the same option there.
           The two tabs talk directly, with no network involved — handy for
           trying multiplayer without a second machine.</p>
        <p class="mode-summary">Pick this in the other tab: <strong>${modeSummary(mode)}</strong>.
           Difficulty counts too.</p>
        <p id="status" style="color:var(--warn)">Waiting for the second tab…</p>
        <button data-act="cancel">Cancel</button>
      `);
      const attempt = abandonOn(dialog, back);

      joinLocalRoom('lan', randomSeed(), modeId(mode), attempt.signal)
        .then(({ transport, seed }) => start(transport, mode, seed))
        .catch((error: unknown) => {
          if (attempt.signal.aborted) return;
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
