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
 * The map, the number of players and which bot plays all change what the
 * simulation computes, so a lobby that let two peers disagree would produce a
 * desync on the first tick rather than a game. The choice is therefore reduced
 * to one opaque string (`modeId`) that the transports compare during their
 * handshake, and a mismatch is refused with a message a person can act on.
 */

import { joinLocalRoom } from '../net/broadcastChannelTransport.js';
import { SoloTransport } from '../net/localTransport.js';
import { generateRoomCode, joinOnlineRoom, seedFromRoomCode } from '../net/trysteroTransport.js';
import type { Transport } from '../net/transport.js';
import type { AgentDeps } from '../ai/factory.js';
import { NeuralAgent } from '../ai/neural/agent.js';
import { loadNeuralRuntime, probeNeuralModel } from '../ai/neural/browser.js';
import type { WorkerRuntime } from '../ai/neural/runtime.js';
import { coopMatch, duelMatch } from '../sim/match.js';
import { BotKind, type MatchConfig } from '../sim/types.js';
import { audio } from '../audio/audio.js';

export interface MatchSetup {
  transport: Transport;
  /** The whole agreed description of the match. */
  config: MatchConfig;
  /** What the bots this peer hosts need that the config cannot carry: a loaded neural runtime. */
  agentDeps?: AgentDeps;
}

/** What the player picked, before a seed is known. */
type LobbyMode =
  /** One human against one AI, on the duel map. */
  | { kind: 'skirmish'; bot: BotKind }
  /** Two humans on the duel map. */
  | { kind: 'versus' }
  /** Two players a side on the four-corner map, against the AI. */
  | { kind: 'coop'; bot: BotKind; withAiPartner: boolean };

const BOT_LABELS: Readonly<Record<BotKind, string>> = {
  [BotKind.Scripted]: 'Scripted',
  [BotKind.Neural]: 'Neural',
};

/**
 * The chips to offer, derived from the labels rather than listed again.
 *
 * `BOT_LABELS` is a `Record<BotKind, string>`, so the compiler makes it
 * exhaustive; a hand-written array is not checked against anything, and a new
 * kind of bot would simply never get a chip with nothing failing to say so.
 * Integer-like keys enumerate in ascending numeric order, which is the order
 * the row should read in anyway.
 */
const BOT_KINDS = Object.keys(BOT_LABELS).map(Number) as BotKind[];

/**
 * Whether a mode has a slot the neural model would play. The chip is shared
 * by every mode with an AI in it, so this is the one question every start
 * path asks before it connects or begins.
 */
function needsNeuralModel(mode: LobbyMode): boolean {
  return mode.kind !== 'versus' && mode.bot === BotKind.Neural;
}

/** Turn a choice plus an agreed seed into the match every peer will run. */
function configFor(mode: LobbyMode, seed: number): MatchConfig {
  if (mode.kind === 'versus') return duelMatch(seed, { botPlayers: [] });
  if (mode.kind === 'skirmish') {
    return duelMatch(seed, { botPlayers: [1], kind: mode.bot });
  }
  // Slot 1 is the second human seat. Filling it with a bot is what lets one
  // person try the co-op map without waiting for a partner — and that partner
  // is always the scripted bot: a known quantity beside you, one model in the
  // tab at most, and the opponents are what the chip chooses.
  return coopMatch(seed, {
    botSlots: [
      ...(mode.withAiPartner ? [{ player: 1, kind: BotKind.Scripted }] : []),
      { player: 2, kind: mode.bot },
      { player: 3, kind: mode.bot },
    ],
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
 * The handshake compares `modeId`, and since the AI choice follows the player
 * between screens the likeliest mismatch is now one neither of them chose
 * deliberately: pick Neural for a skirmish, back out, open co-op, and Neural
 * is already selected. The transport can only report "you chose different
 * modes" — it never interprets the string — so the connect screens have to
 * show what is being compared, or a mismatch is unactionable from anything on
 * screen.
 */
function modeSummary(mode: LobbyMode): string {
  if (mode.kind === 'versus') return 'Versus another player';
  const label = BOT_LABELS[mode.bot];
  return mode.kind === 'skirmish' ? `Skirmish vs AI · ${label}` : `Co-op vs AI · ${label}`;
}

/**
 * One action button on a mode screen.
 *
 * `run` takes no argument and builds its own mode when clicked, rather than
 * being handed one when the screen is drawn: the bot chips can change
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
     * Which bot plays, shared by every screen that offers the choice.
     *
     * One value rather than one per screen, so a player who picks Neural for a
     * skirmish and then opens co-op finds Neural still selected. It also
     * survives the Back button out of the connect screens, which route back to
     * the mode they came from.
     */
    let bot = BotKind.Scripted;

    /**
     * Whether this build ships a model: unknown until the manifest has been
     * asked for, then yes or no. The Neural chip is disabled until it is yes —
     * shown, so the choice is visible; disabled, so a match is never started
     * for a slot nothing can play. A mode screen open when the answer arrives
     * is redrawn so the chip comes alive without a click.
     */
    let neuralAvailable: boolean | null = null;
    let redrawMode: (() => void) | null = null;
    void probeNeuralModel().then((manifest) => {
      neuralAvailable = manifest !== null;
      redrawMode?.();
    });

    /**
     * The model, loaded once per page and shared by every neural slot this
     * peer hosts. Loaded *before* connecting, so a peer that cannot run the
     * model never leaves the other waiting on a match that will not start.
     */
    let runtimeLoading: Promise<WorkerRuntime> | null = null;
    const ensureRuntime = (): Promise<WorkerRuntime> => {
      if (runtimeLoading === null) {
        runtimeLoading = loadNeuralRuntime().catch((error: unknown) => {
          runtimeLoading = null;
          throw error;
        });
      }
      return runtimeLoading.then((runtime) => {
        if (!runtime.disposed) return runtime;
        runtimeLoading = null;
        return ensureRuntime();
      });
    };

    const render = (html: string): HTMLElement => {
      redrawMode = null;
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
      const config = configFor(mode, seed);
      if (!needsNeuralModel(mode)) {
        overlay.remove();
        resolve({ transport, config });
        return;
      }
      // Already loaded by `withModel` on the way here; this only unwraps it.
      void ensureRuntime().then((runtime) => {
        overlay.remove();
        resolve({ transport, config, agentDeps: { neural: () => new NeuralAgent(runtime) } });
      });
    };

    /** Run `then` once the model a mode needs is loaded, or explain why it cannot be. */
    const withModel = (mode: LobbyMode, back: () => void, then: () => void): void => {
      if (!needsNeuralModel(mode)) {
        then();
        return;
      }
      render(`
        <h1>Loading the neural bot…</h1>
        <p>Fetching the model and starting its worker. This happens once.</p>
      `);
      ensureRuntime()
        .then(then)
        .catch((error: unknown) => {
          showError(error instanceof Error ? error.message : String(error), back);
        });
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

    /** Render a mode: what it is, which bot plays, and how to start it. */
    const modeScreen = (spec: {
      title: string;
      blurb: string;
      /** Show the bot picker. True exactly when the mode has an AI in it. */
      hasBots: boolean;
      actions: ModeAction[];
      back: () => void;
    }): void => {
      const chips = spec.hasBots
        ? `<div id="difficulty-row" role="group" aria-label="AI opponent">
             ${BOT_KINDS.map((k) => {
               const off = k === BotKind.Neural && neuralAvailable !== true;
               const title = !off
                 ? ''
                 : neuralAvailable === null
                   ? ' title="Checking whether this build ships a model…"'
                   : ' title="This build ships no neural model. See ml/README.md for how to train and export one."';
               return `<button class="chip" data-bot="${k}"${off ? ' disabled' : ''}${title}>${BOT_LABELS[k]}</button>`;
             }).join('')}
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

      redrawMode = () => modeScreen(spec);

      if (spec.hasBots) {
        const syncChips = (): void => {
          for (const chip of dialog.querySelectorAll<HTMLElement>('[data-bot]')) {
            const value = Number(chip.dataset.bot);
            chip.classList.toggle('active', value === bot);
            chip.setAttribute('aria-pressed', String(value === bot));
          }
        };
        for (const chip of dialog.querySelectorAll<HTMLElement>('[data-bot]')) {
          chip.addEventListener('click', () => {
            bot = Number(chip.dataset.bot) as BotKind;
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
                Scripted plays one fixed, tuned strategy and reads the whole map;
                Neural is a learned player that sees only what you would. Neither
                gets bonus income or extra units.`,
        hasBots: true,
        actions: [
          {
            label: 'Start match',
            primary: true,
            run: () =>
              withModel({ kind: 'skirmish', bot }, skirmish, () =>
                start(new SoloTransport(), { kind: 'skirmish', bot }, randomSeed()),
              ),
          },
        ],
        back: menu,
      });
    };

    const coop = (): void => {
      const withPartner = (withAiPartner: boolean): LobbyMode => ({
        kind: 'coop',
        bot,
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
            run: () =>
              withModel(withPartner(true), coop, () =>
                start(new SoloTransport(), withPartner(true), randomSeed()),
              ),
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

    // Hosting a bot over WebRTC is the same dealing as over two tabs, so the
    // model a neural slot needs is loaded before the room-code screen, as it
    // is before two tabs connect: in online co-op each browser hosts one of
    // the two bots, and a peer that cannot run the model must find out here
    // rather than leave the other waiting on a match that will not start.
    const online = (mode: LobbyMode, back: () => void): void => {
      withModel(mode, back, () => onlineRoom(mode, back));
    };

    const onlineRoom = (mode: LobbyMode, back: () => void): void => {
      const suggested = generateRoomCode();
      const bothHost = needsNeuralModel(mode)
        ? ` Each browser runs one of the two learned bots, so the other player's
           browser loads the model too.`
        : '';
      const dialog = render(`
        <h1>Play online</h1>
        <p>Share a code with the other player. You both enter the same code —
           whoever arrives first hosts. No account, and no server of ours
           involved: once connected, the game runs directly between the two
           browsers.</p>
        <p class="mode-summary">You are starting: <strong>${modeSummary(mode)}</strong>.
           The other player has to pick the same, AI included.${bothHost}</p>
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
      withModel(mode, back, () => localRoom(mode, back));
    };

    const localRoom = (mode: LobbyMode, back: () => void): void => {
      const dialog = render(`
        <h1>Two tabs</h1>
        <p>Open this page in a second tab and choose the same option there.
           The two tabs talk directly, with no network involved — handy for
           trying multiplayer without a second machine.</p>
        <p class="mode-summary">Pick this in the other tab: <strong>${modeSummary(mode)}</strong>.
           The AI choice counts too.</p>
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
