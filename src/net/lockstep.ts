/**
 * The lockstep scheduler.
 *
 * ## How it works
 *
 * The simulation advances in fixed ticks, grouped into *turns*. Commands issued
 * during turn N are not executed at turn N — they are broadcast to be executed
 * at turn `N + INPUT_DELAY_TURNS`. That delay is what makes peer-to-peer work
 * without a server: by the time a turn comes up for execution, every peer has
 * already received everyone's commands for it, so all of them can run the turn
 * independently and get the same answer.
 *
 * The cost is input latency, which is why RTS games feel slightly "heavy"
 * compared to shooters. The benefit is that a 200-unit battle costs the same
 * bandwidth as a single click.
 *
 * How much latency is not fixed. Each peer sizes its own delay from feedback
 * the other sends about how early its packets are landing — 100ms on a LAN,
 * up to 500ms on a link that needs it. See `adaptDelay`, and note that the two
 * peers can hold different values without any risk of divergence: the wire
 * carries the absolute turn each command belongs to, so a schedule is stated
 * rather than inferred.
 *
 * ## The rule that must never bend
 *
 * A turn executes only when *every* player's commands for it have arrived. If
 * one peer is slow, everyone waits. Advancing without a peer's input would mean
 * simulating a different game from theirs, and there is no authority to correct
 * it afterwards — so stalling is not a failure mode here, it is the design.
 */

import type { Command } from '../sim/commands.js';
import { checksumToHex } from '../sim/checksum.js';
import type { Simulation } from '../sim/tick.js';
import { CHECKSUM_INTERVAL, MS_PER_TICK, type PlayerId } from '../sim/types.js';
import type { Packet, Transport, TurnCommands } from './transport.js';

/** Ticks per command turn. */
export const TICKS_PER_TURN = 2;

/**
 * How many turns ahead commands are scheduled at the start of a match.
 *
 * Two turns at 100ms each gives peers ~200ms to get a packet across, which
 * covers most real connections. Lower feels snappier but stalls on any latency
 * spike; higher is mushy to play. From here it adapts — see `adaptDelay` — but
 * the *starting* value has to be a constant every peer shares, because the
 * first few turns are seeded rather than sent and each peer seeds its own.
 */
export const INPUT_DELAY_TURNS = 2;

/**
 * Bounds on the adapted delay, in turns of 100ms.
 *
 * One turn is the floor because commands must land on a turn that has not
 * executed yet; zero would mean scheduling into the present. Five is the
 * ceiling because half a second of input lag is already worse than the stalls
 * it is avoiding, and a link that bad wants the "waiting for player" overlay
 * telling the truth rather than a game that merely feels broken.
 */
export const MIN_INPUT_DELAY_TURNS = 1;
export const MAX_INPUT_DELAY_TURNS = 5;

/** Turns between reconsiderations. Two seconds: slow enough not to oscillate. */
const ADAPT_INTERVAL_TURNS = 20;

/**
 * Spare turns a remote peer's commands must arrive with before we speed up.
 *
 * Headroom is how far ahead of our current turn a packet's newest turn is when
 * it lands. Zero means it arrived exactly as we needed it — no margin at all —
 * so shrinking the delay is only considered when there are turns to spare.
 */
const HEADROOM_TO_SPEED_UP = 2;

/** Older turns repeated in each packet, to ride out packet loss. */
const REDUNDANT_TURNS = 2;

/**
 * Turns of send history retained for recovery retransmission.
 *
 * Piggybacked redundancy alone is not enough. A turn's commands ride in only
 * REDUNDANT_TURNS + 1 consecutive packets; if every one of those is lost, the
 * sender has moved on and will never mention that turn again, while the receiver
 * blocks on it forever — and because the sender then blocks waiting for the
 * receiver, the whole match deadlocks. At 30% packet loss that happens within a
 * minute of play. Keeping a longer history lets a stalled peer re-send the
 * window that is actually missing.
 */
const SEND_HISTORY_TURNS = 24;

/** While stalled, retransmit the history no more often than this. */
const RESEND_INTERVAL_MS = 120;

/** Give up on a stalled peer after this long. */
export const PEER_TIMEOUT_MS = 15000;

/**
 * Wait this long before telling the UI we are stalled.
 *
 * Brief stalls are the *normal* resting state of lockstep, not an error: a peer
 * that is a turn or two ahead blocks at every turn boundary until the other side
 * catches up, so the raw flag flips many times a second. Reporting that directly
 * would strobe a "waiting for player" overlay during completely healthy play.
 * Only a stall that outlasts this threshold is worth a player's attention.
 */
export const STALL_REPORT_DELAY_MS = 250;

export type LockstepState = 'running' | 'stalled' | 'desynced' | 'ended';

export interface LockstepEvents {
  onStall?: (waitingFor: PlayerId[]) => void;
  onResume?: () => void;
  onDesync?: (tick: number, local: number, remote: number, player: PlayerId) => void;
  onPeerTimeout?: (player: PlayerId) => void;
}

export class LockstepRunner {
  state: LockstepState = 'running';

  /** Ticks executed so far. Always equals `simulation.world.tick`. */
  private tick = 0;
  private accumulatorMs = 0;

  /** Commands the local player has issued but not yet broadcast. */
  private pendingLocal: Command[] = [];

  /**
   * turn -> per-player command lists. A `null` entry means "not yet received".
   *
   * Kept as a Map of arrays indexed by player id rather than a Map keyed by
   * player, so that gathering a turn's commands walks players in a fixed
   * numeric order. Iteration order feeds straight into the simulation, and the
   * simulation must not care which packet arrived first.
   */
  private readonly buffer = new Map<number, (Command[] | null)[]>();

  /** Local checksums by tick, kept until a peer confirms or contradicts them. */
  private readonly checksums = new Map<number, number>();

  /**
   * Turns we have already broadcast for, as a contiguous prefix `[0, sentThrough]`.
   *
   * Contiguity is the whole safety argument for a delay that moves. A peer
   * blocks forever on a turn nobody ever sends, so the prefix may be *extended*
   * — by several turns at once when the delay grows — but never left with a
   * hole, and never revisited: a turn already sent may sit in a peer that has
   * already executed it, so superseding it would desync rather than correct.
   *
   * Starts at the last seeded turn, since those are filled locally by every
   * peer instead of being sent.
   */
  private sentThrough = INPUT_DELAY_TURNS - 1;

  /** Turns between issuing a command and executing it. `sentThrough` follows it. */
  private delayTurns = INPUT_DELAY_TURNS;

  /** Newest turn each peer has been seen scheduling, indexed by player. */
  private readonly peerHighWater: number[] = [];

  /** Smallest headroom we have observed from remotes since our last packet. */
  private observedHeadroom = Number.POSITIVE_INFINITY;
  /** Smallest headroom a remote has reported *for us* since the last decision. */
  private reportedHeadroom = Number.POSITIVE_INFINITY;
  private lastAdaptTurn = 0;

  private stalledSinceMs = 0;
  private lastResendMs = 0;
  private lastStallReport: PlayerId[] = [];
  /** True once the current stall has been reported to the UI. */
  private stallReported = false;

  /**
   * How long the current stall has lasted, in milliseconds. Zero when running.
   * Exposed so the UI can show a progressive indicator rather than a hard flip.
   */
  get stalledForMs(): number {
    if (this.state !== 'stalled' || this.stalledSinceMs === 0) return 0;
    return this.now() - this.stalledSinceMs;
  }

  constructor(
    private readonly simulation: Simulation,
    private readonly transport: Transport,
    private readonly events: LockstepEvents = {},
    /** Wall-clock source, injectable so tests can drive time explicitly. */
    private readonly now: () => number = () => Date.now(),
  ) {
    transport.onPacket((p) => this.receive(p));
    transport.onPeerLost((p) => this.events.onPeerTimeout?.(p));

    // The first few turns come up for execution before anyone has had a chance
    // to broadcast for them, so seed them as *received and empty* for every
    // player. Note this fills each slot with `[]`, not `null` — a null slot
    // means "still waiting", so seeding with nulls would stall every peer on
    // turn 0 forever. Every peer does this identically, so they agree.
    for (let t = 0; t < INPUT_DELAY_TURNS; t++) {
      const slots = new Array<Command[] | null>(transport.playerCount);
      for (let p = 0; p < slots.length; p++) slots[p] = [];
      this.buffer.set(t, slots);
    }
  }

  /** Queue a command for the local player. Executed INPUT_DELAY_TURNS later. */
  issue(command: Command): void {
    this.pendingLocal.push(command);
  }

  /** Current simulation tick. */
  get currentTick(): number {
    return this.tick;
  }

  /**
   * Advance real time. Call once per rendered frame.
   *
   * Returns the interpolation alpha for rendering: how far between the last two
   * simulated ticks we currently are, clamped to [0,1]. The clamp matters — when
   * the game stalls waiting for a peer, an unclamped alpha would keep
   * extrapolating units forward, sliding them through walls until the packet
   * arrives.
   */
  update(deltaMs: number): number {
    if (this.state === 'ended' || this.state === 'desynced') return 0;

    // Cap the catch-up burst. After a long pause (tab backgrounded, breakpoint)
    // an uncapped accumulator would try to simulate minutes of game in one
    // frame and freeze the browser.
    this.accumulatorMs = Math.min(this.accumulatorMs + deltaMs, MS_PER_TICK * 10);

    while (this.accumulatorMs >= MS_PER_TICK) {
      if (!this.stepOnce()) break; // stalled; keep the time for later
      this.accumulatorMs -= MS_PER_TICK;
    }

    const alpha = this.accumulatorMs / MS_PER_TICK;
    return alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
  }

  /**
   * Try to advance exactly one tick. Returns false if blocked on a peer.
   */
  private stepOnce(): boolean {
    const isTurnStart = this.tick % TICKS_PER_TURN === 0;
    let commands: Command[] = [];

    if (isTurnStart) {
      const turn = this.tick / TICKS_PER_TURN;
      const slots = this.buffer.get(turn);

      if (!slots || !this.isComplete(slots)) {
        this.enterStall(turn, slots);
        return false;
      }

      if (this.state === 'stalled') {
        this.state = 'running';
        this.stalledSinceMs = 0;
        // Only tell the UI it is over if we ever told it there was a problem.
        if (this.stallReported) {
          this.stallReported = false;
          this.events.onResume?.();
        }
      }

      commands = this.gather(slots);
      this.buffer.delete(turn);

      this.adaptDelay(turn);
      // Broadcast local intent for a future turn, plus redundant copies of the
      // last few, before advancing.
      this.broadcast(turn);
    }

    this.simulation.step(commands);
    this.tick++;

    if (this.tick % CHECKSUM_INTERVAL === 0) {
      this.checksums.set(this.tick, this.simulation.checksum());
      this.pruneChecksums();
    }

    return true;
  }

  private isComplete(slots: (Command[] | null)[]): boolean {
    for (let p = 0; p < slots.length; p++) {
      if (slots[p] === null) return false;
    }
    return true;
  }

  /** Flatten a turn's per-player commands in ascending player order. */
  private gather(slots: (Command[] | null)[]): Command[] {
    const out: Command[] = [];
    for (let p = 0; p < slots.length; p++) {
      const list = slots[p];
      if (!list) continue;
      for (let i = 0; i < list.length; i++) out.push(list[i]!);
    }
    return out;
  }

  private enterStall(turn: number, slots: (Command[] | null)[] | undefined): void {
    const waiting: PlayerId[] = [];
    for (let p = 0; p < this.transport.playerCount; p++) {
      if (!slots || slots[p] === null) {
        if (p !== this.transport.localPlayer) waiting.push(p);
      }
    }

    // Make sure our own commands for this turn went out — if the local player is
    // the one missing, we are stalling on ourselves, which should never happen
    // but is worth self-healing rather than deadlocking.
    if (slots && slots[this.transport.localPlayer] === null) {
      this.setTurn(turn, this.transport.localPlayer, []);
    }

    if (this.state !== 'stalled') {
      this.state = 'stalled';
      this.stalledSinceMs = this.now();
      this.lastStallReport = waiting;
    }

    const stalledForMs = this.now() - this.stalledSinceMs;

    // Surface it to the player only once it has lasted long enough to be worth
    // noticing — see STALL_REPORT_DELAY_MS.
    if (!this.stallReported && stalledForMs >= STALL_REPORT_DELAY_MS) {
      this.stallReported = true;
      this.events.onStall?.(waiting);
    }

    // Retransmit our full recent history while blocked. If we are stalled, the
    // peer is probably stalled on us too, so the packet they are missing will
    // never be re-sent by the normal path — someone has to break the tie.
    if (this.now() - this.lastResendMs >= RESEND_INTERVAL_MS) {
      this.lastResendMs = this.now();
      if (this.recentSent.length > 0) {
        const recovery: Packet = {
          player: this.transport.localPlayer,
          turns: this.recentSent.slice(),
        };
        const latest = this.latestChecksum();
        if (latest) recovery.checksum = latest;
        this.transport.send(recovery);
      }
    }

    if (stalledForMs > PEER_TIMEOUT_MS) {
      for (const p of this.lastStallReport) this.events.onPeerTimeout?.(p);
      this.stalledSinceMs = this.now(); // report at most once per timeout window
    }
  }

  /**
   * Reconsider how far ahead we schedule, from how the link is behaving.
   *
   * This is a purely local decision and needs no agreement, because the wire
   * carries the absolute turn each command belongs to — a peer files what it
   * receives at the stated turn and never infers a schedule. Two peers can run
   * different delays indefinitely without diverging by a bit; the only thing at
   * stake is how each one feels to the player holding it.
   *
   * The input is `peerHeadroom` — what the *other* peer reports about how early
   * our packets reach it. Nothing else will do. Our own stalls are a statement
   * about the other peer's schedule, and acting on them adjusts the wrong end:
   * measured at 120ms, a stall-driven rule parks one peer at the maximum delay,
   * stalling permanently, while the other sits at the minimum and never learns
   * it is the one arriving late.
   */
  private adaptDelay(turn: number): void {
    if (turn - this.lastAdaptTurn < ADAPT_INTERVAL_TURNS) return;
    this.lastAdaptTurn = turn;

    const reported = this.reportedHeadroom;
    this.reportedHeadroom = Number.POSITIVE_INFINITY;

    // Nothing heard back this window — a silent peer says nothing about our
    // schedule, so leave it where it is.
    if (!Number.isFinite(reported)) return;

    if (reported <= 0 && this.delayTurns < MAX_INPUT_DELAY_TURNS) {
      this.delayTurns++;
    } else if (reported >= HEADROOM_TO_SPEED_UP && this.delayTurns > MIN_INPUT_DELAY_TURNS) {
      this.delayTurns--;
    }
  }

  /** Turns between issuing a command and its execution. Adapts during play. */
  get inputDelayTurns(): number {
    return this.delayTurns;
  }

  /**
   * Extend our schedule past `turn`, and send it with recent history attached.
   *
   * How far past is `delayTurns`, so this covers three cases with one rule:
   * normally it adds the single next turn; after the delay grows it adds
   * several at once to keep the prefix contiguous; and after the delay shrinks
   * it adds none at all, letting `turn` catch up to a schedule already sent.
   *
   * A packet still goes out in that last case. Empty turns are the heartbeat
   * that carries redundant copies and the checksum, and going quiet because we
   * had nothing new to schedule would drop both.
   */
  private broadcast(turn: number): void {
    const target = turn + this.delayTurns;
    const first = this.sentThrough + 1;

    for (let t = first; t <= target; t++) {
      // Commands ride the first turn we open — the one we would have sent
      // anyway. Turns beyond it are pure headroom and go out empty.
      const mine = t === first ? this.pendingLocal : [];
      this.recentSent.push({
        turn: t,
        player: this.transport.localPlayer,
        commands: mine,
      });
      while (this.recentSent.length > SEND_HISTORY_TURNS) this.recentSent.shift();

      // Record our own commands locally too — we are a peer like any other, and
      // the turn cannot execute until every slot including ours is filled.
      this.setTurn(t, this.transport.localPlayer, mine);
    }

    if (target >= first) {
      // Only once something was actually scheduled: if the delay just shrank
      // and we opened nothing, the commands must stay pending rather than
      // vanish into a turn we never sent.
      this.pendingLocal = [];
      this.sentThrough = target;
    }

    // Steady-state packets carry only a few turns; the rest of the history is
    // held back for stall recovery so normal packets stay small.
    const packet: Packet = {
      player: this.transport.localPlayer,
      turns: this.recentSent.slice(-(REDUNDANT_TURNS + 1)),
    };

    // Attach the most recent checksum so peers can compare. Sent on the same
    // channel as commands so it cannot arrive out of band.
    const latest = this.latestChecksum();
    if (latest) packet.checksum = latest;

    // Tell the peer how early its packets have been reaching us, which is the
    // only way it can learn whether *its* schedule is tight enough.
    if (Number.isFinite(this.observedHeadroom)) {
      packet.peerHeadroom = this.observedHeadroom;
      this.observedHeadroom = Number.POSITIVE_INFINITY;
    }

    this.transport.send(packet);
  }

  private readonly recentSent: TurnCommands[] = [];

  private latestChecksum(): { tick: number; value: number } | undefined {
    let bestTick = -1;
    for (const t of this.checksums.keys()) {
      if (t > bestTick) bestTick = t;
    }
    if (bestTick < 0) return undefined;
    return { tick: bestTick, value: this.checksums.get(bestTick)! };
  }

  private receive(packet: Packet): void {
    if (packet.player !== this.transport.localPlayer) {
      // How much margin the peer's schedule is arriving with, in turns.
      //
      // Measured against a high-water mark rather than per packet, because the
      // channel is deliberately unordered and every packet repeats the previous
      // turns: a reordered or redundant copy carries old turn numbers and would
      // read as a peer falling behind, which would ratchet the delay up on a
      // link that is in fact perfectly healthy. Only genuine forward progress
      // is a measurement.
      let newest = -1;
      for (const entry of packet.turns) {
        if (entry.turn > newest) newest = entry.turn;
      }
      const seen = this.peerHighWater[packet.player] ?? -1;
      if (newest > seen) {
        this.peerHighWater[packet.player] = newest;
        const headroom = newest - Math.floor(this.tick / TICKS_PER_TURN);
        if (headroom < this.observedHeadroom) this.observedHeadroom = headroom;
      }

      // And what they say about ours, which is the only thing we can act on.
      if (packet.peerHeadroom !== undefined && packet.peerHeadroom < this.reportedHeadroom) {
        this.reportedHeadroom = packet.peerHeadroom;
      }
    }

    for (const entry of packet.turns) {
      // Turns already executed are dropped; redundant copies of them are the
      // normal case, not an error.
      if (entry.turn * TICKS_PER_TURN < this.tick) continue;
      this.setTurn(entry.turn, entry.player, entry.commands);
    }

    if (packet.checksum) this.compareChecksum(packet.player, packet.checksum);
  }

  /**
   * Record a player's commands for a turn.
   *
   * First write wins. A redundant resend carries the same data, so ignoring
   * duplicates is free; but if a duplicate ever *differed*, overwriting would
   * apply different commands than a peer that saw only the first copy.
   */
  private setTurn(turn: number, player: PlayerId, commands: Command[]): void {
    let slots = this.buffer.get(turn);
    if (!slots) {
      slots = this.emptyTurn();
      this.buffer.set(turn, slots);
    }
    if (slots[player] !== null) return;
    // Stamp the issuing player server-side, so a peer cannot move someone
    // else's units by lying in the payload.
    for (const c of commands) c.player = player;
    slots[player] = commands;
  }

  private emptyTurn(): (Command[] | null)[] {
    const slots = new Array<Command[] | null>(this.transport.playerCount);
    for (let p = 0; p < slots.length; p++) slots[p] = null;
    return slots;
  }

  /**
   * Compare a peer's checksum against ours for the same tick.
   *
   * A mismatch means the two simulations have already diverged and every frame
   * from here is fiction. There is no recovery in a peer-to-peer model — no peer
   * has authority to declare its version correct — so the honest thing is to
   * stop and say so rather than let two players keep playing different games.
   */
  private compareChecksum(player: PlayerId, remote: { tick: number; value: number }): void {
    const local = this.checksums.get(remote.tick);
    if (local === undefined) return; // not there yet, or already pruned
    if (local === remote.value) return;

    this.state = 'desynced';
    this.events.onDesync?.(remote.tick, local, remote.value, player);
    // eslint-disable-next-line no-console
    console.error(
      `desync at tick ${remote.tick}: local ${checksumToHex(local)} != ` +
        `player ${player} ${checksumToHex(remote.value)}`,
    );
  }

  /** Keep the checksum history bounded; peers only ever compare recent ticks. */
  private pruneChecksums(): void {
    const cutoff = this.tick - CHECKSUM_INTERVAL * 10;
    for (const t of this.checksums.keys()) {
      if (t < cutoff) this.checksums.delete(t);
    }
  }

  /** Latest local checksum, for tests and the debug overlay. */
  checksumAt(tick: number): number | undefined {
    return this.checksums.get(tick);
  }

  end(): void {
    this.state = 'ended';
    this.transport.close();
  }
}
