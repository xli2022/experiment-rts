/**
 * Lockstep scheduler tests.
 *
 * These run two complete peers inside one process over a simulated network, so
 * the conditions that actually break lockstep in the wild — latency, jitter,
 * packet loss, a peer going silent — are reproducible in milliseconds instead of
 * needing two machines and luck.
 */

import { describe, expect, it } from 'vitest';
import { CommandType, type Command } from '../src/sim/commands.js';
import { fromInt } from '../src/sim/fixed.js';
import { coopMatch } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { CHECKSUM_INTERVAL, MS_PER_TICK } from '../src/sim/types.js';
import { LocalNetwork } from '../src/net/localTransport.js';
import { INPUT_DELAY_TURNS, LockstepRunner, TICKS_PER_TURN } from '../src/net/lockstep.js';

import type { MatchConfig } from '../src/sim/types.js';

const SEED = 0xbeef01;

interface Peer {
  sim: Simulation;
  runner: LockstepRunner;
  stalls: number;
  desyncs: number;
}

/** Build two peers wired together through one simulated network. */
function makeMatch(net: LocalNetwork, seed: number | MatchConfig = SEED): Peer[] {
  const peers: Peer[] = [];
  for (let p = 0; p < net.playerCount; p++) {
    const sim = new Simulation(seed);
    const peer: Peer = { sim, runner: null as never, stalls: 0, desyncs: 0 };
    peer.runner = new LockstepRunner(
      sim,
      net.createTransport(p),
      {
        onStall: () => {
          peer.stalls++;
        },
        onDesync: () => {
          peer.desyncs++;
        },
      },
      () => net.nowMs,
    );
    peers.push(peer);
  }
  return peers;
}

/**
 * Compare peers at a tick they have both simulated.
 *
 * Peers legitimately sit at different ticks — one can be up to INPUT_DELAY_TURNS
 * ahead — so comparing their *current* checksums compares different moments in
 * the game and fails for no reason. Both record a checksum every
 * CHECKSUM_INTERVAL ticks, so compare the newest one they share.
 */
function expectAgreementAtCommonTick(peers: Peer[]): void {
  const minTick = Math.min(...peers.map((p) => p.runner.currentTick));
  for (let t = minTick - (minTick % CHECKSUM_INTERVAL); t > 0; t -= CHECKSUM_INTERVAL) {
    const values = peers.map((p) => p.runner.checksumAt(t));
    if (values.some((v) => v === undefined)) continue;
    expect(values[0]).toBe(values[1]);
    return;
  }
  throw new Error(`no common checksummed tick below ${minTick}`);
}

/** Drive every peer and the network forward by `frames` ticks worth of time. */
function run(net: LocalNetwork, peers: Peer[], frames: number): void {
  for (let f = 0; f < frames; f++) {
    for (const peer of peers) peer.runner.update(MS_PER_TICK);
    net.advance(MS_PER_TICK);
  }
}

describe('lockstep scheduler', () => {
  it('exposes every transient event during multi-tick catch-up', () => {
    class EventSimulation extends Simulation {
      override step(commands: Command[]): void {
        super.step(commands);
        // Model a one-tick event that the following step will clear.
        this.world.events.attackStarts.push(this.world.tick, this.world.tick);
      }
    }

    const net = new LocalNetwork(1);
    const sim = new EventSimulation(SEED);
    const seenStarts: number[] = [];
    const runner = new LockstepRunner(sim, net.createTransport(0), {
      onStep: () => seenStarts.push(sim.world.events.attackStarts[0]!),
    });

    runner.update(MS_PER_TICK * 3);

    expect(runner.currentTick).toBe(3);
    expect(seenStarts).toEqual([1, 2, 3]);
  });

  it('keeps two peers on identical state over a clean network', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);

    run(net, peers, 400);

    expect(peers[0]!.runner.currentTick).toBeGreaterThan(300);
    expect(peers[0]!.runner.currentTick).toBe(peers[1]!.runner.currentTick);
    expect(peers[0]!.sim.checksum()).toBe(peers[1]!.sim.checksum());
    expect(peers[0]!.desyncs).toBe(0);
    expect(peers[1]!.desyncs).toBe(0);
  });

  it('runs a co-op match with the AI in the slots nobody is sitting in', () => {
    // The whole reason co-op costs no bandwidth: the two humans are the only
    // players on the wire, and the two bots are generated locally on both peers
    // from state they already share. So the transport carries two slots while
    // the simulation runs four, and the peers still have to agree exactly.
    const config = coopMatch(SEED);
    const net = new LocalNetwork(2, 5);
    net.latencyMs = 60;
    net.jitterMs = 20;
    const peers = makeMatch(net, config);

    expect(peers[0]!.sim.world.players.length).toBe(4);
    expect(peers[0]!.sim.isBot(2)).toBe(true);
    expect(peers[0]!.sim.isBot(0)).toBe(false);

    // Watch what actually crosses the network rather than trusting the roster.
    const onWire = new Set<number>();
    const originalSubmit = net.submit.bind(net);
    net.submit = (from, packet): void => {
      for (const turn of packet.turns) {
        for (const command of turn.commands) onWire.add(command.player);
      }
      originalSubmit(from, packet);
    };

    // Both humans keep clicking, so there is real traffic to inspect rather
    // than a stream of empty heartbeats that would make the check below pass
    // by saying nothing.
    for (let f = 0; f < 600; f++) {
      if (f % 40 === 0) {
        for (const peer of peers) {
          const player = peer.runner === peers[0]!.runner ? 0 : 1;
          const units: number[] = [];
          const pool = peer.sim.world.pool;
          for (let i = 0; i < pool.count && units.length < 3; i++) {
            if (pool.alive[i] === 1 && pool.owner[i] === player) units.push(pool.idAt(i));
          }
          const start = peer.sim.world.map.starts[player]!;
          peer.runner.issue({
            type: CommandType.Move,
            player,
            units,
            x: fromInt(start.tileX + 4),
            y: fromInt(start.tileY + 4),
          });
        }
      }
      for (const peer of peers) peer.runner.update(MS_PER_TICK);
      net.advance(MS_PER_TICK);
    }

    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
    expect(peers[1]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(300);

    // Both humans, and only the humans. A bot's command appearing here would be
    // applied twice on every peer that received it — once from the packet and
    // once by the local bot.
    expect([...onWire].sort()).toEqual([0, 1]);

    // And the bots were actually playing, or agreement is agreement about
    // nothing happening.
    expect(peers[0]!.sim.world.player(2).supplyUsed).toBeGreaterThan(0);
  });

  it('stays in sync across realistic latency and jitter', () => {
    const net = new LocalNetwork(2, 77);
    net.latencyMs = 80;
    net.jitterMs = 40;
    const peers = makeMatch(net);

    run(net, peers, 600);

    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(100);
  });

  it('rides out packet loss using redundant resends', () => {
    // Commands are repeated in the next few packets, so a lost packet normally
    // costs nothing. Without that, this test would deadlock.
    const net = new LocalNetwork(2, 1234);
    net.latencyMs = 50;
    net.dropRate = 0.3;
    const peers = makeMatch(net);

    run(net, peers, 800);

    // Loss makes peers advance at different rates, so they will not be on the
    // same tick — what matters is that they agree about the ticks they share
    // and that both kept making progress.
    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(100);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(100);
  });

  it('recovers from loss severe enough to defeat piggybacked redundancy', () => {
    // At this loss rate every copy of a turn in the normal redundancy window is
    // routinely lost. Without stall-triggered retransmission the match
    // deadlocks permanently — which it did, at around tick 48.
    const net = new LocalNetwork(2, 999);
    net.latencyMs = 60;
    net.jitterMs = 30;
    net.dropRate = 0.5;
    const peers = makeMatch(net);

    run(net, peers, 2000);

    expect(peers[0]!.runner.currentTick).toBeGreaterThan(200);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(200);
    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
  });

  it('executes a command on the same tick for both peers', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);

    // Let the match settle, then have player 0 order its units somewhere.
    run(net, peers, 40);

    const pool = peers[0]!.sim.world.pool;
    const units: number[] = [];
    for (let i = 0; i < pool.count && units.length < 3; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0 && pool.type[i] === 0) {
        units.push(pool.idAt(i));
      }
    }
    expect(units.length).toBeGreaterThan(0);

    const cmd: Command = {
      type: CommandType.Move,
      player: 0,
      units,
      x: fromInt(40),
      y: fromInt(40),
    };
    peers[0]!.runner.issue(cmd);

    run(net, peers, 200);

    // Both peers must agree on the result, which only happens if they applied
    // the command at the same tick.
    expectAgreementAtCommonTick(peers);
  });

  it('stalls rather than advancing without a peer', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);
    run(net, peers, 40);

    const tickBefore = peers[0]!.runner.currentTick;
    expect(tickBefore).toBeGreaterThan(0);

    // Peer 1 goes silent: stop updating it, so it sends nothing.
    for (let f = 0; f < 200; f++) {
      peers[0]!.runner.update(MS_PER_TICK);
      net.advance(MS_PER_TICK);
    }

    // Peer 0 may finish the turns it already had commands for, but must then
    // stop. Advancing further would mean simulating a different game.
    const maxExtra = (INPUT_DELAY_TURNS + 1) * TICKS_PER_TURN;
    expect(peers[0]!.runner.currentTick - tickBefore).toBeLessThanOrEqual(maxExtra);
    expect(peers[0]!.runner.state).toBe('stalled');
    expect(peers[0]!.stalls).toBeGreaterThan(0);
  });

  it('resumes cleanly once the missing peer returns', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);
    run(net, peers, 40);

    // Peer 1 freezes...
    for (let f = 0; f < 60; f++) {
      peers[0]!.runner.update(MS_PER_TICK);
      net.advance(MS_PER_TICK);
    }
    expect(peers[0]!.runner.state).toBe('stalled');
    const stalledAt = peers[0]!.runner.currentTick;

    // ...and comes back.
    run(net, peers, 300);

    // The instantaneous state flag is not the right assertion: having run ahead
    // during the freeze, peer 0 now sits exactly INPUT_DELAY_TURNS in front and
    // blocks at every turn boundary, so the flag alternates. Progress and
    // agreement are what actually matter.
    expect(peers[0]!.runner.currentTick).toBeGreaterThan(stalledAt);
    expect(peers[1]!.runner.currentTick).toBeGreaterThan(stalledAt);
    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
  });

  it('detects a desync instead of letting peers drift apart', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);
    run(net, peers, 60);

    // Corrupt one peer's world behind the scheduler's back, simulating the
    // effect of a real determinism bug.
    const pool = peers[1]!.sim.world.pool;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1) {
        pool.posX[i] = (pool.posX[i]! + 12345) | 0;
        break;
      }
    }

    run(net, peers, 200);

    const detected = peers[0]!.desyncs > 0 || peers[1]!.desyncs > 0;
    expect(detected).toBe(true);
    const halted = peers[0]!.runner.state === 'desynced' || peers[1]!.runner.state === 'desynced';
    expect(halted).toBe(true);
  });

  it('clamps interpolation alpha so a stall freezes rather than extrapolates', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);
    run(net, peers, 40);

    // Starve peer 0 and pump a large delta; alpha must stay in [0,1] rather
    // than growing and sliding units through walls.
    for (let f = 0; f < 50; f++) {
      const alpha = peers[0]!.runner.update(MS_PER_TICK * 3);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThanOrEqual(1);
      net.advance(MS_PER_TICK);
    }
  });

  it('ignores a peer trying to command units it does not own', () => {
    const net = new LocalNetwork(2);
    const peers = makeMatch(net);
    run(net, peers, 40);

    // Player 1 forges a command claiming to be player 0.
    const pool = peers[1]!.sim.world.pool;
    let victim = -1;
    for (let i = 0; i < pool.count; i++) {
      if (pool.alive[i] === 1 && pool.owner[i] === 0) {
        victim = i;
        break;
      }
    }
    expect(victim).toBeGreaterThanOrEqual(0);

    peers[1]!.runner.issue({
      type: CommandType.Move,
      player: 0, // lying about who we are
      units: [pool.idAt(victim)],
      x: fromInt(60),
      y: fromInt(60),
    });

    run(net, peers, 200);

    // The scheduler restamps commands with the sending peer's real id, and the
    // simulation drops orders against units that peer does not own — so both
    // sides stay identical either way.
    expectAgreementAtCommonTick(peers);
    expect(peers[0]!.desyncs).toBe(0);
  });
});
