/**
 * Bot against bot, headless, from both seats.
 *
 *   npm run ml:arena -- --a scripted@10 --b scripted@20 --seeds 8 --layout lanes
 *
 * Every pairing is played from both seats on every seed: the simulation is
 * rotation-equivariant, so a seat-swapped match should be the same match with
 * the winner swapped, and a pairing that resolves differently from the two
 * seats is a bug report, not a result.
 */

import type { Agent } from '../../src/ai/agent.js';
import { HeadlessMatch } from '../../src/ai/headless.js';
import { ScriptedAgent } from '../../src/ai/scripted.js';
import { matchConfig } from '../../src/sim/match.js';
import { MapLayout, NO_ENTITY } from '../../src/sim/types.js';
import { parseLayout } from './env.js';

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2)
  args.set(process.argv[i]!.replace(/^--/, ''), process.argv[i + 1] ?? '');

function agentFor(spec: string): () => Agent {
  const [kind, arg] = spec.split('@');
  if (kind !== 'scripted') throw new Error(`arena plays scripted bots only, got ${spec}`);
  const thinkInterval = arg === undefined ? undefined : Number(arg);
  return () => new ScriptedAgent(thinkInterval === undefined ? {} : { thinkInterval });
}

const a = agentFor(args.get('a') ?? 'scripted');
const b = agentFor(args.get('b') ?? 'scripted@20');
const layout = parseLayout(args.get('layout') ?? 'lanes');
const seeds = Number(args.get('seeds') ?? 8);
const maxTicks = Number(args.get('maxTicks') ?? 36000);
const perSide = layout === MapLayout.Quarters ? 2 : 1;

let winsA = 0;
let winsB = 0;
let draws = 0;
let mismatches = 0;
let totalTicks = 0;
for (let s = 0; s < seeds; s++) {
  const seed = (0x51ce7a11 + s * 7919) >>> 0;
  const results: { winner: number; ticks: number }[] = [];
  for (const aFirst of [true, false]) {
    const config = matchConfig(layout, seed, { botPlayers: [...Array(perSide * 2).keys()] });
    const agents: [number, Agent][] = [];
    for (let p = 0; p < perSide * 2; p++) {
      const first = p < perSide;
      agents.push([p, first === aFirst ? a() : b()]);
    }
    const match = new HeadlessMatch(config, agents);
    match.run(maxTicks);
    const world = match.world;
    const winner = world.matchOver ? world.winner : NO_ENTITY;
    const aTeam = aFirst ? 0 : 1;
    results.push({ winner, ticks: world.tick });
    totalTicks += world.tick;
    if (winner === NO_ENTITY) draws++;
    else if (winner === aTeam) winsA++;
    else winsB++;
  }
  const [x, y] = results as [{ winner: number; ticks: number }, { winner: number; ticks: number }];
  const swapped = x.winner === NO_ENTITY ? y.winner === NO_ENTITY : y.winner === 1 - x.winner;
  if (!swapped || x.ticks !== y.ticks) mismatches++;
}
console.log(
  `${args.get('a') ?? 'scripted'} vs ${args.get('b') ?? 'scripted@20'} on ${args.get('layout') ?? 'lanes'}: ` +
    `${winsA}-${winsB}-${draws} over ${seeds} seeds × 2 seats, mean ${(totalTicks / (seeds * 2)).toFixed(0)} ticks` +
    (mismatches > 0 ? `, ${mismatches} seat-swap mismatches` : ', seats agree'),
);
