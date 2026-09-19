/**
 * Re-record the scripted bot fixtures under `tests/fixtures/`.
 *
 * The fixtures are a golden master: every command the bot decided on every
 * think of two whole matches, plus the entity pool's checksum after every tick.
 * `tests/agent.test.ts` replays them and asks the live bot the same questions on
 * the same worlds, which pins both the bot's decisions and the simulation's
 * behaviour to the byte.
 *
 * Being a golden master, it is invalidated by any deliberate change to either.
 * A recorded think is only comparable while the replayed world still matches
 * the world it was recorded on, so a change that moves units differently makes
 * the tail of the recording compare a live bot against a question it was never
 * asked. Run this when that happens — and only then, having satisfied yourself
 * that the behaviour change was the one you meant to make, because re-recording
 * is exactly as good at blessing a bug as it is at blessing a fix.
 *
 *     npx vite-node scripts/record-bot-fixtures.ts
 *
 * The original fixtures were recorded by hand from a build that predates this
 * script; it exists so the next person does not have to reconstruct how.
 */

import { writeFileSync } from 'node:fs';
import { botThink, THINK_INTERVAL } from '../src/ai/bot.js';
import { checksumInit } from '../src/sim/checksum.js';
import type { Command } from '../src/sim/commands.js';
import { matchConfig } from '../src/sim/match.js';
import { Simulation } from '../src/sim/tick.js';
import { MapLayout } from '../src/sim/types.js';

interface Recipe {
  name: string;
  layout: MapLayout;
  seed: number;
  players: number;
  ticks: number;
}

/** The two matches the fixtures cover, unchanged from the original recording. */
const RECIPES: readonly Recipe[] = [
  { name: 'bot-hard-duel', layout: MapLayout.Lanes, seed: 1372486161, players: 2, ticks: 4000 },
  {
    name: 'bot-hard-quarters',
    layout: MapLayout.Quarters,
    seed: 305441741,
    players: 4,
    ticks: 2500,
  },
];

function record(recipe: Recipe): void {
  const players = Array.from({ length: recipe.players }, (_, p) => p);
  const sim = new Simulation(matchConfig(recipe.layout, recipe.seed, { botPlayers: players }));
  const world = sim.world;

  const thinks: { tick: number; player: number; commands: Command[] }[] = [];
  const poolChecksums: number[] = [];

  for (let t = 0; t < recipe.ticks; t++) {
    const due: Command[] = [];
    if (t % THINK_INTERVAL === 0) {
      for (const p of players) {
        // What the bot decides, before any chunking — the same question
        // `agent.test.ts` asks on replay.
        const commands = botThink(world, p);
        thinks.push({ tick: t, player: p, commands: structuredClone(commands) });
        for (const c of structuredClone(commands)) due.push(c);
      }
    }
    sim.step(due);
    poolChecksums.push(world.pool.checksum(checksumInit()) >>> 0);
  }

  const out = {
    name: recipe.name,
    layout: recipe.layout,
    seed: recipe.seed,
    players: recipe.players,
    ticks: recipe.ticks,
    thinks,
    poolChecksums,
  };
  const path = new URL(`../tests/fixtures/${recipe.name}.json`, import.meta.url);
  writeFileSync(path, JSON.stringify(out));
  console.log(
    `${recipe.name}: ${thinks.length} thinks, ${poolChecksums.length} ticks -> ${path.pathname}`,
  );
}

for (const recipe of RECIPES) record(recipe);
