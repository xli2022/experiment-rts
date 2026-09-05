#!/usr/bin/env node
/**
 * The neural bot in a real browser.
 *
 *   npm run build && npm run smoke:neural [-- --seconds 30 --seed 1 --allow-random]
 *
 * Serves the build with `vite preview`, opens `?skip=neural` in headless
 * Chromium, lets the match run, and reads the hosted bot's statistics off
 * `window.__game`. It fails on any console error, on a slot that issued no
 * command, on any failed decision, or on more than 1% of decisions skipped
 * because the previous one was still in flight — the sign of a model too slow
 * for the machine. Without `--allow-random` it also fails if the build shipped
 * no model and the random stand-in played instead.
 *
 * `--max-skipped` is that fraction; `--viewport` (default 960x600) sizes the
 * page, which matters on a machine without a GPU, where software rendering
 * competes with the model for the same cores. When the page itself cannot keep
 * the simulation near 20 ticks a second the skip rate says nothing about the
 * model, so the model's own time in the worker is what is checked instead.
 *
 * Playwright is resolved from the project or from the global npm root.
 */

import { execSync, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';

const args = new Map();
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (!arg.startsWith('--')) continue;
  const next = process.argv[i + 1];
  if (next !== undefined && !next.startsWith('--')) {
    args.set(arg.slice(2), next);
    i++;
  } else {
    args.set(arg.slice(2), 'true');
  }
}
const seconds = Number(args.get('seconds') ?? 30);
const seed = Number(args.get('seed') ?? 1);
const allowRandom = args.get('allow-random') === 'true';
const port = Number(args.get('port') ?? 4173);
const maxSkipped = Number(args.get('max-skipped') ?? 0.01);
/** Below this many simulation ticks a second the page, not the model, is the bottleneck. */
const STARVED_TPS = 16;
/** A decision is four ticks; a model slower than this could never keep up. */
const MAX_MODEL_MS = 150;
const [width, height] = String(args.get('viewport') ?? '960x600')
  .split('x')
  .map(Number);
const base = args.get('base') ?? process.env.BASE_PATH ?? '/experiment-rts/';
const url = args.get('url') ?? `http://127.0.0.1:${port}${base.endsWith('/') ? base : `${base}/`}`;

function resolvePlaywright() {
  const require = createRequire(import.meta.url);
  const paths = [process.cwd()];
  try {
    paths.push(execSync('npm root -g', { encoding: 'utf8' }).trim());
  } catch {
    // no global root to look in
  }
  try {
    return require(require.resolve('playwright', { paths }));
  } catch {
    throw new Error('playwright is not installed: `npm i -D playwright` or install it globally');
  }
}

async function waitForServer(target, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(target);
      if (response.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`vite preview did not answer at ${target} within ${ms} ms`);
}

async function main() {
  if (!existsSync('dist/index.html')) throw new Error('no build: run `npm run build` first');
  const { chromium } = resolvePlaywright();
  console.log(`playwright ${chromium.name()} at ${chromium.executablePath()}`);
  // Its own process group, so that killing the preview kills the server npx
  // started under it — and so that a signal to this script takes it along.
  const preview = spawn('npx', ['vite', 'preview', '--port', String(port), '--strictPort'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  });
  const previewLog = [];
  preview.stdout.on('data', (d) => previewLog.push(String(d)));
  preview.stderr.on('data', (d) => previewLog.push(String(d)));
  const stopPreview = () => {
    try {
      process.kill(-preview.pid, 'SIGTERM');
    } catch {
      // already gone
    }
  };
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, () => {
      stopPreview();
      process.exit(130);
    });
  }
  let browser = null;
  const errors = [];
  try {
    await waitForServer(url, 20000);
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width, height } });
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(`console: ${message.text()}`);
    });
    page.on('pageerror', (error) => errors.push(`page: ${error.message}`));
    const target = `${url}?skip=neural&seed=${seed}`;
    console.log(`server up; opening ${target}`);
    await page.goto(target, { waitUntil: 'load' });
    await page.waitForFunction(() => typeof window.__game?.hostedStats === 'function', null, {
      timeout: 60000,
    });
    await page.waitForTimeout(seconds * 1000);
    const stats = await page.evaluate(() => window.__game.hostedStats());
    console.log(JSON.stringify(stats, null, 2));

    const failures = [...errors];
    const slot = stats.find((s) => s.player === 1);
    if (!slot) failures.push('no hosted slot 1');
    else {
      if (slot.kind !== 'neural' && !allowRandom) {
        failures.push(`slot 1 was played by the ${slot.kind} agent, not the model`);
      }
      if (slot.issued === 0) failures.push('slot 1 issued no command');
      if (slot.neural) {
        const n = slot.neural;
        const r = slot.runtime;
        if (n.decisions === 0) failures.push('the model was never asked');
        if (n.failed > 0) failures.push(`${n.failed} decisions failed`);
        // The simulation runs 20 ticks a second when the page keeps up. A page
        // well below that is starved — software rendering on a machine with no
        // GPU — and a decision's round trip then measures the page, not the
        // model; the model's own time inside the worker is what is gated then.
        const ticksPerSecond = slot.tick / seconds;
        const starved = ticksPerSecond < STARVED_TPS;
        if (starved) {
          console.warn(
            `page starved: ${ticksPerSecond.toFixed(1)} ticks/s of 20; gating on model time, not the skip rate`,
          );
          if (r && r.lastInferenceMs > MAX_MODEL_MS) {
            failures.push(
              `the model took ${Math.round(r.lastInferenceMs)} ms, more than ${MAX_MODEL_MS}`,
            );
          }
        } else if (n.skipped > maxSkipped * n.decisions) {
          const why = r
            ? `last decision ${r.lastLatencyMs} ms round trip, ${Math.round(r.lastInferenceMs)} ms in the model`
            : 'no runtime numbers';
          failures.push(
            `${n.skipped} of ${n.decisions} decisions skipped, more than ${maxSkipped * 100}% (${why})`,
          );
        }
      }
    }
    if (failures.length > 0) {
      console.error('FAIL');
      for (const f of failures) console.error(`  ${f}`);
      process.exitCode = 1;
    } else {
      const r = slot.runtime;
      const timing = r
        ? `; warm-up ${Math.round(r.warmupMs)} ms, last decision ${Math.round(r.lastInferenceMs)} ms in the model, ${r.lastLatencyMs} ms round trip`
        : '';
      console.log(
        `PASS — ${seconds}s of play to tick ${slot.tick}, slot 1 (${slot.kind}) issued ${slot.issued} commands${timing}`,
      );
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    for (const e of errors) console.error(`  ${e}`);
    if (previewLog.length > 0) console.error(previewLog.join(''));
    process.exitCode = 1;
  } finally {
    await browser?.close();
    stopPreview();
  }
}

await main();
