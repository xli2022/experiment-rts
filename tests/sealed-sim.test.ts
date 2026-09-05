/**
 * Guards the determinism boundary.
 *
 * The simulation is only reproducible if it never reaches for anything that
 * varies between machines. That rule is easy to state and easy to break by
 * accident months later — a `Math.random()` for a scatter effect, a `Date.now()`
 * in a debug log — and the resulting desync surfaces as an unreproducible bug
 * report from a player, not as a failing test.
 *
 * So the rule is mechanically enforced: this test reads every source file under
 * `src/sim/` and fails if a banned construct appears.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SIM_ROOT = fileURLToPath(new URL('../src/sim', import.meta.url));
const CONFIG_ROOT = fileURLToPath(new URL('../src/config', import.meta.url));
const AI_ROOT = fileURLToPath(new URL('../src/ai', import.meta.url));

/**
 * The scripted bot and everything that hosts it in a headless match.
 *
 * None of this runs inside the simulation any more — a bot is a player, hosted
 * by one peer — so nothing structural needs it to be reproducible. The
 * determinism and mirror probes do: they drive whole matches with the scripted
 * bot and expect the same answer on every run and every engine, and that is
 * only true while these files obey the same rules as the simulation.
 */
const REPRODUCIBLE_AI = [
  'agent.ts',
  'bot.ts',
  'cadence.ts',
  'driver.ts',
  'headless.ts',
  'scripted.ts',
];

/**
 * Constructs that make a simulation non-reproducible.
 *
 * `Math.sqrt` is deliberately absent — IEEE-754 requires it to be correctly
 * rounded, so it is safe and the fixed-point layer depends on it. The
 * transcendentals are not required to be correctly rounded and genuinely differ
 * between engines.
 */
const BANNED: { pattern: RegExp; why: string }[] = [
  { pattern: /\bMath\.random\b/, why: 'use the seeded Rng instead' },
  { pattern: /\bDate\.now\b/, why: 'wall-clock time is not part of the simulation' },
  { pattern: /\bnew Date\b/, why: 'wall-clock time is not part of the simulation' },
  { pattern: /\bperformance\.now\b/, why: 'wall-clock time is not part of the simulation' },
  { pattern: /\bMath\.sin\b/, why: 'not correctly rounded; store facing as a vector' },
  { pattern: /\bMath\.cos\b/, why: 'not correctly rounded; store facing as a vector' },
  { pattern: /\bMath\.tan\b/, why: 'not correctly rounded' },
  { pattern: /\bMath\.atan2?\b/, why: 'not correctly rounded; store facing as a vector' },
  { pattern: /\bMath\.pow\b/, why: 'not correctly rounded' },
  { pattern: /\bMath\.exp\b/, why: 'not correctly rounded' },
  { pattern: /\bMath\.log\b/, why: 'not correctly rounded' },
  { pattern: /\bMath\.hypot\b/, why: 'not correctly rounded; use vecLen' },
  { pattern: /\bMath\.cbrt\b/, why: 'not correctly rounded' },
  { pattern: /\bfor\s*\(\s*(?:const|let|var)\s+\w+\s+in\s/, why: 'for-in order is not guaranteed' },
  { pattern: /\bcrypto\./, why: 'non-deterministic' },
];

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir).sort()) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...sourceFiles(full));
    } else if (entry.endsWith('.ts')) {
      out.push(full);
    }
  }
  return out;
}

/** Strip comments so prose explaining a ban does not trip the ban. */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

describe('simulation determinism boundary', () => {
  const files = [
    ...sourceFiles(SIM_ROOT),
    ...sourceFiles(CONFIG_ROOT),
    ...REPRODUCIBLE_AI.map((name) => join(AI_ROOT, name)),
  ];

  it('finds the simulation sources', () => {
    expect(files.length).toBeGreaterThan(8);
  });

  it('contains no non-deterministic constructs', () => {
    const violations: string[] = [];

    for (const file of files) {
      const code = stripComments(readFileSync(file, 'utf8'));
      const lines = code.split('\n');
      for (const { pattern, why } of BANNED) {
        for (let i = 0; i < lines.length; i++) {
          if (pattern.test(lines[i]!)) {
            violations.push(
              `${file.replace(/^.*\/src\//, 'src/')}:${i + 1} matches ${pattern} — ${why}`,
            );
          }
        }
      }
    }

    expect(violations).toEqual([]);
  });

  it('never imports rendering, DOM, networking or bot code', () => {
    const violations: string[] = [];
    for (const file of sourceFiles(SIM_ROOT)) {
      const code = readFileSync(file, 'utf8');
      const imports = [...code.matchAll(/from\s+['"]([^'"]+)['"]/g)].map((m) => m[1]!);
      for (const spec of imports) {
        // The simulation may import itself and the balance tables, nothing else.
        // Not even the bots: they are players now, and a simulation that ran one
        // would be applying commands a peer never received.
        const ok = spec.startsWith('./') || spec.startsWith('../') || spec === 'node:fs';
        const reachesOutside =
          spec.includes('../render') ||
          spec.includes('../ui') ||
          spec.includes('../input') ||
          spec.includes('../net') ||
          spec.includes('../ai');
        if (!ok || reachesOutside) {
          violations.push(`${file.replace(/^.*\/src\//, 'src/')} imports ${spec}`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it('keeps the reproducible bot code away from the browser and the neural bot', () => {
    // The neural bot samples; the renderer and the UI hold per-peer state. A
    // scripted-bot file that reached either would stop being a pure function
    // of the world, and the probes would start disagreeing with themselves.
    const violations: string[] = [];
    for (const name of REPRODUCIBLE_AI) {
      const file = join(AI_ROOT, name);
      const code = readFileSync(file, 'utf8');
      const imports = [...code.matchAll(/from\s+['"]([^'"]+)['"]/g)].map((m) => m[1]!);
      for (const spec of imports) {
        const forbidden =
          spec.includes('neural') ||
          spec.includes('../vision') ||
          spec.includes('../render') ||
          spec.includes('../ui') ||
          spec.includes('../input');
        if (forbidden) violations.push(`src/ai/${name} imports ${spec}`);
      }
    }
    expect(violations).toEqual([]);
  });

  it('detects a planted violation', () => {
    // Proves the scanner actually works, rather than passing because the regexes
    // never match anything.
    const planted = 'const x = Math.random();';
    const hit = BANNED.some(({ pattern }) => pattern.test(planted));
    expect(hit).toBe(true);
  });
});
