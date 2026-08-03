import { defineConfig } from 'vitest/config';

/**
 * GitHub Pages serves a project site from a subdirectory —
 * `https://<user>.github.io/<repo>/` — so the built asset URLs have to be
 * prefixed with that path or every script and stylesheet 404s. The deploy
 * workflow passes the correct prefix in `BASE_PATH`; the fallback keeps a manual
 * `npm run build` working too.
 *
 * The dev server stays at `/`, because serving it from a subdirectory locally is
 * pure friction for no benefit.
 */
function basePath(isBuild: boolean): string {
  if (!isBuild) return '/';
  const fromEnv = process.env.BASE_PATH;
  if (!fromEnv) return '/experiment-rts/';
  return fromEnv.endsWith('/') ? fromEnv : `${fromEnv}/`;
}

export default defineConfig(({ command }) => ({
  base: basePath(command === 'build'),
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    // The determinism tests simulate thousands of ticks twice over and compare
    // every one. That is genuinely heavy work, not a hang, and shortening it to
    // fit a 5s default would cost exactly the coverage that makes it valuable.
    testTimeout: 60000,
  },
}));
