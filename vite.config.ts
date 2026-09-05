import { defineConfig } from 'vitest/config';

/**
 * GitHub Pages serves a project site from a subdirectory —
 * `https://<user>.github.io/<repo>/` — so the built asset URLs have to be
 * prefixed with that path or every script and stylesheet 404s. The deploy
 * workflow passes the correct prefix in `BASE_PATH`; the fallback keeps a manual
 * `npm run build` working too.
 *
 * The dev server stays at `/`, because serving it from a subdirectory locally is
 * pure friction for no benefit. `vite preview` serves a finished build, whose
 * asset URLs already carry the prefix, so it is served at the prefix too.
 */
function basePath(isBuild: boolean): string {
  if (!isBuild) return '/';
  const fromEnv = process.env.BASE_PATH;
  if (!fromEnv) return '/experiment-rts/';
  return fromEnv.endsWith('/') ? fromEnv : `${fromEnv}/`;
}

export default defineConfig(({ command, isPreview }) => ({
  // `vite preview` serves the build, so it has to serve it at the build's base.
  base: basePath(command === 'build' || isPreview === true),
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  // The neural bot's inference worker is an ES module worker, and ONNX Runtime
  // Web resolves its own WebAssembly binary at runtime, so it is kept out of
  // the dependency pre-bundle and handed the binary's URL explicitly.
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    // The determinism tests simulate thousands of ticks twice over and compare
    // every one. That is genuinely heavy work, not a hang, and shortening it to
    // fit a 5s default would cost exactly the coverage that makes it valuable.
    testTimeout: 60000,
  },
}));
