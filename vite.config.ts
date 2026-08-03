import { defineConfig } from 'vitest/config';

export default defineConfig({
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
});
