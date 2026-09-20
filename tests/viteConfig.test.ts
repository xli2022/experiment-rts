import { afterEach, expect, it, vi } from 'vitest';
import type { UserConfigFnObject } from 'vite';
import config from '../vite.config.js';

const configure = config as UserConfigFnObject;
afterEach(() => vi.unstubAllEnvs());

it('uses the root when GitHub Pages reports an empty base path', () => {
  vi.stubEnv('BASE_PATH', '');
  expect(configure({ command: 'build', mode: 'production' }).base).toBe('/');
  expect(configure({ command: 'serve', mode: 'production', isPreview: true }).base).toBe('/');
});

it('keeps the repository fallback and normalizes configured project paths', () => {
  vi.stubEnv('BASE_PATH', undefined);
  expect(configure({ command: 'build', mode: 'production' }).base).toBe('/experiment-rts/');
  vi.stubEnv('BASE_PATH', '/another-project');
  expect(configure({ command: 'build', mode: 'production' }).base).toBe('/another-project/');
  expect(configure({ command: 'serve', mode: 'development' }).base).toBe('/');
});
