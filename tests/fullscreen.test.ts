import { afterEach, expect, it, vi } from 'vitest';
import { toggleFullscreen } from '../src/ui/fullscreen.js';

afterEach(() => vi.unstubAllGlobals());

it('reports unsupported fullscreen as inactive', async () => {
  vi.stubGlobal('document', { documentElement: {}, fullscreenElement: null });
  expect(await toggleFullscreen()).toBe(false);
});

it('selects the standard API once even when it returns void', async () => {
  const fallback = vi.fn();
  const doc = { fullscreenElement: null as object | null, documentElement: {} };
  doc.documentElement = {
    requestFullscreen() {
      doc.fullscreenElement = this;
    },
    webkitRequestFullscreen: fallback,
  };
  vi.stubGlobal('document', doc);
  expect(await toggleFullscreen()).toBe(true);
  expect(fallback).not.toHaveBeenCalled();
});

it('uses the prefixed API and reports the state actually reached', async () => {
  const doc = {
    webkitFullscreenElement: null as object | null,
    documentElement: {},
    webkitExitFullscreen() {
      this.webkitFullscreenElement = null;
    },
  };
  doc.documentElement = {
    webkitRequestFullscreen() {
      doc.webkitFullscreenElement = this;
    },
  };
  vi.stubGlobal('document', doc);
  expect(await toggleFullscreen()).toBe(true);
  expect(await toggleFullscreen()).toBe(false);
});

it('preserves active state when the browser denies exit', async () => {
  vi.stubGlobal('document', {
    documentElement: {},
    fullscreenElement: {},
    exitFullscreen: async () => {
      throw new Error('denied');
    },
  });
  expect(await toggleFullscreen()).toBe(true);
});
