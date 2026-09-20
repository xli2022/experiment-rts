import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadNeuralRuntime, probeNeuralModel } from '../src/ai/neural/browser.js';
import { SPEC } from '../src/ai/neural/spec.js';
import { MapLayout } from '../src/sim/types.js';

afterEach(() => vi.unstubAllGlobals());

describe('neural model availability', () => {
  it.each([1, 2])(
    'keeps codec %s models unavailable without creating a worker',
    async (specVersion) => {
      const manifest = { specVersion, model: 'policy-lanes.onnx' };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifest }));
      const worker = vi.fn();
      vi.stubGlobal('Worker', worker);
      expect(await probeNeuralModel(MapLayout.Lanes)).toBeNull();
      await expect(loadNeuralRuntime(MapLayout.Lanes)).rejects.toThrow(/older codec/);
      expect(worker).not.toHaveBeenCalled();
    },
  );

  it('offers a model trained for the current production and upgrade vocabulary', async () => {
    const manifest = { specVersion: SPEC.version, model: 'policy-quarters.onnx' };
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => manifest });
    vi.stubGlobal('fetch', fetch);
    expect(await probeNeuralModel(MapLayout.Quarters)).toEqual(manifest);
    expect(fetch).toHaveBeenCalledWith('/models/policy-quarters.json', { cache: 'no-cache' });
  });
});
