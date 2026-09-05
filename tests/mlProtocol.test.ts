/**
 * The pipe to Python: frames survive any split of the byte stream, and the
 * arrays come back with their types and shapes.
 */

import { describe, expect, it } from 'vitest';
import { decodeFrame, encodeFrame, FrameParser, Kind } from '../tools/ml/protocol.js';

describe('the frame protocol', () => {
  it('round-trips a header and typed arrays', () => {
    const f32 = new Float32Array([1.5, -2, 3.25]);
    const u8 = new Uint8Array([1, 0, 1, 1]);
    const i32 = new Int32Array([-1, 7, 123456]);
    const bytes = encodeFrame(Kind.Obs, { tick: 42, envs: [{ done: false }] }, [
      { name: 'a', view: f32, shape: [3] },
      { name: 'b', view: u8, shape: [2, 2] },
      { name: 'c', view: i32 },
    ]);
    const frame = decodeFrame<{ tick: number; envs: { done: boolean }[] }>(bytes);
    expect(frame.kind).toBe(Kind.Obs);
    expect(frame.header.tick).toBe(42);
    expect(frame.header.envs).toEqual([{ done: false }]);
    expect(frame.header.arrays).toEqual([
      { name: 'a', dtype: 'f32', shape: [3] },
      { name: 'b', dtype: 'u8', shape: [2, 2] },
      { name: 'c', dtype: 'i32', shape: [3] },
    ]);
    expect(frame.arrays[0]).toBeInstanceOf(Float32Array);
    expect([...(frame.arrays[0] as Float32Array)]).toEqual([1.5, -2, 3.25]);
    expect([...(frame.arrays[1] as Uint8Array)]).toEqual([1, 0, 1, 1]);
    expect([...(frame.arrays[2] as Int32Array)]).toEqual([-1, 7, 123456]);
  });

  it('reassembles frames from a stream split anywhere', () => {
    const frames = [
      encodeFrame(Kind.Hello, { n: 1 }),
      encodeFrame(Kind.Step, { n: 2 }, [{ name: 'x', view: new Int32Array(29).fill(-1) }]),
      encodeFrame(Kind.Close, { n: 3 }),
    ];
    const stream = new Uint8Array(frames.reduce((n, f) => n + f.length, 0));
    let at = 0;
    for (const f of frames) {
      stream.set(f, at);
      at += f.length;
    }
    for (const chunk of [1, 3, 7, 1000]) {
      const parser = new FrameParser();
      const out: Uint8Array[] = [];
      for (let i = 0; i < stream.length; i += chunk)
        out.push(...parser.push(stream.subarray(i, i + chunk)));
      expect(out.length).toBe(3);
      expect(out.map((f) => decodeFrame(f).kind)).toEqual([Kind.Hello, Kind.Step, Kind.Close]);
      expect(decodeFrame<{ n: number }>(out[1]!).header.n).toBe(2);
    }
  });

  it('refuses a frame whose length lies', () => {
    const bytes = encodeFrame(Kind.Hello, {});
    expect(() => decodeFrame(bytes.subarray(0, bytes.length - 1))).toThrow();
  });
});
