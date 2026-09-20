import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { ScriptedAgent } from '../src/ai/scripted.js';
import { ActionType, SCALARS, SPEC } from '../src/ai/neural/spec.js';
import { CommandType } from '../src/sim/commands.js';
import { EntityType, MapLayout } from '../src/sim/types.js';
import { decodeFrame, FrameParser } from '../tools/ml/protocol.js';
import { recordTeacherMatch } from '../tools/ml/recording.js';

describe('teacher recording', () => {
  it('writes aligned frames and counts valid, invalid and no-op labels separately', () => {
    const directory = mkdtempSync(join(tmpdir(), 'rts-teacher-record-'));
    const think = vi.spyOn(ScriptedAgent.prototype, 'act').mockImplementation((world, player) => {
      const index = world.pool.type.findIndex(
        (type, i) => type === EntityType.Worker && world.pool.owner[i] === player,
      );
      if (world.tick === 4)
        return [{ type: CommandType.Hold, player, units: [world.pool.idAt(index)] }];
      if (world.tick === 8)
        return [{ type: CommandType.CancelBuild, player, building: world.pool.idAt(index) }];
      return [];
    });
    try {
      const stem = join(directory, 'shard');
      const summary = recordTeacherMatch(
        {
          seed: 1,
          layout: MapLayout.Lanes,
          slots: [{ kind: 'teacher' }, { kind: 'idle' }],
          maxTicks: 24,
        },
        stem,
      );
      expect(summary).toMatchObject({
        specVersion: SPEC.version,
        decisions: 5,
        labels: 1,
        dropped: 1,
        noop: 3,
        ticks: 24,
      });
      const frames = new FrameParser()
        .push(readFileSync(`${stem}.bin`))
        .map((bytes) => decodeFrame(bytes));
      expect(frames).toHaveLength(5);
      expect(frames.map((frame) => frame.header.tick)).toEqual([4, 8, 12, 16, 20]);
      expect(frames.map((frame) => (frame.arrays[12] as Int32Array)[0])).toEqual([
        ActionType.Hold,
        -1,
        0,
        0,
        0,
      ]);
      for (const frame of frames) {
        expect(frame.header.specVersion).toBe(SPEC.version);
        expect((frame.arrays[3] as Float32Array)[SCALARS.indexOf('tick')]).toBeCloseTo(
          Number(frame.header.tick) / 24000,
          8,
        );
      }
      expect(JSON.parse(readFileSync(`${stem}.json`, 'utf8'))).toEqual(summary);
    } finally {
      think.mockRestore();
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
