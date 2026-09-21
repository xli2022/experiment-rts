import { describe, expect, it } from 'vitest';
import { defOf, STARTING_WORKERS } from '../src/config/rules.js';
import { fromInt } from '../src/sim/fixed.js';
import { EntityType, MapLayout } from '../src/sim/types.js';
import { MatchEnv, parseSlots, type EnvConfig } from '../tools/ml/env.js';

const gamma = 0.8;
const shaping = 0.3;
const initialUnits = STARTING_WORKERS * defOf(EntityType.Worker).mineralCost;
const initialBuildings = defOf(EntityType.CommandPost).mineralCost;
const initialPotential =
  (0.25 * (initialUnits + initialBuildings) - initialUnits - 3 * initialBuildings) / 1000;

describe('episodic potential-based shaping', () => {
  it.each([
    { name: 'victory', defeated: [1], bonus: 1 },
    { name: 'defeat', defeated: [0], bonus: -1 },
    { name: 'mutual elimination', defeated: [0, 1], bonus: 0 },
  ])('adds only the initial-potential constant over a $name episode', ({ defeated, bonus }) => {
    const env = new MatchEnv({
      seed: 42,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,idle'),
      gamma,
      shaping,
      timeCost: 0,
    });
    try {
      const first = env.step(new Map());
      expect(first.done).toBe(false);
      expect(first.rewards[0]).toBeCloseTo(shaping * (gamma - 1) * initialPotential, 6);
      // Change board potential during the episode, then leave different amounts
      // standing at its end. None may create a terminal shaping windfall.
      env.world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(30));
      const second = env.step(new Map());
      for (const player of defeated) env.world.player(player).defeated = true;
      const terminal = env.step(new Map());
      expect(terminal.done).toBe(true);
      expect(terminal.truncated).toBe(false);
      const discountedShaping =
        first.rewards[0]! +
        gamma * second.rewards[0]! +
        gamma ** 2 * (terminal.rewards[0]! - bonus);
      expect(discountedShaping).toBeCloseTo(-shaping * initialPotential, 6);
    } finally {
      env.dispose();
    }
  });

  it('also closes the potential at the episode cap used as a terminal draw by PPO', () => {
    const env = new MatchEnv({
      seed: 42,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,idle'),
      maxTicks: 12,
      gamma,
      shaping,
      timeCost: 0,
    });
    try {
      const first = env.step(new Map());
      env.world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(30));
      const second = env.step(new Map());
      const terminal = env.step(new Map());
      expect(terminal.done).toBe(true);
      expect(terminal.truncated).toBe(true);
      expect(terminal.winner).toBe(-1);
      expect(
        first.rewards[0]! + gamma * second.rewards[0]! + gamma ** 2 * terminal.rewards[0]!,
      ).toBeCloseTo(-shaping * initialPotential, 6);
    } finally {
      env.dispose();
    }
  });
});

describe('optional win-only episode objective', () => {
  const endings = [
    { name: 'win', defeated: [1], outcome: 1 },
    { name: 'loss', defeated: [0], outcome: -1 },
    { name: 'natural draw', defeated: [0, 1], outcome: -1 },
    { name: 'capped draw', defeated: [], outcome: -1 },
  ];

  it.each(endings)(
    'depends only on win/nonwin for a $name at different lengths',
    ({ defeated, outcome }) => {
      for (const decisions of [1, 3, 17]) {
        const capped = defeated.length === 0;
        const env = new MatchEnv({
          seed: 42,
          layout: MapLayout.Lanes,
          slots: parseSlots('policy,idle'),
          gamma: 1,
          timeCost: 0,
          shaping,
          drawReward: -1,
          maxTicks: capped ? decisions * 4 : 1000,
        });
        try {
          let total = 0;
          for (let step = 0; step < decisions; step++) {
            if (step === 1) env.world.pool.spawn(EntityType.Burstbot, 0, fromInt(30), fromInt(30));
            if (step === decisions - 1) {
              for (const player of defeated) env.world.player(player).defeated = true;
            }
            const result = env.step(new Map());
            total += result.rewards[0]!;
            expect(result.done).toBe(step === decisions - 1);
            if (result.done) {
              expect(result.truncated).toBe(capped);
              // A configured negative draw reward must not turn the reported draw into a loss.
              expect(result.winner).toBe(
                defeated.length === 0 || defeated.length === 2 ? -1 : defeated[0] === 1 ? 0 : 1,
              );
            }
          }
          expect(total).toBeCloseTo(outcome - shaping * initialPotential, 6);
        } finally {
          env.dispose();
        }
      }
    },
  );

  it.each([false, true])('preserves the default zero draw reward exactly (capped=%s)', (capped) => {
    const config: EnvConfig = {
      seed: 42,
      layout: MapLayout.Lanes,
      slots: parseSlots('policy,idle'),
      maxTicks: capped ? 12 : 1000,
      gamma,
      shaping,
      timeCost: 0.00002,
    };
    const a = new MatchEnv(config),
      b = new MatchEnv({ ...config, drawReward: 0 });
    try {
      for (let step = 0; step < 3; step++) {
        if (step === 2 && !capped) {
          for (const env of [a, b])
            for (const player of [0, 1]) env.world.player(player).defeated = true;
        }
        expect(a.step(new Map())).toEqual(b.step(new Map()));
        expect(a.world.checksum()).toBe(b.world.checksum());
      }
    } finally {
      a.dispose();
      b.dispose();
    }
  });

  it.each([false, true])(
    'adds a fractional configured draw reward only at termination (capped=%s)',
    (capped) => {
      const drawReward = -0.4;
      const env = new MatchEnv({
        seed: 42,
        layout: MapLayout.Lanes,
        slots: parseSlots('policy,idle'),
        maxTicks: capped ? 12 : 1000,
        gamma,
        shaping,
        timeCost: 0,
        drawReward,
      });
      try {
        const first = env.step(new Map());
        const second = env.step(new Map());
        if (!capped) for (const player of [0, 1]) env.world.player(player).defeated = true;
        const terminal = env.step(new Map());
        expect(first.done || second.done).toBe(false);
        expect(terminal.done).toBe(true);
        expect(
          first.rewards[0]! + gamma * second.rewards[0]! + gamma ** 2 * terminal.rewards[0]!,
        ).toBeCloseTo(gamma ** 2 * drawReward - shaping * initialPotential, 6);
      } finally {
        env.dispose();
      }
    },
  );

  it.each([NaN, Infinity, -Infinity, -1.01, 1.01, null, '0'])(
    'rejects invalid drawReward %s before creating a match',
    (value) => {
      expect(
        () =>
          new MatchEnv({
            seed: 42,
            layout: MapLayout.Lanes,
            slots: parseSlots('policy,idle'),
            drawReward: value as number,
          }),
      ).toThrow('drawReward must be finite and between -1 and 1');
    },
  );
});
