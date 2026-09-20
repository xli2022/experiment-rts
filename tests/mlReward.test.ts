import { describe, expect, it } from 'vitest';
import { defOf, STARTING_WORKERS } from '../src/config/rules.js';
import { fromInt } from '../src/sim/fixed.js';
import { EntityType, MapLayout } from '../src/sim/types.js';
import { MatchEnv, parseSlots } from '../tools/ml/env.js';

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
