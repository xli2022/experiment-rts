/** Coverage without storing observations: bun run tools/ml/teacher-probe.ts [seeds=2] [seconds=600]. */
import { ActionType, SPEC } from '../../src/ai/neural/spec.js';
import { actionFromInts, allocAction, legalise } from '../../src/ai/neural/actions.js';
import { MapLayout, TICKS_PER_SECOND } from '../../src/sim/types.js';
import { MatchEnv, type SlotSpec } from './env.js';

const seeds = Number(process.argv[2] ?? 2);
const seconds = Number(process.argv[3] ?? 600);
if (!Number.isInteger(seeds) || seeds < 1 || !Number.isFinite(seconds) || seconds <= 0)
  throw new Error('Usage: teacher-probe.ts [positive integer seeds] [positive seconds]');

const action = allocAction();
for (const layout of [MapLayout.Lanes, MapLayout.Quarters]) {
  for (let sample = 0; sample < seeds; sample++) {
    const seed = (0x51ce7a11 + sample * 7919) >>> 0;
    const count = layout === MapLayout.Quarters ? 4 : 2;
    const team = sample % 2;
    const slots: SlotSpec[] = Array.from({ length: count }, (_, player) => ({
      kind: Math.floor(player / (count / 2)) === team ? 'teacher' : 'scripted',
    }));
    const env = new MatchEnv({ seed, layout, slots, maxTicks: seconds * TICKS_PER_SECOND });
    let players = env.observed.map((player) => ({ player, ...env.teacherCoverage(player) }));
    let checked = 0;
    try {
      while (!env.done) {
        env.step(new Map());
        // The live Python bridge resets here too. A terminal decision has no
        // following issue tick, so it is not part of the training dataset.
        if (env.done) break;
        for (const player of env.observed) {
          const slot = env.observe(player);
          if (slot.label[0]! >= ActionType.Noop) {
            actionFromInts(slot.label, action);
            if (!legalise(action, slot.masks))
              throw new Error(
                `illegal teacher label: seed ${seed}, player ${player}, tick ${env.tick}`,
              );
            checked++;
          }
        }
        players = env.observed.map((player) => ({ player, ...env.teacherCoverage(player) }));
      }
      console.log(
        JSON.stringify({
          specVersion: SPEC.version,
          layout: MapLayout[layout],
          seed,
          seconds: env.tick / TICKS_PER_SECOND,
          teacherTeam: team,
          winner: env.world.winner,
          truncated: !env.world.matchOver,
          checked,
          players,
        }),
      );
    } finally {
      env.dispose();
    }
  }
}
