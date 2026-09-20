/** Small reproducible skirmishes for balance work; not a competitive tier list.
 * Run with `npx vite-node scripts/combat-probe.ts`. No focus fire or kiting.
 */
import { defOf } from '../src/config/rules.js';
import { CommandType } from '../src/sim/commands.js';
import { fromFloat } from '../src/sim/fixed.js';
import { Simulation } from '../src/sim/tick.js';
import { EntityType as E, Tile, TICKS_PER_SECOND } from '../src/sim/types.js';

type Squad = readonly (readonly [E, number])[];
interface Setup {
  spacing?: number;
  holdB?: boolean;
}
const cases: readonly [string, Squad, Squad, Setup?][] = [
  ['melee closes on rifles', [[E.Slicebot, 4]], [[E.Burstbot, 6]]],
  ['brawlers against rifles', [[E.Firespout, 3]], [[E.Burstbot, 6]]],
  ['brawlers against melee', [[E.Firespout, 3]], [[E.Slicebot, 4]]],
  ['armour against brawlers', [[E.DarkGolem, 2]], [[E.Firespout, 5]]],
  ['brawlers against air', [[E.Firespout, 3]], [[E.Beamdrone, 3]]],
  // Holding the defenders keeps their spacing intact instead of allowing an
  // attack-move formation to bunch them back up on the approach. This measures
  // whether deliberate positioning offers a meaningful answer to early splash.
  [
    'brawlers into spread rifles',
    [[E.Firespout, 3]],
    [[E.Burstbot, 6]],
    { spacing: 3, holdB: true },
  ],
  [
    'brawlers into spread coils',
    [[E.Firespout, 3]],
    [[E.Arclight, 2]],
    { spacing: 3, holdB: true },
  ],
  ['coils against rifles', [[E.Arclight, 2]], [[E.Burstbot, 6]]],
  ['armour against small hits', [[E.DarkGolem, 1]], [[E.Burstbot, 5]]],
  ['light air against rifles', [[E.Beamdrone, 3]], [[E.Burstbot, 6]]],
  [
    'repair escort',
    [
      [E.Slicebot, 3],
      [E.Fixomatic, 1],
    ],
    [[E.Burstbot, 6]],
  ],
  [
    'artillery escort',
    [
      [E.Sentry, 2],
      [E.Slicebot, 2],
    ],
    [[E.Firespout, 5]],
  ],
  ['railguns against armour', [[E.Piercebot, 2]], [[E.DarkGolem, 1]]],
];

for (const [name, a, b, setup = {}] of cases) {
  const spacing = setup.spacing ?? 1.5;
  const results = [];
  for (const reversed of [false, true]) {
    const sim = new Simulation(1);
    const { world } = sim;
    const { pool } = world;
    world.map.tiles.fill(Tile.Ground);
    world.map.elevation.fill(0);
    const ids: number[][] = [[], []];
    for (let side = 0; side < 2; side++) {
      const squad = side === 0 ? a : b;
      const owner = reversed ? 1 - side : side;
      const direction = owner === 0 ? 1 : -1;
      let row = 0;
      for (const [type, count] of squad) {
        for (let n = 0; n < count; n++, row++) {
          const rank = Math.floor(row / 4);
          const x = 64 - direction * (7 + rank * spacing);
          const y = 64 + direction * ((row % 4) - 1.5) * spacing;
          ids[side]!.push(pool.spawn(type, owner, fromFloat(x), fromFloat(y)));
        }
      }
    }
    const remaining = (side: number) => ids[side]!.filter((id) => pool.isAlive(id));
    sim.step(
      ids.map((units, side) => {
        const player = reversed ? 1 - side : side;
        return {
          type: setup.holdB && side === 1 ? CommandType.Hold : CommandType.AttackMove,
          player,
          units,
          x: fromFloat(player === 0 ? 72 : 56),
          y: fromFloat(64),
        };
      }),
    );
    while (world.tick < 90 * TICKS_PER_SECOND && remaining(0).length && remaining(1).length)
      sim.step([]);
    results.push({
      seconds: world.tick / TICKS_PER_SECOND,
      survivors: [remaining(0).length, remaining(1).length],
      hp: [0, 1].map((side) => remaining(side).reduce((sum, id) => sum + pool.hp[id & 0xffff]!, 0)),
    });
  }
  console.log(
    JSON.stringify({
      name,
      spacing,
      orders: ['attack-move', setup.holdB ? 'hold' : 'attack-move'],
      cost: [a, b].map((squad) =>
        squad.reduce((sum, [type, n]) => sum + defOf(type).mineralCost * n, 0),
      ),
      results,
    }),
  );
}
