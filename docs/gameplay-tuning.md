# Faster openings, more tactical choices

This pass shortens the wait before players can make army decisions, gives the
flanks an economic purpose, and makes more of the existing roster useful. It
keeps the mineral economy, twelve unit identities, visible damage rules, map
dimensions, and mirrored starts.

The measurements below record the original tuning pass. The later production
redesign renamed Foundry to Factory, introduced Airport, and added per-building
level-two upgrades; the current roster and costs are in the README.
The [second tuning pass](gameplay-tuning-pass-2.md) records the subsequent
Firespout adjustment, scripted AI polish and neural-training preparation.

## Opening and production

| Change                              | Before | After  |
| ----------------------------------- | ------ | ------ |
| Starting minerals                   | 50     | 200    |
| Command Post supply                 | 10     | 15     |
| Worker training                     | 12s    | 10s    |
| Barracks construction               | 45s    | 30s    |
| Foundry construction                | 55s    | 40s    |
| Supply Depot construction           | 25s    | 18s    |
| Expansion Command Post construction | 55s    | 45s    |
| Turret construction                 | 28s    | 24s    |
| Light unit training                 | 17–24s | 12–20s |
| Heavy unit training                 | 26–40s | 22–32s |

The starting bank buys a Barracks and one worker immediately. The additional
supply allows a small fighting force before the first depot. Income per trip,
mining time, starting worker count, building costs and main mineral reserves
are unchanged, so expansion and production still compete for the same money.

## Unit roles

- **Fixomatic:** 100 → 75 minerals; 2 → 1 supply. Adding repair support costs
  one light fighter's budget and still requires an escort. Repair rate unchanged.
- **Firespout:** 130 → 115 HP; 14 → 12 damage. Still punishes packed ground
  squads, but an early splash army is easier to contest.
- **Arclight:** 120 → 130 HP; 9 → 10 damage; 4.5 → 5 range. It can trade with
  ranged squads without first walking through an uncontested volley, while its
  value still depends on having several targets.
- **Piercebot:** 20 → 26 damage. Heavy hits better answer armour; exposed
  railguns still need protection from melee.
- **Sentry:** 30 damage every 2.6s → 36 every 2.4s. Siege is a more decisive
  investment. Its minimum range and inability to hit air remain weaknesses.
- **Ice Golem:** 5 → 4 supply. Control is easier to fit into a mixed army.

Combat probes compare fixed formations from both seats. They preserve useful
tradeoffs: rifles beat light air, armour beats small hits, melee catches exposed
railguns, and escorted artillery beats short-range splash. These are scenario
checks, not win-rate estimates: terrain, formation, focus fire and kiting matter.

## Maps and opponents

**Three Lanes** now has four expansions: two naturals and two contested sites at
the outer-lane/river crossroads. Bringing the outer crossroads inward reduces
base-to-flank walking distance from 99 to 87 tiles and natural-to-flank distance
from 75 to 59. The main route to the centre is unchanged. Each contested site
has equal path distance from both starting bases and a full mineral line.

**Four Quarters** adds direct natural-to-flank connections. The measured
natural-to-flank route falls from 41–45 to 21 tiles; defenders can rotate without
returning to their main base. Terrain and expansion usability are checked over
multiple seeds, including symmetry, connectivity and Command Post footprints.

**Scripted AI** chooses composition deficits from living and queued units,
instead of a time-based rotation that repeatedly skipped most unit types.
It adds a small repair contingent after a fighting core, uses the entire heavy
roster, and changes counter-unit weights for enemies in allied sight. Hidden
reinforcements do not change its next counter-unit. It still knows structure
locations for strategic targeting. Worker targets include queued workers.

## Controls and readability

F1 cycles idle owned workers and centers the camera. F2 selects the full owned
army, including support; a second press centers it. Selection supports 200 units;
large orders are split into commands of at most 24 units and delivered through
the existing network budget. Chunks retain distinct formation slots, including
the last short chunk. Workers and allied armies are excluded from F2. Holding
a command key does not flood the order queue or repeatedly spend minerals.

Build and train tooltips explain each role, actual build time and supply.
Selected unit names also carry the role hint. Multiplayer protocol is **8**;
both peers need this version of the game.

## Repeating the measurements

```sh
bun run scripts/gameplay-probe.ts 2 600
npx vite-node scripts/combat-probe.ts
npm test
npm run build
npm run determinism:node
npm run determinism:bun
```

The pacing probe runs two seeds on each map for up to ten simulated minutes.
The two sides use different scripted think intervals so mirror symmetry does
not force a draw. This is a reproducible pacing sample, not a human difficulty
rating or a claim of competitive balance. The combat probe uses open ground,
fixed formations and attack-move without micro, from both seats.

Across the final four-match comparison, first units moved from 110–119s to
45–47s, first combat from 147–164s to 86–103s, and first heavy units from
228–235s to 129–132s. All twelve robot types appeared across the revised sample,
compared with five before. All four revised matches finished in 362–502s;
the mean capped duration fell from 585s to 452s. Human play remains the best test of whether those
choices feel good; the probes make the pacing and available roles measurable.

## Validation

- 744 tests across 65 TypeScript suites passed, including mirror fairness,
  usable expansions, production choices, network budgets and formation arrival.
- Production build and typecheck passed. The existing bundle-size warning remains.
- Node/V8 and Bun/JavaScriptCore produced identical simulation checkpoints and
  the same observation hash (`ff33e645`). Bot replay fixtures were regenerated
  after the final simulation changes.
- Production-browser checks covered F1 selection, real building placement and
  training, role tooltips, F2 selection/centering, co-op activity, and a controlled
  130-unit move on open ground. Every unit received a distinct destination and
  settled; browser console errors were absent. The HUD also fit at 960×600.

No model was retrained. Formation offsets that the neural action vocabulary
cannot represent are excluded from imitation labels; model tensor shapes stay
unchanged. Both multiplayer peers must update to protocol 8.
