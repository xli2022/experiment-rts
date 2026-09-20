# Scripted AI and training tuning pass

This pass follows the Barracks / Factory / Airport redesign. It keeps the faster
opening economy and production times, checks early-unit counters, and improves
the scripted opponent and the demonstrations used to initialize a neural policy.

## Early combat

Firespout damage is **10 every 0.9 seconds**, down from 12. Its 100-mineral cost,
115 HP, 1.6 splash radius, and training time are unchanged. It still rewards
closing on a packed army; investing in armour or spacing ranged units now gives
better answers to a unit available immediately from a level-one Barracks.

`scripts/combat-probe.ts` now includes 13 scenarios, run from both seats. The
Firespout cases use equal mineral budgets. These are fixed-formation encounters,
not competitive win-rate estimates.

| Encounter                                            | Before                                | After                                 |
| ---------------------------------------------------- | ------------------------------------- | ------------------------------------- |
| Three Firespouts vs six clustered Burstbots          | Firespouts win in 3.8s, 237 HP remain | Firespouts win in 4.7s, 207 HP remain |
| Three Firespouts vs four Slicebots                   | Firespouts win in 5.7s, 228 HP remain | Firespouts win in 6.6s, 189 HP remain |
| Three Firespouts vs six spread, stationary Burstbots | Firespouts win with 86 HP             | Firespouts win with 38 HP             |
| Three Firespouts vs two spread, stationary Arclights | Firespouts win with 45 HP             | Arclights win with 40 HP              |
| Five Firespouts vs two Dark Golems                   | Firespouts win with 26 HP             | Dark Golems win with 192 HP           |

Every scenario produced identical mirrored results. Multiplayer protocol is
**10**, so an older peer cannot join and apply the previous damage value.

## Scripted opponent

- Enemy structure targets and composition counters require current allied sight.
  When there is no visible target, the army searches public starting positions
  and expansion coordinates. An expansion worker scouts its footprint before
  construction; another worker is not sent while that scout is already en route.
- Existing army orders are retained when the goal is unchanged or close enough
  to be covered by the same formation. Units finishing a weapon windup keep it.
  Reinforcements and changed destinations still receive orders, and defence
  continues to take priority.
- Attack commitment counts armed units, so repair support cannot trigger an
  unsupported push by itself.
- Worker targets account for remaining visible home patches. Upgrade planning
  no longer idles a production building for technology it cannot afford.
- An orphaned construction site reserves its builder for that think. The bot
  can reassign unfinished Command Posts and rebuild a lost one with its last
  worker when the old footprint is visible and usable.
- New building sites avoid currently visible enemy weapon range. This addresses
  a losing teacher repeatedly replacing an Airport foundation under fire,
  spending 4,000 minerals without completing one. Hidden enemies do not veto
  construction sites. In the same reproduction, new Airport spending fell to
  1,200 minerals across six different sites. The losing bot still could not
  finish one; safe placement does not guarantee its builder survives the route.

The bot remains a deterministic function of the current world and tick. It has
no income or unit bonuses. Its scouting is deliberately simple; it does not
have the neural policy's memory of previously seen enemies.

## Measurement setup

Before changes, the pacing probe ran six seeds on each map, capped at 600
simulated seconds. As in the first pass, one side thinks every 10 ticks and the
other every 20; identical bots on these mirrored maps necessarily draw. Cadence
is an opponent variation, not a monotonic difficulty setting.

The baseline finished 5/6 Lanes games and 6/6 Quarters games within the cap.
Mean capped duration was 466.8s and 474.8s respectively. First contact occurred
at 103.1–103.2s on Lanes and 86.4–93.0s on Quarters. All 36 player slots started
an expansion. The frozen baseline bot and raw measurements are retained locally
under `test-results/tuning-pass-2/` for the before/after comparison.

For the direct bot comparison, both versions use the final unit rules and the
same 10-tick cadence. Each seed is played from both seats, and both team slots
use the same version in Quarters. This separates bot changes from the damage
adjustment; it does not replace playtesting against people.

The updated bot won **12 of 16** direct games: **4/8 on Lanes and 8/8 on
Quarters**, across four seeds per map with seats swapped. All direct games
finished within 600s. The previous bot retained its knowledge of hidden enemy
buildings in this comparison.

Across those same games, army-order commands fell from **33.1 to 22.7 per
active player-minute**, a 31% reduction. Total commands fell from 71.8 to 60.5;
idle-worker time fell from 0.37% to 0.29%. These measures describe the sampled
matches, not a guarantee of strength against a person.

The separate 10-tick-versus-20-tick pacing sample still produced all twelve unit
types on each map. First units appeared at **44.7–47.2s**; first contact was
**103.1–103.2s on Lanes and 89.1s on Quarters**. Lanes finished 6/6 with mean
duration 460.9s. Quarters games were longer: 4/6 finished with mean capped
duration 574.8s. This pass improves the opponent's behaviour and teaching data;
it does not claim shorter matches on both maps. Opening production times are
unchanged. Extending the two capped Quarters games produced victories at
602.05s and 684.7s; both were still in active combat at the cap.

## Training data and initialization

Teacher frames are captured at the decision boundary, and the selected command
is issued on the next tick under the same delay as a policy action. Previously,
some commands were paired with a frame from four ticks earlier. The regression
checks both the source tick and the later simulation execution tick.

Build actions now also represent resuming an owned unfinished site. The masks
allow that operation with an empty bank, and decoding preserves the exact site
without charging for another building. Previously an occupied anchor could
snap to nearby free ground, turning a teacher's repair of its build order into
a different instruction for the student. Completed, destroyed or replaced
sites cannot turn a delayed resume into a new construction order. This adds
no tensor dimensions and keeps codec version 3.

The validation recorder now writes frames incrementally instead of keeping an
entire match in memory. It records the captured frame's tick, omits terminal
decisions that cannot be issued, reports invalid labels separately, and rotates
through player seats on each requested map. A coverage-only probe checks label
legality and reports action, building, upgrade and unit counts without writing
large observation shards:

```sh
bun run tools/ml/teacher-probe.ts 4 600
```

Imitation trains the final partial buffer, including a run with fewer requested
labels than one full buffer. Export rejects a checkpoint whose recorded layout
conflicts with the requested export layout. These prevent a successful-looking
run from saving untrained weights or mislabeling a Lanes model as Quarters.

The final four-seed coverage sample retained **99.140% of 9,995 Lanes
decisions** and **99.182% of 20,776 Quarters decisions**. Every retained label
passed its captured masks. Valid non-Noop counts were 1,870 and 3,556. Both maps
demonstrated all three production buildings and both building upgrades. Build
counts include valid resumes: 104 of 273 on Lanes, and 143 of 359 on Quarters.
They are not counts of newly purchased foundations.

This teacher demonstrates seven of the thirteen policy action types. It does
not teach explicit Attack, Hold, Rally or cancellation commands, so imitation
is an initialization rather than complete coverage of the policy's choices.
The four-seed Lanes sample included eleven combat types, missing Plasmodrone;
Quarters included all twelve. Lanes late-air demonstrations remain a coverage
gap to address with varied or longer training games and scenario sampling.
Match-based evaluation is still required; label accuracy is not playing strength.

Final verification passed **814 TypeScript tests across 72 suites**, **41
Python tests**, and the production build. Simulation checksums matched between
Node and Bun; the observation/mask probe matched at `538ab91e`. Tiny imitation
runs for both layouts completed three training rounds and exported with **8/8
live ONNX parity samples each**. Those smoke artifacts stay in
`runs/training-readiness/`; they are not deployed game opponents. Bundled neural
models remain unchanged, and full training has not been run in this pass.

## Repeating the probes

```sh
bun run scripts/gameplay-probe.ts 6 600
bun run scripts/combat-probe.ts
npm test
npm run build
npm run determinism:node
npm run determinism:bun
```

The [training guide](../ml/README.md) describes the current codec, imitation
initialization, PPO, evaluation and export. New policies must use codec version
3 and be trained and evaluated separately for Lanes and Quarters.
