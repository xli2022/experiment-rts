# Training tools

Everything here runs under Bun (or `vite-node`) and talks to the Python
package in `../../ml`. The codec — what a bot sees and what it can say — lives
in `src/ai/neural`; these are the loops around it.

| script             | does                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| `spec.ts`          | prints `SPEC` as JSON; `npm run ml:spec > ml/rtsml/spec.json`, checked by `tests/spec.test.ts`     |
| `env.ts`           | `MatchEnv`: a headless match one decision at a time, with policy, scripted, teacher and idle slots |
| `protocol.ts`      | the frame layout on the pipe to Python                                                             |
| `serve.ts`         | many `MatchEnv`s behind stdin/stdout; `rtsml.env.BunVectorEnv` spawns one per core                 |
| `record.ts`        | a few teacher matches to disk, for a fixed validation set                                          |
| `recording.ts`     | streams one aligned teacher shard with bounded memory and separate valid/invalid label counts      |
| `teacher-probe.ts` | checks label coverage by action, production type and upgrade without writing observation data      |
| `bench.ts`         | decisions per second, and an observation hash to compare engines                                   |
| `arena.ts`         | scripted bot against scripted bot from both seats                                                  |

## The pipe

Each frame is `u32 length | u8 kind | u32 headerLength | header JSON | payload`,
little-endian. The header's `arrays` list names the payload's arrays in order,
with `dtype` (`f32`, `u8`, `i32`) and `shape`; the payload is those arrays'
raw bytes back to back.

| kind      | direction | header                                                                                          | payload                                                                                             |
| --------- | --------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `Hello` 1 | both      | `{specVersion, spec}` back                                                                      | —                                                                                                   |
| `Reset` 2 | to Bun    | `{envs: EnvConfig[]}`                                                                           | —                                                                                                   |
| `Step` 3  | to Bun    | `{maxObserved}`                                                                                 | `actions i32[envs × maxObserved × ACTION_INTS]`                                                     |
| `Obs` 4   | from Bun  | `{specVersion, slots: [{env, player}], envs: [{tick, done, truncated, winner, issued, reset}]}` | per observed slot: `entities, entity_mask, grid, scalars, mask_*, critic, label`; per env: `reward` |
| `Error` 5 | from Bun  | `{message}`                                                                                     | —                                                                                                   |
| `Close` 6 | to Bun    | —                                                                                               | —                                                                                                   |

An environment that finishes is reset on the next seed before its observation
is sent, with `reset: true` in its status, so a vector of environments never
waits on one that is over. Within one `Obs` frame the arrays run environment by
environment: every observed slot's thirteen arrays, then that environment's
`reward`, one entry per observed slot.

## Slots

A slot is `policy` (Python decides), `scripted@k` (the scripted bot thinking
every k ticks; `scripted` alone is the real bot at `THINK_INTERVAL`),
`teacher@k` (the scripted bot at the student's cadence, its commands handed
out as labels) or `idle`. `arena.ts` takes the same names:

```sh
npm run ml:arena -- --a scripted@10 --b scripted@20 --seeds 8
```

The think interval is not a strength dial — historical measurements had `@20`
beating `@10` from either seat — so the rungs are distinct opponents rather than a ladder; see
`ml/README.md`.

A teacher captures its label, visibility, observation and masks at the same
four-tick decision boundary, then issues the command on the following tick,
matching neural input timing. Tick zero has an invalid label. Commands absent
from the policy vocabulary, stale illegal commands and Build commands that
would decode to a different site are labelled `-1` and skipped. Build can
resume the exact owned unfinished site without charging minerals again.

`bun run tools/ml/teacher-probe.ts 2 600` checks two seeds on each layout for
up to ten minutes of game time, rotating the teacher's team and controlling
both allied slots on Quarters. It emits one JSON report per match, including
valid/non-Noop/dropped counts and command, building, upgrade and unit coverage.

`npm run ml:record -- --layout lanes --matches 4 --out ml/data/validation-lanes`
streams fixed matches to binary shards and JSON indexes. `lanes` is the
default; use `quarters` or `mix` explicitly. The recorded seat rotates within
each layout. Each frame's tick is the captured decision tick; Noops and
dropped labels have separate counts. Terminal decisions that cannot issue
before reset are excluded, matching the live bridge.

## Optional DAgger coverage

`rtsml-dagger --log-coverage` adds a cumulative `coverage` object to each
training log record and its checkpoint metrics. The flag is off by default;
default log fields, checkpoint metadata, sampling and training stay unchanged.
Coverage reads existing actor tensors and teacher labels with NumPy. It does
not run another policy/teacher, draw random numbers, alter loss weights, or
submit actions.

The four counter groups distinguish different stages:

| group | rows counted |
| --- | --- |
| `offered` | Valid teacher labels on nonterminal observations, before Noop retention or the final label-budget trim. |
| `retainedFresh` | Fresh labels actually added to the buffer after retention and budget trimming. Its final `rows` equals the requested fresh-label total. |
| `mixedTraining` | Rows passed to each completed training call, including fresh and sampled replay rows, once per materialized dataset before epoch repeats. Replay can count the same observation again. |
| `expertSelections` | Valid nonterminal teacher rows selected by the existing `use_expert` mask, including Noop. These count chosen substitutions; the learner may already have proposed the same action. They do not prove a command executed or changed behavior. |

Each group reports action-type counts, exact teacher resumes and other Build
labels by building type, unclassified Build labels, and represented-orphan
counts. A resume must match the labelled building type and exact canonical
top-left of a represented owned unfinished site, including its sub-cell;
same-cell proximity is insufficient. This classifier applies only to valid
teacher labels, never sampled actor actions, whose decoder may use fallback
placement. The current maps' public dimensions and building footprints are
explicitly recorded in `rtsml.coverage`; changing those constants requires
updating the classifier and its mirror/coordinate tests.

`representedOrphanObservations` counts data rows with at least one represented
own unfinished building whose public `hasAssignedBuilder` feature is zero.
`representedOrphanSitesByBuilding` sums such represented building rows across
observations, so it is **not** a unique-site count or full-world census. Masked,
allied, completed and staffed rows are excluded. No hidden simulator state is
read. These counters do not measure elapsed orphan duration, worker arrival,
construction completion or the reason an expert proposed a command.

Log snapshots are taken after training a buffer and before that iteration's
action selection. Expert selections through the preceding step are included;
the next log includes subsequent selections. When the fresh-label budget is
reached, the final buffer is trained and the loop exits without sampling or
substituting another action. Thus final offered/retained counts can include
labels that never had an action selected from their observation. Initial/reset
invalid labels and terminal rows never enter these stages. Counts are per
invocation and start at zero when continuing model weights; Adam and the
replay reservoir also retain the existing restart behavior.

## Engines

`bench.ts --hash` prints a hash of the observation stream. Training runs under
Bun and the browser runs under V8 or JavaScriptCore, so the encoder is checked
the way the simulation is: `scripts/cross-engine.sh` runs the same hash under
Node and Bun and diffs them.
