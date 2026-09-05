# Training tools

Everything here runs under Bun (or `vite-node`) and talks to the Python
package in `../../ml`. The codec — what a bot sees and what it can say — lives
in `src/ai/neural`; these are the loops around it.

| script        | does                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `spec.ts`     | prints `SPEC` as JSON; `npm run ml:spec > ml/rtsml/spec.json`, checked by `tests/spec.test.ts`     |
| `env.ts`      | `MatchEnv`: a headless match one decision at a time, with policy, scripted, teacher and idle slots |
| `protocol.ts` | the frame layout on the pipe to Python                                                             |
| `serve.ts`    | many `MatchEnv`s behind stdin/stdout; `rtsml.env.BunVectorEnv` spawns one per core                 |
| `record.ts`   | a few teacher matches to disk, for a fixed validation set                                          |
| `bench.ts`    | decisions per second, and an observation hash to compare engines                                   |
| `arena.ts`    | scripted bot against scripted bot from both seats                                                  |

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

The think interval is not a strength dial — measured, `@20` beats `@10` from
either seat — so the rungs are distinct opponents rather than a ladder; see
`ml/README.md`.

## Engines

`bench.ts --hash` prints a hash of the observation stream. Training runs under
Bun and the browser runs under V8 or JavaScriptCore, so the encoder is checked
the way the simulation is: `scripts/cross-engine.sh` runs the same hash under
Node and Bun and diffs them.
