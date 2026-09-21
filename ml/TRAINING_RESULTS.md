# Neural training wrap-up

Training stopped at the user's request on 2026-09-21 UTC. The training goal and
five-minute progress automation are paused. No model met the required match
win rates, and no experimental checkpoint was exported or promoted into the
game. The bundled legacy model remains incompatible with the current codec.

## Final results

The completed comparison on the repaired game (codec 6, network protocol 12)
used unchanged checkpoint weights, temperature 0.5, both seats, and the
unmodified scripted opponent at cadence 10:

| Layout | Development matches | Wins |
| --- | ---: | ---: |
| Lanes | 16 | 4 (25%) |
| Quarters | 8 | 0 (0%) |

These small development sets do not establish generalization or qualification.
No reserved qualification seeds were used for training or tuning.

Two older codec-5 runs were intentionally stopped:

| Run | Progress at stop | Latest completed match evaluation |
| --- | --- | --- |
| Lanes PPO | 1,030 of 1,100 updates; 2,109,440 decisions | Update 1,000: 2/16 wins |
| Quarters activation DAgger | 853,973 fresh labels | Checkpoint 8: 0/8 wins |

The Lanes `last.pt` is valid at update 1,030. Quarters did not reach its final
save; its latest durable checkpoint is `training/ckpt8.pt` at 800,612 labels.
Both checkpoints load with finite tensors. Optimizer progress and imitation
accuracy are not evidence of stronger play.

## Retained changes and artifacts

The source retains the verified observation-budget, construction routing,
private-path invalidation, checkpoint migration, packaging, and training
correctness fixes. The combined code passed 869 TypeScript tests, 159 Python
tests, the production build, and 26 matching Node/Bun determinism checkpoints.

Optional activation, coverage-logging, and reward experiments remain on their
separate branches. The newly prepared codec-6 adaptation runs were never
launched. Further training requires a new user request.

Local checkpoints, logs, evaluation reports, source hashes, and stop records
are preserved under the ignored `runs/neural-campaign/` directory. Its
`state.json`, `README.md`, and `wrap-up-20260921/` record the final campaign
state. These large experimental artifacts are not included in Git.
