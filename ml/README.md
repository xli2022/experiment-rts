# rtsml — training the neural bot

The TypeScript side decides what the bot sees and what it can say
(`src/ai/neural`, dumped to `rtsml/spec.json`); this package trains a policy
to say the right things and exports it to ONNX for the browser. Matches are
served by Bun processes running `tools/ml/serve.ts`; Python never simulates.

The current codec is **version 5**. It preserves all 73 version-4 entity features
and appends `hasAssignedBuilder`: a 0/1 flag on an own unfinished building when
any living owned worker has a Build order targeting that building's current
handle. It describes the issued public assignment, not whether the worker can
reach the site or is making progress. Complete buildings, units, allies, enemies,
neutral entities and unused rows have zero in this column. This lets the policy
distinguish an orphaned site from one that already has a builder.

Version 4 preserved the first 58 entity features from version 3 and appended
15 columns: canonical Move/AttackMove goal displacement
for owned units, and queue counts for the 13 trainable unit types on owned
producers. These fields are zero for allied, enemy and neutral rows. Queue counts
are divided by the production queue capacity; goal displacement is divided by
the larger map dimension. Upgrade state and production legality still depend on
each owned building's level and whether it is upgrading.

Migrate a version-4 checkpoint from `ml/`:

```sh
python -m rtsml.migrate_observation --ckpt ../runs/bc-lanes/codec4.pt --out ../runs/bc-lanes/codec5.pt
```

Version-3 checkpoints require two explicit steps: first run the same command
with `--to-version 4` to produce an intermediate codec-4 checkpoint, then migrate
that file to version 5. Direct version-3 to version-5 conversion is rejected.

Migration preserves the architecture, layout and existing weights, appending
zero columns to `entity_in.weight` so the added inputs are initially ignored.
It refuses incompatible versions and existing output files, records the source
checkpoint hash, and clears evaluation metrics that no longer qualify the
output. Continue imitation or DAgger with the migrated checkpoint as `--init`,
collect fresh codec-5 observations, and run fresh gameplay evaluation before
exporting. Existing ONNX exports are not migrated. Version 1 and 2 checkpoints
remain incompatible; retrain those with the current `spec.json`.

The bundled Lanes model predates this codec and the lobby disables it. The
win rates and training experiments below are historical results, not evidence
for the current production tree, fog-aware teacher or gameplay balance. A new
model needs fresh imitation, gameplay evaluation and export before deployment.

```sh
cd ml && pip install -e '.[dev]' && pytest      # Python 3.10+; needs bun on PATH for the env tests
```

## The loop

**One model per map.** Every step takes `--layout lanes` or `--layout quarters`
and the loop is run once for each; the export names the file for it. The table
below is the Lanes pass.

| step     | command                                                                            | writes                                     |
| -------- | ---------------------------------------------------------------------------------- | ------------------------------------------ |
| imitate  | `rtsml-imitate --layout lanes --procs 16 --envs 8 --steps 5e6`                     | `runs/bc/{best,last}.pt`, log.jsonl        |
| PPO      | `rtsml-ppo --layout lanes --init runs/bc/best.pt --minibatch 1024 --keep-every 25` | `runs/ppo/{ckpt*,best,last}.pt`, league.pt |
| screen   | `python screen.py runs/ppo/ckpt*.pt --layout lanes --seeds 24 --seed0 1200000`     | a ranking; verify the top on two ranges    |
| evaluate | `rtsml-eval --ckpt <winner> --layout lanes --seeds 48 --out eval.json`             | a table, `eval.json`                       |
| export   | `rtsml-export --ckpt <winner> --layout lanes --evaluation eval.json`               | `public/models/policy-lanes.{onnx,json}`   |

Evaluation JSON records the checkpoint SHA-256, exact map seeds, action-sampling
seed, temperature, time limit, and each match outcome. The CLI saves progress
after each seat and marks interrupted or failed runs as failed; partial results
do not establish a completed evaluation. Use immutable checkpoint files when
comparing candidates.

Export retains the evaluation's sampling temperature in the model manifest,
and the browser uses that temperature by default. An explicit `--temperature`
must agree with a supplied evaluation. The exporter also rejects evaluation
metadata identifying another checkpoint, layout or codec. Reports from older
versions without these fields remain readable, but new qualification runs
should use the complete provenance format.

Every script takes `--smoke` (imitate, ppo) or small `--seeds` (eval) for a
run that finishes in seconds; `tests/test_training.py` runs exactly those.
`--procs` is the number of Bun processes and is what the throughput scales
with — the model is a small part of the wall time.

For a bounded pipeline check, run these from `ml/` (repeat with `quarters`):

```sh
python -m rtsml.imitation --smoke --layout lanes --out ../runs/training-readiness/bc-lanes
python -m rtsml.export --ckpt ../runs/training-readiness/bc-lanes/best.pt --out ../runs/training-readiness/export-lanes --parity-samples 8
```

These outputs stay outside `public/models`. A smoke checkpoint proves the
training and export interfaces work; it is not a playable trained bot. A
starting imitation run is `rtsml-imitate --layout lanes --procs 4 --envs 4
--steps 200000 --buffer 2048 --keep-every 25000 --out runs/bc-lanes`. The final
partial buffer is trained even if `--steps` is smaller than `--buffer`. Keep
checkpoints for match-based selection; label accuracy alone is not a promotion
gate. Train and evaluate Quarters separately with `--layout quarters`.

When a clone reproduces teacher examples but fails after its own mistakes,
collect corrective examples from learner-controlled matches:

```sh
python -m rtsml.dagger --layout lanes --init ../runs/bc-lanes/best.pt --out ../runs/dagger-lanes --steps 200000 --keep-every 25000
```

The environment's optional `expertLabels` mode runs a shadow teacher that
labels the current learner observation without issuing commands. DAgger pairs
each label with that observation before the next action, trains on fresh
examples plus a bounded reservoir of previous examples, and gradually reduces
the fraction of expert actions used during collection. The checkpoint retains
the initial model architecture. Expert assistance is training-only; evaluate
the saved policy with `rtsml-eval`, which uses the neural policy alone. Its
`best.pt` is selected by imitation loss; screen the intermediate checkpoints
in actual matches before choosing a model. Current validation accuracy and
training losses do not establish playing strength.

For a controlled experiment on construction recall, DAgger accepts
`--build-weight 4`. The default is `1`, preserving uniform weighting. A finite
positive weight changes only Build examples' full imitation loss, including
entropy; other actions keep weight 1. Each minibatch is normalized by its total
weight. Validation remains unweighted and adds `perActionType` label, prediction
and correct counts, recall and precision, so extra Build predictions can be
checked for false positives. Compare matched runs and full-match results before
choosing a nondefault weight.

Before a longer run, inspect the teacher without allocating a dataset (from
the repository root): `bun run tools/ml/teacher-probe.ts 2 600`. Each JSON line
reports one seeded match, valid/non-Noop/dropped decisions, action types,
Build labels (including resumes), upgrades and trained unit labels. The probe rotates the
teacher's team, uses both teammates on Quarters, and checks each accepted
label against the masks captured with it.

`npm run ml:record -- --layout lanes --matches 4 --out ml/data/validation-lanes`
records fixed validation matches one frame at a time, so memory does not grow
with match length. It defaults to Lanes; `quarters` and `mix` are explicit
options. Seats rotate within each layout. Shard metadata distinguishes valid
non-Noop labels, Noops and dropped labels; frame ticks describe the labelled
observation. Terminal unissued decisions are excluded, as in the live Python
bridge.

## What the pieces are

| module         | role                                                                                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec.py`      | `SPEC`, read from `spec.json`; every shape, name and index the model depends on                                                                                         |
| `protocol.py`  | the frame codec on the pipe (`tools/ml/protocol.ts` in Python)                                                                                                          |
| `env.py`       | `BunVectorEnv`: many environments in many processes; one step is one decision per observed slot, four ticks                                                             |
| `model.py`     | `Policy`: entity transformer + map CNN + scalar MLP → torso → autoregressive heads (type, selection, entity type, target, cell, sub-cell), and an asymmetric value head |
| `sampling.py`  | masked heads, Gumbel-max with supplied noise, Bernoulli selection with top-k, log-probabilities and entropies                                                           |
| `imitation.py` | behaviour cloning from the scripted teacher, streamed from live matches                                                                                                 |
| `dagger.py`    | corrective expert labels on learner states, with bounded replay of previous examples                                                                                      |
| `ppo.py`       | clipped PPO with GAE, entropy bonus and an annealed KL leash to the imitation policy, against a league                                                                  |
| `league.py`    | the scripted ladder plus snapshots, drawn by prioritised fictitious self-play `(1 − winrate)²`                                                                          |
| `evaluate.py`  | win rate versus each ladder rung from both seats on hold-out seeds, match length, commands per minute                                                                   |
| `export.py`    | the act graph to ONNX (opset 18, batch 1, fixed shapes), exact parity against onnxruntime, optional int8, `policy.json`                                                 |

## Decisions worth knowing

**The policy is the human vocabulary, one command at a time.** A decision is
`[type, entityType, target, cell, sub, selection × 24]` — the same integers
`src/ai/neural/actions.ts` decodes in the browser — with every head masked to
what is legal at that moment, so an illegal decision has probability zero.
The environment marks a head the type does not use as −1 and so does the
model.

**Sampling is inside the graph and its noise is an input.** Each categorical
head takes `argmax(masked logits / T + Gumbel)`; the selection head keeps
every legal row whose `logit / T + Logistic` is positive, at most 24 of them
and never none. The exported ONNX therefore carries no random number
generator, the browser fills the noise from `crypto.getRandomValues`, and
parity between torch and onnxruntime is an exact comparison of integers
(`tests/test_export.py`).

**PPO scores the draws before selection is capped or forced nonempty.** The
final unit set is not an independent Bernoulli sample: when all draws fail it
still names one unit, and when more than 24 pass it drops some. PPO therefore
stores `X = selection logits / T + GumbelA` alongside each rollout action and
scores its location-Gumbel density at every update. The selection is a fixed
projection of `X` and independent `GumbelB`, whose density cancels in the
likelihood ratio. This covers both the cap and the fallback without changing
the action tensors or the exported model interface. Latent Gumbel entropy is
constant with respect to the logits; imitation continues to use Bernoulli
membership supervision for teacher labels. Every PPO head is scored at the
same temperature used for sampling. Historical training scores here do not
establish the win rate of a model trained with this corrected objective.

**The teacher uses visible threats and public scouting locations.** A teacher
slot is the scripted bot at the student's cadence, and each
command it releases is encoded against the student's own frame and masks. A
label the student could not express — a Train the bank no longer covers, a
Build in fog, or a later formation chunk with an offset absent from the action
vocabulary — arrives as type −1 and is skipped. Noop is most of what any
player does between commands and is kept at `--noop-keep` of its natural
rate.

The paced teacher drains its pending command plan before asking for another,
so repeated planning cannot crowd queued production orders out of the bounded
queue. This option affects the training teacher only; the full scripted
opponent and asynchronous neural-agent polling keep their normal behavior.

Build preserves the teacher's exact site: an owned unfinished structure can
be resumed with no mineral charge, and a completed or destroyed resume target
does not turn into a new nearby foundation when inference arrives late.

**Teacher labels and observations describe the exact same decision tick.**
At each four-tick boundary, `MatchEnv` captures current visibility, the frame,
the masks and the teacher's released command together. The teacher issues that
command on the next tick, matching the neural policy's issue delay. Recent
action features are updated after capture so they describe the previous
decision. Tick zero has no teacher label. Pairing a command chosen on tick 4
with an observation from tick 0 is wrong even when its rows and masks happen
to remain legal; the timing regression checks the actual source tick.

**Rewards and the critic see everything; the policy does not.** Terminal ±1
for the slot's team, potential shaping on the mineral value each side has
_committed to the board_, and a small cost per decision so a draw is never
free. The critic vector is the whole truth about every player's economy and
army; it exists only in training.

The bank is deliberately outside the potential. With it counted, harvesting
raised the potential directly and spending it never did, so the shaped
optimum was to hoard: a PPO run against the banked potential converged to
Harvest on 95.6% of its decisions, built nothing at all, and lost every
match. It was not failing — it was succeeding at the wrong objective.

**The critic starts untrained, so PPO fits it before the policy moves.**
Imitation trains the policy heads only; `critic` is not in its `STORED` list.
Measured at the first PPO update, the value head's predictions have std 0.18
against a real advantage signal of std 2e-5 — four orders of magnitude of
noise, which normalising advantages then rescales to unit variance and
optimises at full strength. `--value-warmup` fits the value head with every
other gradient dropped, so the clone comes out of it unchanged. Watch
`explainedVariance` in the log: near zero means the advantages are noise and
the policy is about to sharpen onto it.

**The discount is per decision, and a match is two thousand of them.** This is
the one that cost every PPO run before it. `--gamma` is applied once per
decision, a decision is four ticks, and matches run to a median 8,383 ticks — so
`0.99` discounted the terminal ±1 by `0.99^2100 = 7e-10` and left an effective
horizon of 100 decisions, about twenty seconds of a seven-minute match. Winning
was not in the objective at all; the best PPO could do was climb the shaping
potential greedily, which is why no run ever beat the clone it started from. At
`0.999` the same terminal is worth 0.12, the same order as the shaping a whole
match accumulates, and the outcome competes with the hint instead of vanishing
underneath it. The time cost is a discounted sum too: at `1e-4` it came to 0.086
against a win worth 0.12, so the clock was nearly as loud as the result.

**Advantage normalisation turns a signal-free window into noise, at full
strength.** `(adv - mean) / (std + 1e-8)` is the standard line and it is a trap
here. Measured on a real rollout, the first 32-decision window of a match has a
reward that is _exactly_ constant — one unique value across every row and every
step, std 0.0 — because neither side has committed anything to the board yet and
the potential has not moved. The critic fits that constant trivially,
`explainedVariance` reaches 0.994, the advantages are float32 rounding, and
dividing by their own std rescales that rounding to unit variance and hands it
to PPO for three epochs. Entropy collapsed to 2e-5, the ratio ran away, `pg`
reached 4e7, and the win rate went to zero and stayed there. `--adv-floor`
divides by the std _or_ the floor, whichever is larger: a healthy window is
untouched, a signal-free one becomes the near-no-op it should always have been.

**The evaluation used to reset every environment.** `--eval-every` called
`env.reset(groups)` when it was done, but `play` builds and tears down its own
Bun processes and never touches the training environment. The reset threw away
every match in flight — and a match needs about 65 updates to finish while the
eval landed every 50, so most matches never reached a terminal ±1 and the league
was never told who won. It is exactly the failure the `--refresh` note below
describes, re-entered through a different door.

**The joint log-ratio is an ordinary log-ratio.** A note here used to claim that
`logp`, summed over six heads plus up to `N_ENT` Bernoulli selection rows, sat
in the tens _by construction_ and so had broken PPO's clip. Measured from the
clone, it does not: `logp` is −2.4 on average with a minimum of −12.5, the
selection head contributes −2.3 of that because a decision has about five legal
rows rather than 160, and one Adam step moves the log-ratio by 0.05. Log-ratios
in the tens were a symptom of the runaway above, not its cause, and the ±20
clamp that was meant to contain them permitted a ratio of 4.8e8. What the joint
_does_ do is make the tail heavy, which is why the early stop reads half the
mean square log-ratio rather than the usual `exp(-r) - 1 + r`: two unlikely rows
drove that estimator to 17 in a minibatch whose clip fraction was 0.06, stopping
every update after a single step.

**A PPO run here has a peak and then decays, so evaluate often enough to catch
it.** The entropy bonus is the only gradient in the loss that points the same
way every step, so given long enough it outvotes a modest advantage signal:
entropy climbs from 0.86 to 2.02 over seventy updates and the win rate goes
0.25 → 0.42 → 0.00 as the policy spreads into near-random play. The obvious
answer — a smaller `--ent`, a smaller `--lr`, a tighter `--kl-end` — was tried
and is wrong. All three together kept the policy within a Huber KL of 0.08 of
its initialisation over 150 updates of policy training and scored 4% against
`scripted@10`, where the aggressive settings reached 42% in forty updates. The
distance is what buys the improvement; the decay afterwards is the price. So the
defaults stay aggressive, `--eval-every` is set fine enough to sample the peak,
and `best.pt` is what you ship.

Two things follow. `beta` anneals over `--updates`, in fractions of the run
rather than in updates, so a long run holds the leash tight for far longer per
update than a short one — `--updates 2000` left it near 0.9 where `--updates
160` had reached 0.55 by the same update. And the first evaluation of a run
lands _during_ `--value-warmup` if `--eval-every` divides into it, which scores
the frozen clone and anchors `best` to it; every earlier run in this repo did
exactly that, which is why every `runs/*/best.pt` was the clone with a trained
critic and nothing else.

**The scripted ladder is a set of opponents, not a scale.** `scripted@k` is
the one scripted bot thinking every k ticks. Measured over eight seeds from
both seats, `@20` beats `@10` 8–0 and `@40` beats `@10` 5–3 while `@30`
loses 8–0, so a longer interval is not a weaker bot — the interval changes
_when_ it commits, and some cadences happen to suit its strategy. The league
does not assume an order: it weights every member by how often the learner
still loses to it. Gates should be stated per rung, and `@10` — the bot the
game ships — is the one that matters.

**Memory.** A rollout keeps every learner observation on the host: about
150 KB each (the 13 × 40 × 40 map is most of it), so `--rollout 32` over 128
rows is roughly 600 MB. Imitation buffers `--buffer` labels the same way.

**A league redraw resets every environment, so it must not land mid-match.**
The server already resets an environment of its own accord when its match
ends, keeping the same opponent, so the only cost of a long `--refresh` is a
stale draw. The cost of a short one is total: at `--refresh 10` a redraw
arrives every `10 × --rollout × 4` = 1,280 ticks against matches that run to
24,000, so no match ever finished, no terminal ±1 was ever collected, and the
league's `(1 − winrate)²` had nothing to weight on — 230 updates recorded zero
results. `--refresh 0` auto-sizes to one full match.

## Gates

| stage   | check                                                                                                                                                                        |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| infra   | `pytest` green; `npm run ml:bench` ≥ 800 decisions/s per core; the same `--hash` under Bun and Node (`scripts/cross-engine.sh`)                                              |
| imitate | non-Noop type accuracy ≥ 0.8; top-1 selection accuracy well clear of chance; validation entropy still well off zero; wins some matches against `scripted@10` from both seats |
| PPO     | ≥ 70% versus `scripted@10` from both seats on 64 hold-out seeds, seat bias within ±5%; ≥ 55% versus the previous snapshot                                                    |
| export  | fp32 parity exact; `npm test` green with the new `policy.json`; see the size note below                                                                                      |

**Where the gates actually stand — on Lanes.** The model in `public/models` is
a PPO checkpoint six generations on from the imitation clone. Over 48 hold-out
seeds from both seats it beats `scripted@10` 85.4%, `scripted@20` 83.3% and
`scripted@40` 77.1%, with a seat bias of 0.0 on every rung — the first time the
≥70% gate has been met. Against the clone's 24.0% that is 278 wins in 336
matches across five independent seed ranges, versus 46 in 192.

Every one of those numbers is the 1v1 map. `rtsml-eval` defaults to
`--layout lanes`, the in-run eval hardcodes `LANES`, and so did the screening
that picked all six champions.

**On Quarters the same model wins 3%.** Measured over 64 matches against
`scripted@10`: the clone 0/64, generation 1 2/64, the champion 2/64. Six
generations moved the 1v1 rate sixty points and left the 2v2 rate at zero.

This is not interference — the clone was already at zero before any of the
tuning, and the supervision is sound: teacher labels on Quarters are 98.3% valid
against 98.0% on Lanes, with the same 14% non-Noop share, so the encoding, the
masks and label reporting worked there. It is simply a mode
that got a quarter of the gradient (`--quarters-share 0.25`) and none of the
selection pressure, in a network that is _told_ which layout it is in — the
scalars carry a `layout:Lanes`/`layout:Quarters` one-hot, plus `allies` and
`seatInHalf`, and the critic gets its own layout bit. Nothing forces one policy
to be the other, so neglecting one is free.

**So a model is now trained per layout.** `rtsml-imitate`, `rtsml-ppo` and
`rtsml-export` each take `--layout lanes|quarters`, a checkpoint records which
map it is for, and the export names the file after it —
`public/models/policy-lanes.{onnx,json}` and `policy-quarters.{onnx,json}`. The
browser asks for the model belonging to the map it is about to play
(`modelStem` in `src/ai/neural/browser.ts`), keeps one worker per layout rather
than one per page, and the lobby's Neural chip is live per map: Co-op offers it
only when a Quarters model is present. `rtsml-export` refuses a checkpoint that
does not say which map it plays, because a Lanes model served as the Quarters
one is a 3% bot that looks like it loaded correctly.

The cost is honest: two downloads of 5.9 MB where the 4 MB budget already could
not be met by one, and each is fetched only when a match on that map starts.
What it buys is that neither map can be neglected by a run scored on the other,
which is the failure above.

Two details the split brought out. The Quarters teacher patterns now include the
teacher playing _both_ slots of its team, which is how `rtsml-eval` and the
browser use the model — the old mix only ever paired a teacher with a scripted
ally, so the student was cloned on a game it would never be asked to play. And
the export's parity check runs on the map the model is for: the layouts are
different sizes, 128 tiles against 152, so a Lanes match exercises neither the
scalars nor the region of the cell head a Quarters model uses.

**Iterating past the first PPO model took six generations, and four of them
failed.** The log is worth keeping because the failures are more informative
than the successes.

| gen | change from the previous champion        | outcome                           |
| --- | ---------------------------------------- | --------------------------------- |
| 1   | the fixes above, from `bc5`              | 0.240 → 0.646, decisive           |
| 2   | `--ent-end`, annealing the entropy bonus | 0.688, z = 1.26 — not established |
| 3   | `--rollout` 32 → 64                      | failed; most checkpoints 0/48     |
| 4   | `--minibatch` 1024, `--lr` 2e-4          | 0.875, z = +5.26 — decisive       |
| 5   | generation 4's recipe again              | failed                            |
| 6   | gentler still: `--lr` 8e-5, `--ent` 1e-3 | 0.802, below the champion         |

What moved it was **the minibatch, not the rollout or the learning rate**. At
`--minibatch 256` an update abandoned after two of its sixty minibatches —
`--target-kl` tripping on a gradient estimated from too few decisions — so the
policy took a handful of large, noisy steps per rollout and mostly destroyed
itself. Doubling the rollout did not help: it left the minibatch the same size
and merely raised the count that went unused, two of a hundred and twenty three.
At 1024 the whole update runs inside the trust region. Lowering `--lr` instead
does not substitute for it, because the step count is not what was wrong: at
8e-5 updates still stopped after two minibatches.

**Three generations then failed to beat generation 4, which is what a plateau
looks like here.** Generations 5 and 6 both started from it and both came back
worse, and the good checkpoints of every run sit in the stretch where `steps`
equals `planned` and the Huber KL stays under about 0.1 — visible in the log
before any match is played, and a better guide to where the peak is than a
short evaluation.

**Screening candidates on their best score is a trap.** Every run produces a
checkpoint that looks superb on the range it was screened on and regresses on a
fresh one: generation 2's `ckpt180` screened 0.792 and verified 0.562,
generation 6's `ckpt140` screened 0.875 and verified 0.802. Picking the maximum
of nine noisy estimates is biased upward by roughly the amount that matters. So
`--keep-every` writes checkpoints, a screen ranks them with the current champion
included _in the same run_ as a control, and nothing is promoted until it holds
up on the two established ranges. Pairing does not rescue a small evaluation
either — on identical seeds and seats two checkpoints disagree on about half
their matches, so McNemar buys almost nothing over the two-proportion test.

**Sampling colder does not help.** The exported graph takes `temperature` as an
input and the browser passes 1. Measured at 0.8 and 0.6 the champion of the day
scored 0.542 and 0.625 against 0.635 at 1.0 — neutral at best. The entropy the
bonus leaves behind is not the kind that greedier sampling recovers.

**The Lanes model was not improved by training it Lanes-only.** With the split
in place the whole loop was rerun for Lanes: a fresh clone, then PPO, and
separately PPO continued from the existing champion on 100% Lanes rather than
75%. Neither beat it — from scratch 0.417, continued 0.833, the incumbent 0.917
on the same 48 matches. That is the right result rather than a disappointing
one: Lanes already had three quarters of the data _and_ all of the checkpoint
selection, so there was nothing for the split to give back. It also shows what
the champion actually is — four generations of accumulated iteration, not one
invocation of the pipeline. Regenerating it from scratch means budgeting for the
generations, not the run.

**Quarters is winnable, and PPO still cannot learn it.** Worth stating in that
order, because the first half was checked before the second was believed: the
teacher slot _is_ the scripted bot, so `[teacher, teacher]` against
`[scripted@10, scripted@10]` is the bot playing itself, and team 0 takes 8 of 12.
The mode is balanced, the seats are right, and the labels are sound (98.3% valid
against 98.0% on Lanes).

What fails is the reinforcement step, and the log says exactly how. Over 450
updates the learner finished 1,148 matches and won **4** — and the breakdown by
opponent is the finding:

| opponent      | learner's record |
| ------------- | ---------------- |
| `scripted@10` | 0 / 364          |
| `scripted@20` | 0 / 362          |
| `scripted@40` | 0 / 306          |
| `imitation`   | 4 / 76           |
| `ppo75`       | 0 / 28           |
| `ppo150`      | 0 / 12           |

It loses to the checkpoint it started from and to its own earlier selves. PPO is
not failing to improve the policy here; it is making it worse.

The reason is that a terminal reward nobody ever earns is a constant, and a
constant has no gradient. On Lanes the clone wins a quarter of its matches, so
±1 varies and the outcome is in the objective beside the shaping. On Quarters
the clone wins none, so every episode ends −1 and the only thing left to
optimise is the potential — and optimising board presence alone, measured
against held-out opponents, makes the policy worse. This is the banked-potential
failure in another dress: not failing, succeeding at the wrong objective. The
diagnostics that would normally catch it all look healthy, which is the point
worth remembering — `advStd` sits at 1.7e-2 against a 1e-3 floor, the critic's
`explainedVariance` is 0.995, and the updates run their full sixty minibatches.

So nothing ships for Quarters, and the Co-op Neural chip stays disabled: a bot
that loses every match is worse than an honestly greyed-out button. The way in
is a curriculum that lets the learner win _something_ first — its own imitation
snapshot is the obvious first rung, since 5% is not zero — so that the terminal
signal has variance before the ladder is asked for.

**Selection F1 is not the gate it looks like.** The scripted teacher names
exactly one unit per command — 336 of 336 sampled multi-select labels, never
two — while the head is trained as an independent Bernoulli over the ~6 legal
rows. One positive against five negatives drives every logit negative, so at
F1's `logit > 0` threshold the model names 0.21 units and recall collapses to
0.14, however well it has learned. The sampler never uses that threshold:
`select_many` falls back to the argmax and so names one. Measured that way the
same checkpoint puts the teacher's unit first 62% of the time against a 16.5%
chance baseline, and in its top three 83% of the time. Gate on that, not F1.

**Two export gates, one model, and they conflict.** At 1.5M parameters fp32 is
5.9 MB and int8 is 2.0 MB, so `policy.onnx ≤ 4 MB` and exact parity cannot
both hold. int8 parity on a _trained_ model is 183/200 — quantisation flips
8.5% of decisions, so the browser would play a subtly different bot from the
one the eval measured. The model that ships is fp32: correctness over
download size. Shrink the model if the 4 MB matters.
