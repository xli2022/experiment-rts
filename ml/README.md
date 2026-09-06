# rtsml — training the neural bot

The TypeScript side decides what the bot sees and what it can say
(`src/ai/neural`, dumped to `rtsml/spec.json`); this package trains a policy
to say the right things and exports it to ONNX for the browser. Matches are
served by Bun processes running `tools/ml/serve.ts`; Python never simulates.

```sh
cd ml && pip install -e '.[dev]' && pytest      # Python 3.10+; needs bun on PATH for the env tests
```

## The loop

| step     | command                                                                | writes                               |
| -------- | ---------------------------------------------------------------------- | ------------------------------------ |
| imitate  | `rtsml-imitate --procs 16 --envs 8 --steps 5e6`                        | `runs/bc/{best,last}.pt`, log.jsonl  |
| PPO      | `rtsml-ppo --init runs/bc/best.pt --procs 16 --envs 8 --updates 2000`  | `runs/ppo/{best,last}.pt`, league.pt |
| evaluate | `rtsml-eval --ckpt runs/ppo/best.pt --seeds 64 --out eval.json`        | a table, `eval.json`                 |
| export   | `rtsml-export --ckpt runs/ppo/best.pt --evaluation eval.json [--int8]` | `public/models/policy.{onnx,json}`   |

Every script takes `--smoke` (imitate, ppo) or small `--seeds` (eval) for a
run that finishes in seconds; `tests/test_training.py` runs exactly those.
`--procs` is the number of Bun processes and is what the throughput scales
with — the model is a small part of the wall time.

## What the pieces are

| module         | role                                                                                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec.py`      | `SPEC`, read from `spec.json`; every shape, name and index the model depends on                                                                                         |
| `protocol.py`  | the frame codec on the pipe (`tools/ml/protocol.ts` in Python)                                                                                                          |
| `env.py`       | `BunVectorEnv`: many environments in many processes; one step is one decision per observed slot, four ticks                                                             |
| `model.py`     | `Policy`: entity transformer + map CNN + scalar MLP → torso → autoregressive heads (type, selection, entity type, target, cell, sub-cell), and an asymmetric value head |
| `sampling.py`  | masked heads, Gumbel-max with supplied noise, Bernoulli selection with top-k, log-probabilities and entropies                                                           |
| `imitation.py` | behaviour cloning from the scripted teacher, streamed from live matches                                                                                                 |
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

**The teacher sees everything; the label is what the student could have
said.** A teacher slot is the scripted bot at the student's cadence, and each
command it releases is encoded against the student's own frame and masks. A
label the student could not express — a Train the bank no longer covers, a
Build in fog — arrives as type −1 and is skipped. Noop is most of what any
player does between commands and is kept at `--noop-keep` of its natural
rate.

**A teacher slot is reported one decision late, and has to be.** The command
the teacher takes from an observation is only known once the world has been
stepped, and a label's selection and target are row indices into the frame it
was encoded against — so the pair can only be sent after the fact. `MatchEnv`
holds each teacher observation back and reports it once its answer is known.
Reporting the live observation with the last command instead teaches the
student to reply with the _previous_ state's move, which scores 1.00 on every
validation head — the shifted task is self-consistent — and then stands
perfectly still in a real match, because a student that does nothing never
advances its own world. That bug cost a full pipeline: 0.0 commands per
minute, and a PPO run that could only ever be 0%.

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
