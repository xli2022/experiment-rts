# rtsml — training the neural bot

The TypeScript side decides what the bot sees and what it can say
(`src/ai/neural`, dumped to `rtsml/spec.json`); this package trains a policy
to say the right things and exports it to ONNX for the browser. Matches are
served by Bun processes running `tools/ml/serve.ts`; Python never simulates.

```sh
cd ml && pip install -e '.[dev]' && pytest      # needs bun on PATH for the env tests
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

**Rewards and the critic see everything; the policy does not.** Terminal ±1
for the slot's team, potential shaping on the mineral value of what each
side has, and a small cost per decision so a draw is never free. The critic
vector is the whole truth about every player's economy and army; it exists
only in training.

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

## Gates

| stage   | check                                                                                                                           |
| ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| infra   | `pytest` green; `npm run ml:bench` ≥ 800 decisions/s per core; the same `--hash` under Bun and Node (`scripts/cross-engine.sh`) |
| imitate | non-Noop type accuracy ≥ 0.8 and selection F1 ≥ 0.7 on validation; wins some matches against `scripted@10` from both seats      |
| PPO     | ≥ 70% versus `scripted@10` from both seats on 64 hold-out seeds, seat bias within ±5%; ≥ 55% versus the previous snapshot       |
| export  | parity exact; `policy.onnx` ≤ 4 MB; `npm test` green with the new `policy.json`                                                 |
