"""Behaviour cloning from the scripted teacher, streamed from live matches.

    rtsml-imitate --procs 16 --envs 8 --steps 5e6

The teacher is the scripted bot playing through the student's own cadence and
vocabulary; every command it releases arrives as a label against the very
observation the student would have seen, already checked against the masks
(a label the student could not have expressed is dropped by the environment).
Labels stream straight from the environments into a buffer that is trained
on and emptied, so there is no dataset on disk; validation uses a fixed set
of labels from a hold-out seed range. Noop is most of what any player does
between commands and is kept at a fraction, so the loss is about what to do
rather than about waiting.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import sampling as S
from .env import LANES, QUARTERS, BunVectorEnv, EnvConfig, noop_actions, slot
from .model import Policy, parameter_count
from .spec import MULTI_SELECT, NOOP, SPEC, USES_ENTITY_TYPE, USES_LOCATION, USES_TARGET
from .util import load_checkpoint, pick_device, save_checkpoint, set_seed, to_torch

VAL_SEED0 = 500_000
SEED_STRIDE = 4096
STORED = (
    "entities",
    "entity_mask",
    "grid",
    "scalars",
    "mask_type",
    "mask_selection",
    "mask_target",
    "mask_cell",
    "mask_build_cell",
    "mask_row_entity_type",
    "mask_build_type",
    "label",
)


# Teacher match patterns per layout. `None` is the historical mix — three
# quarters Lanes, one quarter Quarters — which is what trained a clone that
# played Lanes at 25% and Quarters at 0%: a quarter of the gradient and none of
# the checkpoint selection is not enough to learn a mode the policy is told it
# is in. A model is now trained per layout, so a run names one.
LANES_PATTERNS = (
    [slot("teacher"), slot("scripted", 10)],
    [slot("scripted", 10), slot("teacher")],
    [slot("teacher"), slot("teacher")],
)
# Both orders of the team a teacher sits on, and — unlike the mix, where a
# teacher always had a scripted ally — a pattern where the teacher plays *both*
# slots of its team, which is how `rtsml-eval` and the browser use it. Training
# only alongside a scripted partner teaches the student a game it will not be
# asked to play.
QUARTERS_PATTERNS = (
    [slot("teacher"), slot("scripted", 10), slot("scripted", 10), slot("teacher")],
    [slot("scripted", 10), slot("teacher"), slot("teacher"), slot("scripted", 10)],
    [slot("teacher"), slot("teacher"), slot("scripted", 10), slot("scripted", 10)],
    [slot("scripted", 10), slot("scripted", 10), slot("teacher"), slot("teacher")],
)


def teacher_configs(
    procs: int, envs: int, seed0: int, max_ticks: int = 24_000, layout: str | None = None
) -> list[list[EnvConfig]]:
    """Teacher matches for one layout, or the historical mix when `layout` is None."""
    groups: list[list[EnvConfig]] = []
    i = 0
    for _ in range(procs):
        group: list[EnvConfig] = []
        for _ in range(envs):
            if layout == "lanes":
                lay, slots = LANES, LANES_PATTERNS[i % len(LANES_PATTERNS)]
            elif layout == "quarters":
                lay, slots = QUARTERS, QUARTERS_PATTERNS[i % len(QUARTERS_PATTERNS)]
            else:
                pattern = i % 4
                if pattern < 3:
                    lay, slots = LANES, LANES_PATTERNS[pattern]
                else:
                    lay, slots = QUARTERS, QUARTERS_PATTERNS[(i // 4) % 2]
            group.append(EnvConfig(seed=seed0 + i * SEED_STRIDE, layout=lay, slots=list(slots), max_ticks=max_ticks))
            i += 1
        groups.append(group)
    return groups


class LabelBuffer:
    """Labelled rows, appended per step and materialised for training."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in STORED}
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def add(self, arrays: dict[str, np.ndarray], keep: np.ndarray) -> None:
        if not keep.any():
            return
        for name in STORED:
            self.parts[name].append(arrays[name][keep])
        self.n += int(keep.sum())

    def materialise(self) -> dict[str, np.ndarray]:
        return {name: np.concatenate(parts) for name, parts in self.parts.items()}

    def clear(self) -> None:
        for parts in self.parts.values():
            parts.clear()
        self.n = 0


def select_labels(arrays: dict[str, np.ndarray], rng: np.random.Generator, noop_keep: float) -> np.ndarray:
    """Valid labels, Noop kept at `noop_keep`."""
    label = arrays["label"][:, 0]
    valid = label >= 0
    noop = label == NOOP
    return valid & (~noop | (rng.random(len(label)) < noop_keep))


def collect_labels(env: BunVectorEnv, n: int, rng: np.random.Generator, noop_keep: float, max_steps: int = 100_000) -> dict[str, np.ndarray]:
    buffer = LabelBuffer()
    batch = env.reset()
    steps = 0
    while len(buffer) < n and steps < max_steps:
        batch = env.step(noop_actions(len(batch)))
        buffer.add(batch.arrays, select_labels(batch.arrays, rng, noop_keep))
        steps += 1
    data = buffer.materialise()
    return {name: array[:n] for name, array in data.items()}


def train_on(policy: Policy, opt: torch.optim.Optimizer, data: dict[str, np.ndarray], batch_size: int, epochs: int, device: torch.device, rng: np.random.Generator, ent_coef: float, *, weights: np.ndarray | None = None) -> float:
    """Fit complete action labels, optionally weighting rows within each minibatch.

    Weights include every used action head and its entropy bonus. Normalizing
    their sum keeps an all-Build batch on the same scale as an ordinary batch.
    The default retains the original unweighted loss operations exactly.
    """
    n = len(data["label"])
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("weights must contain one finite positive value per label")
        if n and (weights == weights[0]).all():
            weights = None
    losses: list[float] = []
    policy.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            obs = to_torch({k: v for k, v in data.items() if k != "label"}, device, idx)
            labels = torch.from_numpy(data["label"][idx].astype(np.int64)).to(device)
            out = policy.evaluate(obs, labels)
            # The teacher is deterministic, so the cross-entropy optimum is a delta
            # function and enough labels will drive the policy to it: probability
            # exactly 1, entropy exactly 0, logits past +/-100. That clones the
            # teacher well and is a dead initialisation for PPO, whose ratio is then
            # identically 1 with no gradient to push on. A small entropy bonus keeps
            # the cloned policy stochastic enough to be improvable.
            if weights is None:
                loss = -out["logp"].mean() - ent_coef * out["entropy"].mean()
            else:
                batch_weights = weights[idx]
                # Scale before summing so even large finite CLI weights cannot
                # overflow; only relative weights affect this objective.
                scaled = batch_weights / batch_weights.max()
                normalized = torch.as_tensor(scaled / scaled.sum(), dtype=out["logp"].dtype, device=device)
                loss = ((-out["logp"] - ent_coef * out["entropy"]) * normalized).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


def validate(policy: Policy, data: dict[str, np.ndarray], device: torch.device, batch_size: int = 256) -> dict[str, Any]:
    """Negative log-likelihood and per-head accuracies on labelled rows."""
    n = len(data["label"])
    nll = 0.0
    ent = 0.0
    type_hits = 0
    non_noop = 0
    non_noop_hits = 0
    type_counts = np.zeros(SPEC.n_types, dtype=np.int64)
    type_predicted = np.zeros_like(type_counts)
    type_correct = np.zeros_like(type_counts)
    tp = fp = fn = 0
    multi_rows = 0
    target_rows = target_hits = 0
    cell_rows = cell_hits = 0
    et_rows = et_hits = 0
    multi = torch.zeros(SPEC.n_types, dtype=torch.bool, device=device)
    multi[list(MULTI_SELECT)] = True
    policy.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(n, start + batch_size))
            obs = to_torch({k: v for k, v in data.items() if k != "label"}, device, idx)
            labels = torch.from_numpy(data["label"][idx].astype(np.int64)).to(device)
            out = policy.evaluate(obs, labels)
            t = labels[:, 0]
            nll += float(-out["logp"].sum())
            ent += float(out["entropy"].sum())
            pred = out["type_logits"].argmax(-1)
            type_hits += int((pred == t).sum())
            type_counts += torch.bincount(t, minlength=SPEC.n_types).cpu().numpy()
            type_predicted += torch.bincount(pred, minlength=SPEC.n_types).cpu().numpy()
            type_correct += torch.bincount(t[pred == t], minlength=SPEC.n_types).cpu().numpy()
            nn_rows = t != NOOP
            non_noop += int(nn_rows.sum())
            non_noop_hits += int((pred[nn_rows] == t[nn_rows]).sum())
            is_multi = multi[t]
            if is_multi.any():
                truth = S.membership_of(labels[:, 5:], SPEC.n_ent)[is_multi]
                guess = (out["selection_logits"] > 0)[is_multi] & out["selection_mask"][is_multi]
                tp += int((truth & guess).sum())
                fp += int((~truth & guess).sum())
                fn += int((truth & ~guess).sum())
                multi_rows += int(is_multi.sum())
            for rows, logits, column in (
                (torch.isin(t, torch.tensor(USES_TARGET, device=device)), out["target_logits"], 2),
                (torch.isin(t, torch.tensor(USES_LOCATION, device=device)), out["cell_logits"], 3),
                (torch.isin(t, torch.tensor(USES_ENTITY_TYPE, device=device)), out["entity_type_logits"], 1),
            ):
                if rows.any():
                    hits = int((logits[rows].argmax(-1) == labels[rows, column]).sum())
                    if column == 2:
                        target_rows += int(rows.sum())
                        target_hits += hits
                    elif column == 3:
                        cell_rows += int(rows.sum())
                        cell_hits += hits
                    else:
                        et_rows += int(rows.sum())
                        et_hits += hits
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "labels": n,
        "nll": nll / max(1, n),
        "typeAccuracy": type_hits / max(1, n),
        "nonNoopTypeAccuracy": non_noop_hits / max(1, non_noop),
        "selectionF1": 2 * precision * recall / max(1e-9, precision + recall),
        "targetAccuracy": target_hits / max(1, target_rows),
        "cellAccuracy": cell_hits / max(1, cell_rows),
        "entityTypeAccuracy": et_hits / max(1, et_rows),
        "entropy": ent / max(1, n),
        "perActionType": {
            name: {
                "labels": int(type_counts[i]),
                "predicted": int(type_predicted[i]),
                "correct": int(type_correct[i]),
                "recall": float(type_correct[i] / type_counts[i]) if type_counts[i] else None,
                "precision": float(type_correct[i] / type_predicted[i]) if type_predicted[i] else None,
            }
            for i, name in enumerate(SPEC.action_types)
        },
    }


def model_hparams(args: argparse.Namespace) -> dict[str, Any]:
    return {"d": args.d, "heads": args.heads, "layers": args.layers, "torso": args.torso}


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--torso", type=int, default=256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--envs", type=int, default=4, help="environments per process")
    parser.add_argument("--steps", type=float, default=2e5, help="labels to train on")
    parser.add_argument("--buffer", type=int, default=2048, help="labels per training round")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1, help="passes over each buffer")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noop-keep", type=float, default=0.25)
    parser.add_argument(
        "--ent",
        type=float,
        default=0.01,
        help="entropy bonus; keeps the clone stochastic enough for PPO to improve. 0 reproduces the old collapse",
    )
    parser.add_argument("--val-labels", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=5, help="buffers between validations")
    parser.add_argument(
        "--keep-every",
        type=float,
        default=0,
        help="also write ckpt<labels>.pt every N labels; 0 disables. Validation does not say which "
        "clone plays: screen these against the scripted bot and pick on win rate",
    )
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=24_000)
    parser.add_argument(
        "--layout",
        choices=["lanes", "quarters", "mix"],
        default="lanes",
        help="which map to clone the teacher on; a model is trained per layout",
    )
    parser.add_argument("--init", type=Path, help="continue from a checkpoint")
    parser.add_argument("--out", type=Path, default=Path("runs/bc"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="a tiny run that exercises everything")
    add_model_args(parser)
    args = parser.parse_args(argv)
    if args.smoke:
        args.procs, args.envs, args.steps, args.buffer, args.batch = 1, 2, 48, 16, 8
        args.val_labels, args.val_every, args.max_ticks = 8, 1, 400
        args.d, args.heads, args.layers, args.torso = 32, 2, 1, 64

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    layout = None if args.layout == "mix" else args.layout
    hparams = {
        "model": model_hparams(args),
        "lr": args.lr,
        "noopKeep": args.noop_keep,
        "buffer": args.buffer,
        "batch": args.batch,
        # Recorded so a checkpoint says which map it is for; a Lanes model and a
        # Quarters model are different files with the same shape.
        "layout": args.layout,
    }
    policy = Policy(**hparams["model"]).to(device)
    if args.init:
        policy.load_state_dict(load_checkpoint(args.init, device)["model"])
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"policy: {parameter_count(policy)} parameters on {device}")

    val_env = BunVectorEnv(teacher_configs(1, max(1, min(4, args.envs)), VAL_SEED0, args.max_ticks, layout))
    try:
        val = collect_labels(val_env, args.val_labels, rng, args.noop_keep)
    finally:
        val_env.close()
    print(f"validation: {len(val['label'])} labels")

    env = BunVectorEnv(teacher_configs(args.procs, args.envs, args.seed0, args.max_ticks, layout))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    best = math.inf

    def checkpoint(metrics: dict[str, float]) -> None:
        """Write `last.pt`, and `best.pt` when this is the best clone so far.

        Scored on the quantity training actually minimises, entropy bonus
        included, rather than on nll alone. Nll alone picks the most collapsed
        policy on offer, which is precisely the one PPO cannot improve: a clone
        that answers with probability 1 has a ratio of exactly 1 and so no
        gradient to push on. With the bonus in the score a collapsed checkpoint
        loses on its own merits, and there is no separate floor to tune.
        """
        nonlocal best
        save_checkpoint(args.out / "last.pt", policy, "bc", hparams, metrics)
        score = metrics["nll"] - args.ent * metrics["entropy"]
        if score < best:
            best = score
            save_checkpoint(args.out / "best.pt", policy, "bc", hparams, metrics)
    labels = 0
    kept = 0
    rounds = 0
    buffer = LabelBuffer()
    t0 = time.time()
    try:
        batch = env.reset()
        while labels < args.steps:
            batch = env.step(noop_actions(len(batch)))
            keep = select_labels(batch.arrays, rng, args.noop_keep)
            buffer.add(batch.arrays, keep)
            labels += int(keep.sum())
            # The final round can be smaller than the configured buffer. It
            # still contains requested training labels, even when steps < buffer.
            if len(buffer) < args.buffer and labels < args.steps:
                continue
            loss = train_on(policy, opt, buffer.materialise(), args.batch, args.epochs, device, rng, args.ent)
            buffer.clear()
            rounds += 1
            # Validation is not the gate, and on a single layout it is actively
            # misleading: cloning one map is an easier fit than cloning the mix,
            # so nll runs down to 0.11 while entropy collapses to 0.12, and the
            # clone that wins on `nll - ent * entropy` is a policy that issues a
            # third of the teacher's commands and wins nothing. Keep the run's
            # intermediates and choose between them on win rate.
            if args.keep_every and labels >= (kept + 1) * args.keep_every:
                kept = int(labels // args.keep_every)
                save_checkpoint(args.out / f"ckpt{kept}.pt", policy, "bc", hparams, {"labels": labels})
            record: dict[str, Any] = {"round": rounds, "labels": labels, "loss": loss, "seconds": round(time.time() - t0, 1)}
            if rounds % args.val_every == 0:
                metrics = validate(policy, val, device)
                record["val"] = metrics
                checkpoint(metrics)
                print(
                    f"round {rounds} labels {labels} loss {loss:.3f} val nll {metrics['nll']:.3f} "
                    f"type {metrics['nonNoopTypeAccuracy']:.2f} sel F1 {metrics['selectionF1']:.2f} "
                    f"cell {metrics['cellAccuracy']:.2f} ({record['seconds']}s)"
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
        metrics = validate(policy, val, device)
        checkpoint(metrics)
        print(f"done: {labels} labels, final val {json.dumps(metrics)}")
    finally:
        log.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
