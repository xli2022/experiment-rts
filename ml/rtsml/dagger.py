"""Correct a clone on states visited by its own policy, with bounded replay.

The expert only supplies labels during training. Evaluation and exported
policies use the neural network alone. Keep intermediate checkpoints and
select them by full matches, not by the supervised validation score.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .env import LANES, QUARTERS, BunVectorEnv, EnvConfig, slot
from .imitation import (LabelBuffer, STORED, VAL_SEED0, SEED_STRIDE,
                        collect_labels, select_labels, teacher_configs, train_on, validate)
from .model import Policy
from .spec import BUILD
from .util import decide, load_checkpoint, pick_device, save_checkpoint, set_seed


class Replay:
    """Uniform reservoir of older labelled states, independent of fresh buffers."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.data: dict[str, np.ndarray] = {}
        self.size = self.seen = 0

    def add(self, fresh: dict[str, np.ndarray]) -> None:
        if not self.data:
            self.data = {name: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                         for name, value in fresh.items()}
        for i in range(len(fresh["label"])):
            self.seen += 1
            if self.size < self.capacity:
                target = self.size
                self.size += 1
            else:
                target = int(self.rng.integers(self.seen))
            if target < self.capacity:
                for name in STORED:
                    self.data[name][target] = fresh[name][i]

    def mix(self, fresh: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
        count = min(count, self.size)
        if count == 0:
            return fresh
        indices = self.rng.choice(self.size, count, replace=False)
        return {name: np.concatenate((fresh[name], old[indices])) for name, old in self.data.items()}


def learner_configs(procs: int, envs: int, seed0: int, max_ticks: int, layout: str) -> list[list[EnvConfig]]:
    groups: list[list[EnvConfig]] = []
    for p in range(procs):
        group = []
        for e in range(envs):
            i = p * envs + e
            teammates = 2 if layout == "quarters" else 1
            mine = [slot("policy") for _ in range(teammates)]
            theirs = [slot("scripted", 10) for _ in range(teammates)]
            group.append(EnvConfig(seed=seed0 + i * SEED_STRIDE,
                                   layout=QUARTERS if layout == "quarters" else LANES,
                                   slots=mine + theirs if i % 2 == 0 else theirs + mine,
                                   max_ticks=max_ticks, expert_labels=True))
        groups.append(group)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=["lanes", "quarters"], required=True)
    parser.add_argument("--steps", type=int, default=200_000, help="fresh expert labels")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--replay", type=int, default=8192)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument("--build-weight", type=float, default=1.0,
                        help="relative loss weight for Build labels, including entropy; 1 preserves uniform weighting")
    parser.add_argument("--noop-keep", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--expert-start", type=float, default=0.5,
                        help="initial probability of executing a valid expert action during collection")
    parser.add_argument("--expert-end", type=float, default=0.0)
    parser.add_argument("--val-labels", type=int, default=4096)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--keep-every", type=int, default=25_000)
    parser.add_argument("--max-ticks", type=int, default=24_000)
    parser.add_argument("--seed0", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if not math.isfinite(args.build_weight) or args.build_weight <= 0:
        parser.error("build-weight must be finite and positive")
    for name in ("steps", "procs", "envs", "buffer", "replay", "batch", "epochs", "val_labels", "val_every", "keep_every", "max_ticks"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not (0 <= args.expert_start <= 1 and 0 <= args.expert_end <= 1 and 0 < args.noop_keep <= 1):
        parser.error("expert probabilities must be in [0,1] and noop-keep in (0,1]")
    if args.temperature <= 0 or args.replay_ratio < 0:
        parser.error("temperature must be positive and replay-ratio nonnegative")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    replay_rng = np.random.default_rng(args.seed + 1)
    generator = torch.Generator().manual_seed(args.seed)
    device = pick_device(args.device)
    ckpt = load_checkpoint(args.init, device)
    initial_layout = ckpt["hparams"].get("layout")
    if initial_layout not in (None, "mix", args.layout):
        parser.error(f"initial checkpoint is for {initial_layout}, requested {args.layout}")
    hparams = {**vars(args), "model": ckpt["hparams"].get("model", {}),
               "init": str(args.init), "out": str(args.out), "algorithm": "dagger-reservoir"}
    policy = Policy(**hparams["model"]).to(device)
    policy.load_state_dict(ckpt["model"])
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    val_env = BunVectorEnv(teacher_configs(1, min(4, args.envs), VAL_SEED0, args.max_ticks, args.layout))
    try:
        val = collect_labels(val_env, args.val_labels, np.random.default_rng(0), args.noop_keep)
    finally:
        val_env.close()
    print(f"validation: {len(val['label'])} corrected teacher labels; device: {device}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    env = BunVectorEnv(learner_configs(args.procs, args.envs, args.seed0, args.max_ticks, args.layout))
    replay = Replay(args.replay, replay_rng)
    buffer = LabelBuffer()
    labels = rounds = kept = 0
    best = math.inf
    t0 = time.time()
    metrics: dict = {}
    try:
        batch = env.reset()
        with (args.out / "log.jsonl").open("a", encoding="utf-8") as log:
            while labels < args.steps:
                # Expert labels belong to this observation, before the learner
                # takes its next action. The environment excludes terminal and
                # initial/reset frames, which cannot provide a useful label.
                keep = select_labels(batch.arrays, rng, args.noop_keep) & ~batch.done
                remaining = args.steps - labels
                if int(keep.sum()) > remaining:
                    indices = np.flatnonzero(keep)
                    keep[indices[remaining:]] = False
                buffer.add(batch.arrays, keep)
                labels += int(keep.sum())
                beta = args.expert_start + (args.expert_end - args.expert_start) * labels / args.steps

                if len(buffer) >= args.buffer or labels == args.steps:
                    fresh = buffer.materialise()
                    data = replay.mix(fresh, int(len(fresh["label"]) * args.replay_ratio))
                    weights = None if args.build_weight == 1 else np.where(data["label"][:, 0] == BUILD, args.build_weight, 1.0)
                    loss = train_on(policy, opt, data, args.batch, args.epochs, device, rng, args.ent, weights=weights)
                    replay.add(fresh)
                    buffer.clear()
                    rounds += 1
                    metrics = {"round": rounds, "labels": labels, "loss": loss,
                               "replayLabels": replay.size, "trainedRows": len(data["label"]),
                               "expertProbability": beta, "seconds": round(time.time() - t0, 1)}
                    if rounds % args.val_every == 0 or labels == args.steps:
                        validation = validate(policy, val, device)
                        metrics["val"] = validation
                        score = validation["nll"] - args.ent * validation["entropy"]
                        if score < best:
                            best = score
                            save_checkpoint(args.out / "best.pt", policy, "dagger", hparams, metrics)
                    if labels >= (kept + 1) * args.keep_every:
                        kept = labels // args.keep_every
                        save_checkpoint(args.out / f"ckpt{kept}.pt", policy, "dagger", hparams, metrics)
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    print(json.dumps(metrics), flush=True)
                    if labels == args.steps:
                        break

                policy.eval()
                actions = decide(policy, batch.arrays, np.arange(len(batch)), device, args.temperature, generator)
                expert = batch.arrays["label"]
                use_expert = (expert[:, 0] >= 0) & ~batch.done & (rng.random(len(batch)) < beta)
                actions[use_expert] = expert[use_expert]
                batch = env.step(actions)
        save_checkpoint(args.out / "last.pt", policy, "dagger", hparams, metrics)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
