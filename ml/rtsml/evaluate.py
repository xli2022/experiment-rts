"""How a policy plays: win rate against the scripted ladder, from both seats.

    rtsml-eval --ckpt runs/ppo/best.pt --ladder 10,20,40 --seeds 32

Every match is one environment; a policy plays one seat on Lanes or both
slots of one team on Quarters, against the scripted bot at the named think
interval. Seeds vary the map, so a hold-out seed range is a real test. The
result is a table and, with `--out`, a JSON file `rtsml-export` can embed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import LANES, QUARTERS, BunVectorEnv, EnvConfig, slot, team_of
from .model import Policy
from .spec import SPEC
from .util import decide, load_checkpoint, pick_device

TICKS_PER_SECOND = 20


@dataclass
class MatchResult:
    seed: int
    layout: int
    seat: int
    won: bool | None
    ticks: int
    commands: int

    @property
    def commands_per_minute(self) -> float:
        return self.commands / max(1, self.ticks) * TICKS_PER_SECOND * 60


def match_config(seed: int, layout: int, seat: int, opponent: dict[str, Any], max_ticks: int) -> EnvConfig:
    """The learner in `seat` (its team on Quarters), the opponent everywhere else."""
    if layout == QUARTERS:
        mine = [slot("policy"), slot("policy")]
        theirs = [dict(opponent), dict(opponent)]
    else:
        mine = [slot("policy")]
        theirs = [dict(opponent)]
    slots = mine + theirs if seat == 0 else theirs + mine
    return EnvConfig(seed=seed, layout=layout, slots=slots, max_ticks=max_ticks, shaping=0.0, time_cost=0.0)


def play(
    policy: Policy,
    opponent: dict[str, Any],
    seeds: list[int],
    layout: int,
    seat: int,
    procs: int,
    device: torch.device,
    temperature: float = 1.0,
    max_ticks: int = 24_000,
    seed: int = 0,
) -> list[MatchResult]:
    """One match per seed, `procs` processes; returns a result per seed, in seed order."""
    groups: list[list[EnvConfig]] = [[] for _ in range(max(1, min(procs, len(seeds))))]
    keys: list[tuple[int, int]] = []
    for i, s in enumerate(seeds):
        p = i % len(groups)
        keys.append((p, len(groups[p])))
        groups[p].append(match_config(s, layout, seat, opponent, max_ticks))
    env = BunVectorEnv(groups)
    generator = torch.Generator().manual_seed(seed)
    results: dict[tuple[int, int], MatchResult] = {}
    commands: dict[tuple[int, int], int] = {k: 0 for k in keys}
    players = 4 if layout == QUARTERS else 2
    try:
        batch = env.reset()
        rows = np.arange(len(batch))
        while len(results) < len(keys):
            actions = decide(policy, batch.arrays, rows, device, temperature, generator)
            batch = env.step(actions)
            # Sum every teammate's commands before recording a terminal match;
            # otherwise its first row marks it complete and hides the last
            # decision from the second Quarters slot.
            for r, (p, e, _) in enumerate(batch.slots):
                if (p, e) not in results:
                    commands[(p, e)] += int(batch.issued[r])
            for r, (p, e, player) in enumerate(batch.slots):
                key = (p, e)
                if key in results:
                    continue
                if batch.done[r]:
                    winner = int(batch.winner[r])
                    won = None if winner < 0 else winner == team_of(player, players)
                    results[key] = MatchResult(
                        seed=groups[p][e].seed,
                        layout=layout,
                        seat=seat,
                        won=won,
                        ticks=int(batch.tick[r]),
                        commands=commands[key],
                    )
    finally:
        env.close()
    return [results[k] for k in keys]


def summarise(results: list[MatchResult]) -> dict[str, Any]:
    wins = sum(1 for r in results if r.won)
    draws = sum(1 for r in results if r.won is None)
    return {
        "matches": len(results),
        "wins": wins,
        "draws": draws,
        "winRate": wins / max(1, len(results)),
        "medianTicks": statistics.median(r.ticks for r in results) if results else 0,
        "commandsPerMinute": statistics.median(r.commands_per_minute for r in results) if results else 0,
    }


def evaluation_metadata(seeds: list[int], layout: int, procs: int, device: torch.device,
                        temperature: float, max_ticks: int, sampling_seed: int) -> dict[str, Any]:
    return {
        "layout": "quarters" if layout == QUARTERS else "lanes", "seeds": len(seeds),
        "mapSeeds": list(seeds), "temperature": temperature, "maxTicks": max_ticks,
        "samplingSeed": sampling_seed, "specVersion": SPEC.version,
        # Process grouping changes row ordering and therefore which random draws
        # reach each map. Retain it alongside the sampling seed for reproduction.
        "procs": procs, "device": str(device), "torchVersion": str(torch.__version__),
    }


def evaluate_ladder(
    policy: Policy,
    ladder: tuple[int, ...],
    seeds: list[int],
    layout: int,
    procs: int,
    device: torch.device,
    temperature: float = 1.0,
    max_ticks: int = 24_000,
    sampling_seed: int = 0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    out = {**evaluation_metadata(seeds, layout, procs, device, temperature, max_ticks, sampling_seed),
           "state": "running", "rungs": {}}
    for k in ladder:
        rung: dict[str, Any] = {}
        out["rungs"][f"scripted@{k}"] = rung
        for seat in (0, 1):
            results = play(policy, slot("scripted", k), seeds, layout, seat, procs, device, temperature, max_ticks, sampling_seed)
            rung[f"seat{seat}"] = {**summarise(results), "results": [asdict(result) for result in results]}
            if on_progress is not None:
                on_progress(out)
        rung["winRate"] = (rung["seat0"]["winRate"] + rung["seat1"]["winRate"]) / 2
        rung["seatBias"] = rung["seat0"]["winRate"] - rung["seat1"]["winRate"]
    out["state"] = "completed"
    return out


def print_table(report: dict[str, Any]) -> None:
    print(f"{'opponent':<14}{'seat 0':>8}{'seat 1':>8}{'draws':>7}{'ticks':>8}{'cmd/min':>9}")
    for name, rung in report["rungs"].items():
        s0, s1 = rung["seat0"], rung["seat1"]
        draws = s0["draws"] + s1["draws"]
        ticks = (s0["medianTicks"] + s1["medianTicks"]) / 2
        cpm = (s0["commandsPerMinute"] + s1["commandsPerMinute"]) / 2
        print(f"{name:<14}{s0['winRate']:>8.2f}{s1['winRate']:>8.2f}{draws:>7}{ticks:>8.0f}{cpm:>9.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ladder", default="10,20,40")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed0", type=int, default=1_000_000, help="first hold-out seed")
    parser.add_argument("--layout", choices=["lanes", "quarters"], default="lanes")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=0, help="seed for neural action sampling")
    parser.add_argument("--max-ticks", type=int, default=24_000)
    parser.add_argument("--device")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if min(args.seeds, args.procs, args.max_ticks) <= 0:
        parser.error("seeds, procs and max-ticks must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    try:
        ladder = tuple(int(k) for k in args.ladder.split(","))
    except ValueError:
        parser.error("ladder must contain comma-separated positive think intervals")
    if any(k <= 0 for k in ladder) or len(set(ladder)) != len(ladder):
        parser.error("ladder intervals must be positive and distinct")

    device = pick_device(args.device)
    fingerprint = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    ckpt = load_checkpoint(args.ckpt, device)
    if fingerprint != hashlib.sha256(args.ckpt.read_bytes()).hexdigest():
        parser.error("checkpoint changed while loading; evaluate a frozen checkpoint")
    recorded_layout = ckpt["hparams"].get("layout")
    if recorded_layout in ("lanes", "quarters") and recorded_layout != args.layout:
        parser.error(f"checkpoint is for {recorded_layout}, requested {args.layout}")
    policy = Policy(**ckpt["hparams"].get("model", {})).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    layout = QUARTERS if args.layout == "quarters" else LANES
    t0 = time.time()
    metadata = {"checkpoint": str(args.ckpt.resolve()), "checkpointSha256": fingerprint}
    latest = {**evaluation_metadata(seeds, layout, args.procs, device, args.temperature,
                                   args.max_ticks, args.sampling_seed),
              "state": "running", "rungs": {}}

    def save_progress(report: dict[str, Any]) -> None:
        nonlocal latest
        latest = report
        report.update(metadata, seconds=round(time.time() - t0, 1))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out.with_suffix(args.out.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.out)
        if report["state"] == "running" and report["rungs"]:
            name, rung = next(reversed(report["rungs"].items()))
            seat = "seat1" if "seat1" in rung else "seat0"
            result = rung[seat]
            print(f"{name} {seat}: {result['wins']}/{result['matches']} wins, {result['draws']} draws", flush=True)

    save_progress(latest)
    try:
        report = evaluate_ladder(policy, ladder, seeds, layout, args.procs, device, args.temperature,
                                 args.max_ticks, args.sampling_seed, save_progress)
    except BaseException as error:
        latest.update(state="failed", error=str(error))
        save_progress(latest)
        raise
    save_progress(report)
    print_table(report)
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
