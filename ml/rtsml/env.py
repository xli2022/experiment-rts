"""Environments, many at once, each a Bun process running `tools/ml/serve.ts`.

Every observed slot of every environment becomes one row of the batch the
policy sees; `slots` says which (process, environment, player) each row is.
Environments that finish are reset by the server before their next
observation, so the batch never shrinks and never waits.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import Kind, encode_frame, read_frame
from .spec import SPEC

REPO_ROOT = Path(__file__).resolve().parents[2]

ARRAY_NAMES = (
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
    "critic",
    "label",
)


@dataclass
class EnvConfig:
    seed: int
    layout: int
    slots: list[dict[str, Any]]
    max_ticks: int = 24_000
    shaping: float = 1e-3
    gamma: float = 0.99
    time_cost: float = 1e-4

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "layout": self.layout,
            "slots": self.slots,
            "maxTicks": self.max_ticks,
            "shaping": self.shaping,
            "gamma": self.gamma,
            "timeCost": self.time_cost,
        }


def observed_slots(cfg: EnvConfig) -> int:
    """Slots an observation is produced for: policy and teacher."""
    return sum(1 for s in cfg.slots if s["kind"] in ("policy", "teacher"))


def team_of(player: int, players: int) -> int:
    """The first half of the roster against the second, as the simulation derives it."""
    return 0 if player < players // 2 else 1


def slot(kind: str, think_interval: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind}
    if think_interval is not None:
        out["thinkInterval"] = think_interval
    return out


LANES = 0
QUARTERS = 1


@dataclass
class Batch:
    """One step's worth of observations for every observed slot."""

    arrays: dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    truncated: np.ndarray
    reset: np.ndarray
    winner: np.ndarray
    tick: np.ndarray
    issued: np.ndarray
    slots: list[tuple[int, int, int]] = field(default_factory=list)
    """(process, env, player) per row."""

    def __len__(self) -> int:
        return len(self.slots)


class _Process:
    def __init__(self, command: list[str], cwd: Path):
        self.proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.stdin = self.proc.stdin
        self.stdout = self.proc.stdout
        self.observed = 0

    def send(self, frame: bytes) -> None:
        self.stdin.write(frame)
        self.stdin.flush()

    def recv(self):
        frame = read_frame(self.stdout)
        if frame.kind == Kind.ERROR:
            raise RuntimeError(f"environment error: {frame.header.get('message')}")
        return frame

    def close(self) -> None:
        try:
            self.send(encode_frame(Kind.CLOSE, {}))
        except (BrokenPipeError, OSError):
            pass
        self.proc.wait(timeout=10)


class BunVectorEnv:
    """`n_procs` processes, each stepping the environments it was given."""

    def __init__(
        self,
        configs: list[list[EnvConfig]],
        command: list[str] | None = None,
        cwd: Path = REPO_ROOT,
    ):
        command = command or [os.environ.get("RTSML_BUN", "bun"), "run", "tools/ml/serve.ts"]
        self.procs = [_Process(command, cwd) for _ in configs]
        self.configs = configs
        self.max_observed = max(observed_slots(cfg) for group in configs for cfg in group)
        for proc in self.procs:
            proc.send(encode_frame(Kind.HELLO, {"specVersion": SPEC.version}))
            hello = proc.recv()
            if hello.header.get("specVersion") != SPEC.version:
                raise RuntimeError(f"environment spec {hello.header.get('specVersion')} != {SPEC.version}")
        self.last: Batch | None = None

    def reset(self, configs: list[list[EnvConfig]] | None = None) -> Batch:
        """Start every environment over — on new configs, one group per process, if given."""
        if configs is not None:
            if len(configs) != len(self.procs):
                raise ValueError(f"{len(configs)} config groups for {len(self.procs)} processes")
            self.configs = configs
            self.max_observed = max(observed_slots(cfg) for group in configs for cfg in group)
        for proc, group in zip(self.procs, self.configs):
            proc.send(encode_frame(Kind.RESET, {"envs": [cfg.to_json() for cfg in group]}))
        return self._collect()

    def step(self, actions: np.ndarray) -> Batch:
        """`actions` is int32 [rows, ACTION_INTS], rows in the order of the last batch."""
        if self.last is None:
            raise RuntimeError("reset first")
        if actions.shape != (len(self.last), SPEC.action_ints):
            raise ValueError(f"actions must be {(len(self.last), SPEC.action_ints)}, got {actions.shape}")
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        row = 0
        threads = []
        for p, (proc, group) in enumerate(zip(self.procs, self.configs)):
            block = np.full((len(group), self.max_observed, SPEC.action_ints), -1, dtype=np.int32)
            for e, cfg in enumerate(group):
                observed = observed_slots(cfg)
                block[e, :observed] = actions[row : row + observed]
                row += observed
            frame = encode_frame(Kind.STEP, {"maxObserved": self.max_observed}, [("actions", block.reshape(-1))])
            t = threading.Thread(target=proc.send, args=(frame,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self._collect()

    def _collect(self) -> Batch:
        frames = [None] * len(self.procs)

        def read(i: int) -> None:
            frames[i] = self.procs[i].recv()

        threads = [threading.Thread(target=read, args=(i,)) for i in range(len(self.procs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        arrays: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
        reward, done, truncated, reset, winner, tick, issued, slots = [], [], [], [], [], [], [], []
        for p, frame in enumerate(frames):
            assert frame is not None
            if frame.kind != Kind.OBS:
                raise RuntimeError(f"expected an observation, got {frame.kind}")
            per_slot = frame.header["slots"]
            envs = frame.header["envs"]
            # The server writes each environment's observed slots and then that
            # environment's rewards, one entry per observed slot.
            at = 0
            k = 0
            for e, status in enumerate(envs):
                observed = 0
                while k < len(per_slot) and per_slot[k]["env"] == e:
                    for name in ARRAY_NAMES:
                        arrays[name].append(frame.arrays[at])
                        at += 1
                    slots.append((p, e, per_slot[k]["player"]))
                    k += 1
                    observed += 1
                r = frame.arrays[at]
                at += 1
                if len(r) != observed:
                    raise RuntimeError(f"env {e} sent {len(r)} rewards for {observed} observed slots")
                for j in range(observed):
                    reward.append(float(r[j]))
                    done.append(bool(status["done"]))
                    truncated.append(bool(status["truncated"]))
                    reset.append(bool(status["reset"]))
                    winner.append(int(status["winner"]))
                    tick.append(int(status["tick"]))
                    issued.append(int(status["issued"][j]) if "issued" in status else 0)
            if at != len(frame.arrays):
                raise RuntimeError(f"frame carried {len(frame.arrays) - at} unexpected arrays")
        batch = Batch(
            arrays={name: np.stack(values) for name, values in arrays.items()},
            reward=np.asarray(reward, dtype=np.float32),
            done=np.asarray(done, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            reset=np.asarray(reset, dtype=bool),
            winner=np.asarray(winner, dtype=np.int32),
            tick=np.asarray(tick, dtype=np.int32),
            issued=np.asarray(issued, dtype=np.int32),
            slots=slots,
        )
        self.last = batch
        return batch

    def close(self) -> None:
        for proc in self.procs:
            proc.close()


def noop_actions(rows: int) -> np.ndarray:
    actions = np.full((rows, SPEC.action_ints), -1, dtype=np.int32)
    actions[:, 0] = 0
    return actions
