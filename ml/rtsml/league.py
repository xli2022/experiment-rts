"""Who the learner plays.

The pool holds the scripted ladder — the one scripted bot thinking at
different intervals — and snapshots of the learner itself, the imitation
policy first and then checkpoints as training goes. Opponents are drawn by
prioritised fictitious self-play: weight `(1 - winrate)²`, so the ones the
learner still loses to are played most, with a floor so none is forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import slot

DEFAULT_LADDER = (10, 20, 40)


@dataclass
class Member:
    name: str
    kind: str
    """`scripted` or `snapshot`."""
    think_interval: int | None = None
    state: dict[str, torch.Tensor] | None = None
    hparams: dict[str, Any] = field(default_factory=dict)
    wins: float = 0.0
    games: int = 0

    @property
    def winrate(self) -> float:
        """The learner's win rate against this member, with a Laplace prior."""
        return (self.wins + 1.0) / (self.games + 2.0)

    def slot(self) -> dict[str, Any]:
        if self.kind == "scripted":
            return slot("scripted", self.think_interval)
        return slot("policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "think_interval": self.think_interval,
            "state": self.state,
            "hparams": self.hparams,
            "wins": self.wins,
            "games": self.games,
        }


class League:
    def __init__(self, ladder: tuple[int, ...] = DEFAULT_LADDER, floor: float = 0.05, seed: int = 0):
        self.members: list[Member] = [Member(f"scripted@{k}", "scripted", think_interval=k) for k in ladder]
        self.floor = floor
        self.rng = np.random.default_rng(seed)

    def add_snapshot(self, name: str, policy: torch.nn.Module, hparams: dict[str, Any]) -> Member:
        state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        member = Member(name, "snapshot", state=state, hparams=dict(hparams))
        self.members.append(member)
        return member

    def weights(self) -> np.ndarray:
        w = np.array([(1.0 - m.winrate) ** 2 + self.floor for m in self.members], dtype=np.float64)
        return w / w.sum()

    def sample(self, n: int) -> list[Member]:
        picks = self.rng.choice(len(self.members), size=n, p=self.weights())
        return [self.members[int(i)] for i in picks]

    def report(self, member: Member, learner_won: bool | None) -> None:
        """A finished match; a draw counts half."""
        member.games += 1
        member.wins += 1.0 if learner_won else 0.5 if learner_won is None else 0.0

    def summary(self) -> str:
        parts = [f"{m.name} {m.winrate:.2f}({m.games})" for m in self.members]
        return "  ".join(parts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"members": [m.to_dict() for m in self.members], "floor": self.floor}, path)

    @classmethod
    def load(cls, path: Path, seed: int = 0) -> "League":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        league = cls(ladder=(), floor=raw["floor"], seed=seed)
        league.members = [Member(**m) for m in raw["members"]]
        return league
