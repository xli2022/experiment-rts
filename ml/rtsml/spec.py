"""The codec spec, read from the JSON that `npm run ml:spec` writes.

Every shape and every name the model depends on comes from here, so the
Python side cannot drift from the TypeScript side without `tests/spec.test.ts`
saying so on one side and this module raising on the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("spec.json")


@dataclass(frozen=True)
class NoiseSegment:
    name: str
    size: int
    offset: int


@dataclass(frozen=True)
class Spec:
    version: int
    decision_ticks: int
    n_ent: int
    entity_features: tuple[str, ...]
    grid: int
    cell_tiles: int
    grid_channels: tuple[str, ...]
    scalars: tuple[str, ...]
    critic_len: int
    action_types: tuple[str, ...]
    selection_max: int
    entity_types: int
    sub: int
    action_ints: int
    heads: tuple[str, ...]
    noise: tuple[NoiseSegment, ...]
    noise_len: int

    @property
    def f(self) -> int:
        return len(self.entity_features)

    @property
    def c(self) -> int:
        return len(self.grid_channels)

    @property
    def s(self) -> int:
        return len(self.scalars)

    @property
    def n_types(self) -> int:
        return len(self.action_types)

    @property
    def cells(self) -> int:
        return self.grid * self.grid

    def type_index(self, name: str) -> int:
        return self.action_types.index(name)

    def noise_segment(self, name: str) -> NoiseSegment:
        for seg in self.noise:
            if seg.name == name:
                return seg
        raise KeyError(name)


def load_spec(path: Path = SPEC_PATH) -> Spec:
    raw = json.loads(path.read_text())
    segments = []
    offset = 0
    for seg in raw["noise"]["segments"]:
        segments.append(NoiseSegment(seg["name"], int(seg["size"]), offset))
        offset += int(seg["size"])
    if offset != raw["noise"]["length"]:
        raise ValueError(f"noise segments sum to {offset}, spec says {raw['noise']['length']}")
    spec = Spec(
        version=int(raw["version"]),
        decision_ticks=int(raw["decisionTicks"]),
        n_ent=int(raw["entities"]["rows"]),
        entity_features=tuple(raw["entities"]["features"]),
        grid=int(raw["grid"]["size"]),
        cell_tiles=int(raw["grid"]["cellTiles"]),
        grid_channels=tuple(raw["grid"]["channels"]),
        scalars=tuple(raw["scalars"]),
        critic_len=int(raw["critic"]["length"]),
        action_types=tuple(raw["actions"]["types"]),
        selection_max=int(raw["actions"]["selectionMax"]),
        entity_types=int(raw["actions"]["entityTypes"]),
        sub=int(raw["actions"]["sub"]),
        action_ints=int(raw["actions"]["ints"]),
        heads=tuple(raw["actions"]["heads"]),
        noise=tuple(segments),
        noise_len=int(raw["noise"]["length"]),
    )
    if spec.action_ints != 5 + spec.selection_max:
        raise ValueError("action layout is five choices plus the selection")
    return spec


SPEC = load_spec()

# Types by name, so the model never hard-codes an index.
NOOP = SPEC.type_index("Noop")
MOVE = SPEC.type_index("Move")
ATTACK_MOVE = SPEC.type_index("AttackMove")
ATTACK = SPEC.type_index("Attack")
HARVEST = SPEC.type_index("Harvest")
BUILD = SPEC.type_index("Build")
STOP = SPEC.type_index("Stop")
HOLD = SPEC.type_index("Hold")
TRAIN = SPEC.type_index("Train")
CANCEL_TRAIN = SPEC.type_index("CancelTrain")
SET_RALLY = SPEC.type_index("SetRally")

MULTI_SELECT = (MOVE, ATTACK_MOVE, ATTACK, HARVEST, STOP, HOLD)
SINGLE_SELECT = (BUILD, TRAIN, CANCEL_TRAIN, SET_RALLY)
USES_TARGET = (ATTACK, HARVEST)
USES_LOCATION = (MOVE, ATTACK_MOVE, BUILD, SET_RALLY)
USES_ENTITY_TYPE = (BUILD, TRAIN)
