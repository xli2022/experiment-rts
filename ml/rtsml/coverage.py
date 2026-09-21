"""Optional DAgger coverage, derived only from actor tensors and teacher labels.

The exact-anchor classifier is for valid teacher labels, never sampled actions:
a sampled Build may decode through a different sub-cell fallback. Telemetry
counts observations/data rows, not unique sites, game events or executed orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import json

import numpy as np

from .spec import SPEC_PATH

INVALID_LABEL = -1
UNSUPPORTED_OBSERVATION = -2
NON_BUILD = 0
OTHER_BUILD = 1
EXACT_RESUME = 2
NAMES = {
    INVALID_LABEL: "invalidLabelSkipped",
    UNSUPPORTED_OBSERVATION: "unsupportedObservationSkipped",
    NON_BUILD: "nonBuild",
    OTHER_BUILD: "otherBuild",
    EXACT_RESUME: "exactResume",
}
# Public gameplay constants, not hidden state. The generated SPEC currently
# records feature names but does not export map dimensions or footprints.
MAP_SIZES = {"Lanes": 128, "Quarters": 152}
FOOTPRINTS = {"CommandPost": 4, "Depot": 2, "Barracks": 3,
              "Turret": 2, "Factory": 3, "Airport": 3}


@lru_cache(maxsize=1)
def read_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def classify_teacher_labels(arrays, spec: dict | None = None) -> np.ndarray:
    """One status per batch row; inputs must be paired with valid teacher labels.

    Negative labels and malformed Build label fields are skipped. `otherBuild`
    means no exact represented own unfinished site matches; only the upstream
    valid-teacher contract establishes that this is a fresh foundation intent.
    This is not a replacement for simulator legality or exact execution checks.
    """
    spec = read_spec() if spec is None else spec
    features = {name: i for i, name in enumerate(spec["entities"]["features"])}
    scalars = {name: i for i, name in enumerate(spec["scalars"])}
    action_types = spec["actions"]["types"]
    n, f = spec["entities"]["rows"], len(features)
    cells, step, sub_count = spec["grid"]["size"], spec["grid"]["cellTiles"], spec["actions"]["sub"]
    if sub_count != step * step:
        raise ValueError("unsupported cell/sub geometry")
    def actor(name):
        key = name if name in arrays else "obs_" + name
        if key not in arrays:
            raise ValueError(f"missing actor array {name}")
        return np.asarray(arrays[key])
    if "label" not in arrays:
        raise ValueError("requires teacher `label`; sampled `action`/`actions` cannot be classified")
    labels = np.asarray(arrays["label"])
    if labels.ndim != 2 or labels.shape[1] != spec["actions"]["ints"] or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("teacher label must be integer [B,ACTION_INTS]")
    b = len(labels)
    entities, entity_mask, hud = actor("entities"), actor("entity_mask"), actor("scalars")
    if entities.shape != (b, n, f) or entity_mask.shape != (b, n) or hud.shape != (b, len(scalars)):
        raise ValueError("actor shapes do not match the supplied SPEC and label batch")
    result = np.full(b, INVALID_LABEL, dtype=np.int8)
    known = (labels[:, 0] >= 0) & (labels[:, 0] < len(action_types))
    result[known] = NON_BUILD
    build = action_types.index("Build")
    type_names = [name for name in spec["entities"]["features"] if name.startswith("type:")]
    type_columns = {entity_type: features[name] for entity_type, name in enumerate(type_names)}
    by_type = {features["type:" + name]: footprint for name, footprint in FOOTPRINTS.items()}
    for i in np.flatnonzero(known & (labels[:, 0] == build)):
        result[i] = INVALID_LABEL
        entity_type, cell, sub, worker = map(int, labels[i, [1, 3, 4, 5]])
        type_column = type_columns.get(entity_type)
        if (type_column not in by_type or not 0 <= cell < cells * cells
                or not 0 <= sub < sub_count or not 0 <= worker < n
                or labels[i, 2] != -1 or np.any(labels[i, 6:] != -1)):
            continue
        selected = entities[i, worker]
        if (entity_mask[i, worker] != 1 or selected[features["rel:own"]] != 1
                or selected[features["type:Worker"]] != 1):
            continue
        footprint = by_type[type_column]
        layout = [name for name in MAP_SIZES if hud[i, scalars["layout:" + name]] == 1]
        if len(layout) != 1 or any(hud[i, scalars["layout:" + name]] not in (0, 1) for name in MAP_SIZES):
            result[i] = UNSUPPORTED_OBSERVATION
            continue
        size = MAP_SIZES[layout[0]]
        tx, ty = step * (cell % cells) + sub % step, step * (cell // cells) + sub // step
        if not (0 <= tx <= size - footprint and 0 <= ty <= size - footprint):
            continue
        e = entities[i]
        sites = ((entity_mask[i] == 1) & (e[:, features["rel:own"]] == 1)
                 & (e[:, type_column] == 1) & (e[:, features["build:Complete"]] == 0)
                 & ((e[:, features["build:Site"]] == 1) | (e[:, features["build:UnderConstruction"]] == 1)))
        positions = e[sites][:, [features["x"], features["y"]]].astype(np.float64) * size
        # The encoder floors building centers to canonical integer tiles before
        # float32 normalization. Round, never floor the reconstructed floats.
        rounded = np.rint(positions)
        if not np.isfinite(positions).all() or np.any(np.abs(positions - rounded) > 1e-3):
            result[i] = UNSUPPORTED_OBSERVATION
            continue
        anchors = rounded.astype(np.int64) - footprint // 2
        result[i] = EXACT_RESUME if np.any(np.all(anchors == (tx, ty), axis=1)) else OTHER_BUILD
    return result


@dataclass(frozen=True)
class CoverageBatch:
    action_types: np.ndarray
    building_types: np.ndarray
    resume_status: np.ndarray
    represented_orphans: np.ndarray
    """[observations, building types], counts only the represented own rows."""


class DaggerCoverage:
    """Cumulative counters; methods never sample or modify their input arrays."""

    STAGES = ("offered", "retainedFresh", "mixedTraining", "expertSelections")

    def __init__(self):
        self.spec = read_spec()
        self.features = {name: i for i, name in enumerate(self.spec["entities"]["features"])}
        self.action_names = self.spec["actions"]["types"]
        self.build = self.action_names.index("Build")
        type_names = [name[5:] for name in self.spec["entities"]["features"] if name.startswith("type:")]
        self.building_types = {name: type_names.index(name) for name in FOOTPRINTS}
        self.counters = {stage: {
            "rows": 0,
            "actionTypes": {name: 0 for name in self.action_names},
            "exactResumesByBuilding": {name: 0 for name in FOOTPRINTS},
            "otherBuildsByBuilding": {name: 0 for name in FOOTPRINTS},
            "unclassifiedBuilds": 0,
            "representedOrphanObservations": 0,
            "representedOrphanSitesByBuilding": {name: 0 for name in FOOTPRINTS},
        } for stage in self.STAGES}

    def classify(self, arrays: dict[str, np.ndarray]) -> CoverageBatch:
        """Read one actor/teacher batch once, then reuse it for several stages."""
        status = classify_teacher_labels(arrays, self.spec)
        entities, mask = arrays["entities"], arrays["entity_mask"]
        f = self.features
        own_orphan = ((mask == 1) & (entities[:, :, f["rel:own"]] == 1)
                      & (entities[:, :, f["build:Complete"]] == 0)
                      & ((entities[:, :, f["build:Site"]] == 1)
                         | (entities[:, :, f["build:UnderConstruction"]] == 1))
                      & (entities[:, :, f["hasAssignedBuilder"]] == 0))
        orphans = np.stack([(own_orphan & (entities[:, :, f["type:" + name]] == 1)).sum(axis=1)
                            for name in self.building_types], axis=1)
        return CoverageBatch(arrays["label"][:, 0].copy(), arrays["label"][:, 1].copy(), status, orphans)

    def add(self, stage: str, batch: CoverageBatch, selected: np.ndarray | None = None) -> None:
        """Count selected data rows once; mixedTraining is before epoch repeats."""
        keep = np.ones(len(batch.action_types), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
        if keep.shape != batch.action_types.shape:
            raise ValueError("coverage selection must have one flag per observation")
        types = batch.action_types[keep]
        if ((types < 0) | (types >= len(self.action_names))).any():
            raise ValueError("coverage stages require valid teacher action types")
        counter = self.counters[stage]
        counter["rows"] += len(types)
        for action, name in enumerate(self.action_names):
            counter["actionTypes"][name] += int((types == action).sum())
        status, building = batch.resume_status[keep], batch.building_types[keep]
        counter["unclassifiedBuilds"] += int(((types == self.build) & (status < 0)).sum())
        orphans = batch.represented_orphans[keep]
        counter["representedOrphanObservations"] += int((orphans.sum(axis=1) > 0).sum())
        for column, (name, entity_type) in enumerate(self.building_types.items()):
            counter["exactResumesByBuilding"][name] += int(((status == EXACT_RESUME) & (building == entity_type)).sum())
            counter["otherBuildsByBuilding"][name] += int(((status == OTHER_BUILD) & (building == entity_type)).sum())
            counter["representedOrphanSitesByBuilding"][name] += int(orphans[:, column].sum())

    def snapshot(self) -> dict:
        """Detached so later counter updates cannot change earlier log/checkpoint data."""
        return {"version": 1, **copy.deepcopy(self.counters)}

