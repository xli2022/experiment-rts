"""Migrate codec-3 policy weights to the additive codec-4 observations.

New entity input columns start at zero, retaining the learned policy while
allowing further training to use its own movement goals and production queue.
This is not gameplay qualification: evaluate the continued model before export.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .model import Policy
from .spec import SPEC, load_spec


def migrate(checkpoint: dict[str, Any]) -> dict[str, Any]:
    legacy = load_spec(Path(__file__).with_name("spec-v3.json"))
    if legacy.version != 3 or SPEC.version != 4 or checkpoint.get("spec_version") != 3:
        raise ValueError("this migration accepts codec-3 checkpoints and produces codec 4 only")
    previous, current = asdict(legacy), asdict(SPEC)
    for metadata in (previous, current):
        metadata.pop("version")
        metadata.pop("entity_features")
    if previous != current or SPEC.entity_features[:legacy.f] != legacy.entity_features or SPEC.f <= legacy.f:
        raise ValueError("codec changes are not an additive entity-feature extension")

    architecture = checkpoint["hparams"].get("model", {})
    old_policy = Policy(spec=legacy, **architecture)
    old_policy.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("checkpoint contains non-finite parameters")
    state = dict(checkpoint["model"])
    original = state["entity_in.weight"]
    extended = original.new_zeros((original.shape[0], SPEC.f))
    extended[:, :legacy.f] = original
    state["entity_in.weight"] = extended
    migrated_policy = Policy(**architecture)
    migrated_policy.load_state_dict(state, strict=True)
    return {
        **checkpoint,
        "spec_version": SPEC.version,
        "model": state,
        "metrics": {},
        "migration": {"fromSpec": 3, "toSpec": 4, "newFeatures": list(SPEC.entity_features[legacy.f:]),
                      "initialization": "zero new entity input columns", "requiresFreshEvaluation": True,
                      "sourceMetrics": checkpoint.get("metrics", {})},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; choose a new checkpoint path")
    payload = args.ckpt.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    try:
        migrated = migrate(checkpoint)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    migrated["migration"].update(source=str(args.ckpt.resolve()),
                                 sourceSha256=hashlib.sha256(payload).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.out)
    print(f"wrote {args.out}; new feature columns are zero, fresh gameplay evaluation is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
