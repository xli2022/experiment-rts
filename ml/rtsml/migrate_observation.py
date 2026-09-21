"""Migrate policy weights through the additive codec-3 -> 4 -> 5 observations.

New entity input columns start at zero, retaining the learned policy while
allowing further training to use added public own-state information.
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


def migrate(checkpoint: dict[str, Any], *, to_version: int = 5) -> dict[str, Any]:
    source_version = checkpoint.get("spec_version")
    if (source_version, to_version) not in ((3, 4), (4, 5)) or SPEC.version != 5:
        raise ValueError("migration requires adjacent codec versions: 3 -> 4 or 4 -> 5")
    legacy = load_spec(Path(__file__).with_name(f"spec-v{source_version}.json"))
    target = SPEC if to_version == 5 else load_spec(Path(__file__).with_name("spec-v4.json"))
    if legacy.version != source_version or target.version != to_version:
        raise ValueError("frozen codec version does not match its migration endpoint")
    previous, current = asdict(legacy), asdict(target)
    for metadata in (previous, current):
        metadata.pop("version")
        metadata.pop("entity_features")
    if previous != current or target.entity_features[:legacy.f] != legacy.entity_features or target.f <= legacy.f:
        raise ValueError("codec changes are not an additive entity-feature extension")

    architecture = checkpoint["hparams"].get("model", {})
    old_policy = Policy(spec=legacy, **architecture)
    old_policy.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("checkpoint contains non-finite parameters")
    state = dict(checkpoint["model"])
    original = state["entity_in.weight"]
    extended = original.new_zeros((original.shape[0], target.f))
    extended[:, :legacy.f] = original
    state["entity_in.weight"] = extended
    migrated_policy = Policy(spec=target, **architecture)
    migrated_policy.load_state_dict(state, strict=True)
    return {
        **checkpoint,
        "spec_version": target.version,
        "model": state,
        "metrics": {},
        "migration": {"fromSpec": source_version, "toSpec": target.version,
                      "newFeatures": list(target.entity_features[legacy.f:]),
                      "initialization": "zero new entity input columns", "requiresFreshEvaluation": True,
                      "sourceMetrics": checkpoint.get("metrics", {}),
                      "sourceMigration": checkpoint.get("migration")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--to-version", type=int, choices=(4, 5), default=5,
                        help="target codec; migrate codec 3 to 4 explicitly before migrating to 5")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; choose a new checkpoint path")
    payload = args.ckpt.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    try:
        migrated = migrate(checkpoint, to_version=args.to_version)
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
