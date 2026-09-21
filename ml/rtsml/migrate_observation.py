"""Migrate policy weights through adjacent codec-3 -> 4 -> 5 -> 6 observations.

New entity input columns in versions 4 and 5 start at zero. Version 6 retains
all weights and shapes but changes row selection and public map summaries;
identical weights therefore do not imply identical gameplay behavior.
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


def migrate(checkpoint: dict[str, Any], *, to_version: int = 6) -> dict[str, Any]:
    source_version = checkpoint.get("spec_version")
    if (source_version, to_version) not in ((3, 4), (4, 5), (5, 6)) or SPEC.version != 6:
        raise ValueError("migration requires adjacent codec versions: 3 -> 4, 4 -> 5 or 5 -> 6")
    legacy = load_spec(Path(__file__).with_name(f"spec-v{source_version}.json"))
    target = SPEC if to_version == SPEC.version else load_spec(Path(__file__).with_name(f"spec-v{to_version}.json"))
    if legacy.version != source_version or target.version != to_version:
        raise ValueError("frozen codec version does not match its migration endpoint")
    previous, current = asdict(legacy), asdict(target)
    for metadata in (previous, current):
        metadata.pop("version")
        metadata.pop("entity_features")
    semantic_only = (source_version, to_version) == (5, 6)
    if semantic_only:
        if previous != current or target.entity_features != legacy.entity_features:
            raise ValueError("codec-6 migration requires identical tensor and action contracts")
    elif previous != current or target.entity_features[:legacy.f] != legacy.entity_features or target.f <= legacy.f:
        raise ValueError("codec changes are not an additive entity-feature extension")

    architecture = checkpoint["hparams"].get("model", {})
    old_policy = Policy(spec=legacy, **architecture)
    old_policy.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("checkpoint contains non-finite parameters")
    state = dict(checkpoint["model"])
    if not semantic_only:
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
                      "initialization": "preserve all weights; observation semantics changed" if semantic_only else "zero new entity input columns",
                      "requiresFreshEvaluation": True,
                      **({"behaviorChanged": True, "parityClaimed": False} if semantic_only else {}),
                      "sourceMetrics": checkpoint.get("metrics", {}),
                      "sourceMigration": checkpoint.get("migration")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--to-version", type=int, choices=(4, 5, 6), default=6,
                        help="target codec; migrate adjacent versions explicitly before migrating to 6")
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
    print(f"wrote {args.out}; {migrated['migration']['initialization']}; fresh gameplay evaluation is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
