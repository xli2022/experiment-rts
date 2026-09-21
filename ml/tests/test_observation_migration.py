"""Adjacent codec migration preserves weights without reusing old win claims."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from rtsml.export import example_inputs
from rtsml.migrate_observation import migrate, main
from rtsml.model import Policy
from rtsml.spec import SPEC, load_spec


def old_checkpoint(version=3):
    legacy = load_spec(Path(__file__).parents[1] / "rtsml" / f"spec-v{version}.json")
    architecture = {"d": 32, "heads": 2, "layers": 1, "torso": 64}
    torch.manual_seed(117)
    policy = Policy(spec=legacy, **architecture).eval()
    return policy, {"spec_version": version, "kind": "bc", "hparams": {"model": architecture, "layout": "quarters"},
                    "model": policy.state_dict(), "metrics": {"labels": 400000}}


@pytest.mark.parametrize("source_version,target_version", [(3, 4), (4, 5)])
def test_migration_retains_actions_and_logit_values_with_unseen_features_present(source_version, target_version):
    original, source = old_checkpoint(source_version)
    migrated = migrate(source, to_version=target_version)
    target = load_spec(Path(__file__).parents[1] / "rtsml" / f"spec-v{target_version}.json")
    policy = Policy(spec=target, **migrated["hparams"]["model"]).eval()
    policy.load_state_dict(migrated["model"])
    assert migrated["spec_version"] == target_version and migrated["metrics"] == {}
    assert migrated["kind"] == source["kind"] and migrated["hparams"] == source["hparams"]
    assert source["spec_version"] == source_version and source["metrics"] == {"labels": 400000}
    assert migrated["migration"]["fromSpec"] == source_version
    assert migrated["migration"]["toSpec"] == target_version
    assert migrated["migration"]["newFeatures"] == list(target.entity_features[original.spec.f:])
    assert migrated["migration"]["sourceMetrics"] == {"labels": 400000}
    assert migrated["migration"]["requiresFreshEvaluation"]
    old_width = original.spec.f
    for name, weight in source["model"].items():
        got = migrated["model"][name]
        if name == "entity_in.weight":
            torch.testing.assert_close(got[:, :old_width], weight, rtol=0, atol=0)
            assert got.shape[1] == target.f and torch.count_nonzero(got[:, old_width:]) == 0
        else:
            torch.testing.assert_close(got, weight, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(14)
    with torch.no_grad():
        for temperature in (0.5, 1.0):
            inputs = example_inputs(8, generator)
            inputs["entities"] = inputs["entities"][..., :target.f].contiguous()
            # Include both staffing states. Even nonzero new input columns
            # must have no effect before continuation learns nonzero weights.
            if target_version == 5:
                inputs["entities"][..., -1] = torch.arange(target.n_ent) % 2
            noise = inputs.pop("noise")
            inputs["temperature"].fill_(temperature)
            t = inputs.pop("temperature")
            previous = {**inputs, "entities": inputs["entities"][..., :old_width]}
            torch.testing.assert_close(policy.type_logits(policy.encode(inputs)),
                                       original.type_logits(original.encode(previous)), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(policy.act(inputs, noise, t), original.act(previous, noise, t), rtol=0, atol=0)


@pytest.mark.parametrize("source_version,target_version", [(3, 4), (4, 5), (5, 6)])
def test_migration_rejects_nonfinite_weights(source_version, target_version):
    _, source = old_checkpoint(source_version)
    source["model"]["entity_in.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        migrate(source, to_version=target_version)


@pytest.mark.parametrize("source,target", [(2, 4), (3, 5), (4, 4), (5, 5), (5, 4), (4, 6), (6, 6), (None, 5)])
def test_migration_rejects_nonadjacent_or_unknown_codecs(source, target):
    with pytest.raises(ValueError, match="adjacent codec"):
        migrate({"spec_version": source}, to_version=target)


@pytest.mark.parametrize("change", [
    {"action_types": tuple(reversed(SPEC.action_types))},
    {"scalars": SPEC.scalars + ("unexpected",)},
    {"entity_features": ("renamed",) + SPEC.entity_features[1:]},
    {"entity_features": SPEC.entity_features[:-1]},
])
def test_migration_refuses_nonadditive_contract_changes(monkeypatch, change):
    import rtsml.migrate_observation as migration
    _, source = old_checkpoint(4)
    original_loader = migration.load_spec
    monkeypatch.setattr(migration, "load_spec", lambda path: replace(original_loader(path), **change)
                        if path.name == "spec-v5.json" else original_loader(path))
    with pytest.raises(ValueError, match="not an additive"):
        migrate(source, to_version=5)


def test_sequential_migration_keeps_source_provenance_and_original_weights():
    original, source = old_checkpoint(3)
    intermediate = migrate(source, to_version=4)
    intermediate["migration"]["sourceSha256"] = "original-source-hash"
    codec5 = migrate(intermediate, to_version=5)
    final = migrate(codec5)
    assert final["spec_version"] == 6
    assert final["migration"]["sourceMigration"] == codec5["migration"]
    assert final["migration"]["sourceMigration"]["sourceMigration"] == intermediate["migration"]
    assert final["migration"]["sourceMigration"]["sourceMigration"]["sourceMetrics"] == {"labels": 400000}
    weight = final["model"]["entity_in.weight"]
    torch.testing.assert_close(weight[:, :original.spec.f], source["model"]["entity_in.weight"], rtol=0, atol=0)
    assert torch.count_nonzero(weight[:, original.spec.f:]) == 0


@pytest.mark.parametrize("source_version,target_version", [(3, 4), (4, 5), (5, 6)])
def test_cli_preserves_source_and_will_not_overwrite_checkpoint(tmp_path, source_version, target_version):
    _, source = old_checkpoint(source_version)
    initial = tmp_path / "source.pt"
    output = tmp_path / "new" / "migrated.pt"
    torch.save(source, initial)
    before = initial.read_bytes()
    options = ["--to-version", str(target_version)]
    assert main(["--ckpt", str(initial), "--out", str(output), *options]) == 0
    assert initial.read_bytes() == before
    migrated = torch.load(output, weights_only=False)
    assert migrated["migration"]["source"] == str(initial.resolve())
    assert migrated["migration"]["sourceSha256"] == hashlib.sha256(before).hexdigest()
    assert migrated["spec_version"] == target_version
    with pytest.raises(SystemExit):
        main(["--ckpt", str(initial), "--out", str(output), *options])


def test_codec6_retains_every_weight_but_explicitly_requires_new_behavior_evaluation():
    _, source = old_checkpoint(5)
    source["migration"] = {"fromSpec": 4, "sourceSha256": "parent-checkpoint"}
    migrated = migrate(source)
    assert migrated["spec_version"] == 6
    assert migrated["metrics"] == {}
    assert source["metrics"] == {"labels": 400000}
    assert migrated["hparams"] == source["hparams"]
    assert set(migrated["model"]) == set(source["model"])
    for name, weight in source["model"].items():
        torch.testing.assert_close(migrated["model"][name], weight, rtol=0, atol=0)
    proof = migrated["migration"]
    assert proof["newFeatures"] == []
    assert proof["behaviorChanged"] is True
    assert proof["parityClaimed"] is False
    assert proof["requiresFreshEvaluation"] is True
    assert proof["sourceMigration"] == source["migration"]


@pytest.mark.parametrize("change", [
    {"action_types": tuple(reversed(SPEC.action_types))},
    {"n_ent": SPEC.n_ent + 1},
    {"entity_features": ("renamed",) + SPEC.entity_features[1:]},
    {"scalars": SPEC.scalars + ("unexpected",)},
])
def test_codec6_refuses_shape_or_feature_contract_changes(monkeypatch, change):
    import rtsml.migrate_observation as migration
    _, source = old_checkpoint(5)
    monkeypatch.setattr(migration, "SPEC", replace(SPEC, **change))
    with pytest.raises(ValueError, match="identical tensor and action"):
        migrate(source)


def test_codec6_refuses_an_incomplete_checkpoint():
    _, source = old_checkpoint(5)
    del source["model"]["type_head.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        migrate(source)
