"""Additive own-order features retain learned weights without claiming new wins."""

from pathlib import Path

import pytest
import torch

from rtsml.export import example_inputs
from rtsml.migrate_observation import migrate, main
from rtsml.model import Policy
from rtsml.spec import SPEC, load_spec


def old_checkpoint():
    legacy = load_spec(Path(__file__).parents[1] / "rtsml" / "spec-v3.json")
    architecture = {"d": 32, "heads": 2, "layers": 1, "torso": 64}
    torch.manual_seed(117)
    policy = Policy(spec=legacy, **architecture).eval()
    return policy, {"spec_version": 3, "kind": "bc", "hparams": {"model": architecture, "layout": "quarters"},
                    "model": policy.state_dict(), "metrics": {"labels": 400000}}


def test_migration_retains_actions_and_logit_values_with_unseen_features_present():
    original, source = old_checkpoint()
    migrated = migrate(source)
    policy = Policy(**migrated["hparams"]["model"]).eval()
    policy.load_state_dict(migrated["model"])
    assert migrated["spec_version"] == 4 and migrated["metrics"] == {}
    assert migrated["migration"]["sourceMetrics"] == {"labels": 400000}
    assert migrated["migration"]["requiresFreshEvaluation"]
    old_width = original.spec.f
    for name, weight in source["model"].items():
        got = migrated["model"][name]
        if name == "entity_in.weight":
            torch.testing.assert_close(got[:, :old_width], weight, rtol=0, atol=0)
            assert got.shape[1] == SPEC.f and torch.count_nonzero(got[:, old_width:]) == 0
        else:
            torch.testing.assert_close(got, weight, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(14)
    with torch.no_grad():
        for temperature in (0.5, 1.0):
            inputs = example_inputs(8, generator)
            noise = inputs.pop("noise")
            inputs["temperature"].fill_(temperature)
            t = inputs.pop("temperature")
            previous = {**inputs, "entities": inputs["entities"][..., :old_width]}
            torch.testing.assert_close(policy.type_logits(policy.encode(inputs)),
                                       original.type_logits(original.encode(previous)), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(policy.act(inputs, noise, t), original.act(previous, noise, t), rtol=0, atol=0)


def test_migration_rejects_other_codec_and_nonfinite_weights():
    _, source = old_checkpoint()
    with pytest.raises(ValueError, match="codec-3"):
        migrate({**source, "spec_version": 2})
    source["model"]["entity_in.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        migrate(source)


def test_cli_preserves_source_and_will_not_overwrite_checkpoint(tmp_path):
    _, source = old_checkpoint()
    initial = tmp_path / "source.pt"
    output = tmp_path / "new" / "migrated.pt"
    torch.save(source, initial)
    before = initial.read_bytes()
    assert main(["--ckpt", str(initial), "--out", str(output)]) == 0
    assert initial.read_bytes() == before
    migrated = torch.load(output, weights_only=False)
    assert migrated["migration"]["source"] == str(initial.resolve())
    with pytest.raises(SystemExit):
        main(["--ckpt", str(initial), "--out", str(output)])
