"""DAgger trains expert targets from learner states and replays older visits."""

import json

import numpy as np
import pytest
import torch

from rtsml import dagger
from rtsml.env import LANES, QUARTERS
from rtsml.imitation import STORED
from rtsml.model import Policy
from rtsml.spec import BUILD
from rtsml.util import load_checkpoint, save_checkpoint

from conftest import requires_bun


@pytest.mark.parametrize("layout,map_id,teammates", [("lanes", LANES, 1), ("quarters", QUARTERS, 2)])
def test_learner_configs_cover_both_teams_with_experts_only_as_labels(layout, map_id, teammates):
    groups = dagger.learner_configs(2, 2, 700_000, 24_000, layout)
    configs = [cfg for group in groups for cfg in group]
    assert len({cfg.seed for cfg in configs}) == 4
    for i, cfg in enumerate(configs):
        assert cfg.layout == map_id and cfg.expert_labels
        kinds = [spec["kind"] for spec in cfg.slots]
        expected = ["policy"] * teammates + ["scripted"] * teammates
        assert kinds == (expected if i % 2 == 0 else list(reversed(expected)))
        assert all(spec["thinkInterval"] == 10 for spec in cfg.slots if spec["kind"] == "scripted")


def test_replay_mixes_previous_visits_and_keeps_their_targets_paired():
    replay = dagger.Replay(3, np.random.default_rng(17))
    older = {name: np.arange(8, dtype=np.float32).reshape(8, 1) for name in STORED}
    replay.add(older)
    assert replay.size == 3 and replay.seen == 8
    fresh = {name: np.array([[100], [101]], dtype=np.float32) for name in STORED}
    mixed = replay.mix(fresh, 2)
    assert len(mixed["label"]) == 4
    np.testing.assert_array_equal(mixed["label"][:2], fresh["label"])
    assert (mixed["label"][2:] < 8).all()
    for name in STORED:
        np.testing.assert_array_equal(mixed[name], mixed["label"])
    # Sampling cannot mutate either the fresh observations or the reservoir.
    mixed["label"].fill(-1)
    assert (fresh["label"] >= 100).all() and (replay.data["label"] >= 0).all()


@requires_bun
@pytest.mark.parametrize("build_weight", [1.0, 4.0])
def test_dagger_continues_checkpoint_on_learner_states_with_fresh_and_replay_rows(tmp_path, monkeypatch, build_weight):
    shape = {"d": 32, "heads": 2, "layers": 1, "torso": 64}
    torch.manual_seed(13)
    initial = Policy(**shape)
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, initial, "bc", {"model": shape, "layout": "lanes"})
    sampled = []
    trained = []
    original_act = Policy.act
    original_train = dagger.train_on

    def checked_act(self, *args, **kwargs):
        assert not torch.is_grad_enabled() and not self.training
        sampled.append(True)
        return original_act(self, *args, **kwargs)

    def checked_train(policy, opt, data, *args, **kwargs):
        labels = data["label"]
        assert (labels[:, 0] >= 0).all()
        assert data["mask_type"][np.arange(len(labels)), labels[:, 0]].all()
        if build_weight == 1:
            assert kwargs["weights"] is None
        else:
            np.testing.assert_array_equal(kwargs["weights"], np.where(labels[:, 0] == BUILD, build_weight, 1.0))
        trained.append(labels.copy())
        return original_train(policy, opt, data, *args, **kwargs)

    monkeypatch.setattr(Policy, "act", checked_act)
    monkeypatch.setattr(dagger, "train_on", checked_train)
    out = tmp_path / "dagger"
    assert dagger.main([
        "--init", str(checkpoint), "--out", str(out), "--layout", "lanes",
        "--steps", "12", "--procs", "1", "--envs", "2", "--buffer", "4",
        "--replay", "8", "--batch", "4", "--epochs", "1", "--noop-keep", "1",
        "--expert-start", "0", "--expert-end", "0", "--val-labels", "4",
        "--val-every", "1", "--keep-every", "4", "--max-ticks", "400",
        "--seed", "23", "--device", "cpu",
        *([] if build_weight == 1 else ["--build-weight", str(build_weight)]),
    ]) == 0
    assert sampled and trained and any((labels[:, 0] > 0).any() for labels in trained)
    records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [row["labels"] for row in records] == [4, 8, 12]
    assert [row["trainedRows"] for row in records] == [4, 8, 8]
    assert all(np.isfinite(row["loss"]) for row in records)
    saved = load_checkpoint(out / "last.pt")
    assert saved["kind"] == "dagger" and saved["hparams"]["model"] == shape
    assert saved["hparams"]["layout"] == "lanes"
    assert saved["hparams"]["build_weight"] == build_weight
    assert saved["metrics"]["val"]["perActionType"]["Build"]["labels"] >= 0
    assert all(torch.isfinite(value).all() for value in saved["model"].values())
    assert any(not torch.equal(value, initial.state_dict()[name]) for name, value in saved["model"].items()
               if not name.startswith(("critic_mlp.", "value_head.")))
    assert (out / "best.pt").exists() and (out / "ckpt3.pt").exists()


@pytest.mark.parametrize("weight", ["0", "-1", "nan", "inf", "-inf"])
def test_build_weight_rejects_nonpositive_or_nonfinite_before_loading(weight, tmp_path, monkeypatch, capsys):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid options must fail before checkpoint or environment work")
    monkeypatch.setattr(dagger, "load_checkpoint", unexpected_load)
    with pytest.raises(SystemExit) as error:
        dagger.main(["--init", str(tmp_path / "absent.pt"), "--out", str(tmp_path),
                     "--layout", "lanes", f"--build-weight={weight}"])
    assert error.value.code == 2
    assert "build-weight must be finite and positive" in capsys.readouterr().err
