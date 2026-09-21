"""Optional training objectives preserve default behavior and real outcomes.

No game or optimizer steps: CLI setup is intercepted before its environment
is created; episode reward behavior is covered by tests/mlReward.test.ts.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from rtsml import ppo
from rtsml.env import LANES, EnvConfig, slot
from rtsml.evaluate import MatchResult, summarise
from rtsml.league import League
from rtsml.util import save_checkpoint


def config(**kwargs):
    return EnvConfig(seed=42, layout=LANES, slots=[slot("policy"), slot("idle")], **kwargs)


@pytest.mark.parametrize("reward", [0, -1, -.4, 1])
def test_draw_reward_crosses_the_environment_configuration_boundary(reward):
    assert config().to_json()["drawReward"] == 0
    assert config(draw_reward=reward).to_json()["drawReward"] == reward


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1.01, 1.01, True, None, "0"])
def test_invalid_draw_reward_is_rejected_at_construction_and_serialization(bad):
    with pytest.raises(ValueError, match="draw_reward must be finite"):
        config(draw_reward=bad)
    cfg = config()
    cfg.draw_reward = bad
    with pytest.raises(ValueError, match="draw_reward must be finite"):
        cfg.to_json()


def test_default_league_sampling_matches_explicit_half_credit():
    a, b = League(seed=7), League(seed=7, draw_credit=.5)
    for i, outcomes in enumerate(([True, None, False], [None, None], [False, False])):
        for outcome in outcomes:
            a.report(a.members[i], outcome)
            b.report(b.members[i], outcome)
    assert a.members[0].wins == 1.5
    np.testing.assert_array_equal(a.weights(), b.weights())
    assert [m.name for m in a.sample(100)] == [m.name for m in b.sample(100)]


@pytest.mark.parametrize("credit", [0, .25, .5, 1])
def test_league_persists_draw_credit_and_sampling_statistics(tmp_path, credit):
    league = League(seed=8, draw_credit=credit)
    for outcome in [True, None, False]:
        league.report(league.members[0], outcome)
    path = tmp_path / "league.pt"
    league.save(path)
    restored = League.load(path, seed=8)
    assert restored.draw_credit == credit
    assert restored.members[0].wins == 1 + credit
    assert restored.members[0].games == 3
    np.testing.assert_array_equal(restored.weights(), league.weights())
    restored.report(restored.members[0], None)
    assert restored.members[0].wins == 1 + 2 * credit


def test_old_league_files_keep_half_credit_and_existing_totals(tmp_path):
    legacy = League()
    for outcome in [True, None, False]:
        legacy.report(legacy.members[0], outcome)
    path = tmp_path / "legacy.pt"
    torch.save({"members": [m.to_dict() for m in legacy.members], "floor": legacy.floor}, path)
    restored = League.load(path)
    assert restored.draw_credit == .5
    assert restored.members[0].wins == 1.5 and restored.members[0].games == 3
    restored.report(restored.members[0], None)
    assert restored.members[0].wins == 2 and restored.members[0].games == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -.01, 1.01, True, None, "0"])
def test_league_rejects_invalid_draw_credit_including_saved_files(tmp_path, bad):
    with pytest.raises(ValueError, match="draw_credit must be finite"):
        League(draw_credit=bad)
    path = tmp_path / "invalid.pt"
    torch.save({"members": [], "floor": .05, "draw_credit": bad}, path)
    with pytest.raises(ValueError, match="draw_credit must be finite"):
        League.load(path)


def test_draws_remain_reported_as_draws_when_their_league_credit_is_zero():
    outcomes = [True, None, False, None]
    league = League(ladder=(10,), draw_credit=0)
    for outcome in outcomes:
        league.report(league.members[0], outcome)
    results = [MatchResult(i, LANES, 0, outcome, 400, 0) for i, outcome in enumerate(outcomes)]
    report = summarise(results)
    assert report["wins"] == 1 and report["draws"] == 2 and report["winRate"] == .25
    assert league.members[0].wins == 1 and league.members[0].games == 4
    assert outcomes == [True, None, False, None]


@pytest.mark.parametrize("options,reward,credit,gamma,time_cost", [
    ([], 0, .5, .999, .00002),
    (["--draw-reward=-1", "--league-draw-credit=0", "--gamma=1", "--time-cost=0"], -1, 0, 1, 0),
])
def test_ppo_cli_passes_objective_controls_and_records_checkpoint_hparams(
    tmp_path, monkeypatch, tiny_policy, options, reward, credit, gamma, time_cost,
):
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, tiny_policy, "bc", {
        "model": {"d": 32, "heads": 2, "layers": 1, "torso": 64}, "layout": "lanes",
    })
    captured = {}
    add_snapshot = League.add_snapshot

    def capture_snapshot(self, name, policy, hparams):
        captured["hparams"] = deepcopy(hparams)
        captured["credit"] = self.draw_credit
        return add_snapshot(self, name, policy, hparams)

    class BeforeAnyGameOrTraining(Exception):
        pass

    def stop_at_environment(groups):
        captured["groups"] = groups
        raise BeforeAnyGameOrTraining

    monkeypatch.setattr(League, "add_snapshot", capture_snapshot)
    monkeypatch.setattr(ppo, "BunVectorEnv", stop_at_environment)
    with pytest.raises(BeforeAnyGameOrTraining):
        ppo.main([
            "--init", str(checkpoint), "--out", str(tmp_path / "unused"), "--device", "cpu",
            "--procs", "1", "--envs", "2", *options,
        ])
    assert not (tmp_path / "unused").exists()
    assert captured["credit"] == credit
    settings = captured["hparams"]["ppo"]
    assert settings["draw_reward"] == reward and settings["league_draw_credit"] == credit
    for group in captured["groups"]:
        for cfg in group:
            wire = cfg.to_json()
            assert wire["drawReward"] == reward
            assert wire["gamma"] == gamma and wire["timeCost"] == time_cost


@pytest.mark.parametrize("option,bad", [
    ("draw-reward", "nan"), ("draw-reward", "inf"), ("draw-reward", "-inf"),
    ("draw-reward", "-1.01"), ("draw-reward", "1.01"),
    ("league-draw-credit", "nan"), ("league-draw-credit", "inf"),
    ("league-draw-credit", "-inf"), ("league-draw-credit", "-.01"), ("league-draw-credit", "1.01"),
])
def test_cli_rejects_invalid_objective_controls_before_model_or_environment_setup(monkeypatch, capsys, option, bad):
    def unexpected_setup(*args):
        raise AssertionError("invalid controls must fail before any setup")
    monkeypatch.setattr(ppo, "set_seed", unexpected_setup)
    with pytest.raises(SystemExit) as error:
        ppo.main([f"--{option}={bad}"])
    assert error.value.code == 2
    assert f"{option} must be finite" in capsys.readouterr().err
