"""Evaluation artifacts identify the actual policy, sampling and match results."""

import hashlib
import json

import pytest
import torch

from rtsml import evaluate
from rtsml.env import LANES
from rtsml.model import Policy
from rtsml.util import save_checkpoint

from conftest import requires_bun


def test_ladder_preserves_each_seed_seat_and_draw(monkeypatch):
    observed = []

    def fake_play(policy, opponent, seeds, layout, seat, procs, device, temperature, max_ticks, seed):
        observed.append((opponent["thinkInterval"], seat, temperature, max_ticks, seed))
        return [evaluate.MatchResult(s, layout, seat, outcome, 100 + i, 4)
                for i, (s, outcome) in enumerate(zip(seeds, [True, False, None]))]

    monkeypatch.setattr(evaluate, "play", fake_play)
    partials = []
    result = evaluate.evaluate_ladder(None, (10, 20), [101, 105, 109], LANES, 2,
                                     torch.device("cpu"), 0.5, 400, 17,
                                     lambda report: partials.append(json.loads(json.dumps(report))))
    assert result["state"] == "completed"
    assert result["mapSeeds"] == [101, 105, 109]
    assert result["temperature"] == 0.5 and result["maxTicks"] == 400
    assert result["samplingSeed"] == 17
    assert result["procs"] == 2 and result["device"] == "cpu"
    assert observed == [(rung, seat, 0.5, 400, 17) for rung in (10, 20) for seat in (0, 1)]
    for rung in result["rungs"].values():
        for seat in (0, 1):
            summary = rung[f"seat{seat}"]
            assert summary["wins"] == 1 and summary["draws"] == 1 and summary["matches"] == 3
            assert [row["seed"] for row in summary["results"]] == [101, 105, 109]
            assert [row["won"] for row in summary["results"]] == [True, False, None]
            assert all(row["seat"] == seat for row in summary["results"])
    assert len(partials) == 4 and partials[0]["state"] == "running"
    assert "seat1" not in partials[0]["rungs"]["scripted@10"]


def initial_checkpoint(tmp_path):
    shape = {"d": 32, "heads": 2, "layers": 1, "torso": 64}
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(checkpoint, Policy(**shape), "bc", {"model": shape, "layout": "lanes"})
    return checkpoint


@requires_bun
def test_cli_records_frozen_policy_identity_and_actual_capped_matches(tmp_path):
    checkpoint = initial_checkpoint(tmp_path)
    report_file = tmp_path / "nested" / "evaluation.json"
    assert evaluate.main(["--ckpt", str(checkpoint), "--ladder", "10", "--seeds", "2",
                          "--seed0", "177", "--procs", "1", "--temperature", "0.5",
                          "--sampling-seed", "9", "--max-ticks", "8", "--device", "cpu",
                          "--out", str(report_file)]) == 0
    report = json.loads(report_file.read_text())
    assert report["state"] == "completed"
    assert report["checkpointSha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["temperature"] == 0.5 and report["samplingSeed"] == 9
    for seat in (0, 1):
        summary = report["rungs"]["scripted@10"][f"seat{seat}"]
        assert summary["wins"] == 0 and summary["draws"] == 2
        assert [row["seed"] for row in summary["results"]] == [177, 178]
        assert all(row["ticks"] == 8 and row["won"] is None for row in summary["results"])


def test_failed_evaluation_retains_completed_seat_results(tmp_path, monkeypatch):
    checkpoint = initial_checkpoint(tmp_path)
    output = tmp_path / "evaluation.json"

    def fake_play(policy, opponent, seeds, layout, seat, *args):
        if seat == 1:
            raise RuntimeError("bridge exited")
        return [evaluate.MatchResult(seeds[0], layout, seat, False, 200, 5)]

    monkeypatch.setattr(evaluate, "play", fake_play)
    with pytest.raises(RuntimeError, match="bridge exited"):
        evaluate.main(["--ckpt", str(checkpoint), "--ladder", "10", "--seeds", "1",
                       "--device", "cpu", "--out", str(output)])
    result = json.loads(output.read_text())
    assert result["state"] == "failed" and result["error"] == "bridge exited"
    assert result["rungs"]["scripted@10"]["seat0"]["matches"] == 1
    assert "seat1" not in result["rungs"]["scripted@10"]


def test_evaluation_refuses_wrong_layout_before_playing(tmp_path):
    checkpoint = initial_checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        evaluate.main(["--ckpt", str(checkpoint), "--layout", "quarters", "--device", "cpu"])
