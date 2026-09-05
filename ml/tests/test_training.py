"""The loops end to end, at toy size: they run, write what they should, and play a match."""

import torch

from rtsml import evaluate, imitation, ppo
from rtsml.env import LANES, slot
from rtsml.model import Policy
from rtsml.util import load_checkpoint

from conftest import requires_bun

pytestmark = requires_bun


def test_imitation_smoke(tmp_path):
    assert imitation.main(["--smoke", "--out", str(tmp_path)]) == 0
    ckpt = load_checkpoint(tmp_path / "best.pt")
    assert ckpt["kind"] == "bc" and "nll" in ckpt["metrics"]
    assert (tmp_path / "log.jsonl").read_text().count("\n") >= 1


def test_ppo_smoke_from_scratch_and_from_a_checkpoint(tmp_path):
    assert imitation.main(["--smoke", "--out", str(tmp_path / "bc")]) == 0
    assert ppo.main(["--smoke", "--out", str(tmp_path / "ppo0")]) == 0
    assert ppo.main(["--smoke", "--init", str(tmp_path / "bc" / "best.pt"), "--out", str(tmp_path / "ppo1")]) == 0
    ckpt = load_checkpoint(tmp_path / "ppo1" / "last.pt")
    assert ckpt["kind"] == "ppo" and ckpt["metrics"]["update"] == 2
    assert (tmp_path / "ppo1" / "league.pt").exists()


def test_evaluate_plays_a_truncated_match():
    torch.manual_seed(0)
    policy = Policy(d=32, heads=2, layers=1, torso=64).eval()
    results = evaluate.play(policy, slot("scripted", 10), [7, 8], LANES, 1, 1, torch.device("cpu"), max_ticks=200)
    assert [r.seed for r in results] == [7, 8]
    for r in results:
        assert r.won is None and r.ticks == 200 and r.seat == 1
    summary = evaluate.summarise(results)
    assert summary["matches"] == 2 and summary["draws"] == 2
