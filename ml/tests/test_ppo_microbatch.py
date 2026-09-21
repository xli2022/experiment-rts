"""Microbatches change memory usage, not PPO's logical optimiser step."""

import copy
import json

import numpy as np
import pytest
import torch

from rtsml import ppo
from rtsml.evaluate import MatchResult
from rtsml.model import Policy
from rtsml.spec import SPEC
from rtsml.util import load_checkpoint, save_checkpoint

from conftest import requires_bun


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("inf")


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(2, 3)
        self.actor = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(3, 1)
        self.bad_backward = False
        self.calls = 0

    def evaluate(self, obs, actions, temperature, selection_scores):
        self.calls += 1
        hidden = torch.tanh(self.shared(obs["entities"]))
        lp = torch.log_softmax(self.actor(hidden) / temperature, dim=-1)
        logp = lp.gather(1, actions[:, :1]).squeeze(1)
        if self.bad_backward and (obs["entities"][:, 0] == 3).any():
            logp = InfiniteGradient.apply(logp)
        return {"logp": logp, "entropy": -(lp.exp() * lp).sum(-1),
                "value": self.critic(hidden).squeeze(-1)}


def inputs(policy, n=7):
    rollout = ppo.Rollout(1, n)
    rollout.obs = {"entities": np.column_stack((np.arange(n), np.linspace(-1, 1, n))).astype(np.float32)}
    rollout.actions[:, 0] = np.arange(n) % 2
    with torch.no_grad():
        out = policy.evaluate({"entities": torch.from_numpy(rollout.obs["entities"])},
                              torch.from_numpy(rollout.actions).long(), 0.5,
                              torch.zeros(n, SPEC.n_ent))
    rollout.logp[0] = out["logp"].numpy() + np.linspace(-0.4, 0.4, n)
    rollout.value[0] = out["value"].numpy() + np.linspace(-0.3, 0.3, n)
    advantages = np.array([-1.3, 0.2, 1.4, -0.8, 0.6, -0.1, 0.9], dtype=np.float32)[:n]
    returns = np.linspace(-0.7, 0.8, n, dtype=np.float32)
    return rollout, advantages, returns


def update(policy, reference, opt, rollout, advantages, returns, microbatch, warming=False):
    return ppo.train_minibatch(
        policy, reference, opt, rollout, np.arange(rollout.rows), advantages, returns,
        torch.device("cpu"), temperature=0.5, clip=0.2, vf_coef=0.5,
        ent_coef=0.03, beta=0.7, warming=warming, microbatch=microbatch,
    )


@pytest.mark.parametrize("warming", [False, True])
@pytest.mark.parametrize("microbatch", [1, 3, 20])
def test_uneven_microbatches_match_full_gradients_adam_update_and_metrics(warming, microbatch):
    torch.manual_seed(19)
    full = ToyPolicy()
    split = copy.deepcopy(full)
    reference = copy.deepcopy(full)
    with torch.no_grad():
        reference.actor.bias[0] += 0.7
    if warming:
        for policy in (full, split):
            for parameter in (*policy.shared.parameters(), *policy.actor.parameters()):
                parameter.requires_grad_(False)
    rollout, advantages, returns = inputs(full)
    full_opt = torch.optim.Adam(full.parameters(), lr=0.01, eps=1e-5)
    split_opt = torch.optim.Adam(split.parameters(), lr=0.01, eps=1e-5)
    want = update(full, reference, full_opt, rollout, advantages, returns, 0, warming)
    got = update(split, reference, split_opt, rollout, advantages, returns, microbatch, warming)
    assert got == pytest.approx(want, abs=1e-7, rel=1e-6)
    assert got["kl"] == 0 if warming else got["kl"] > 0
    # The logical KL stop decision uses weighted rows, including the last
    # one-row chunk of a seven-row minibatch split into threes.
    for threshold in (got["approxKl"] / 2, got["approxKl"] * 2):
        assert (got["approxKl"] > threshold) == (want["approxKl"] > threshold)
    for a, b in zip(full.parameters(), split.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        if a.grad is None:
            assert b.grad is None
            continue
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)
        assert full_opt.state[a]["step"] == split_opt.state[b]["step"] == 1
        torch.testing.assert_close(full_opt.state[a]["exp_avg"], split_opt.state[b]["exp_avg"], atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_later_chunk_discards_all_accumulated_gradients_without_a_partial_step(failure):
    torch.manual_seed(9)
    policy = ToyPolicy()
    rollout, advantages, returns = inputs(policy)
    original = copy.deepcopy(policy.state_dict())
    if failure == "loss":
        rollout.obs["entities"][3, 0] = float("nan")
    else:
        policy.bad_backward = True
    policy.calls = 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    result = update(policy, None, optimizer, rollout, advantages, returns, 2)
    assert result is None and policy.calls >= 2
    assert not optimizer.state
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in policy.parameters())


@requires_bun
def test_ppo_cli_accumulates_logical_steps_and_records_the_selected_evaluation_seed(tmp_path, monkeypatch):
    shape = {"d": 32, "heads": 2, "layers": 1, "torso": 64}
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, Policy(**shape), "bc", {"model": shape, "layout": "lanes"})
    evaluated = []

    def fake_play(policy, opponent, seeds, layout, seat, procs, device, temperature, max_ticks):
        evaluated.append((seeds, seat))
        return [MatchResult(seed, layout, seat, None, max_ticks, 0) for seed in seeds]

    monkeypatch.setattr(ppo, "play", fake_play)
    out = tmp_path / "ppo"
    assert ppo.main([
        "--init", str(checkpoint), "--out", str(out), "--device", "cpu",
        "--procs", "1", "--envs", "2", "--rollout", "3", "--updates", "2",
        "--epochs", "1", "--minibatch", "5", "--microbatch", "3",
        "--value-warmup", "1", "--max-ticks", "20", "--target-kl", "0",
        "--eval-every", "1", "--eval-seeds", "1", "--eval-seed0", "1000500",
    ]) == 0
    assert evaluated == [([1000500], seat) for _ in range(2) for seat in (0, 1)]
    records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [row["warming"] for row in records] == [True, False]
    assert all(row["steps"] == row["planned"] == 2 and row["skipped"] == 0 for row in records)
    saved = load_checkpoint(out / "last.pt")
    assert saved["hparams"]["ppo"]["microbatch"] == 3
    assert saved["hparams"]["ppo"]["eval_seed0"] == 1000500
