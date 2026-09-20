"""Exact training likelihoods for the unchanged top-k/nonempty selection sampler."""

import math

import pytest
import torch

from rtsml import sampling as S
from rtsml.export import example_inputs
from rtsml.model import ACT_INPUTS
from rtsml.spec import SPEC, STOP


def test_selection_score_density_and_gradient_match_torch_gumbel():
    logits = torch.tensor([[0.3, -0.8, 2.0], [-1.0, 0.4, 0.0]], dtype=torch.float64, requires_grad=True)
    scores = torch.tensor([[0.7, -0.2, 1.2], [0.0, 1.0, 0.6]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    expected = torch.distributions.Gumbel(logits, torch.ones_like(logits)).log_prob(scores)
    expected = expected.masked_fill(~mask, 0).sum(-1)
    actual = S.selection_score_logp(logits, mask, scores)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.equal(actual_gradient[~mask], torch.zeros_like(actual_gradient[~mask]))


def test_illegal_scores_cannot_overflow_the_density_or_its_gradient():
    logits = torch.zeros((1, 2), requires_grad=True)
    mask = torch.tensor([[True, False]])
    scores = torch.tensor([[0.5, -1e30]])
    logp = S.selection_score_logp(logits, mask, scores)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0


def test_importance_ratio_recovers_the_new_gumbel_distribution():
    # Midpoint quantiles integrate the old density without random test error.
    n = 20000
    uniform = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    old_location = torch.full((n, 1), -0.4, dtype=torch.float64)
    new_location = torch.full((n, 1), 0.3, dtype=torch.float64)
    scores = old_location - torch.log(-torch.log(uniform)).unsqueeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    old_logp = S.selection_score_logp(old_location, mask, scores)
    new_logp = S.selection_score_logp(new_location, mask, scores)
    ratio = (new_logp - old_logp).exp()
    assert abs(ratio.mean().item() - 1.0) < 1e-4
    expected_positive = 1.0 - math.exp(-math.exp(0.3))
    actual_positive = (ratio * (scores[:, 0] > 0)).mean().item()
    assert abs(actual_positive - expected_positive) < 1e-4


@pytest.mark.parametrize("legal_rows, bias", [(1, -40.0), (8, -40.0), (SPEC.selection_max + 5, 40.0)])
def test_training_sample_preserves_browser_actions_and_scores_its_latent_draw(
    tiny_policy, monkeypatch, legal_rows, bias
):
    inputs = example_inputs(2, torch.Generator().manual_seed(7))
    obs = {name: inputs[name] for name in ACT_INPUTS if name not in ("noise", "temperature")}
    obs["mask_type"].zero_()
    obs["mask_type"][:, STOP] = 1
    obs["entity_mask"].zero_()
    obs["entity_mask"][:, :legal_rows] = 1
    obs["mask_selection"].zero_()
    obs["mask_selection"][:, STOP, :legal_rows] = 1
    logits = torch.full((2, SPEC.n_ent), bias)
    monkeypatch.setattr(tiny_policy, "selection_logits", lambda enc, ctx: logits)
    noise = inputs["noise"]
    temperature = torch.ones(2)
    with torch.no_grad():
        browser_actions = tiny_policy.act(obs, noise, temperature)
        actions, scores = tiny_policy.sample(obs, noise, temperature)
        out = tiny_policy.evaluate(obs, actions, selection_scores=scores)
    torch.testing.assert_close(actions, browser_actions, rtol=0, atol=0)
    # Negative logits force the nonempty fallback; positive logits exercise the cap.
    expected_count = 1 if bias < 0 else SPEC.selection_max
    assert ((actions[:, 5:] >= 0).sum(-1) == expected_count).all()
    selection_noise = noise[:, S.noise_slices()["selection"]][:, :SPEC.n_ent]
    torch.testing.assert_close(scores, logits + selection_noise)
    mask = obs["mask_selection"][:, STOP].bool()
    expected_logp = torch.distributions.Gumbel(logits, torch.ones_like(logits)).log_prob(scores)
    expected_logp = expected_logp.masked_fill(~mask, 0).sum(-1)
    torch.testing.assert_close(out["logp"], expected_logp)
    expected_entropy = torch.full((2,), legal_rows * (1.0 + 0.5772156649015329))
    torch.testing.assert_close(out["entropy"], expected_entropy)
