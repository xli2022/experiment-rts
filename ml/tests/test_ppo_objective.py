"""PPO's reference leash must continue acting after a large policy change."""

import torch

from rtsml.ppo import reference_penalty


def test_reference_huber_keeps_its_restoring_gradient_beyond_the_ratio_clamp():
    reference = torch.full((7,), -30.0)
    offsets = torch.tensor([-20.0, -2.0, -0.25, 0.0, 0.25, 2.0, 20.0])
    logp = (reference + offsets).requires_grad_()
    penalty = reference_penalty(logp, reference)
    expected = torch.tensor([19.5, 1.5, 0.03125, 0.0, 0.03125, 1.5, 19.5]).mean()
    torch.testing.assert_close(penalty, expected)
    penalty.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([-1., -1., -0.25, 0., 0.25, 1., 1.]) / 7)
    # A policy beyond the old +/-10 cutoff is pulled back, not left frozen.
    updated = logp.detach() - 0.1 * logp.grad
    assert torch.all((updated - reference).abs()[offsets != 0] < offsets.abs()[offsets != 0])


def test_reference_huber_is_finite_and_linear_for_very_large_finite_differences():
    logp = torch.tensor([-1e20, 1e20], requires_grad=True)
    penalty = reference_penalty(logp, torch.zeros_like(logp))
    assert torch.isfinite(penalty)
    penalty.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([-0.5, 0.5]))
