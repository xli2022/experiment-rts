"""Exact optional marginalization, including its parameter-dependent branch mass."""

import copy
import json

import numpy as np
import pytest
import torch

from rtsml import ppo, sampling as S
from rtsml.export import example_inputs
from rtsml.model import ACT_INPUTS, Policy
from rtsml.spec import BUILD, SPEC, STOP
from rtsml.util import load_checkpoint, save_checkpoint

from conftest import requires_bun


@pytest.mark.parametrize("old,new", [([-0.7, 0.2], [-0.2, -0.1]), ([-3., -2.], [-2.5, -2.4]), ([2., 3.], [1.7, 3.2])])
def test_mixed_branch_normalization_score_mass_and_importance_ratios(old, new):
    # Quasi-Monte Carlo integrates the actual two-Gumbel sampler, independently
    # of the likelihood implementation, including the rare fallback branch.
    u = torch.quasirandom.SobolEngine(4, scramble=True, seed=71).draw(2**17).double()
    noise = -torch.log(-torch.log(u.clamp(1e-12, 1-1e-12)))
    mu = torch.tensor(old, dtype=torch.float64, requires_grad=True)
    other = torch.tensor(new, dtype=torch.float64)
    x = mu + noise[:, :2]
    b = noise[:, 2:]
    mask = torch.ones_like(x, dtype=torch.bool)
    stored, fallback = S.hybrid_selection_scores(x, mask, b)
    assert not stored.requires_grad and not fallback.requires_grad
    lp = S.hybrid_selection_logp(mu, mask, stored, fallback)
    new_lp = S.hybrid_selection_logp(other, mask, stored, fallback)
    ratio = (new_lp-lp).exp()
    p0 = torch.sigmoid(-mu).prod()
    new_p0 = torch.sigmoid(-other).prod()
    # Density on each branch is unnormalized. Conditional normalization would
    # make these branch gradients zero and fail the importance-mass checks.
    nonempty_score = torch.autograd.grad((lp * (~fallback)).mean(), mu, retain_graph=True)[0]
    fallback_score = torch.autograd.grad((lp * fallback).mean(), mu)[0]
    expected = p0.detach() * torch.sigmoid(mu.detach())
    torch.testing.assert_close(nonempty_score, expected, atol=8e-4, rtol=0)
    torch.testing.assert_close(fallback_score, -expected, atol=8e-4, rtol=0)
    torch.testing.assert_close(nonempty_score+fallback_score, torch.zeros_like(mu), atol=8e-4, rtol=0)
    assert abs(fallback.double().mean().item()-p0.item()) < 8e-4
    assert abs(ratio.mean().item()-1) < 8e-4
    assert abs((ratio*fallback).mean().item()-new_p0.item()) < 8e-4
    assert abs((ratio*(~fallback)).mean().item()-(1-new_p0.item())) < 8e-4
    old_action = S.select_many_from_scores(x.detach(), mask, b, 1)[:, 0]
    new_action = S.select_many_from_scores(other+noise[:, :2], mask, b, 1)[:, 0]
    assert abs((ratio*(old_action == 0)).mean().item()-(new_action == 0).double().mean().item()) < 1.5e-3


def test_logistic_branch_has_bounded_exact_gradient_and_masks_inactive_exponentials():
    mu = torch.tensor([[0.2, -0.3, 0.5], [0.1, 0.4, -0.5]], dtype=torch.float64, requires_grad=True)
    stored = torch.tensor([[0.8, -1e30, -1e30], [0.5, -0.2, -1e30]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    fallback = torch.tensor([False, True])
    got = S.hybrid_selection_logp(mu, mask, stored, fallback)
    z = stored.detach()-mu
    logistic = torch.nn.functional.logsigmoid(z[0, :2])+torch.nn.functional.logsigmoid(-z[0, :2])
    gumbel = torch.distributions.Gumbel(mu[1, :2], torch.ones(2)).log_prob(stored.detach()[1, :2])
    torch.testing.assert_close(got, torch.stack((logistic.sum(), gumbel.sum())))
    got.sum().backward()
    assert torch.isfinite(got).all() and torch.isfinite(mu.grad).all()
    torch.testing.assert_close(mu.grad[0, :2], 2*torch.sigmoid(z.detach()[0, :2])-1)
    assert (mu.grad[0].abs() <= 1).all()
    assert torch.equal(mu.grad[:, 2], torch.zeros(2, dtype=torch.float64))
    assert stored.grad is None


def test_branch_is_stored_and_never_recomputed_from_new_policy_logits():
    scores = torch.tensor([[1., -2.], [-1., -3.]])
    mask = torch.ones_like(scores, dtype=torch.bool)
    stored, fallback = S.hybrid_selection_scores(scores, mask, torch.zeros_like(scores))
    assert fallback.tolist() == [False, True]
    new_logits = torch.tensor([[10., -10.], [20., 20.]])
    got = S.hybrid_selection_logp(new_logits, mask, stored, fallback)
    z = stored[0]-new_logits[0]
    expected_positive = (-torch.nn.functional.softplus(z)-torch.nn.functional.softplus(-z)).sum()
    expected_fallback = torch.distributions.Gumbel(new_logits[1], torch.ones(2)).log_prob(stored[1]).sum()
    torch.testing.assert_close(got, torch.stack((expected_positive, expected_fallback)))


def forced_multi_obs(batch, legal_rows):
    inputs = example_inputs(batch, torch.Generator().manual_seed(21))
    obs = {name: inputs[name] for name in ACT_INPUTS if name not in ("noise", "temperature")}
    obs["mask_type"].zero_(); obs["mask_type"][:, STOP] = 1
    obs["entity_mask"].zero_(); obs["entity_mask"][:, :legal_rows] = 1
    obs["mask_selection"].zero_(); obs["mask_selection"][:, STOP, :legal_rows] = 1
    return obs, inputs["noise"]


@pytest.mark.parametrize("legal_rows,bias", [(1, -40.), (8, -40.), (8, 0.), (SPEC.selection_max+5, 40.)])
def test_optional_sampler_preserves_exact_actions_order_cap_and_fallback(tiny_policy, monkeypatch, legal_rows, bias):
    obs, noise = forced_multi_obs(3, legal_rows)
    logits = torch.full((3, SPEC.n_ent), bias)
    monkeypatch.setattr(tiny_policy, "selection_logits", lambda enc, ctx: torch.full((len(ctx), SPEC.n_ent), bias))
    temperature = torch.tensor([.5, 1., 2.])
    with torch.no_grad():
        browser = tiny_policy.act(obs, noise, temperature)
        legacy, x = tiny_policy.sample(obs, noise, temperature)
        hybrid, stored, fallback = tiny_policy.sample_hybrid(obs, noise, temperature)
        torch.testing.assert_close(browser, legacy, atol=0, rtol=0)
        torch.testing.assert_close(browser, hybrid, atol=0, rtol=0)
        b = noise[:, S.noise_slices()["selection"]][:, SPEC.n_ent:]
        mask = obs["mask_selection"][:, STOP].bool()
        expected, branch = S.hybrid_selection_scores(x, mask, b)
        torch.testing.assert_close(stored, expected, atol=0, rtol=0)
        assert torch.equal(fallback, branch)
        # Evaluate each temperature separately: PPO uses one scalar temperature.
        for row in range(3):
            single = {k:v[row:row+1] for k,v in obs.items()}
            out = tiny_policy.evaluate(single, hybrid[row:row+1], float(temperature[row]), stored[row:row+1], selection_fallback=fallback[row:row+1])
            expected_lp = S.hybrid_selection_logp(logits[row:row+1]/temperature[row], mask[row:row+1], stored[row:row+1], fallback[row:row+1])
            torch.testing.assert_close(out["logp"], expected_lp)
            assert out["entropy"].item() == 0  # STOP has no other stochastic heads.
    if bias < 0:
        assert fallback.all() and ((hybrid[:, 5:]>=0).sum(-1)==1).all()
    elif bias > 0:
        assert not fallback.any() and ((hybrid[:, 5:]>=0).sum(-1)==SPEC.selection_max).all()


def test_hybrid_minibatches_and_reference_use_same_fixed_branch_with_uneven_chunks(tiny_policy):
    torch.manual_seed(91)
    obs, noise = forced_multi_obs(3, 7)
    obs["critic"] = torch.randn(3, SPEC.critic_len)
    # Force both branches without modifying the policy: B beats every X in row0.
    segment = S.noise_slices()["selection"]
    noise[0, segment.start+SPEC.n_ent:segment.stop] = 100
    noise[1:, segment.start+SPEC.n_ent:segment.stop] = -100
    actions, scores, fallback = tiny_policy.sample_hybrid(obs, noise, torch.full((3,), .5))
    assert fallback.tolist() == [True, False, False]
    rollout = ppo.Rollout(1, 3, hybrid_selection=True)
    rollout.obs = {k:v.numpy() for k,v in obs.items()}
    rollout.actions[:] = actions.detach().numpy(); rollout.selection_scores[:] = scores.numpy()
    rollout.selection_fallback[:] = fallback.numpy()
    with torch.no_grad():
        before = tiny_policy.evaluate(obs, actions, .5, scores, selection_fallback=fallback)
        rollout.logp[0] = before["logp"].numpy(); rollout.value[0] = before["value"].numpy()
    advantages = np.array([-.5, 1., -.5], dtype=np.float32)
    returns = np.array([.2, -.1, .5], dtype=np.float32)
    reference = copy.deepcopy(tiny_policy)
    with torch.no_grad(): reference.sel_query.bias.add_(.01)
    policies = [copy.deepcopy(tiny_policy) for _ in range(2)]
    results = []
    for policy, microbatch in zip(policies, [0, 2]):
        result = ppo.train_minibatch(policy, reference, torch.optim.Adam(policy.parameters(), lr=1e-5, eps=1e-5), rollout,
                    np.array([2, 0, 1]), advantages, returns, torch.device("cpu"), temperature=.5, clip=.15,
                    vf_coef=.5, ent_coef=.0005, beta=1., warming=False, microbatch=microbatch)
        assert result is not None and result["kl"] > 0
        results.append(result)
    assert results[0] == pytest.approx(results[1], abs=2e-5, rel=2e-4)
    for a,b in zip(*[p.parameters() for p in policies]):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


def test_hybrid_evaluation_rejects_missing_latents(tiny_policy):
    obs, noise = forced_multi_obs(1, 2)
    actions = tiny_policy.act(obs, noise, torch.ones(1))
    with pytest.raises(ValueError, match="stored scores"):
        tiny_policy.evaluate(obs, actions, selection_fallback=torch.tensor([False]))


@pytest.mark.parametrize("likelihood", ["gumbel", "hybrid"])
def test_unused_latent_density_cannot_poison_single_selection_gradients(tiny_policy, likelihood):
    obs, noise = forced_multi_obs(1, 2)
    obs["mask_type"].zero_(); obs["mask_type"][:, BUILD] = 1
    obs["mask_selection"][:, BUILD] = obs["mask_selection"][:, STOP]
    actions = tiny_policy.act(obs, noise, torch.ones(1))
    # Multi-selection latents are irrelevant to this single-builder command.
    # Even an extreme inactive Gumbel value must not make its gradients NaN.
    kwargs = {"selection_fallback": torch.tensor([True])} if likelihood == "hybrid" else {}
    out = tiny_policy.evaluate(obs, actions, selection_scores=torch.full((1, SPEC.n_ent), -1e30), **kwargs)
    out["logp"].sum().backward()
    assert torch.isfinite(out["logp"]).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in tiny_policy.parameters())


def test_default_numerical_guard_preserves_exact_normal_outputs_and_gradients(tiny_policy, monkeypatch):
    obs, noise = forced_multi_obs(2, 7)
    obs["critic"] = torch.randn(2, SPEC.critic_len)
    obs["mask_type"][0].zero_(); obs["mask_type"][0, BUILD] = 1
    obs["mask_selection"][:, BUILD] = obs["mask_selection"][:, STOP]
    actions, scores = tiny_policy.sample(obs, noise, torch.ones(2))
    original_mask = tiny_policy.selection_mask(obs, actions[:, 0], obs["entity_mask"].bool())
    guarded = tiny_policy.evaluate(obs, actions, selection_scores=scores)
    (guarded["logp"].sum()+guarded["entropy"].sum()+guarded["value"].sum()).backward()
    guarded_gradients = {name:None if p.grad is None else p.grad.clone() for name,p in tiny_policy.named_parameters()}
    tiny_policy.zero_grad(set_to_none=True)
    original_density = S.selection_score_logp
    # Reproduce the pre-guard implementation: all legal rows are scored even
    # for the Build row, whose multi-selection density is later discarded.
    monkeypatch.setattr(S,"selection_score_logp",lambda logits, mask, latent: original_density(logits, original_mask, latent))
    legacy = tiny_policy.evaluate(obs, actions, selection_scores=scores)
    (legacy["logp"].sum()+legacy["entropy"].sum()+legacy["value"].sum()).backward()
    for name in guarded:
        torch.testing.assert_close(guarded[name], legacy[name], atol=0, rtol=0)
    for name, parameter in tiny_policy.named_parameters():
        if guarded_gradients[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(guarded_gradients[name], parameter.grad, atol=0, rtol=0)


@requires_bun
def test_hybrid_two_update_cli_uses_live_bun_and_records_partial_entropy(tmp_path, tiny_policy):
    shape = {"d":32,"heads":2,"layers":1,"torso":64}
    initial = tmp_path/"initial.pt"
    save_checkpoint(initial,tiny_policy,"bc",{"model":shape,"layout":"lanes"})
    out = tmp_path/"ppo"
    assert ppo.main(["--smoke","--selection-likelihood","hybrid","--init",str(initial),
                     "--out",str(out),"--device","cpu","--microbatch","3","--seed","43"]) == 0
    records = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert len(records)==2 and [r["warming"] for r in records]==[True,False]
    assert all(r["selectionLikelihood"]=="hybrid" and r["entropyKind"]=="categorical-heads-only" for r in records)
    assert all(r["skipped"]==0 and r["steps"]==r["planned"]==2 for r in records)
    saved=load_checkpoint(out/"last.pt")
    assert saved["hparams"]["ppo"]["selection_likelihood"]=="hybrid"
    assert saved["hparams"]["ppo"]["entropy_kind"]=="categorical-heads-only"
    assert all(torch.isfinite(value).all() for value in saved["model"].values())
    assert any(not torch.equal(value,tiny_policy.state_dict()[name]) for name,value in saved["model"].items()
               if not name.startswith(("critic_mlp.","value_head.")))
