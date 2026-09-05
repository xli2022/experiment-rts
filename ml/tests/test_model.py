import torch

from rtsml import sampling as S
from rtsml.export import example_inputs
from rtsml.model import ACT_INPUTS, Policy
from rtsml.spec import (
    BUILD,
    MULTI_SELECT,
    NOOP,
    SINGLE_SELECT,
    SPEC,
    USES_ENTITY_TYPE,
    USES_LOCATION,
    USES_TARGET,
)


def random_obs(batch: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    inputs = example_inputs(batch, g)
    obs = {name: inputs[name] for name in ACT_INPUTS if name not in ("noise", "temperature")}
    obs["critic"] = torch.randn((batch, SPEC.critic_len), generator=g)
    return obs, inputs["noise"], inputs["temperature"]


def assert_legal(obs: dict[str, torch.Tensor], actions: torch.Tensor) -> None:
    """Every decision respects every mask, and unused heads are -1."""
    n = SPEC.n_ent
    for b in range(actions.shape[0]):
        a = actions[b].tolist()
        t, et, target, cell, sub, sel = a[0], a[1], a[2], a[3], a[4], a[5:]
        assert obs["mask_type"][b, t] == 1
        rows = [r for r in sel if r >= 0]
        if t in MULTI_SELECT or t in SINGLE_SELECT:
            assert rows, "a type that needs units named none"
            for r in rows:
                assert obs["mask_selection"][b, t, r] == 1 and obs["entity_mask"][b, r] == 1
            if t in SINGLE_SELECT:
                assert len(rows) == 1
        else:
            assert not rows
        if t in USES_ENTITY_TYPE:
            if t == BUILD:
                assert obs["mask_build_type"][b, et] == 1
            else:
                assert obs["mask_row_entity_type"][b, rows[0], et] == 1
        else:
            assert et == -1
        if t in USES_TARGET:
            assert obs["mask_target"][b, t, target] == 1
        else:
            assert target == -1
        if t in USES_LOCATION:
            assert 0 <= sub < SPEC.sub
            if t == BUILD:
                assert obs["mask_build_cell"][b, et, cell] == 1
            else:
                assert obs["mask_cell"][b, t, cell] == 1
        else:
            assert cell == -1 and sub == -1


def test_act_never_breaks_a_mask(tiny_policy: Policy):
    for seed in range(8):
        obs, noise, temperature = random_obs(16, seed)
        with torch.no_grad():
            actions = tiny_policy.act(obs, noise, temperature)
        assert actions.shape == (16, SPEC.action_ints)
        assert_legal(obs, actions)


def test_act_covers_every_type(tiny_policy: Policy):
    seen = set()
    for seed in range(20):
        obs, noise, temperature = random_obs(32, seed)
        with torch.no_grad():
            seen.update(tiny_policy.act(obs, noise, temperature)[:, 0].tolist())
    assert seen == set(range(SPEC.n_types))


def test_masked_type_is_never_sampled(tiny_policy: Policy):
    obs, noise, temperature = random_obs(64, 3)
    obs["mask_type"][:] = 0
    obs["mask_type"][:, NOOP] = 1
    obs["mask_type"][:, BUILD] = 1
    with torch.no_grad():
        actions = tiny_policy.act(obs, noise, temperature)
    assert set(actions[:, 0].tolist()) <= {NOOP, BUILD}
    assert_legal(obs, actions)


def test_evaluate_scores_its_own_samples(tiny_policy: Policy):
    obs, noise, temperature = random_obs(16, 5)
    with torch.no_grad():
        actions = tiny_policy.act(obs, noise, temperature)
        out = tiny_policy.evaluate(obs, actions)
    assert out["logp"].shape == (16,) and torch.isfinite(out["logp"]).all()
    assert (out["logp"] <= 1e-5).all()
    assert torch.isfinite(out["entropy"]).all() and (out["entropy"] >= -1e-5).all()
    assert out["value"].shape == (16,)
    # A Noop's log-probability is exactly the type head's.
    noop_rows = actions[:, 0] == NOOP
    if noop_rows.any():
        lp_type = S.categorical_logp(out["type_logits"], obs["mask_type"].bool(), actions[:, 0])
        assert torch.allclose(out["logp"][noop_rows], lp_type[noop_rows], atol=1e-5)


def test_temperature_zero_limit_is_the_argmax(tiny_policy: Policy):
    obs, noise, _ = random_obs(8, 9)
    cold = torch.full((8,), 1e-3)
    with torch.no_grad():
        a = tiny_policy.act(obs, noise, cold)
        b = tiny_policy.act(obs, torch.zeros_like(noise), cold)
    assert torch.equal(a[:, 0], b[:, 0])


def test_select_many_keeps_at_least_one_and_only_legal_rows():
    logits = torch.full((4, 10), -50.0)
    mask = torch.zeros((4, 10), dtype=torch.bool)
    mask[:, 3] = True
    mask[:, 7] = True
    g = torch.Generator().manual_seed(1)
    a = torch.rand((4, 10), generator=g)
    b = torch.rand((4, 10), generator=g)
    sel = S.select_many(logits, mask, a, b, torch.ones(4), 5)
    assert sel.shape == (4, 5)
    for row in sel.tolist():
        assert row[0] in (3, 7)
        assert all(r in (3, 7, -1) for r in row)


def test_selection_logp_counts_only_legal_rows():
    logits = torch.zeros((2, 6))
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    membership = torch.zeros((2, 6), dtype=torch.bool)
    lp = S.selection_logp(logits, mask, membership)
    assert torch.allclose(lp, torch.tensor([2, 6]) * torch.log(torch.tensor(0.5)))
