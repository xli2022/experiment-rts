import numpy as np
import pytest
import torch

from rtsml.env import LANES, QUARTERS, BunVectorEnv, EnvConfig, noop_actions, slot
from rtsml.model import Policy
from rtsml.spec import BUILD, NOOP, SPEC, TRAIN
from rtsml.util import decide

from conftest import requires_bun

pytestmark = requires_bun


@pytest.fixture(scope="module")
def env():
    configs = [
        [
            EnvConfig(seed=1, layout=LANES, slots=[slot("policy"), slot("scripted", 20)], max_ticks=2000),
            EnvConfig(seed=2, layout=LANES, slots=[slot("teacher"), slot("scripted", 10)], max_ticks=2000),
        ]
    ]
    env = BunVectorEnv(configs)
    yield env
    env.close()


def test_reset_gives_one_row_per_observed_slot(env):
    batch = env.reset()
    assert batch.slots == [(0, 0, 0), (0, 1, 0)]
    a = batch.arrays
    assert a["entities"].shape == (2, SPEC.n_ent, SPEC.f)
    assert a["grid"].shape == (2, SPEC.c, SPEC.grid, SPEC.grid)
    assert a["scalars"].shape == (2, SPEC.s)
    assert a["mask_cell"].shape == (2, SPEC.n_types, SPEC.cells)
    assert a["critic"].shape == (2, SPEC.critic_len)
    assert a["label"].shape == (2, SPEC.action_ints)
    assert (a["mask_type"][:, NOOP] == 1).all()
    assert batch.reset.all()


def test_steps_advance_four_ticks_and_label_the_teacher(env):
    batch = env.reset()
    labelled = 0
    for k in range(30):
        batch = env.step(noop_actions(len(batch)))
        assert batch.tick.tolist() == [4 * (k + 1)] * 2
        assert np.isfinite(batch.reward).all()
        label = batch.arrays["label"]
        assert label[0, 0] == NOOP and (label[0, 1:] == -1).all(), "a policy slot carries no label"
        if label[1, 0] > NOOP:
            labelled += 1
    assert labelled > 0


def test_a_policy_can_play_and_reconfigure(env):
    torch.manual_seed(0)
    policy = Policy(d=32, heads=2, layers=1, torso=64).eval()
    batch = env.reset()
    issued = 0
    generator = torch.Generator().manual_seed(0)
    for _ in range(40):
        actions = decide(policy, batch.arrays, np.arange(len(batch)), torch.device("cpu"), 1.0, generator)
        batch = env.step(actions)
        issued += int(batch.issued[0])
    assert issued > 0
    batch = env.reset([[EnvConfig(seed=5, layout=LANES, slots=[slot("scripted", 10), slot("policy")], max_ticks=400)]])
    assert batch.slots == [(0, 0, 1)]
    for _ in range(100):
        batch = env.step(noop_actions(1))
    assert batch.done.all() and batch.truncated.all() and batch.reset.all()


@pytest.mark.parametrize("layout,players", [(LANES, 2), (QUARTERS, 4)])
def test_expert_labels_cross_the_pipe_without_controlling_policy_slots(layout, players):
    learners = players // 2
    slots = [slot("policy") for _ in range(learners)] + [slot("idle") for _ in range(learners)]
    plain = EnvConfig(seed=31, layout=layout, slots=slots, max_ticks=400)
    labelled = EnvConfig(seed=31, layout=layout, slots=slots, max_ticks=400, expert_labels=True)
    assert plain.to_json()["expertLabels"] is False
    assert labelled.to_json()["expertLabels"] is True
    env = BunVectorEnv([[labelled, plain]])
    try:
        batch = env.reset()
        assert (batch.arrays["label"][:learners, 0] == -1).all()
        for step in range(60):
            batch = env.step(noop_actions(len(batch)))
            assert not batch.issued.any(), "a shadow expert must never issue its suggestion"
            for name, values in batch.arrays.items():
                if name != "label":
                    np.testing.assert_array_equal(values[:learners], values[learners:], err_msg=name)
            labels = batch.arrays["label"]
            assert (labels[learners:, 0] == NOOP).all()
            if step >= 8 and labels[0, 0] in (BUILD, TRAIN):
                break
        else:
            pytest.fail("the expert supplied no production or construction suggestion")

        # The same current-state label becomes an action only when the learner
        # sends it back; this also checks observation/label row alignment.
        action = noop_actions(len(batch))
        action[0] = labels[0]
        assert batch.arrays["mask_type"][0, action[0, 0]]
        assert batch.arrays["mask_selection"][0, action[0, 0], action[0, 5]]
        batch = env.step(action)
        assert batch.issued[0] == 1 and not batch.issued[1:].any()
        for _ in range(3):
            batch = env.step(noop_actions(len(batch)))
        assert not np.array_equal(batch.arrays["entities"][0], batch.arrays["entities"][learners])

        # A server auto-reset returns the new match's observation, so a label
        # from the just-ended match must not leak into that next observation.
        for _ in range(100):
            if batch.reset.all():
                break
            batch = env.step(noop_actions(len(batch)))
        assert batch.done.all() and batch.truncated.all() and batch.reset.all()
        assert (batch.arrays["label"][:learners, 0] == -1).all()
    finally:
        env.close()
