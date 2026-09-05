import numpy as np
import pytest
import torch

from rtsml.env import LANES, BunVectorEnv, EnvConfig, noop_actions, slot
from rtsml.model import Policy
from rtsml.spec import NOOP, SPEC
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
