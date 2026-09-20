from types import SimpleNamespace

import numpy as np
import torch

from rtsml import evaluate
from rtsml.env import LANES, QUARTERS, slot
from rtsml.league import Member
from rtsml.ppo import Assignment, Roles


def test_league_records_one_result_per_match_in_either_layout():
    member = Member("scripted@10", "scripted", think_interval=10)
    assignments = [[Assignment(member, 1, QUARTERS), Assignment(member, 0, LANES)]]
    roles = Roles([(0, 0, 2), (0, 0, 3), (0, 1, 0)], assignments)
    assert roles.learner.tolist() == [0, 1, 2]
    assert roles.matches == [0, 2]


def test_evaluation_counts_both_teammates_final_commands(monkeypatch):
    class Initial:
        arrays = {}

        def __len__(self):
            return 2

    class FakeEnv:
        def __init__(self, groups):
            pass

        def reset(self):
            return Initial()

        def step(self, actions):
            return SimpleNamespace(
                slots=[(0, 0, 0), (0, 0, 1)], issued=np.array([1, 1]),
                done=np.array([True, True]), winner=np.array([0, 0]), tick=np.array([4, 4]),
            )

        def close(self):
            pass

    monkeypatch.setattr(evaluate, "BunVectorEnv", FakeEnv)
    monkeypatch.setattr(evaluate, "decide", lambda *args: None)
    results = evaluate.play(None, slot("scripted", 10), [7], QUARTERS, 0, 1, torch.device("cpu"))
    assert len(results) == 1
    assert results[0].won and results[0].commands == 2
