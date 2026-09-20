"""Pipe errors must reach training instead of disappearing in helper threads."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rtsml.env import LANES, BunVectorEnv, EnvConfig, noop_actions, slot


def vector_with(proc):
    env = BunVectorEnv.__new__(BunVectorEnv)
    env.procs = [proc]
    env.configs = [[EnvConfig(seed=1, layout=LANES, slots=[slot("policy"), slot("idle")])]]
    env.max_observed = 1
    env.last = [None]
    return env


def test_step_propagates_write_failure_without_waiting_for_an_observation():
    proc = SimpleNamespace(send=Mock(side_effect=BrokenPipeError("closed stdin")), recv=Mock())
    env = vector_with(proc)
    with pytest.raises(BrokenPipeError, match="closed stdin"):
        env.step(noop_actions(1))
    proc.recv.assert_not_called()


@pytest.mark.parametrize("error", [EOFError("closed stdout"), RuntimeError("environment error: bad slot")])
def test_collect_preserves_the_environment_error(error):
    env = vector_with(SimpleNamespace(recv=Mock(side_effect=error)))
    with pytest.raises(type(error), match=str(error)):
        env._collect()
