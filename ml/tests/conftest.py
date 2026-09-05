import shutil

import pytest
import torch

from rtsml.model import Policy


def has_bun() -> bool:
    return shutil.which("bun") is not None


requires_bun = pytest.mark.skipif(not has_bun(), reason="bun is not installed")


@pytest.fixture
def tiny_policy() -> Policy:
    torch.manual_seed(0)
    return Policy(d=32, heads=2, layers=1, torso=64).eval()
