"""Small things every training script needs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .spec import SPEC

# Observation arrays the model consumes, and the dtype each is fed as.
OBS_FLOAT = ("entities", "grid", "scalars", "critic")
OBS_MASK = (
    "entity_mask",
    "mask_type",
    "mask_selection",
    "mask_target",
    "mask_cell",
    "mask_build_cell",
    "mask_row_entity_type",
    "mask_build_type",
)


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def to_torch(arrays: dict[str, np.ndarray], device: torch.device, index: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Observation arrays as tensors: floats as f32, masks as u8, labels/actions as i64."""
    out: dict[str, torch.Tensor] = {}
    for name, array in arrays.items():
        if index is not None:
            array = array[index]
        t = torch.from_numpy(np.ascontiguousarray(array))
        if name in OBS_FLOAT:
            t = t.to(torch.float32)
        elif name in OBS_MASK:
            t = t.to(torch.uint8)
        else:
            t = t.to(torch.int64)
        out[name] = t.to(device)
    return out


def gumbel_noise(rows: int, generator: torch.Generator | None = None, device: torch.device | None = None) -> torch.Tensor:
    """Gumbel(0, 1) noise, one full vector per row, as the act graph expects."""
    u = torch.rand((rows, SPEC.noise_len), generator=generator, device=device)
    u = u.clamp(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(u))


def save_checkpoint(path: Path, model: torch.nn.Module, kind: str, hparams: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": kind,
            "spec_version": SPEC.version,
            "hparams": hparams,
            "metrics": metrics or {},
            "model": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    if ckpt.get("spec_version") != SPEC.version:
        raise RuntimeError(f"checkpoint was trained for spec {ckpt.get('spec_version')}, this is {SPEC.version}")
    return ckpt


def decide(
    policy: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    index: np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Sample one decision per row of `index` from `policy`, as int32 [rows, ACTION_INTS]."""
    if len(index) == 0:
        return np.zeros((0, SPEC.action_ints), dtype=np.int32)
    obs = to_torch({k: v for k, v in arrays.items() if k not in ("label", "critic")}, device, index)
    noise = gumbel_noise(len(index), generator).to(device)
    t = torch.full((len(index),), float(temperature), device=device)
    with torch.no_grad():
        return policy.act(obs, noise, t).to(torch.int32).cpu().numpy()
