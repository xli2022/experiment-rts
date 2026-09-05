"""The head maths, shared by training, sampling and the exported graph.

Every head is a categorical over a fixed vocabulary with a legality mask, and
sampling is done with noise supplied from outside: a categorical head adds
Gumbel(0, 1) noise to its masked, temperature-scaled logits and takes the
argmax, which is an exact draw from its softmax. The selection head, for a
type that names many units, keeps every legal row whose logit plus the
difference of two Gumbels — a Logistic(0, 1) — is above zero, an exact
Bernoulli draw per row, capped at the top `selection_max` scores and never
empty. Because the noise is an input, the ONNX graph carries no random number
generator and parity between torch and onnxruntime is an exact comparison of
integers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import BUILD, MULTI_SELECT, SINGLE_SELECT, SPEC, USES_ENTITY_TYPE, USES_LOCATION, USES_TARGET

NEG = -1e9


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Logits with every illegal entry pushed far below any legal one."""
    return torch.where(mask, logits, torch.full_like(logits, NEG))


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(masked(logits, mask), dim=-1)


def gumbel_argmax(logits: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """An exact sample from softmax(masked logits / temperature). `temperature` is [B]."""
    scaled = logits / temperature.unsqueeze(-1)
    return torch.argmax(masked(scaled, mask) + noise, dim=-1)


def select_many(
    logits: torch.Tensor,
    mask: torch.Tensor,
    noise_a: torch.Tensor,
    noise_b: torch.Tensor,
    temperature: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Independent Bernoulli draws per legal row, the top `k` kept, at least one.

    Returns [B, k] row indices, -1 padded, in descending score order. When no
    row's draw comes up, the single-row draw (argmax with `noise_a` alone)
    names one, so a type that needs units always has one.
    """
    scaled = logits / temperature.unsqueeze(-1)
    score = masked(scaled + (noise_a - noise_b), mask)
    values, index = torch.topk(score, k, dim=-1)
    kept = torch.where(values > 0, index, torch.full_like(index, -1))
    best = torch.argmax(masked(scaled + noise_a, mask), dim=-1)
    none = values[:, 0] <= 0
    first = torch.where(none, best, kept[:, 0])
    return torch.cat([first.unsqueeze(1), kept[:, 1:]], dim=1)


def select_one(logits: torch.Tensor, mask: torch.Tensor, noise_a: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return gumbel_argmax(logits, mask, noise_a, temperature)


def categorical_logp(logits: torch.Tensor, mask: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """log p(index) under the masked softmax; `index` is [B], negative entries read as 0."""
    lp = masked_log_softmax(logits, mask)
    return lp.gather(1, index.clamp(min=0).unsqueeze(1)).squeeze(1)


def categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = masked_log_softmax(logits, mask)
    return -(lp.exp() * lp).sum(-1)


def selection_logp(logits: torch.Tensor, mask: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Sum over legal rows of the Bernoulli log-probability of `membership` [B, N] bool."""
    per = torch.where(membership, F.logsigmoid(logits), F.logsigmoid(-logits))
    return (per * mask.to(logits.dtype)).sum(-1)


def selection_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    h = -(p * F.logsigmoid(logits) + (1 - p) * F.logsigmoid(-logits))
    return (h * mask.to(logits.dtype)).sum(-1)


def membership_of(selection: torch.Tensor, n: int) -> torch.Tensor:
    """[B, k] row indices (-1 padded) → [B, n] bool."""
    rows = torch.arange(n, device=selection.device).view(1, 1, n)
    return (selection.unsqueeze(-1) == rows).any(dim=1)


def gather_rows(mask3: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """mask3 [B, T, X] indexed by `index` [B] along T → [B, X]."""
    b, _, x = mask3.shape
    return torch.gather(mask3, 1, index.clamp(min=0).view(b, 1, 1).expand(b, 1, x)).squeeze(1)


def usage_tables(n_types: int = SPEC.n_types) -> dict[str, torch.Tensor]:
    """Which heads each action type uses, as bool tables indexed by type."""

    def table(types: tuple[int, ...]) -> torch.Tensor:
        t = torch.zeros(n_types, dtype=torch.bool)
        for i in types:
            t[i] = True
        return t

    return {
        "multi": table(MULTI_SELECT),
        "single": table(SINGLE_SELECT),
        "target": table(USES_TARGET),
        "location": table(USES_LOCATION),
        "entity_type": table(USES_ENTITY_TYPE),
        "build": table((BUILD,)),
    }


def noise_slices() -> dict[str, slice]:
    return {seg.name: slice(seg.offset, seg.offset + seg.size) for seg in SPEC.noise}
