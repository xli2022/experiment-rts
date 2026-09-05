"""The policy and its value head.

Three encoders — a small transformer over the entity table, a convolution
over the coarse map, an MLP over the HUD scalars — meet in a torso, and the
heads then decide in the spec's order: type, selection, entity type, target,
cell, sub-cell. Each head is conditioned on the choices before it, so the
policy is autoregressive over one decision, and each is masked to what the
environment says is legal, so an illegal decision has probability zero and is
never sampled.

`evaluate` scores given decisions (imitation labels, or PPO's own samples)
and `act` draws new ones from supplied noise; both walk the same heads with
the same masks, and `ActGraph` wraps `act` for export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sampling as S
from .spec import SPEC, Spec

NEG = S.NEG


class Attention(nn.Module):
    """Multi-head self-attention written out in full, so it exports as plain ops."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        # Keys without an entity get a large negative bias rather than -inf, so
        # a table with no entities at all still softmaxes to something finite.
        bias = torch.where(key_mask, torch.zeros_like(key_mask, dtype=att.dtype), torch.full_like(key_mask, NEG, dtype=att.dtype))
        att = torch.softmax(att + bias.view(b, 1, 1, n), dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b, n, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_mask)
        return x + self.ffn(self.ln2(x))


@dataclass
class Encoded:
    rows: torch.Tensor
    """[B, N, d] entity embeddings."""
    row_mask: torch.Tensor
    """[B, N] bool, rows that hold an entity."""
    torso: torch.Tensor
    """[B, torso]"""
    gfeat: torch.Tensor
    """[B, 64, G/4, G/4] map features."""
    grid_skip: torch.Tensor
    """[B, 8, G, G] a shallow view of the map for the cell head."""


class Policy(nn.Module):
    def __init__(self, spec: Spec = SPEC, d: int = 128, heads: int = 4, layers: int = 2, torso: int = 256):
        super().__init__()
        self.spec = spec
        self.d = d
        n_types, n_ent, e = spec.n_types, spec.n_ent, spec.entity_types
        cells, sub = spec.cells, spec.sub

        self.entity_in = nn.Linear(spec.f, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.row_norm = nn.LayerNorm(d)

        self.grid_conv = nn.Sequential(
            nn.Conv2d(spec.c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        q = spec.grid // 4
        self.grid_pool = nn.AvgPool2d(2)
        self.grid_out = nn.Sequential(nn.Linear(64 * (q // 2) * (q // 2), 256), nn.ReLU())
        self.grid_skip = nn.Sequential(nn.Conv2d(spec.c, 8, 3, padding=1), nn.ReLU())
        self.scalar_mlp = nn.Sequential(nn.Linear(spec.s, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.torso = nn.Sequential(nn.Linear(d + 256 + 64, 512), nn.ReLU(), nn.Linear(512, torso), nn.ReLU())

        self.type_head = nn.Linear(torso, n_types)
        self.type_emb = nn.Embedding(n_types, 64)
        ctx = torso + 64
        self.sel_query = nn.Linear(ctx, d)
        self.sel_key = nn.Linear(d, d)
        ctx2 = ctx + d
        self.et_head = nn.Sequential(nn.Linear(ctx2, 128), nn.ReLU(), nn.Linear(128, e))
        self.et_emb = nn.Embedding(e, 32)
        self.tgt_query = nn.Linear(ctx2, d)
        self.tgt_key = nn.Linear(d, d)
        ctx3 = ctx2 + 32
        self.cell_query = nn.Linear(ctx3, 64)
        self.cell_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.cell_out = nn.Conv2d(16 + 8, 1, 1)
        self.cell_emb = nn.Embedding(cells, 32)
        self.sub_head = nn.Sequential(nn.Linear(ctx3 + 32, 64), nn.ReLU(), nn.Linear(64, sub))

        self.critic_mlp = nn.Sequential(nn.Linear(spec.critic_len, 64), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(torso + 64, 128), nn.ReLU(), nn.Linear(128, 1))

        for name, table in S.usage_tables(n_types).items():
            self.register_buffer(f"uses_{name}", table, persistent=False)

    # --- encoders -----------------------------------------------------------------

    def encode(self, obs: dict[str, torch.Tensor]) -> Encoded:
        row_mask = obs["entity_mask"].to(torch.bool)
        x = self.entity_in(obs["entities"])
        for block in self.blocks:
            x = block(x, row_mask)
        rows = self.row_norm(x)
        keep = row_mask.unsqueeze(-1).to(rows.dtype)
        pooled = (rows * keep).sum(1) / keep.sum(1).clamp(min=1.0)

        grid = obs["grid"]
        gfeat = self.grid_conv(grid)
        gvec = self.grid_out(self.grid_pool(gfeat).flatten(1))
        svec = self.scalar_mlp(obs["scalars"])
        torso = self.torso(torch.cat([pooled, gvec, svec], dim=1))
        return Encoded(rows, row_mask, torso, gfeat, self.grid_skip(grid))

    def value(self, enc: Encoded, critic: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([enc.torso, self.critic_mlp(critic)], dim=1)).squeeze(-1)

    def value_of(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """The value alone, for bootstrapping a rollout."""
        return self.value(self.encode(obs), obs["critic"])

    # --- heads, in order --------------------------------------------------------------

    def type_logits(self, enc: Encoded) -> torch.Tensor:
        return self.type_head(enc.torso)

    def context(self, enc: Encoded, type_: torch.Tensor) -> torch.Tensor:
        return torch.cat([enc.torso, self.type_emb(type_)], dim=1)

    def pointer(self, rows: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return torch.matmul(rows, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d)

    def selection_logits(self, enc: Encoded, ctx: torch.Tensor) -> torch.Tensor:
        return self.pointer(self.sel_key(enc.rows), self.sel_query(ctx))

    def summary(self, enc: Encoded, membership: torch.Tensor) -> torch.Tensor:
        keep = membership.unsqueeze(-1).to(enc.rows.dtype)
        return (enc.rows * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def entity_type_logits(self, ctx2: torch.Tensor) -> torch.Tensor:
        return self.et_head(ctx2)

    def target_logits(self, enc: Encoded, ctx2: torch.Tensor) -> torch.Tensor:
        return self.pointer(self.tgt_key(enc.rows), self.tgt_query(ctx2))

    def cell_logits(self, enc: Encoded, ctx3: torch.Tensor) -> torch.Tensor:
        q = self.cell_query(ctx3).unsqueeze(-1).unsqueeze(-1)
        up = self.cell_up(F.relu(enc.gfeat + q))
        return self.cell_out(torch.cat([up, enc.grid_skip], dim=1)).flatten(1)

    def sub_logits(self, ctx3: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return self.sub_head(torch.cat([ctx3, self.cell_emb(cell.clamp(min=0))], dim=1))

    # --- conditional masks -----------------------------------------------------------------

    def usage(self, type_: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: torch.index_select(getattr(self, f"uses_{name}"), 0, type_)
            for name in ("multi", "single", "target", "location", "entity_type", "build")
        }

    @staticmethod
    def selection_mask(obs: dict[str, torch.Tensor], type_: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        return S.gather_rows(obs["mask_selection"], type_).to(torch.bool) & row_mask

    # The two conditional masks are chosen in uint8 and only then read as bool:
    # onnxruntime has no Where kernel for bool tensors.
    @staticmethod
    def entity_type_mask(obs: dict[str, torch.Tensor], is_build: torch.Tensor, first_row: torch.Tensor) -> torch.Tensor:
        by_row = S.gather_rows(obs["mask_row_entity_type"], first_row).to(torch.uint8)
        return torch.where(is_build.unsqueeze(-1), obs["mask_build_type"].to(torch.uint8), by_row).to(torch.bool)

    @staticmethod
    def target_mask(obs: dict[str, torch.Tensor], type_: torch.Tensor) -> torch.Tensor:
        return S.gather_rows(obs["mask_target"], type_).to(torch.bool)

    @staticmethod
    def cell_mask(obs: dict[str, torch.Tensor], type_: torch.Tensor, is_build: torch.Tensor, entity_type: torch.Tensor) -> torch.Tensor:
        by_type = S.gather_rows(obs["mask_cell"], type_).to(torch.uint8)
        by_building = S.gather_rows(obs["mask_build_cell"], entity_type).to(torch.uint8)
        return torch.where(is_build.unsqueeze(-1), by_building, by_type).to(torch.bool)

    # --- scoring given decisions ------------------------------------------------------------------

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Log-probability and entropy of `actions` [B, ACTION_INTS] under the policy, plus the value.

        Every head the action's type does not use contributes zero to both.
        """
        spec = self.spec
        enc = self.encode(obs)
        type_ = actions[:, 0].clamp(min=0)
        entity_type = actions[:, 1]
        target = actions[:, 2]
        cell = actions[:, 3]
        sub = actions[:, 4]
        selection = actions[:, 5:]
        use = self.usage(type_)
        zero = torch.zeros_like(type_, dtype=enc.torso.dtype)

        type_mask = obs["mask_type"].to(torch.bool)
        tl = self.type_logits(enc)
        logp = S.categorical_logp(tl, type_mask, type_)
        entropy = S.categorical_entropy(tl, type_mask)

        ctx = self.context(enc, type_)
        sel_logits = self.selection_logits(enc, ctx)
        sel_mask = self.selection_mask(obs, type_, enc.row_mask)
        membership = S.membership_of(selection, spec.n_ent)
        lp_many = S.selection_logp(sel_logits, sel_mask, membership)
        lp_one = S.categorical_logp(sel_logits, sel_mask, selection[:, 0])
        logp = logp + torch.where(use["multi"], lp_many, torch.where(use["single"], lp_one, zero))
        entropy = entropy + torch.where(
            use["multi"],
            S.selection_entropy(sel_logits, sel_mask),
            torch.where(use["single"], S.categorical_entropy(sel_logits, sel_mask), zero),
        )

        ctx2 = torch.cat([ctx, self.summary(enc, membership)], dim=1)
        et_logits = self.entity_type_logits(ctx2)
        et_mask = self.entity_type_mask(obs, use["build"], selection[:, 0])
        logp = logp + torch.where(use["entity_type"], S.categorical_logp(et_logits, et_mask, entity_type), zero)
        entropy = entropy + torch.where(use["entity_type"], S.categorical_entropy(et_logits, et_mask), zero)

        tgt_logits = self.target_logits(enc, ctx2)
        tgt_mask = self.target_mask(obs, type_)
        logp = logp + torch.where(use["target"], S.categorical_logp(tgt_logits, tgt_mask, target), zero)
        entropy = entropy + torch.where(use["target"], S.categorical_entropy(tgt_logits, tgt_mask), zero)

        et_vec = self.et_emb(entity_type.clamp(min=0)) * use["entity_type"].unsqueeze(-1).to(enc.torso.dtype)
        ctx3 = torch.cat([ctx2, et_vec], dim=1)
        cell_logits = self.cell_logits(enc, ctx3)
        cell_mask = self.cell_mask(obs, type_, use["build"], entity_type)
        logp = logp + torch.where(use["location"], S.categorical_logp(cell_logits, cell_mask, cell), zero)
        entropy = entropy + torch.where(use["location"], S.categorical_entropy(cell_logits, cell_mask), zero)

        sub_logits = self.sub_logits(ctx3, cell)
        sub_mask = torch.ones_like(sub_logits, dtype=torch.bool)
        logp = logp + torch.where(use["location"], S.categorical_logp(sub_logits, sub_mask, sub), zero)
        entropy = entropy + torch.where(use["location"], S.categorical_entropy(sub_logits, sub_mask), zero)

        out = {
            "logp": logp,
            "entropy": entropy,
            "type_logits": S.masked(tl, type_mask),
            "selection_logits": S.masked(sel_logits, sel_mask),
            "selection_mask": sel_mask,
            "target_logits": S.masked(tgt_logits, tgt_mask),
            "cell_logits": S.masked(cell_logits, cell_mask),
            "entity_type_logits": S.masked(et_logits, et_mask),
        }
        if "critic" in obs:
            out["value"] = self.value(enc, obs["critic"])
        return out

    # --- sampling ---------------------------------------------------------------------------------

    def act(self, obs: dict[str, torch.Tensor], noise: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """Draw one decision per row from supplied noise: [B, ACTION_INTS] int64, unused heads -1."""
        spec = self.spec
        seg = S.noise_slices()
        n = spec.n_ent
        enc = self.encode(obs)
        minus = torch.full_like(obs["mask_type"][:, 0], -1, dtype=torch.int64)

        type_ = S.gumbel_argmax(self.type_logits(enc), obs["mask_type"].to(torch.bool), noise[:, seg["type"]], temperature)
        use = self.usage(type_)

        ctx = self.context(enc, type_)
        sel_logits = self.selection_logits(enc, ctx)
        sel_mask = self.selection_mask(obs, type_, enc.row_mask)
        sel_noise = noise[:, seg["selection"]]
        many = S.select_many(sel_logits, sel_mask, sel_noise[:, :n], sel_noise[:, n:], temperature, spec.selection_max)
        one = S.select_one(sel_logits, sel_mask, sel_noise[:, :n], temperature)
        one_padded = torch.cat([one.unsqueeze(1), torch.full_like(many[:, 1:], -1)], dim=1)
        none = torch.full_like(many, -1)
        selection = torch.where(use["multi"].unsqueeze(1), many, torch.where(use["single"].unsqueeze(1), one_padded, none))
        membership = S.membership_of(selection, n)

        ctx2 = torch.cat([ctx, self.summary(enc, membership)], dim=1)
        et_mask = self.entity_type_mask(obs, use["build"], selection[:, 0])
        entity_type = S.gumbel_argmax(self.entity_type_logits(ctx2), et_mask, noise[:, seg["entityType"]], temperature)
        entity_type = torch.where(use["entity_type"], entity_type, minus)

        target = S.gumbel_argmax(self.target_logits(enc, ctx2), self.target_mask(obs, type_), noise[:, seg["target"]], temperature)
        target = torch.where(use["target"], target, minus)

        et_vec = self.et_emb(entity_type.clamp(min=0)) * use["entity_type"].unsqueeze(-1).to(enc.torso.dtype)
        ctx3 = torch.cat([ctx2, et_vec], dim=1)
        cell_mask = self.cell_mask(obs, type_, use["build"], entity_type)
        cell = S.gumbel_argmax(self.cell_logits(enc, ctx3), cell_mask, noise[:, seg["cell"]], temperature)
        cell = torch.where(use["location"], cell, minus)

        sub_logits = self.sub_logits(ctx3, cell)
        sub = S.gumbel_argmax(sub_logits, torch.ones_like(sub_logits, dtype=torch.bool), noise[:, seg["sub"]], temperature)
        sub = torch.where(use["location"], sub, minus)

        return torch.cat([type_.unsqueeze(1), entity_type.unsqueeze(1), target.unsqueeze(1), cell.unsqueeze(1), sub.unsqueeze(1), selection], dim=1)


ACT_INPUTS = (
    "entities",
    "entity_mask",
    "grid",
    "scalars",
    "mask_type",
    "mask_selection",
    "mask_target",
    "mask_cell",
    "mask_build_cell",
    "mask_row_entity_type",
    "mask_build_type",
    "noise",
    "temperature",
)
ACT_OUTPUTS = ("action",)


class ActGraph(nn.Module):
    """`Policy.act` as a function of positional tensors, for export. Output is int32 [B, ACTION_INTS]."""

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        obs = dict(zip(ACT_INPUTS, tensors))
        noise = obs.pop("noise")
        temperature = obs.pop("temperature")
        return self.policy.act(obs, noise, temperature).to(torch.int32)


def act_input_shapes(spec: Spec = SPEC, batch: int = 1) -> dict[str, tuple[tuple[int, ...], str]]:
    """Shape and dtype of every act-graph input, in `ACT_INPUTS` order."""
    n, t, e, cells = spec.n_ent, spec.n_types, spec.entity_types, spec.cells
    return {
        "entities": ((batch, n, spec.f), "float32"),
        "entity_mask": ((batch, n), "uint8"),
        "grid": ((batch, spec.c, spec.grid, spec.grid), "float32"),
        "scalars": ((batch, spec.s), "float32"),
        "mask_type": ((batch, t), "uint8"),
        "mask_selection": ((batch, t, n), "uint8"),
        "mask_target": ((batch, t, n), "uint8"),
        "mask_cell": ((batch, t, cells), "uint8"),
        "mask_build_cell": ((batch, e, cells), "uint8"),
        "mask_row_entity_type": ((batch, n, e), "uint8"),
        "mask_build_type": ((batch, e), "uint8"),
        "noise": ((batch, spec.noise_len), "float32"),
        "temperature": ((batch,), "float32"),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
