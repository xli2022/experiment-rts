"""PPO from the imitation policy, against a league, on a leash.

    rtsml-ppo --init runs/bc/best.pt --procs 16 --envs 8

Each environment pits the learner against an opponent the league drew — the
scripted bot at some think interval, the imitation snapshot, or an earlier
checkpoint — from a seat that alternates per environment; a snapshot
opponent is played here, from the same batch, by its frozen weights. The
loss is clipped PPO with GAE over the summed per-head log-probabilities,
plus an entropy bonus and a KL term to the imitation policy, annealed, that
keeps the play human-shaped while it gets stronger.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import LANES, QUARTERS, BunVectorEnv, EnvConfig, slot, team_of
from .evaluate import match_config, play, summarise
from .imitation import SEED_STRIDE, add_model_args, model_hparams
from .league import League, Member
from .model import Policy, parameter_count
from .spec import NOOP, SPEC
from .util import decide, gumbel_noise, load_checkpoint, pick_device, save_checkpoint, set_seed, to_torch

OBS_NAMES = (
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
    "critic",
)
EVAL_SEED0 = 900_000


@dataclass
class Assignment:
    member: Member
    seat: int
    layout: int


class Roles:
    """Which rows of a batch the learner plays, and which each snapshot opponent plays."""

    def __init__(self, batch_slots: list[tuple[int, int, int]], assignments: list[list[Assignment]]):
        learner: list[int] = []
        by_member: dict[str, list[int]] = {}
        self.row_key: list[tuple[int, int]] = []
        self.row_team: list[int] = []
        for r, (p, e, player) in enumerate(batch_slots):
            a = assignments[p][e]
            players = 4 if a.layout == QUARTERS else 2
            team = team_of(player, players)
            self.row_key.append((p, e))
            self.row_team.append(team)
            if team == a.seat:
                learner.append(r)
            else:
                by_member.setdefault(a.member.name, []).append(r)
        self.learner = np.asarray(learner, dtype=np.int64)
        self.snapshots = {name: np.asarray(rows, dtype=np.int64) for name, rows in by_member.items()}
        self.assignments = assignments


class Rollout:
    """`steps × rows` of everything PPO needs, flattened, allocated from the first batch."""

    def __init__(self, steps: int, rows: int):
        self.steps, self.rows = steps, rows
        self.obs: dict[str, np.ndarray] | None = None
        self.actions = np.zeros((steps * rows, SPEC.action_ints), dtype=np.int32)
        self.logp = np.zeros((steps, rows), dtype=np.float32)
        self.value = np.zeros((steps, rows), dtype=np.float32)
        self.reward = np.zeros((steps, rows), dtype=np.float32)
        self.done = np.zeros((steps, rows), dtype=np.float32)

    def store_obs(self, t: int, arrays: dict[str, np.ndarray], index: np.ndarray) -> None:
        if self.obs is None:
            self.obs = {name: np.empty((self.steps * self.rows, *arrays[name].shape[1:]), dtype=arrays[name].dtype) for name in OBS_NAMES}
        for name in OBS_NAMES:
            self.obs[name][t * self.rows : (t + 1) * self.rows] = arrays[name][index]

    def advantages(self, next_value: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(self.reward)
        last = np.zeros(self.rows, dtype=np.float32)
        for t in reversed(range(self.steps)):
            nonterminal = 1.0 - self.done[t]
            nv = next_value if t == self.steps - 1 else self.value[t + 1]
            delta = self.reward[t] + gamma * nv * nonterminal - self.value[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
        return adv, adv + self.value


def build_configs(league: League, procs: int, envs: int, rng: np.random.Generator, seed_base: int, args: argparse.Namespace) -> tuple[list[list[EnvConfig]], list[list[Assignment]]]:
    members = league.sample(procs * envs)
    groups: list[list[EnvConfig]] = []
    assignments: list[list[Assignment]] = []
    i = 0
    for _ in range(procs):
        group: list[EnvConfig] = []
        assigned: list[Assignment] = []
        for _ in range(envs):
            layout = QUARTERS if rng.random() < args.quarters_share else LANES
            seat = i % 2
            cfg = match_config(seed_base + i * SEED_STRIDE, layout, seat, members[i].slot(), args.max_ticks)
            cfg.shaping, cfg.time_cost, cfg.gamma = args.shaping, args.time_cost, args.gamma
            group.append(cfg)
            assigned.append(Assignment(members[i], seat, layout))
            i += 1
        groups.append(group)
        assignments.append(assigned)
    return groups, assignments


def snapshot_policies(league: League, roles: Roles, device: torch.device, cache: dict[str, Policy]) -> dict[str, Policy]:
    out: dict[str, Policy] = {}
    for member in league.members:
        if member.name not in roles.snapshots:
            continue
        if member.name not in cache:
            policy = Policy(**member.hparams.get("model", {})).to(device)
            assert member.state is not None
            policy.load_state_dict(member.state)
            policy.eval()
            cache[member.name] = policy
        out[member.name] = cache[member.name]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, help="the imitation checkpoint to start from and stay close to")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=32, help="decisions per row per update")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf", type=float, default=0.5)
    parser.add_argument("--ent", type=float, default=0.003)
    parser.add_argument("--kl-start", type=float, default=1.0)
    parser.add_argument("--kl-end", type=float, default=0.1)
    parser.add_argument("--shaping", type=float, default=1e-3)
    parser.add_argument("--time-cost", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ladder", default="10,20,40")
    parser.add_argument("--quarters-share", type=float, default=0.25)
    parser.add_argument("--refresh", type=int, default=10, help="updates between league redraws")
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50, help="0 disables")
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--max-ticks", type=int, default=24_000)
    parser.add_argument("--seed0", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="a tiny run that exercises everything")
    add_model_args(parser)
    args = parser.parse_args(argv)
    if args.smoke:
        args.procs, args.envs, args.rollout, args.updates, args.epochs, args.minibatch = 1, 2, 4, 2, 1, 4
        args.refresh, args.snapshot_every, args.eval_every, args.max_ticks = 1, 1, 0, 400
        args.d, args.heads, args.layers, args.torso = 32, 2, 1, 64

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = pick_device(args.device)
    if args.init:
        ckpt = load_checkpoint(args.init, device)
        hparams = dict(ckpt["hparams"])
        policy = Policy(**hparams.get("model", {})).to(device)
        policy.load_state_dict(ckpt["model"])
        reference: Policy | None = copy.deepcopy(policy).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    else:
        hparams = {"model": model_hparams(args)}
        policy = Policy(**hparams["model"]).to(device)
        reference = None
    hparams.update({"ppo": {k: v for k, v in vars(args).items() if not isinstance(v, Path)}})
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    print(f"policy: {parameter_count(policy)} parameters on {device}; KL reference: {'yes' if reference else 'none'}")

    league = League(tuple(int(k) for k in args.ladder.split(",")), seed=args.seed)
    if reference is not None:
        league.add_snapshot("imitation", policy, hparams)
    cache: dict[str, Policy] = {}

    seed_counter = args.seed0
    groups, assignments = build_configs(league, args.procs, args.envs, rng, seed_counter, args)
    seed_counter += args.procs * args.envs * SEED_STRIDE
    env = BunVectorEnv(groups)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    best = -1.0
    t0 = time.time()
    try:
        batch = env.reset()
        roles = Roles(batch.slots, assignments)
        opponents = snapshot_policies(league, roles, device, cache)
        for update in range(1, args.updates + 1):
            if update > 1 and (update - 1) % args.refresh == 0:
                groups, assignments = build_configs(league, args.procs, args.envs, rng, seed_counter, args)
                seed_counter += args.procs * args.envs * SEED_STRIDE
                batch = env.reset(groups)
                roles = Roles(batch.slots, assignments)
                opponents = snapshot_policies(league, roles, device, cache)

            rollout = Rollout(args.rollout, len(roles.learner))
            results: dict[str, list[bool | None]] = {}
            policy.eval()
            for t in range(args.rollout):
                obs = to_torch({k: v for k, v in batch.arrays.items() if k != "label"}, device, roles.learner)
                with torch.no_grad():
                    noise = gumbel_noise(len(roles.learner), generator).to(device)
                    temperature = torch.full((len(roles.learner),), args.temperature, device=device)
                    actions = policy.act(obs, noise, temperature)
                    out = policy.evaluate(obs, actions)
                full = np.full((len(batch), SPEC.action_ints), -1, dtype=np.int32)
                full[:, 0] = NOOP
                full[roles.learner] = actions.to(torch.int32).cpu().numpy()
                for name, rows in roles.snapshots.items():
                    full[rows] = decide(opponents[name], batch.arrays, rows, device, args.temperature, generator)
                rollout.store_obs(t, batch.arrays, roles.learner)
                rollout.actions[t * rollout.rows : (t + 1) * rollout.rows] = full[roles.learner]
                rollout.logp[t] = out["logp"].cpu().numpy()
                rollout.value[t] = out["value"].cpu().numpy()

                batch = env.step(full)
                rollout.reward[t] = batch.reward[roles.learner]
                rollout.done[t] = batch.done[roles.learner]
                for r in roles.learner:
                    if batch.done[r]:
                        p, e = roles.row_key[r]
                        a = assignments[p][e]
                        winner = int(batch.winner[r])
                        won = None if winner < 0 else winner == roles.row_team[r]
                        league.report(a.member, won)
                        results.setdefault(a.member.name, []).append(won)

            with torch.no_grad():
                obs = to_torch({k: v for k, v in batch.arrays.items() if k != "label"}, device, roles.learner)
                next_value = policy.value_of(obs).cpu().numpy()
            adv, returns = rollout.advantages(next_value, args.gamma, args.lam)
            adv = adv.reshape(-1)
            returns = returns.reshape(-1)
            old_logp = rollout.logp.reshape(-1)
            old_value = rollout.value.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            beta = args.kl_start + (args.kl_end - args.kl_start) * (update - 1) / max(1, args.updates - 1)

            assert rollout.obs is not None
            n = rollout.steps * rollout.rows
            stats: dict[str, list[float]] = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "approxKl": []}
            policy.train()
            for _ in range(args.epochs):
                order = rng.permutation(n)
                for start in range(0, n, args.minibatch):
                    idx = order[start : start + args.minibatch]
                    obs = to_torch(rollout.obs, device, idx)
                    actions = torch.from_numpy(rollout.actions[idx].astype(np.int64)).to(device)
                    out = policy.evaluate(obs, actions)
                    logp = out["logp"]
                    lp_old = torch.from_numpy(old_logp[idx]).to(device)
                    a = torch.from_numpy(adv[idx]).to(device)
                    ratio = torch.exp(logp - lp_old)
                    pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                    v_old = torch.from_numpy(old_value[idx]).to(device)
                    ret = torch.from_numpy(returns[idx]).to(device)
                    v_clipped = v_old + torch.clamp(out["value"] - v_old, -args.clip, args.clip)
                    vf = 0.5 * torch.max((out["value"] - ret) ** 2, (v_clipped - ret) ** 2).mean()
                    ent = out["entropy"].mean()
                    loss = pg + args.vf * vf - args.ent * ent
                    kl = torch.zeros((), device=device)
                    if reference is not None:
                        with torch.no_grad():
                            lp_ref = reference.evaluate(obs, actions)["logp"]
                        r = lp_ref - logp
                        kl = (torch.exp(r) - 1 - r).mean()
                        loss = loss + beta * kl
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()
                    stats["pg"].append(float(pg.detach()))
                    stats["vf"].append(float(vf.detach()))
                    stats["ent"].append(float(ent.detach()))
                    stats["kl"].append(float(kl.detach()))
                    stats["clip"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
                    stats["approxKl"].append(float((lp_old - logp).mean().detach()))

            record: dict[str, Any] = {
                "update": update,
                "decisions": update * n,
                "reward": float(rollout.reward.mean()),
                "beta": beta,
                "seconds": round(time.time() - t0, 1),
                "results": {k: [None if w is None else bool(w) for w in v] for k, v in results.items()},
                **{k: float(np.mean(v)) for k, v in stats.items()},
            }
            if update % args.snapshot_every == 0:
                league.add_snapshot(f"ppo{update}", policy, hparams)
                league.save(args.out / "league.pt")
            if args.eval_every and update % args.eval_every == 0:
                seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.eval_seeds))
                policy.eval()
                rates = []
                for seat in (0, 1):
                    rates.append(summarise(play(policy, slot("scripted", 10), seeds, LANES, seat, args.procs, device, args.temperature, args.max_ticks))["winRate"])
                record["eval"] = {"seat0": rates[0], "seat1": rates[1]}
                score = sum(rates) / 2
                if score > best:
                    best = score
                    save_checkpoint(args.out / "best.pt", policy, "ppo", hparams, {"winRateVsScripted10": score, "update": update})
                policy.train()
                batch = env.reset(groups)
                roles = Roles(batch.slots, assignments)
                opponents = snapshot_policies(league, roles, device, cache)
            save_checkpoint(args.out / "last.pt", policy, "ppo", hparams, {"update": update})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"update {update} reward {record['reward']:+.4f} pg {record['pg']:+.3f} vf {record['vf']:.3f} "
                f"ent {record['ent']:.2f} kl {record['kl']:.4f} clip {record['clip']:.2f} beta {beta:.2f} "
                f"({record['seconds']}s)  league: {league.summary()}"
            )
    finally:
        log.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
