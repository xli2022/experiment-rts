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
LOG_RATIO_CLAMP = 10.0
"""Log-ratios are clamped to this before `exp`. float32 `exp` overflows above ~88.

This clamp once carried a note claiming the joint log-probability is large *by
construction* — summed over six heads plus up to `N_ENT` Bernoulli selection
rows — and that a per-decision log-ratio therefore sits in the tens. Measured on
a real rollout from the imitation clone, that is not so: `logp` is -2.4 on
average with a minimum of -12.5, the selection head contributes -2.3 of it
because a decision has about five legal rows rather than 160, and one Adam step
moves the log-ratio by 0.05. The joint ratio is a perfectly ordinary PPO ratio.

Log-ratios in the tens are a *symptom*: they appear only after the policy has
already run away, which it did because the advantages were noise (see
`--adv-floor`). The clamp is a guard against the arithmetic going non-finite on
the way down, not a trust region. At the old +/-20 it permitted a ratio of 4.8e8
and so a policy-gradient term of the same order, which is precisely the
`pg` = 1e7 recorded in `runs/aligned/log.jsonl`; 10 keeps a blown-up minibatch
merely large. `approxKl` is logged unclamped so the runaway stays visible.
"""


def reference_penalty(logp: torch.Tensor, reference_logp: torch.Tensor) -> torch.Tensor:
    """Huber leash with a restoring gradient even far from the reference.

    Only the PPO importance ratio needs a clamp before exponentiation. Clamping
    this log-ratio would make its gradient zero precisely where the policy is
    furthest from the reference. Huber is already linear in that region.
    """
    return torch.nn.functional.huber_loss(logp, reference_logp, reduction="mean", delta=1.0)


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
        # Teammates receive the same terminal result. Count a Quarters match
        # once, just like a Lanes match, when updating league statistics.
        seen: set[tuple[int, int]] = set()
        self.matches: list[int] = []
        for r in learner:
            if self.row_key[r] not in seen:
                seen.add(self.row_key[r])
                self.matches.append(r)
        self.snapshots = {name: np.asarray(rows, dtype=np.int64) for name, rows in by_member.items()}
        self.assignments = assignments


class Rollout:
    """`steps × rows` of everything PPO needs, flattened, allocated from the first batch."""

    def __init__(self, steps: int, rows: int):
        self.steps, self.rows = steps, rows
        self.obs: dict[str, np.ndarray] | None = None
        self.actions = np.zeros((steps * rows, SPEC.action_ints), dtype=np.int32)
        self.selection_scores = np.zeros((steps * rows, SPEC.n_ent), dtype=np.float32)
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
            if args.layout == "lanes":
                layout = LANES
            elif args.layout == "quarters":
                layout = QUARTERS
            else:
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
    # 2.5e-4 trips `--target-kl` after a few minibatches of sixty, so most of a
    # rollout is collected and thrown away, and it is still the right number.
    # 1e-4 spends the whole update inside the trust region and gets a policy
    # that has barely left the clone: measured over 150 updates of policy
    # training it drifted to a Huber KL of 0.08 and scored 4% against
    # `scripted@10`, where 2.5e-4 reached 42% in forty. The wasted rollouts buy
    # the distance.
    parser.add_argument("--lr", type=float, default=2.5e-4)
    # Per *decision*, not per tick, and a decision is SPEC.decision_ticks ticks.
    # A match runs to a median 8,383 ticks, so ~2,100 decisions: at 0.99 the
    # terminal +/-1 arrives discounted by 0.99^2100 = 7e-10 and the effective
    # horizon is 100 decisions, about 20 seconds of a seven-minute match. Winning
    # was literally not in the objective; every earlier PPO run could do no more
    # than climb the shaping potential greedily. At 0.999 the same terminal is
    # worth 0.12, which is the same order as the shaping a whole match
    # accumulates, so the outcome competes with the hint instead of vanishing
    # underneath it.
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf", type=float, default=0.5)
    # This is the only gradient in the loss that points the same way every step,
    # so given long enough it outvotes the advantage signal: entropy climbs from
    # 0.86 to 2.02 over seventy updates and the win rate goes 0.25 -> 0.42 ->
    # 0.00 as the policy spreads out into near-random play. Dropping it to 5e-4
    # does not fix that — it only stops the policy ever getting anywhere, 4%
    # against `scripted@10` versus 42%. A PPO run here has a peak and then
    # decays, so evaluate often enough to catch the peak and trust `best.pt` to
    # hold it; that is what `--eval-every` is for.
    parser.add_argument("--ent", type=float, default=0.003)
    # Annealed to `--ent-end` over the run, on the same schedule as beta. A fixed
    # coefficient is what ends every run here: exploration is worth most early,
    # when the policy is still near the clone, and worth least once it has found
    # something, at which point the bonus is simply a constant pressure toward
    # uniform play that nothing opposes.
    parser.add_argument("--ent-end", type=float, default=2e-4)
    parser.add_argument("--kl-start", type=float, default=1.0)
    # The leash has to loosen for the policy to improve on the clone at all: held
    # near 0.9 by a long `--updates`, the run stayed within a Huber KL of 0.08 of
    # its initialisation and scored 4%. Note that beta anneals over `--updates`,
    # so a long run loosens the leash more slowly per update than a short one —
    # the schedule is in fractions of the run, not in updates.
    parser.add_argument("--kl-end", type=float, default=0.1)
    # Measured at 1e-3: the per-decision shaping term has std 2e-5, a fifth of the
    # constant time cost, and a terminal +/-1 reaches a 32-step rollout window
    # essentially never. The dense term is the only signal most updates get, so it
    # has to be worth more than rounding. Potential-based shaping leaves the
    # optimal policy alone, so scaling it is free.
    parser.add_argument("--shaping", type=float, default=1e-2)
    # Paid every decision, so what matters is the discounted sum over a match,
    # not the per-step figure. At gamma 0.999 over 2,100 decisions, 1e-4 sums to
    # 0.086 against a terminal win worth 0.12 — the clock would have been nearly
    # as loud as the result, and losing quickly nearly as good as winning slowly.
    # 2e-5 sums to 0.017: enough that a draw is never free, quiet enough that the
    # result decides.
    parser.add_argument("--time-cost", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ladder", default="10,20,40")
    parser.add_argument(
        "--layout",
        choices=["lanes", "quarters", "mix"],
        default="lanes",
        help="which map to train and evaluate on; a model is trained per layout",
    )
    # Only consulted for `--layout mix`. The mixed run is what shipped a model
    # that played Lanes at 85% and Quarters at 3%: the policy is *told* its
    # layout by a one-hot scalar, so it can neglect the quarter of the data it
    # is scored on least, and the in-run eval scored only Lanes.
    parser.add_argument("--quarters-share", type=float, default=0.25)
    parser.add_argument(
        "--refresh",
        type=int,
        default=0,
        help="updates between league redraws; 0 auto-sizes to one full match so matches finish",
    )
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument(
        "--keep-every",
        type=int,
        default=0,
        help="also write ckpt<update>.pt every N updates; 0 disables. The in-run eval is too few "
        "seeds to pick a winner — screen these offline with rtsml-eval instead",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.5,
        help="abandon the rest of an update once approxKl passes this; 0 disables",
    )
    parser.add_argument(
        "--adv-floor",
        type=float,
        default=1e-3,
        help="advantages are divided by their std or this, whichever is larger; stops a flat window being rescaled into noise",
    )
    parser.add_argument(
        "--value-warmup",
        type=int,
        default=100,
        help="updates fitting the value head alone before the policy moves; BC never trains the critic",
    )
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
        args.value_warmup = 1
        args.d, args.heads, args.layers, args.torso = 32, 2, 1, 64

    # A redraw resets every environment, so a redraw that lands mid-match throws
    # that match away. The server already resets an environment of its own accord
    # when its match ends (`tools/ml/serve.ts`), keeping the same opponent, so the
    # only cost of a long refresh is a stale league draw. Auto-size it to one full
    # match; anything shorter means no match ever finishes, no terminal +/-1 reward
    # is ever collected, and the league is never told who won.
    match_updates = math.ceil(args.max_ticks / (args.rollout * SPEC.decision_ticks))
    if args.refresh <= 0:
        args.refresh = match_updates
    elif args.refresh < match_updates:
        print(
            f"note: --refresh {args.refresh} resets environments every "
            f"{args.refresh * args.rollout * SPEC.decision_ticks} ticks; matches longer "
            f"than that are cut short and yield no terminal reward "
            f"({match_updates} would cover the {args.max_ticks}-tick cap)."
        )

    # A mixed run has no single map to score, so it is scored on the one the
    # game ships most of; a per-layout run is scored on its own.
    eval_layout = QUARTERS if args.layout == "quarters" else LANES

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
    # Top level too, beside the model shape: `rtsml-export` reads it to name the
    # file, and a checkpoint that does not say which map it plays is a checkpoint
    # someone will ship to the wrong one.
    hparams["layout"] = args.layout
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    critic_params = {id(q) for q in (*policy.critic_mlp.parameters(), *policy.value_head.parameters())}
    trunk = [q for q in policy.parameters() if id(q) not in critic_params]
    frozen = False
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
                    actions, selection_scores = policy.sample(obs, noise, temperature)
                    out = policy.evaluate(obs, actions, args.temperature, selection_scores)
                full = np.full((len(batch), SPEC.action_ints), -1, dtype=np.int32)
                full[:, 0] = NOOP
                full[roles.learner] = actions.to(torch.int32).cpu().numpy()
                for name, rows in roles.snapshots.items():
                    full[rows] = decide(opponents[name], batch.arrays, rows, device, args.temperature, generator)
                rollout.store_obs(t, batch.arrays, roles.learner)
                rollout.actions[t * rollout.rows : (t + 1) * rollout.rows] = full[roles.learner]
                rollout.selection_scores[t * rollout.rows : (t + 1) * rollout.rows] = selection_scores.cpu().numpy()
                rollout.logp[t] = out["logp"].cpu().numpy()
                rollout.value[t] = out["value"].cpu().numpy()

                batch = env.step(full)
                rollout.reward[t] = batch.reward[roles.learner]
                rollout.done[t] = batch.done[roles.learner]
                for r in roles.matches:
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
            # How much of the return the critic actually explains. Near zero means
            # the advantages are noise, and PPO will sharpen the policy onto that
            # noise: entropy collapses while the win rate does not move.
            var_ret = float(returns.var())
            explained = float(1.0 - (returns - old_value).var() / var_ret) if var_ret > 1e-20 else 0.0
            # Dividing by the std alone is what killed every earlier run. Measured
            # on a real rollout, the first 32-decision window of a match has a
            # reward that is *exactly* constant — one unique value across every
            # row and every step, std 0.0 — because no side has committed
            # anything to the board yet and the potential has not moved. Once the
            # critic has fitted that constant (explainedVariance reaches 0.994),
            # the advantages are float32 rounding, and `/(std + 1e-8)` rescales
            # that rounding to unit variance and hands it to PPO at full
            # strength for three epochs. Entropy collapsed to 2e-5, the ratio ran
            # away, and the win rate went to zero and stayed there.
            #
            # Dividing by whichever is larger of the std and a floor leaves a
            # healthy window untouched — its std is well above the floor — and
            # turns a signal-free one into the near-no-op it should always have
            # been.
            adv_std = float(adv.std())
            adv = (adv - adv.mean()) / max(adv_std, args.adv_floor)
            progress = (update - 1) / max(1, args.updates - 1)
            beta = args.kl_start + (args.kl_end - args.kl_start) * progress
            ent_coef = args.ent + (args.ent_end - args.ent) * progress
            # BC trains the policy heads only, so the critic arrives at PPO
            # random: its predictions (std ~0.18) swamp the real advantage signal
            # (std ~2e-5) by four orders of magnitude, and normalising advantages
            # then rescales that noise to unit variance and optimises it. Fit the
            # value head first, with the policy pinned.
            warming = update <= args.value_warmup
            # The value head shares the trunk with the policy, so training it
            # would move the policy too. Freezing the rest for the duration of
            # the warmup keeps the clone byte-for-byte unchanged, and stops
            # autograd at the torso rather than backpropagating through the
            # whole encoder only to discard the result. Taken from the modules
            # themselves: a name prefix would silently train everything again
            # the day either is renamed.
            if warming != frozen:
                for q in trunk:
                    q.requires_grad_(not warming)
                frozen = warming

            assert rollout.obs is not None
            n = rollout.steps * rollout.rows
            stats: dict[str, list[float]] = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "approxKl": []}
            planned = args.epochs * math.ceil(n / args.minibatch)
            skipped = 0
            policy.train()
            stopped = False
            for _ in range(args.epochs):
                if stopped:
                    break
                order = rng.permutation(n)
                for start in range(0, n, args.minibatch):
                    idx = order[start : start + args.minibatch]
                    obs = to_torch(rollout.obs, device, idx)
                    actions = torch.from_numpy(rollout.actions[idx].astype(np.int64)).to(device)
                    selection_scores = torch.from_numpy(rollout.selection_scores[idx]).to(device)
                    out = policy.evaluate(obs, actions, args.temperature, selection_scores)
                    logp = out["logp"]
                    lp_old = torch.from_numpy(old_logp[idx]).to(device)
                    a = torch.from_numpy(adv[idx]).to(device)
                    # The summed per-head log-probability has a wide dynamic range
                    # (the selection head alone sums up to N_ENT Bernoulli terms), so
                    # the log-ratio is clamped before it is exponentiated: float32
                    # exp overflows to inf above ~88, and one inf here poisons every
                    # gradient in the batch.
                    ratio = torch.exp(torch.clamp(logp - lp_old, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP))
                    pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                    v_old = torch.from_numpy(old_value[idx]).to(device)
                    ret = torch.from_numpy(returns[idx]).to(device)
                    v_clipped = v_old + torch.clamp(out["value"] - v_old, -args.clip, args.clip)
                    vf = 0.5 * torch.max((out["value"] - ret) ** 2, (v_clipped - ret) ** 2).mean()
                    ent = out["entropy"].mean()
                    loss = vf if warming else pg + args.vf * vf - ent_coef * ent
                    kl = torch.zeros((), device=device)
                    if reference is not None and not warming:
                        with torch.no_grad():
                            lp_ref = reference.evaluate(obs, actions, args.temperature, selection_scores)["logp"]
                        # A Huber penalty on the log-ratio, not the k3 KL estimator.
                        # k3's gradient is exp(r) - 1, and `logp` here is a joint
                        # log-probability summed over six heads and up to N_ENT
                        # selection rows, so r sits in the tens rather than near
                        # zero: at r = 20 that is a gradient multiplier of 5e8, and
                        # the leash drowns out the policy gradient entirely
                        # (measured: kl ~ 1e6 against pg ~ 0.1). Huber is quadratic
                        # near zero and linear beyond, so the pull is bounded by
                        # beta however far the policy has drifted.
                        kl = reference_penalty(logp, lp_ref)
                        loss = loss + beta * kl
                    if not torch.isfinite(loss):
                        skipped += 1
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # clip_grad_norm_ scales by 1/total_norm, so a NaN norm makes
                    # every gradient NaN rather than stopping it: a single bad
                    # minibatch would otherwise write NaN into all 81 tensors and
                    # the run would train on happily for hours.
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    if not torch.isfinite(norm):
                        skipped += 1
                        opt.zero_grad(set_to_none=True)
                        continue
                    opt.step()
                    stats["pg"].append(float(pg.detach()))
                    stats["vf"].append(float(vf.detach()))
                    stats["ent"].append(float(ent.detach()))
                    stats["kl"].append(float(kl.detach()))
                    stats["clip"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
                    # Half the mean square log-ratio: non-negative per row, so
                    # unlike the plain mean of (lp_old - logp) it cannot cancel.
                    # That signed mean is why the early stop never fired — the
                    # logs record it reaching -524, having passed the 0.5 target
                    # thousands of updates earlier, because rows that had run one
                    # way offset rows that had run the other.
                    #
                    # Not the more usual k3, exp(-r) - 1 + r, whose exponential
                    # makes it a tail statistic: `logp` here reaches -12.5 on an
                    # unlikely joint decision, and a couple of such rows drove k3
                    # to 17 in a minibatch whose clip fraction was 0.06 and whose
                    # pg was 0.03 — a policy that had barely moved, reported as a
                    # catastrophe, stopping every update after one step.
                    with torch.no_grad():
                        r_kl = (logp - lp_old).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
                        stats["approxKl"].append(float(0.5 * (r_kl * r_kl).mean()))
                    # PPO's trust region is only enforced through the ratio clip,
                    # which stops bounding anything once the policy has moved far:
                    # measured, approxKl runs 0.07-0.35 while the run is healthy and
                    # then climbs through 2, 6, 8 as it destroys itself. Abandon the
                    # rest of an update that has already moved too far.
                    if not warming and args.target_kl > 0 and stats["approxKl"][-1] > args.target_kl:
                        stopped = True
                        break

            # A single non-finite parameter makes every later decision garbage, and
            # the checkpoints saved from here on are worthless. Stop while `best.pt`
            # still holds a policy that plays.
            bad = [name for name, p in policy.named_parameters() if not torch.isfinite(p).all()]
            if bad:
                raise RuntimeError(
                    f"update {update}: {len(bad)} parameter tensors went non-finite "
                    f"(first: {bad[0]}). {skipped} minibatches were skipped this update. "
                    f"runs/ppo/best.pt still holds the last policy that evaluated."
                )

            record: dict[str, Any] = {
                "update": update,
                "decisions": update * n,
                "reward": float(rollout.reward.mean()),
                "beta": beta,
                "entCoef": ent_coef,
                "seconds": round(time.time() - t0, 1),
                "skipped": skipped,
                "warming": warming,
                "explainedVariance": explained,
                "advStd": adv_std,
                "stopped": stopped,
                # Optimiser steps actually taken against the number the epochs
                # asked for. A run that early-stops after one minibatch of forty
                # eight is not training, however healthy its other numbers look.
                "steps": len(stats["pg"]),
                "planned": planned,
                "results": {k: [None if w is None else bool(w) for w in v] for k, v in results.items()},
                **{k: (float(np.mean(v)) if v else math.nan) for k, v in stats.items()},
            }
            if update % args.snapshot_every == 0:
                league.add_snapshot(f"ppo{update}", policy, hparams)
                league.save(args.out / "league.pt")
            if args.eval_every and update % args.eval_every == 0:
                seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.eval_seeds))
                policy.eval()
                rates = []
                for seat in (0, 1):
                    rates.append(summarise(play(policy, slot("scripted", 10), seeds, eval_layout, seat, args.procs, device, args.temperature, args.max_ticks))["winRate"])
                record["eval"] = {"seat0": rates[0], "seat1": rates[1]}
                score = sum(rates) / 2
                if score > best:
                    best = score
                    save_checkpoint(args.out / "best.pt", policy, "ppo", hparams, {"winRateVsScripted10": score, "update": update, "warming": warming})
                policy.train()
                # `play` builds and tears down its own BunVectorEnv, so the
                # training env was never touched and `batch` is still the live
                # observation. Resetting here used to throw away every match in
                # flight: a match needs about 65 updates to finish and the eval
                # landed every 50, so most matches never reached a terminal +/-1
                # and the league was never told who won — the same failure the
                # `--refresh` note warns about, re-entered through the eval.
            if args.keep_every and update % args.keep_every == 0 and not warming:
                save_checkpoint(args.out / f"ckpt{update}.pt", policy, "ppo", hparams, {"update": update})
            save_checkpoint(args.out / "last.pt", policy, "ppo", hparams, {"update": update})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"update {update} reward {record['reward']:+.4f} pg {record['pg']:+.3f} vf {record['vf']:.3f} "
                f"ent {record['ent']:.2f} kl {record['kl']:.4f} clip {record['clip']:.2f} beta {beta:.2f} "
                f"({record['seconds']}s)  league: {league.summary()}"
            )
    finally:
        for q in trunk:
            q.requires_grad_(True)
        log.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
