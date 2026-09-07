"""Screen several checkpoints against scripted@10 and print a table.

    python screen.py --seeds 24 --seed0 1200000 --procs 20 ckpt1.pt ckpt2.pt ...
"""
import argparse, json, math, time
from pathlib import Path
import torch
from rtsml.env import LANES, slot
from rtsml.evaluate import play, summarise
from rtsml.model import Policy
from rtsml.util import load_checkpoint, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+", type=Path)
ap.add_argument("--seeds", type=int, default=24)
ap.add_argument("--seed0", type=int, default=1200000)
ap.add_argument("--procs", type=int, default=20)
ap.add_argument("--rung", type=int, default=10)
ap.add_argument("--out", type=Path)
ap.add_argument("--temperature", type=float, default=1.0)
a = ap.parse_args()

dev = pick_device()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
rows = []
for path in a.ckpts:
    ck = load_checkpoint(path, dev)
    pol = Policy(**ck["hparams"].get("model", {})).to(dev)
    pol.load_state_dict(ck["model"]); pol.eval()
    t0 = time.time(); wins = 0; n = 0; ticks = []; cpm = []; outcomes = {}
    for seat in (0, 1):
        res = play(pol, slot("scripted", a.rung), seeds, LANES, seat, a.procs, dev, a.temperature, 24000)
        # Per-match outcomes keyed by (seed, seat): the same key across checkpoints
        # is the same map from the same side, so candidates can be compared pairwise
        # rather than as two independent rates. McNemar on the disagreements is far
        # more powerful than a two-proportion z on 192 matches.
        for m in res: outcomes[f"{m.seed}:{seat}"] = bool(m.won)
        r = summarise(res)
        wins += r["wins"]; n += r["matches"]; ticks.append(r["medianTicks"]); cpm.append(r["commandsPerMinute"])
    rate = wins / n
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    rows.append({"ckpt": str(path), "wins": wins, "matches": n, "winRate": rate, "se": se,
                 "medianTicks": sum(ticks)/2, "cpm": sum(cpm)/2, "seconds": round(time.time()-t0, 1),
                 "outcomes": outcomes})
    print(f"{path.name:<16} {wins:>3}/{n}  {rate:.3f} +/- {se:.3f}   ticks {sum(ticks)/2:.0f}  cpm {sum(cpm)/2:.1f}  ({rows[-1]['seconds']}s)", flush=True)

rows.sort(key=lambda r: -r["winRate"])
print("\nranked:")
for r in rows: print(f"  {Path(r['ckpt']).name:<16} {r['winRate']:.3f}  ({r['wins']}/{r['matches']})")
if a.out: a.out.write_text(json.dumps({"seeds": a.seeds, "seed0": a.seed0, "rung": a.rung, "rows": rows}, indent=2))
