"""Export the act graph to ONNX for the browser, and prove it says the same thing.

    rtsml-export --ckpt runs/ppo/best.pt --out ../public/models [--int8]

Sampling lives inside the graph and its noise is an input, so parity between
torch and onnxruntime is an exact comparison of the integers each returns
for the same observation and the same noise. The result is `policy.onnx` and
a `policy.json` beside it naming the spec version the model was trained
against, its inputs and outputs, its hash and, if `--evaluate` ran, how it
plays; the browser refuses a model whose spec version is not its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ACT_INPUTS, ACT_OUTPUTS, ActGraph, Policy, act_input_shapes, parameter_count
from .spec import SPEC
from .util import gumbel_noise, load_checkpoint, pick_device

OPSET = 18


def export_onnx(policy: Policy, opset: int = OPSET) -> bytes:
    """The act graph as ONNX bytes, batch 1, every shape fixed."""
    graph = ActGraph(policy).eval().cpu()
    args = tuple(example_inputs(1).values())
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            graph,
            args,
            buf,
            input_names=list(ACT_INPUTS),
            output_names=list(ACT_OUTPUTS),
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    return buf.getvalue()


def example_inputs(batch: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Random inputs of the right shapes; masks a coin flip each, Noop always legal."""
    out: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in act_input_shapes(batch=batch).items():
        if name == "noise":
            out[name] = gumbel_noise(batch, generator)
        elif name == "temperature":
            out[name] = torch.ones(shape)
        elif dtype == "float32":
            out[name] = torch.randn(shape, generator=generator)
        else:
            out[name] = (torch.rand(shape, generator=generator) < 0.5).to(torch.uint8)
    out["mask_type"][:, 0] = 1
    return out


def run_onnx(session: Any, inputs: dict[str, torch.Tensor]) -> np.ndarray:
    feed = {name: inputs[name].cpu().numpy() for name in ACT_INPUTS}
    return session.run(list(ACT_OUTPUTS), feed)[0]


def parity(policy: Policy, onnx_bytes: bytes, samples: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """How many of `samples` (batch-1 inputs) the ONNX graph decides identically to torch."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    graph = ActGraph(policy).eval().cpu()
    agree = 0
    first_diff = None
    for k, inputs in enumerate(samples):
        with torch.no_grad():
            want = graph(*[inputs[name] for name in ACT_INPUTS]).numpy()
        got = run_onnx(session, inputs)
        if got.shape == want.shape and (got == want).all():
            agree += 1
        elif first_diff is None:
            first_diff = {"sample": k, "torch": want[0].tolist(), "onnx": got[0].tolist()}
    return {"samples": len(samples), "agree": agree, "firstDifference": first_diff}


def quantize_int8(onnx_bytes: bytes) -> bytes:
    """Dynamic int8 weights for the matmuls; the graph is otherwise unchanged."""
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "fp32.onnx"
        dst = Path(tmp) / "int8.onnx"
        src.write_bytes(onnx_bytes)
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        return dst.read_bytes()


def manifest(onnx_bytes: bytes, policy: Policy, extra: dict[str, Any]) -> dict[str, Any]:
    shapes = act_input_shapes(batch=1)
    return {
        "specVersion": SPEC.version,
        "model": "policy.onnx",
        "sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "bytes": len(onnx_bytes),
        "parameters": parameter_count(policy),
        "opset": OPSET,
        "inputs": [{"name": n, "shape": list(s), "dtype": d} for n, (s, d) in shapes.items()],
        "outputs": [{"name": "action", "shape": [1, SPEC.action_ints], "dtype": "int32"}],
        "noise": {"length": SPEC.noise_len, "distribution": "gumbel"},
        "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def live_samples(n: int, seed: int = 1) -> list[dict[str, torch.Tensor]]:
    """Observations from a real match against the scripted bot, for a parity check that means something."""
    from .env import LANES, BunVectorEnv, EnvConfig, noop_actions, slot
    from .util import to_torch

    env = BunVectorEnv([[EnvConfig(seed=seed, layout=LANES, slots=[slot("policy"), slot("scripted", 10)])]])
    generator = torch.Generator().manual_seed(seed)
    out: list[dict[str, torch.Tensor]] = []
    try:
        batch = env.reset()
        while len(out) < n:
            for _ in range(25):
                batch = env.step(noop_actions(len(batch)))
            obs = to_torch(batch.arrays, torch.device("cpu"))
            sample = {name: obs[name] for name in ACT_INPUTS if name in obs}
            sample["noise"] = gumbel_noise(1, generator)
            sample["temperature"] = torch.ones(1)
            out.append(sample)
    finally:
        env.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "public" / "models")
    parser.add_argument("--int8", action="store_true", help="dynamic int8 weights (parity is then approximate)")
    parser.add_argument("--parity-samples", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true", help="parity on random inputs rather than a live match")
    parser.add_argument("--evaluation", type=Path, help="a JSON file from rtsml-eval to embed in policy.json")
    args = parser.parse_args(argv)

    ckpt = load_checkpoint(args.ckpt, pick_device("cpu"))
    policy = Policy(**ckpt["hparams"].get("model", {}))
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    fp32 = export_onnx(policy)
    generator = torch.Generator().manual_seed(0)
    samples = [example_inputs(1, generator) for _ in range(args.parity_samples)] if args.synthetic else live_samples(args.parity_samples)
    report = parity(policy, fp32, samples)
    print(f"fp32 parity: {report['agree']}/{report['samples']}")
    if report["agree"] != report["samples"]:
        print(json.dumps(report["firstDifference"], indent=1))
        return 1

    final = fp32
    extra: dict[str, Any] = {"parity": report, "quantized": False, "checkpoint": str(args.ckpt), "training": ckpt.get("kind")}
    if args.int8:
        final = quantize_int8(fp32)
        q = parity(policy, final, samples)
        print(f"int8 parity: {q['agree']}/{q['samples']} ({len(final) / 1e6:.2f} MB)")
        extra.update({"quantized": True, "int8Parity": q})
    if args.evaluation:
        extra["evaluation"] = json.loads(args.evaluation.read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "policy.onnx").write_bytes(final)
    (args.out / "policy.json").write_text(json.dumps(manifest(final, policy, extra), indent=2) + "\n")
    print(f"wrote {args.out / 'policy.onnx'} ({len(final) / 1e6:.2f} MB) and policy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
