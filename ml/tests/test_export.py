import hashlib
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from rtsml.export import example_inputs, export_onnx, live_samples, main, manifest, parity, run_onnx
from rtsml import export as exporter
from rtsml.model import ACT_INPUTS, ACT_OUTPUTS, Policy
from rtsml.spec import SPEC

from conftest import requires_bun


def test_export_refuses_to_relabel_a_checkpoint_for_the_other_map(tmp_path):
    checkpoint = tmp_path / "lanes.pt"
    torch.save({"spec_version": SPEC.version, "hparams": {"layout": "lanes"}}, checkpoint)
    out = tmp_path / "output"
    assert main(["--ckpt", str(checkpoint), "--layout", "quarters", "--out", str(out)]) == 1
    assert not out.exists()


@pytest.fixture(scope="module")
def exported():
    torch.manual_seed(0)
    policy = Policy(d=32, heads=2, layers=1, torso=64).eval()
    return policy, export_onnx(policy)


def test_graph_has_the_declared_interface(exported):
    _, onnx_bytes = exported
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == list(ACT_INPUTS)
    assert [o.name for o in session.get_outputs()] == list(ACT_OUTPUTS)
    out = session.get_outputs()[0]
    assert out.shape == [1, SPEC.action_ints] and out.type == "tensor(int32)"


def test_onnx_decides_exactly_what_torch_does(exported):
    policy, onnx_bytes = exported
    g = torch.Generator().manual_seed(7)
    samples = [example_inputs(1, g) for _ in range(40)]
    report = parity(policy, onnx_bytes, samples)
    assert report["agree"] == report["samples"], report["firstDifference"]


@requires_bun
@pytest.mark.parametrize("layout", ["lanes", "quarters"])
def test_live_parity_uses_browser_batch_size(exported, layout):
    policy, onnx_bytes = exported
    samples = live_samples(3, layout=layout, temperature=0.5)
    assert len(samples) == 3
    for sample in samples:
        assert all(t.shape[0] == 1 for t in sample.values())
        torch.testing.assert_close(sample["temperature"], torch.tensor([0.5]))
    report = parity(policy, onnx_bytes, samples)
    assert report["agree"] == 3, report["firstDifference"]


def test_onnx_output_is_a_legal_decision(exported):
    _, onnx_bytes = exported
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(11)
    for _ in range(10):
        inputs = example_inputs(1, g)
        action = run_onnx(session, inputs)
        assert action.dtype == np.int32 and action.shape == (1, SPEC.action_ints)
        t = int(action[0, 0])
        assert inputs["mask_type"][0, t] == 1


def test_manifest_names_the_spec_and_the_hash(exported):
    policy, onnx_bytes = exported
    m = manifest(onnx_bytes, policy, {"quantized": False})
    text = json.dumps(m)
    assert m["specVersion"] == SPEC.version
    assert m["bytes"] == len(onnx_bytes)
    assert [i["name"] for i in m["inputs"]] == list(ACT_INPUTS)
    assert "sha256" in text and m["noise"]["length"] == SPEC.noise_len
    assert m["defaultTemperature"] == 1


@pytest.mark.parametrize("evaluation,temperature", [
    ({"temperature": 0.5}, "1"),
    ({"temperature": 0}, None),
    ({"temperature": -0.5}, None),
    ({"temperature": float("nan")}, None),
    ({"temperature": float("inf")}, None),
    ({"temperature": True}, None),
    ({"temperature": "0.5"}, None),
    ({"temperature": None}, None),
    ({"checkpointSha256": "0" * 64}, None),
    ({"layout": "quarters"}, None),
    ({"specVersion": -1}, None),
    ({"state": "running"}, None),
    ({"state": "failed"}, None),
    ({"state": None}, None),
    ({}, "nan"),
    ({}, "inf"),
    ({}, "0"),
    ({}, "-1"),
])
def test_export_rejects_incompatible_evaluation_before_model_or_onnx_work(tmp_path, evaluation, temperature):
    checkpoint = tmp_path / "lanes.pt"
    # Deliberately lacks model weights: rejected metadata must be checked first.
    torch.save({"spec_version": SPEC.version, "hparams": {"layout": "lanes"}}, checkpoint)
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps(evaluation))
    out = tmp_path / "output"
    args = ["--ckpt", str(checkpoint), "--evaluation", str(report), "--out", str(out)]
    if temperature is not None:
        args += ["--temperature", temperature]
    assert main(args) == 1
    assert not out.exists()


@pytest.mark.parametrize("samples", [0, -1])
def test_export_requires_at_least_one_parity_sample_before_reading_model(tmp_path, samples):
    assert main(["--ckpt", str(tmp_path / "unread.pt"), "--parity-samples", str(samples)]) == 1


def test_export_rejects_checkpoint_replaced_while_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "changing.pt"
    torch.save({"spec_version": SPEC.version, "hparams": {"layout": "lanes"}}, checkpoint)

    def changing_checkpoint(path, device):
        loaded = torch.load(path, weights_only=False)
        path.write_bytes(b"new checkpoint appeared during load")
        return loaded

    monkeypatch.setattr(exporter, "load_checkpoint", changing_checkpoint)
    out = tmp_path / "output"
    assert main(["--ckpt", str(checkpoint), "--out", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("evaluated,explicit,expected", [(0.5, None, 0.5), (None, None, 1), (None, 0.75, 0.75), (0.5, 0.5, 0.5)])
def test_export_persists_selected_temperature_and_checks_parity_at_it(tmp_path, monkeypatch, exported, synthetic, evaluated, explicit, expected):
    policy, onnx_bytes = exported
    checkpoint = tmp_path / "lanes.pt"
    torch.save({"spec_version": SPEC.version, "kind": "bc", "model": policy.state_dict(),
                "hparams": {"layout": "lanes", "model": {"d": 32, "heads": 2, "layers": 1, "torso": 64}}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluation = {"layout": "lanes", "checkpointSha256": digest, "specVersion": SPEC.version}
    if evaluated is not None:
        evaluation["temperature"] = evaluated
    elif explicit is None:
        # Older reports have neither provenance nor a sampling temperature.
        evaluation = {"layout": "lanes"}
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps(evaluation))
    out = tmp_path / "output"
    sampled = []

    def checked_parity(model, graph, samples):
        assert graph == onnx_bytes
        for sample in samples:
            assert sample["temperature"].dtype == torch.float32
            torch.testing.assert_close(sample["temperature"], torch.tensor([float(expected)]))
        sampled.append(True)
        return {"samples": len(samples), "agree": len(samples), "firstDifference": None}

    def checked_live_samples(n, layout, temperature):
        assert not synthetic and layout == "lanes" and temperature == expected
        return [example_inputs(1, temperature=temperature) for _ in range(n)]

    monkeypatch.setattr(exporter, "export_onnx", lambda model: onnx_bytes)
    monkeypatch.setattr(exporter, "parity", checked_parity)
    monkeypatch.setattr(exporter, "live_samples", checked_live_samples)
    args = ["--ckpt", str(checkpoint), "--out", str(out), "--evaluation", str(report), "--parity-samples", "2"]
    if explicit is not None:
        args += ["--temperature", str(explicit)]
    if synthetic:
        args += ["--synthetic"]
    assert main(args) == 0 and sampled
    emitted = json.loads((out / "policy-lanes.json").read_text())
    assert emitted["defaultTemperature"] == expected
    assert emitted["checkpointSha256"] == digest
    assert emitted["evaluation"] == evaluation
