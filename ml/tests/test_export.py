import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from rtsml.export import example_inputs, export_onnx, manifest, parity, run_onnx
from rtsml.model import ACT_INPUTS, ACT_OUTPUTS, Policy
from rtsml.spec import SPEC


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
