from argparse import Namespace

import onnx
import pytest
import torch
from torch import nn

from rtsml import imitation
from rtsml.export import example_inputs, export_onnx, parity
from rtsml.model import ACT_INPUTS, Policy
from rtsml.spec import SPEC
from rtsml.util import load_checkpoint, save_checkpoint


SHAPE = {"d": 32, "heads": 2, "layers": 1, "torso": 64}


def observations(batch=3):
    inputs = example_inputs(batch, torch.Generator().manual_seed(81), temperature=0.5)
    obs = {key: inputs[key] for key in ACT_INPUTS if key not in ("noise", "temperature")}
    obs["critic"] = torch.ones(batch, SPEC.critic_len)
    return obs, inputs["noise"], inputs["temperature"]


def test_default_retains_legacy_relu_parameters_rng_outputs_and_gradients():
    torch.manual_seed(123)
    default = Policy(**SHAPE)
    default_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    explicit_zero = Policy(**SHAPE, grid_negative_slope=0)
    assert torch.equal(torch.random.get_rng_state(), default_rng)
    # Legacy reference: the same graph and initial weights with an ordinary
    # ReLU at grid_out[1], independent of the new zero-slope branch.
    torch.manual_seed(123)
    legacy = Policy(**SHAPE, grid_negative_slope=0.01)
    legacy.grid_out[1] = nn.ReLU()
    assert torch.equal(torch.random.get_rng_state(), default_rng)
    assert type(default.grid_out[1]) is nn.ReLU
    assert type(explicit_zero.grid_out[1]) is nn.ReLU
    assert default.state_dict().keys() == legacy.state_dict().keys()
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_zero.state_dict()[name])
        assert torch.equal(value, legacy.state_dict()[name])
    obs, noise, temperature = observations()
    with torch.no_grad():
        actions = default.act(obs, noise, temperature)
        assert torch.equal(actions, legacy.act(obs, noise, temperature))
        assert torch.equal(actions, explicit_zero.act(obs, noise, temperature))
    result = default.evaluate(obs, actions, temperature=0.5)
    reference = legacy.evaluate(obs, actions, temperature=0.5)
    for key in result:
        assert torch.equal(result[key], reference[key]), key
    for values in (result, reference):
        (values["logp"].sum() + values["value"].sum() + values["entropy"].sum()).backward()
    for (name, parameter), (_, old_parameter) in zip(default.named_parameters(), legacy.named_parameters()):
        assert (parameter.grad is None) == (old_parameter.grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, old_parameter.grad), name


def force_inactive_global_grid(policy):
    """Route one type logit through a negative global grid feature and live torso."""
    with torch.no_grad():
        for layer in policy.grid_conv:
            if isinstance(layer, nn.Conv2d):
                layer.weight.fill_(0.001)
                layer.bias.fill_(0.1)
        policy.grid_out[0].weight.fill_(-0.001)
        policy.grid_out[0].bias.fill_(-1)
        for layer in policy.torso:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        policy.torso[0].weight[0, policy.d] = 1
        policy.torso[0].bias[0] = 2
        policy.torso[2].weight[0, 0] = 1
        policy.type_head.weight.zero_()
        policy.type_head.bias.zero_()
        policy.type_head.weight[0, 0] = 1


@pytest.mark.parametrize("slope", [0.0, 0.001, 0.01, 1.0])
def test_negative_global_branch_receives_gradient_only_when_opted_in(slope):
    policy = Policy(**SHAPE, grid_negative_slope=slope)
    force_inactive_global_grid(policy)
    obs, _, _ = observations(2)
    obs["grid"].fill_(1)
    preactivation = policy.grid_out[0](policy.grid_pool(policy.grid_conv(obs["grid"])).flatten(1))
    assert (preactivation < 0).all()
    encoded = policy.encode(obs)
    policy.type_logits(encoded).sum().backward()
    for parameter in (policy.grid_out[0].weight, policy.grid_out[0].bias, policy.grid_conv[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert bool(parameter.grad.abs().sum() > 0) == (slope > 0)
    assert sum(isinstance(layer, nn.LeakyReLU) for layer in policy.modules()) == (1 if slope else 0)


@pytest.mark.parametrize("bad", [-0.001, 1.001, float("nan"), float("inf"), -float("inf"), True, None, "0.01"])
def test_invalid_grid_activation_setting_is_rejected_without_consuming_rng(bad):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="grid_negative_slope"):
        Policy(**SHAPE, grid_negative_slope=bad)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("slope", [0.001, 0.01])
def test_checkpoint_roundtrip_preserves_optin_activation(tmp_path, tiny_policy, slope):
    config = {**SHAPE, "grid_negative_slope": slope}
    policy = Policy(**config)
    policy.load_state_dict(tiny_policy.state_dict())
    path = tmp_path / "configured.pt"
    save_checkpoint(path, policy, "bc", {"model": config, "layout": "lanes"})
    loaded = load_checkpoint(path)
    restored = Policy(**loaded["hparams"]["model"])
    restored.load_state_dict(loaded["model"])
    assert restored.grid_out[1].negative_slope == slope
    obs, noise, temperature = observations()
    assert torch.equal(policy.act(obs, noise, temperature), restored.act(obs, noise, temperature))
    assert torch.equal(policy.encode(obs).torso, restored.encode(obs).torso)


def test_imitation_keeps_cli_dimensions_and_initializer_activation():
    args = Namespace(**SHAPE)
    assert imitation.model_hparams(args) == SHAPE
    assert imitation.model_hparams(args, {"d": 256, "heads": 8, "layers": 4, "torso": 512}) == SHAPE
    assert imitation.model_hparams(args, {"d": 256, "grid_negative_slope": 0.01}) == {
        **SHAPE, "grid_negative_slope": 0.01,
    }


def test_imitation_init_instantiates_checkpoint_activation_before_any_environment(tmp_path, tiny_policy, monkeypatch):
    path = tmp_path / "configured.pt"
    save_checkpoint(path, tiny_policy, "bc", {"model": {**SHAPE, "grid_negative_slope": 0.01}, "layout": "lanes"})
    constructed = []
    def capture(**config):
        policy = Policy(**config)
        constructed.append(policy)
        return policy
    class StopBeforeEnvironment(Exception):
        pass
    def no_environment(*args, **kwargs):
        raise StopBeforeEnvironment()
    monkeypatch.setattr(imitation, "Policy", capture)
    monkeypatch.setattr(imitation, "BunVectorEnv", no_environment)
    with pytest.raises(StopBeforeEnvironment):
        imitation.main(["--init", str(path), "--device", "cpu", "--d", "32", "--heads", "2",
                        "--layers", "1", "--torso", "64", "--out", str(tmp_path / "unused")])
    assert len(constructed) == 1 and constructed[0].grid_out[1].negative_slope == 0.01
    for name, value in tiny_policy.state_dict().items():
        assert torch.equal(value, constructed[0].state_dict()[name])
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("slope", [0.001, 0.01])
def test_configured_activation_onnx_action_parity(tiny_policy, slope):
    policy = Policy(**SHAPE, grid_negative_slope=slope).eval()
    policy.load_state_dict(tiny_policy.state_dict())
    with torch.no_grad():
        # Exercise LeakyReLU's negative side for every global grid feature.
        policy.grid_out[0].weight.fill_(-0.001)
        policy.grid_out[0].bias.fill_(-1)
    graph = export_onnx(policy)
    activations = [node for node in onnx.load_model_from_string(graph).graph.node if node.op_type == "LeakyRelu"]
    assert len(activations) == 1
    alpha = next(attribute.f for attribute in activations[0].attribute if attribute.name == "alpha")
    assert alpha == pytest.approx(slope)
    generator = torch.Generator().manual_seed(31)
    samples = [example_inputs(1, generator, temperature=0.5) for _ in range(40)]
    report = parity(policy, graph, samples)
    assert report["agree"] == report["samples"] == 40, report["firstDifference"]
