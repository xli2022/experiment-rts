"""Optional row weighting changes the supervised objective, not its scale."""

import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from rtsml.imitation import train_on, validate
from rtsml.spec import BUILD, NOOP, SPEC, TRAIN
from rtsml.util import to_torch


class TwoHeadPolicy(torch.nn.Module):
    """Two Bernoulli heads with shared parameters and nonconstant entropy."""

    def __init__(self):
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float64))

    def evaluate(self, obs, labels):
        z = obs["entities"].double() @ self.coefficients
        auxiliary = 0.7 * z + 0.2
        signs = 2 * (labels[:, 0] == BUILD).double() - 1
        auxiliary_signs = 2 * labels[:, 1].double() - 1
        logp = F.logsigmoid(signs * z) + F.logsigmoid(auxiliary_signs * auxiliary)
        entropy = sum(F.softplus(head) - head * torch.sigmoid(head) for head in (z, auxiliary))
        return {"logp": logp, "entropy": entropy}


def examples():
    return {
        "entities": np.array([[0.1, 0.2], [-0.2, 0.1], [0.1, -0.3], [0.3, 0.2], [-0.1, -0.2]], dtype=np.float32),
        "label": np.array([[BUILD, 1], [TRAIN, 0], [NOOP, 1], [BUILD, 0], [TRAIN, 1]], dtype=np.int64),
    }


@pytest.mark.parametrize("weights", [None, np.ones(5), np.full(5, 7.0)])
def test_default_and_constant_weights_preserve_legacy_loss_gradients_and_adam_updates(weights):
    current = TwoHeadPolicy()
    reference = copy.deepcopy(current)
    current_opt = torch.optim.Adam(current.parameters(), lr=0.03)
    reference_opt = torch.optim.Adam(reference.parameters(), lr=0.03)
    data = examples()
    device = torch.device("cpu")
    loss = train_on(current, current_opt, data, 2, 2, device, np.random.default_rng(17), 0.17, weights=weights)
    old_losses = []
    rng = np.random.default_rng(17)
    # The pre-weighting implementation, including its RNG order, uneven tail,
    # exact expression order, entropy coefficient, clipping and Adam updates.
    for _ in range(2):
        order = rng.permutation(5)
        for start in range(0, 5, 2):
            indices = order[start:start + 2]
            obs = to_torch({"entities": data["entities"]}, device, indices)
            labels = torch.from_numpy(data["label"][indices])
            out = reference.evaluate(obs, labels)
            old_loss = -out["logp"].mean() - 0.17 * out["entropy"].mean()
            reference_opt.zero_grad(set_to_none=True)
            old_loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_opt.step()
            old_losses.append(float(old_loss.item()))
    assert loss == float(np.mean(old_losses))
    assert torch.equal(current.coefficients, reference.coefficients)
    assert torch.equal(current.coefficients.grad, reference.coefficients.grad)
    for name, value in reference_opt.state[reference.coefficients].items():
        assert torch.equal(current_opt.state[current.coefficients][name], value)


class FixedOrder:
    def permutation(self, n):
        assert n == 5
        return np.array([3, 4, 2, 0, 1])


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("scale", [1.0, 2.5e307])
def test_build_weight_matches_closed_form_full_loss_and_gradient_with_uneven_rows(batch_size, scale):
    policy = TwoHeadPolicy()
    data = examples()
    weights = np.where(data["label"][:, 0] == BUILD, 4.0, 1.0)
    coefficients = policy.coefficients.detach().numpy().copy()
    expected_losses = []
    order = FixedOrder().permutation(5)
    for start in range(0, 5, batch_size):
        rows = order[start:start + batch_size]
        x = data["entities"][rows].astype(np.float64)
        signs = 2 * (data["label"][rows, 0] == BUILD) - 1
        auxiliary_signs = 2 * data["label"][rows, 1] - 1
        z = x @ coefficients
        a = 0.7 * z + 0.2
        p, q = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-a))
        nll = np.logaddexp(0, -signs * z) + np.logaddexp(0, -auxiliary_signs * a)
        entropy = np.logaddexp(0, z) - z * p + np.logaddexp(0, a) - a * q
        w = weights[rows] / weights[rows].sum()
        expected_losses.append(float(np.sum(w * (nll - 0.17 * entropy))))
        # d[-log sigmoid(s*z) - e*H(sigmoid(z))]/dz, for both
        # used heads. This independently checks weighting the entropy too.
        derivative = (-signs / (1 + np.exp(signs * z))
                      - 0.7 * auxiliary_signs / (1 + np.exp(auxiliary_signs * a))
                      + 0.17 * (z * p * (1 - p) + 0.7 * a * q * (1 - q)))
        expected_gradient = x.T @ (w * derivative)
        assert np.linalg.norm(expected_gradient) < 1.0
        coefficients -= 0.07 * expected_gradient
    opt = torch.optim.SGD(policy.parameters(), lr=0.07)
    got = train_on(policy, opt, data, batch_size, 1, torch.device("cpu"), FixedOrder(), 0.17, weights=weights * scale)
    assert got == pytest.approx(np.mean(expected_losses), abs=1e-12)
    np.testing.assert_allclose(policy.coefficients.detach().numpy(), coefficients, atol=1e-12, rtol=0)
    np.testing.assert_allclose(policy.coefficients.grad.numpy(), expected_gradient, atol=1e-12, rtol=0)


@pytest.mark.parametrize("weights", [np.ones(4), np.ones((5, 1)), [1, 1, 0, 1, 1],
                                    [1, -1, 1, 1, 1], [1, 1, np.nan, 1, 1], [1, np.inf, 1, 1, 1]])
def test_invalid_weights_fail_before_an_optimizer_update(weights):
    policy = TwoHeadPolicy()
    before = policy.coefficients.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters())
    with pytest.raises(ValueError, match="finite positive value per label"):
        train_on(policy, optimizer, examples(), 3, 1, torch.device("cpu"), FixedOrder(), 0.17, weights=weights)
    assert torch.equal(before, policy.coefficients)
    assert not optimizer.state


class ValidationPolicy(torch.nn.Module):
    def evaluate(self, obs, labels):
        n = len(labels)
        predicted = obs["scalars"][:, 0].long()
        type_logits = torch.zeros(n, SPEC.n_types).scatter_(1, predicted[:, None], 1)
        return {
            "logp": torch.full((n,), -0.5), "entropy": torch.full((n,), 0.25),
            "type_logits": type_logits, "selection_logits": torch.zeros(n, SPEC.n_ent),
            "selection_mask": torch.zeros(n, SPEC.n_ent, dtype=torch.bool),
            "target_logits": torch.zeros(n, SPEC.n_ent), "cell_logits": torch.zeros(n, SPEC.cells),
            "entity_type_logits": torch.zeros(n, SPEC.entity_types),
        }


def test_validation_exposes_rare_action_recall_and_false_positives_across_batches():
    upgrade = SPEC.type_index("UpgradeBuilding")
    labels = np.full((5, SPEC.action_ints), -1, dtype=np.int64)
    labels[:, 0] = [BUILD, BUILD, TRAIN, NOOP, upgrade]
    labels[:, 1] = labels[:, 3] = 0
    data = {"label": labels, "scalars": np.array([[BUILD], [TRAIN], [TRAIN], [BUILD], [NOOP]], dtype=np.float32)}
    result = validate(ValidationPolicy(), data, torch.device("cpu"), batch_size=2)
    assert result["nll"] == 0.5 and result["entropy"] == 0.25
    assert result["typeAccuracy"] == 2 / 5 and result["nonNoopTypeAccuracy"] == 2 / 4
    types = result["perActionType"]
    assert types["Build"] == {"labels": 2, "predicted": 2, "correct": 1, "recall": 0.5, "precision": 0.5}
    assert types["Train"] == {"labels": 1, "predicted": 2, "correct": 1, "recall": 1.0, "precision": 0.5}
    assert types["UpgradeBuilding"] == {"labels": 1, "predicted": 0, "correct": 0, "recall": 0.0, "precision": None}
    assert types["AttackMove"] == {"labels": 0, "predicted": 0, "correct": 0, "recall": None, "precision": None}
    assert sum(row["labels"] for row in types.values()) == result["labels"] == 5
    assert sum(row["predicted"] for row in types.values()) == 5
    assert json.loads(json.dumps(result)) == result
