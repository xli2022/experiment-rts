"""Coverage is a passive observer of the real loop, with no games or inference."""
import copy
import json

import numpy as np
import pytest
import torch

from rtsml import dagger
from rtsml.coverage import DaggerCoverage, EXACT_RESUME, OTHER_BUILD
from rtsml.env import Batch
from rtsml.model import act_input_shapes
from rtsml.spec import BUILD, NOOP, SPEC, TRAIN

F = {name: i for i, name in enumerate(SPEC.entity_features)}
S = {name: i for i, name in enumerate(SPEC.scalars)}
AIRPORT = F['type:Airport']
DEPOT = F['type:Depot']


def examples():
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in act_input_shapes(batch=4).items()
              if name not in ('noise', 'temperature')}
    arrays['label'] = np.full((4, SPEC.action_ints), -1, dtype=np.int32)
    arrays['mask_type'].fill(1)
    arrays['entity_mask'][:, :2] = 1
    e = arrays['entities']
    e[:, 0, F['type:Worker']] = e[:, 0, F['rel:own']] = 1
    e[:, 1, AIRPORT] = e[:, 1, F['rel:own']] = e[:, 1, F['build:Site']] = 1
    e[:, 1, F['x']] = np.float32(32 / 152)
    e[:, 1, F['y']] = np.float32(17 / 152)
    arrays['scalars'][:, S['layout:Quarters']] = 1
    arrays['label'][:, 0] = (BUILD, NOOP, BUILD, TRAIN)
    # Airport top-left31,16; fresh Depot28,16, sharing its coarse cell.
    arrays['label'][0, [1, 3, 4, 5]] = (AIRPORT, 4 * SPEC.grid + 7, 3, 0)
    arrays['label'][2, [1, 3, 4, 5]] = (DEPOT, 4 * SPEC.grid + 7, 0, 0)
    arrays['label'][3, [1, 5]] = (1, 1)
    return arrays


def test_counts_have_separate_stages_and_detached_snapshots_without_mutation():
    arrays = examples()
    before = {k: v.copy() for k, v in arrays.items()}
    # Read-only inputs make accidental telemetry mutations fail immediately.
    for value in arrays.values():
        value.flags.writeable = False
    coverage = DaggerCoverage()
    classified = coverage.classify(arrays)
    assert classified.resume_status[[0, 2]].tolist() == [EXACT_RESUME, OTHER_BUILD]
    coverage.add('offered', classified, np.array([True, True, True, False]))
    coverage.add('retainedFresh', classified, np.array([True, False, True, False]))
    coverage.add('expertSelections', classified, np.array([True, True, False, False]))
    first = coverage.snapshot()
    assert first['offered']['rows'] == 3
    assert first['offered']['exactResumesByBuilding']['Airport'] == 1
    assert first['offered']['otherBuildsByBuilding']['Depot'] == 1
    assert first['retainedFresh']['actionTypes']['Noop'] == 0
    assert first['expertSelections']['actionTypes']['Noop'] == 1
    coverage.add('mixedTraining', classified)
    assert first['mixedTraining']['rows'] == 0  # previous checkpoint stays detached
    assert coverage.snapshot()['mixedTraining']['rows'] == 4
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, before[name])


@pytest.mark.parametrize('reason', ['masked', 'ally', 'completed', 'staffed'])
def test_represented_orphan_counts_exclude_ineligible_rows(reason):
    arrays = examples()
    if reason == 'masked': arrays['entity_mask'][:, 1] = 0
    elif reason == 'ally': arrays['entities'][:, 1, F['rel:own']] = 0
    elif reason == 'completed': arrays['entities'][:, 1, F['build:Complete']] = 1
    else: arrays['entities'][:, 1, F['hasAssignedBuilder']] = 1
    coverage = DaggerCoverage()
    coverage.add('offered', coverage.classify(arrays))
    counts = coverage.snapshot()['offered']
    assert counts['representedOrphanObservations'] == 0
    assert counts['representedOrphanSitesByBuilding']['Airport'] == 0


def test_orphan_counts_are_observation_and_site_rows_not_unique_site_identity():
    arrays = examples()
    arrays['entity_mask'][0, 2] = 1
    arrays['entities'][0, 2] = arrays['entities'][0, 1]
    arrays['entities'][0, 2, AIRPORT] = 0
    arrays['entities'][0, 2, DEPOT] = 1
    coverage = DaggerCoverage()
    coverage.add('offered', coverage.classify(arrays))
    counts = coverage.snapshot()['offered']
    assert counts['representedOrphanObservations'] == 4
    assert counts['representedOrphanSitesByBuilding']['Airport'] == 4
    assert counts['representedOrphanSitesByBuilding']['Depot'] == 1


@pytest.mark.parametrize('beta', [0.0, 0.25, 1.0])
@pytest.mark.parametrize('build_weight', [1.0, 4.0])
def test_fake_loop_coverage_on_off_preserves_data_actions_rng_and_optimizer_inputs(tmp_path, monkeypatch, beta, build_weight):
    """Execute dagger.main against fixed fake frames, not Bun or a neural model."""
    trace = {}

    class FakePolicy(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class FakeEnv:
        def __init__(self, groups):
            self.index = 0
            self.current = None

        def frame(self):
            arrays = examples()
            done = np.zeros(4, dtype=bool)
            if self.index == 0:
                arrays['label'].fill(-1)
            elif self.index % 3 == 0:
                # Even a syntactically valid label on a terminal/reset row is
                # excluded by the loop. Another row remains an ordinary Noop.
                done[0] = True
            if self.index % 2 == 0:
                arrays['entities'][:, 1, F['hasAssignedBuilder']] = 1
            self.current = Batch(arrays, np.zeros(4), done, done.copy(), done.copy(),
                                 np.full(4, -1), np.full(4, self.index * 4), np.zeros(4),
                                 [(0, 0, 0), (0, 0, 1), (0, 1, 2), (0, 1, 3)])
            return self.current

        def reset(self):
            return self.frame()

        def step(self, actions):
            trace['actions'].append(actions.copy())
            eligible = (self.current.arrays['label'][:, 0] >= 0) & ~self.current.done
            trace['executedEligible'].append(self.current.arrays['label'][eligible].copy())
            self.index += 1
            assert self.index < 30, 'fake loop unexpectedly failed to collect bounded labels'
            return self.frame()

        def close(self):
            pass

    original_select = dagger.select_labels
    def select(arrays, rng, noop_keep):
        trace['rng'].append(copy.deepcopy(rng.bit_generator.state))
        selected = original_select(arrays, rng, noop_keep)
        trace['retention'].append(selected.copy())
        return selected

    def decide(policy, arrays, rows, device, temperature, generator):
        trace['noise'].append(torch.rand((len(rows), SPEC.noise_len), generator=generator).numpy())
        actions = np.full((len(rows), SPEC.action_ints), -1, dtype=np.int32)
        actions[:, 0] = NOOP  # Expert Noop selections need not change this action.
        return actions

    def train(policy, opt, data, batch_size, epochs, device, rng, entropy, *, weights):
        trace['rng'].append(copy.deepcopy(rng.bit_generator.state))
        order = rng.permutation(len(data['label']))
        trace['train'].append(({k: v.copy() for k, v in data.items()}, order.copy(),
                               None if weights is None else weights.copy(), policy.weight.detach().clone()))
        # A scalar optimizer smoke verifies parity without policy inference or
        # a game/training rollout. Match the same deterministic target/order.
        target = torch.tensor(float(data['label'][order, 0].mean()))
        opt.zero_grad()
        loss = (policy.weight - target).square()
        loss.backward(); opt.step()
        return float(loss.detach())

    def save(path, policy, kind, hparams, metrics):
        trace['saves'].append((path.name, copy.deepcopy(hparams), copy.deepcopy(metrics),
                               policy.weight.detach().clone()))

    monkeypatch.setattr(dagger, 'Policy', FakePolicy)
    monkeypatch.setattr(dagger, 'BunVectorEnv', FakeEnv)
    monkeypatch.setattr(dagger, 'collect_labels', lambda *a, **k: examples())
    monkeypatch.setattr(dagger, 'load_checkpoint', lambda *a, **k: {'hparams': {'layout': 'quarters'}, 'model': {'weight': torch.tensor(1.0)}})
    monkeypatch.setattr(dagger, 'save_checkpoint', save)
    monkeypatch.setattr(dagger, 'validate', lambda *a, **k: {'nll': 1.0, 'entropy': 0.0})
    monkeypatch.setattr(dagger, 'select_labels', select)
    monkeypatch.setattr(dagger, 'decide', decide)
    monkeypatch.setattr(dagger, 'train_on', train)
    monkeypatch.setattr(dagger.time, 'time', lambda: 100.0)
    results = []
    for enabled in (False, True):
        trace = {key: [] for key in ('actions', 'executedEligible', 'rng', 'retention', 'noise', 'train', 'saves')}
        out = tmp_path / ('on' if enabled else 'off')
        assert dagger.main(['--init', str(tmp_path/'initial.pt'), '--out', str(out), '--layout', 'quarters',
                            '--steps', '13', '--procs', '1', '--envs', '1', '--buffer', '4', '--replay', '8',
                            '--batch', '4', '--epochs', '1', '--noop-keep', '.5', '--expert-start', str(beta),
                            '--expert-end', str(beta), '--val-labels', '4', '--val-every', '1', '--keep-every', '4',
                            '--max-ticks', '400', '--seed', '23', '--device', 'cpu', '--build-weight', str(build_weight),
                            *(['--log-coverage'] if enabled else [])]) == 0
        logs = [json.loads(line) for line in (out/'log.jsonl').read_text().splitlines()]
        results.append((copy.deepcopy(trace), logs, torch.get_rng_state().clone(), np.random.get_state()))
    off, on = results
    assert off[0]['rng'] == on[0]['rng']
    for key in ('actions', 'executedEligible', 'retention', 'noise'):
        assert len(off[0][key]) == len(on[0][key])
        for a, b in zip(off[0][key], on[0][key]): np.testing.assert_array_equal(a, b)
    for a, b in zip(off[0]['train'], on[0]['train']):
        for key in a[0]: np.testing.assert_array_equal(a[0][key], b[0][key])
        np.testing.assert_array_equal(a[1], b[1])
        if a[2] is None: assert b[2] is None
        else: np.testing.assert_array_equal(a[2], b[2])
        assert torch.equal(a[3], b[3])
    assert torch.equal(off[2], on[2])
    assert off[3][0] == on[3][0] and off[3][2:] == on[3][2:]
    np.testing.assert_array_equal(off[3][1], on[3][1])
    assert all('coverage' not in row for row in off[1])
    assert off[1] == [{k: v for k, v in row.items() if k != 'coverage'} for row in on[1]]
    assert 'log_coverage' not in off[0]['saves'][-1][1]
    assert on[0]['saves'][-1][1]['log_coverage'] is True
    assert torch.equal(off[0]['saves'][-1][3], on[0]['saves'][-1][3])
    final = on[1][-1]['coverage']
    assert final['retainedFresh']['rows'] == 13
    assert final['mixedTraining']['rows'] == sum(len(row[0]['label']) for row in on[0]['train'])
    assert final['offered']['rows'] >= final['retainedFresh']['rows']
    if beta == 0:
        assert final['expertSelections']['rows'] == 0
    if beta == 1:
        eligible = np.concatenate(on[0]['executedEligible'])
        assert final['expertSelections']['rows'] == len(eligible)
        assert final['expertSelections']['actionTypes']['Noop'] == int((eligible[:, 0] == NOOP).sum()) > 0
        assert final['offered']['rows'] > len(eligible)  # final buffer trained, no final action
    # Earlier snapshots remain detached while subsequent offers/updates accrue.
    assert on[1][0]['coverage']['retainedFresh']['rows'] < final['retainedFresh']['rows']
