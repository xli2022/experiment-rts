"""Expert targets remain attached to the observations the learner visited."""

import numpy as np

from rtsml.imitation import STORED, LabelBuffer, select_labels
from rtsml.spec import BUILD, NOOP, SPEC, TRAIN


def test_label_buffer_keeps_valid_observation_target_pairs_and_owns_their_storage():
    arrays = {name: np.arange(12, dtype=np.float32).reshape(4, 3) for name in STORED}
    arrays["label"] = np.full((4, SPEC.action_ints), -1, dtype=np.int32)
    arrays["label"][:, 0] = [-1, NOOP, BUILD, TRAIN]
    keep = select_labels(arrays, np.random.default_rng(5), noop_keep=0)
    np.testing.assert_array_equal(keep, [False, False, True, True])
    expected = {name: value[keep].copy() for name, value in arrays.items()}
    buffer = LabelBuffer()
    buffer.add(arrays, keep)
    for value in arrays.values():
        value.fill(-99)
    actual = buffer.materialise()
    assert len(buffer) == 2
    for name in STORED:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_noop_sampling_never_turns_invalid_expert_labels_into_targets():
    arrays = {"label": np.array([[-1], [NOOP], [TRAIN]], dtype=np.int32)}
    keep = select_labels(arrays, np.random.default_rng(5), noop_keep=1)
    np.testing.assert_array_equal(keep, [False, True, True])
