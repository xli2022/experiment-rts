"""Exact teacher resume anchors, including mirror and malformed-label cases."""
import numpy as np
import pytest
from rtsml import coverage as c

SPEC = c.read_spec()
F = {name: i for i, name in enumerate(SPEC["entities"]["features"])}
S = {name: i for i, name in enumerate(SPEC["scalars"])}
TYPES = [name[5:] for name in SPEC["entities"]["features"] if name.startswith("type:")]


def example(layout="Quarters", building="Airport", flip=False, state="Site"):
    size = c.MAP_SIZES[layout]
    footprint = c.FOOTPRINTS[building]
    world_top = (31, 16)
    canonical = tuple(size - footprint - t if flip else t for t in world_top)
    # Independently reproduce canonical floor of the world footprint center.
    center = [np.floor(size - (t + footprint / 2) if flip else t + footprint / 2) for t in world_top]
    arrays = {"entities": np.zeros((1, SPEC["entities"]["rows"], len(F)), np.float32),
              "entity_mask": np.zeros((1, SPEC["entities"]["rows"]), np.uint8),
              "scalars": np.zeros((1, len(S)), np.float32),
              "label": np.full((1, SPEC["actions"]["ints"]), -1, np.int32)}
    e = arrays["entities"][0]
    arrays["entity_mask"][0, :2] = 1
    e[0, F["type:Worker"]] = e[0, F["rel:own"]] = 1
    e[1, F["type:" + building]] = e[1, F["rel:own"]] = e[1, F["build:" + state]] = 1
    e[1, [F["x"], F["y"]]] = np.array(center) / size
    arrays["scalars"][0, S["layout:" + layout]] = 1
    arrays["label"][0, [0, 1, 5]] = (SPEC["actions"]["types"].index("Build"), TYPES.index(building), 0)
    anchor(arrays, *canonical)
    return arrays


def anchor(arrays, tx, ty):
    step, grid = SPEC["grid"]["cellTiles"], SPEC["grid"]["size"]
    arrays["label"][0, 3:5] = ((ty // step) * grid + tx // step, (ty % step) * step + tx % step)


@pytest.mark.parametrize("layout", ["Lanes", "Quarters"])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("building", list(c.FOOTPRINTS))
def test_exact_resume_all_buildings_maps_and_canonical_mirrors(layout, flip, building):
    assert c.classify_teacher_labels(example(layout, building, flip)).tolist() == [c.EXACT_RESUME]


def test_exact_subcell_is_required_even_when_same_cell_contains_site():
    a = example(building="Depot")
    # The site is at31,16;28,16 is a distinct nonoverlapping2x2 foundation in
    # the same coarse cell. A coarse-cell-only detector would overcount it.
    old_cell = a["label"][0, 3]
    anchor(a, 28, 16)
    assert a["label"][0, 3] == old_cell
    assert c.classify_teacher_labels(a).tolist() == [c.OTHER_BUILD]


@pytest.mark.parametrize("relation", ["ally", "enemy", "neutral"])
def test_other_owner_is_not_own_resume(relation):
    a = example()
    a["entities"][0, 1, F["rel:own"]] = 0
    a["entities"][0, 1, F["rel:" + relation]] = 1
    assert c.classify_teacher_labels(a).tolist() == [c.OTHER_BUILD]


def test_wrong_type_at_exact_anchor_is_not_resume():
    a = example()
    a["label"][0, 1] = TYPES.index("Factory")  # same footprint, distinct type
    assert c.classify_teacher_labels(a).tolist() == [c.OTHER_BUILD]


def test_complete_zero_progress_is_not_confused_with_unstarted_site():
    site = example(state="Site")
    complete = example(state="Complete")
    assert site["entities"][0, 1, F["buildProgress"]] == complete["entities"][0, 1, F["buildProgress"]] == 0
    assert c.classify_teacher_labels(site)[0] == c.EXACT_RESUME
    assert c.classify_teacher_labels(complete)[0] == c.OTHER_BUILD


@pytest.mark.parametrize("staffed", [0, 1])
@pytest.mark.parametrize("state", ["Site", "UnderConstruction"])
def test_staffing_does_not_change_resume_classification(staffed, state):
    a = example(state=state)
    a["entities"][0, 1, F["hasAssignedBuilder"]] = staffed
    a["entities"][0, 1, F["buildProgress"]] = 0.25 if state == "UnderConstruction" else 0
    assert c.classify_teacher_labels(a)[0] == c.EXACT_RESUME


def test_masked_site_is_not_counted_and_noop_not_in_build_denominator():
    a = example()
    a["entity_mask"][0, 1] = 0
    assert c.classify_teacher_labels(a)[0] == c.OTHER_BUILD
    a["label"][0, 0] = SPEC["actions"]["types"].index("Noop")
    assert c.classify_teacher_labels(a).tolist() == [c.NON_BUILD]


@pytest.mark.parametrize("field,value", [(0, -1), (0, 999), (1, -1), (2, 0), (3, -1), (3, 1600),
                                        (4, -1), (4, 16), (5, -1), (5, 160), (6, 0)])
def test_invalid_teacher_labels_are_skipped(field, value):
    a = example()
    a["label"][0, field] = value
    assert c.classify_teacher_labels(a)[0] == c.INVALID_LABEL


def test_invalid_worker_and_unknown_layout_are_skipped():
    a = example()
    a["entity_mask"][0, 0] = 0
    assert c.classify_teacher_labels(a)[0] == c.INVALID_LABEL
    a = example()
    a["scalars"].fill(0)
    assert c.classify_teacher_labels(a)[0] == c.UNSUPPORTED_OBSERVATION


def test_obs_prefix_supported_but_sampled_action_never_used_as_label():
    a = example()
    prefixed = {("label" if k == "label" else "obs_" + k): v for k, v in a.items()}
    assert c.classify_teacher_labels(prefixed)[0] == c.EXACT_RESUME
    prefixed["action"] = prefixed.pop("label")
    with pytest.raises(ValueError, match="requires teacher"):
        c.classify_teacher_labels(prefixed)


def test_float32_coordinate_roundtrip_for_every_legal_axis_anchor():
    for size in c.MAP_SIZES.values():
        for footprint in set(c.FOOTPRINTS.values()):
            for flip in (False, True):
                top = np.arange(size - footprint + 1)
                center = size - top - footprint / 2 if flip else top + footprint / 2
                normalized = (np.floor(center) / size).astype(np.float32)
                recovered = np.rint(normalized.astype(np.float64) * size).astype(int) - footprint // 2
                expected = size - footprint - top if flip else top
                assert np.array_equal(recovered, expected)


