import copy
import json

import numpy as np
import pytest

from engine.early_stopping import EarlyStopping


def test_state_round_trip_resumes_exact_decision_sequence():
    original = EarlyStopping(
        mode="min", patience=2, min_delta=0.1, threshold_mode="abs"
    )
    for metric in (5.0, 4.5, 4.45):
        assert original.step(metric) is False

    state = original.state_dict()
    resumed = EarlyStopping(mode="max", patience=0)
    assert resumed.load_state_dict(state) is resumed

    assert resumed.mode == original.mode
    assert resumed.patience == original.patience
    assert resumed.min_delta == original.min_delta
    assert resumed.threshold_mode == original.threshold_mode
    assert resumed.state_dict() == state

    for metric in (4.44, 4.43, 4.42):
        assert resumed.step(metric) == original.step(metric)
        assert resumed.state_dict() == original.state_dict()


def test_stopped_state_round_trip_remains_sticky():
    original = EarlyStopping(patience=0)
    original.step(1.0)
    assert original.step(1.0) is True

    resumed = EarlyStopping()
    resumed.load_state_dict(original.state_dict())
    before = resumed.state_dict()

    assert resumed.step(0.0) is True
    assert resumed.state_dict() == before


def test_json_round_trip_state_loads():
    stopper = EarlyStopping(mode="max", patience=3, min_delta=0.05, threshold_mode="rel")
    stopper.step(-1.3e308)
    stopper.step(-1.0e308)

    payload = json.loads(json.dumps(stopper.state_dict(), allow_nan=False))
    restored = EarlyStopping()
    restored.load_state_dict(payload)
    assert restored.state_dict() == stopper.state_dict()


def test_unknown_extra_state_metadata_is_tolerated():
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    state = stopper.state_dict()
    state["future_metadata"] = {"schema": 7}

    restored = EarlyStopping()
    restored.load_state_dict(state)
    assert restored.state_dict() == stopper.state_dict()


def test_numpy_scalar_state_values_are_normalized():
    state = EarlyStopping(mode="max", patience=2, min_delta=0.25).state_dict()
    state.update(
        {
            "format_version": np.int64(1),
            "patience": np.int64(2),
            "min_delta": np.float32(0.25),
            "best": np.float32(3.5),
            "num_bad_epochs": np.int64(1),
            "step_count": np.int64(2),
            "stopped": np.bool_(False),
            "stopped_step": None,
        }
    )

    restored = EarlyStopping()
    restored.load_state_dict(state)
    assert restored.mode == "max"
    assert restored.best == 3.5
    assert restored.num_bad_epochs == 1
    assert restored.step_count == 2


def _assert_rejected_without_mutation(stopper, state, expected_exception, match):
    before = stopper.state_dict()
    with pytest.raises(expected_exception, match=match):
        stopper.load_state_dict(state)
    assert stopper.state_dict() == before


def test_missing_required_key_is_rejected_transactionally():
    stopper = EarlyStopping(mode="max", patience=4)
    stopper.step(2.0)
    state = stopper.state_dict()
    del state["best"]
    _assert_rejected_without_mutation(stopper, state, ValueError, "missing required keys")


@pytest.mark.parametrize(
    ("key", "value", "exc", "match"),
    [
        ("format_version", 2, ValueError, "unsupported"),
        ("format_version", True, TypeError, "format_version"),
        ("type", "Other", ValueError, "state type"),
        ("mode", "sideways", ValueError, "mode"),
        ("patience", -1, ValueError, "patience"),
        ("patience", True, TypeError, "patience"),
        ("min_delta", -0.1, ValueError, "min_delta"),
        ("min_delta", float("inf"), ValueError, "min_delta"),
        ("threshold_mode", "fraction", ValueError, "threshold_mode"),
        ("step_count", -1, ValueError, "step_count"),
        ("num_bad_epochs", -1, ValueError, "num_bad_epochs"),
        ("stopped", 1, TypeError, "stopped"),
        ("best", float("nan"), ValueError, "best"),
        ("stopped_step", -1, ValueError, "stopped_step"),
    ],
)
def test_state_field_validation_is_transactional(key, value, exc, match):
    stopper = EarlyStopping(mode="min", patience=2)
    stopper.step(5.0)
    state = stopper.state_dict()
    state[key] = value
    _assert_rejected_without_mutation(stopper, state, exc, match)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"step_count": 0, "best": 1.0}, "best=None"),
        ({"step_count": 0, "best": None, "num_bad_epochs": 1}, "num_bad_epochs=0"),
        (
            {"step_count": 0, "best": None, "num_bad_epochs": 0, "stopped": True, "stopped_step": 0},
            "cannot be stopped",
        ),
        ({"step_count": 2, "best": None}, "must have a best metric"),
        ({"step_count": 2, "best": 1.0, "num_bad_epochs": 3}, "cannot exceed"),
        ({"step_count": 2, "best": 1.0, "num_bad_epochs": 0, "stopped_step": 2}, "stopped_step=None"),
    ],
)
def test_active_state_invariants_are_enforced(updates, match):
    stopper = EarlyStopping(patience=2)
    base = stopper.state_dict()
    base.update(updates)
    with pytest.raises(ValueError, match=match):
        EarlyStopping().load_state_dict(base)


def test_stopped_state_requires_exact_terminal_invariants():
    stopper = EarlyStopping(patience=1)
    stopper.step(1.0)
    stopper.step(2.0)
    assert stopper.step(3.0) is True
    state = stopper.state_dict()

    bad_count = copy.deepcopy(state)
    bad_count["num_bad_epochs"] = 1
    with pytest.raises(ValueError, match=r"patience\+1"):
        EarlyStopping().load_state_dict(bad_count)

    bad_step = copy.deepcopy(state)
    bad_step["stopped_step"] = 2
    with pytest.raises(ValueError, match="current step_count"):
        EarlyStopping().load_state_dict(bad_step)

    impossible = copy.deepcopy(state)
    impossible["step_count"] = 2
    impossible["stopped_step"] = 2
    with pytest.raises(ValueError, match="impossible step_count"):
        EarlyStopping().load_state_dict(impossible)


def test_load_state_does_not_consume_numpy_rng():
    state = EarlyStopping(patience=2).state_dict()

    np.random.seed(99)
    expected = np.random.random(3)
    np.random.seed(99)
    EarlyStopping().load_state_dict(state)
    actual = np.random.random(3)

    np.testing.assert_array_equal(actual, expected)
