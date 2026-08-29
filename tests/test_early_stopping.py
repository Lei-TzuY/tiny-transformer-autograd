import json

import numpy as np
import pytest

from engine.early_stopping import EarlyStopping


def test_min_mode_patience_and_improvement_reset():
    stopper = EarlyStopping(mode="min", patience=2)

    assert stopper.step(5.0) is False
    assert stopper.best == 5.0
    assert stopper.step(5.5) is False
    assert stopper.num_bad_epochs == 1
    assert stopper.step(4.0) is False
    assert stopper.best == 4.0
    assert stopper.num_bad_epochs == 0
    assert stopper.step(4.0) is False
    assert stopper.step(4.1) is False
    assert stopper.step(4.2) is True

    assert stopper.stopped is True
    assert stopper.should_stop is True
    assert stopper.stopped_step == 6
    assert stopper.step_count == 6
    assert stopper.num_bad_epochs == 3


def test_max_mode_and_zero_patience():
    stopper = EarlyStopping(mode="max", patience=0)
    assert stopper.step(1.0) is False
    assert stopper.step(2.0) is False
    assert stopper.best == 2.0
    assert stopper.step(2.0) is True
    assert stopper.stopped_step == 3


def test_absolute_min_delta_is_strict():
    stopper = EarlyStopping(mode="min", patience=3, min_delta=0.5)
    assert stopper.step(10.0) is False

    assert stopper.step(9.5) is False
    assert stopper.best == 10.0
    assert stopper.num_bad_epochs == 1

    assert stopper.step(9.499999999999) is False
    assert stopper.best == pytest.approx(9.499999999999)
    assert stopper.num_bad_epochs == 0


def test_relative_min_delta_uses_absolute_best_for_negative_metrics():
    stopper = EarlyStopping(
        mode="min", patience=2, min_delta=0.05, threshold_mode="rel"
    )
    assert stopper.step(-100.0) is False

    assert stopper.step(-104.0) is False
    assert stopper.best == -100.0
    assert stopper.num_bad_epochs == 1

    assert stopper.step(-106.0) is False
    assert stopper.best == -106.0
    assert stopper.num_bad_epochs == 0


def test_relative_threshold_at_zero_requires_any_strict_improvement():
    stopper = EarlyStopping(mode="max", patience=1, min_delta=0.9, threshold_mode="rel")
    assert stopper.step(0.0) is False
    assert stopper.step(np.nextafter(0.0, 1.0)) is False
    assert stopper.best == np.nextafter(0.0, 1.0)


def test_extreme_min_comparison_avoids_subtraction_overflow():
    stopper = EarlyStopping(mode="min", patience=0, min_delta=1.0e308)
    with np.errstate(all="raise"):
        assert stopper.step(1.3e308) is False
        assert stopper.step(-1.3e308) is False
    assert stopper.best == -1.3e308


def test_extreme_relative_comparison_avoids_threshold_overflow():
    stopper = EarlyStopping(
        mode="max", patience=0, min_delta=1.5, threshold_mode="rel"
    )
    with np.errstate(all="raise"):
        assert stopper.step(-1.3e308) is False
        assert stopper.step(1.3e308) is False
    assert stopper.best == 1.3e308


def test_stopped_state_is_sticky_without_advancing_counters():
    stopper = EarlyStopping(patience=0)
    stopper.step(1.0)
    assert stopper.step(2.0) is True

    before = stopper.state_dict()
    assert stopper.step(-100.0) is True
    assert stopper.state_dict() == before


def test_reset_preserves_configuration_and_clears_runtime_state():
    stopper = EarlyStopping(
        mode="max", patience=4, min_delta=0.25, threshold_mode="rel"
    )
    stopper.step(1.0)
    stopper.step(0.0)

    assert stopper.reset() is stopper
    assert stopper.mode == "max"
    assert stopper.patience == 4
    assert stopper.min_delta == 0.25
    assert stopper.threshold_mode == "rel"
    assert stopper.best is None
    assert stopper.num_bad_epochs == 0
    assert stopper.step_count == 0
    assert stopper.stopped is False
    assert stopper.stopped_step is None


def test_numpy_metric_scalars_are_accepted_and_normalized():
    stopper = EarlyStopping(mode="max")
    assert stopper.step(np.float32(1.5)) is False
    assert isinstance(stopper.best, float)
    assert stopper.best == 1.5


def test_state_dict_is_strict_json_safe():
    stopper = EarlyStopping(mode="min", patience=1, min_delta=1e-4, threshold_mode="rel")
    stopper.step(1.3e308)
    stopper.step(1.2e308)
    encoded = json.dumps(stopper.state_dict(), allow_nan=False)
    restored = json.loads(encoded)
    assert restored["type"] == "EarlyStopping"
    assert restored["best"] == 1.2e308


def test_step_and_state_are_numpy_rng_neutral():
    np.random.seed(1234)
    expected = np.random.random(4)

    np.random.seed(1234)
    stopper = EarlyStopping(patience=1)
    stopper.step(2.0)
    stopper.step(1.0)
    stopper.state_dict()
    stopper.reset()
    actual = np.random.random(4)

    np.testing.assert_array_equal(actual, expected)
