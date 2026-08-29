import numpy as np
import pytest

from engine.early_stopping import EarlyStopping


@pytest.mark.parametrize("mode", [None, 1, "minimum", "MAX"])
def test_mode_validation(mode):
    expected = TypeError if not isinstance(mode, str) else ValueError
    with pytest.raises(expected, match="mode"):
        EarlyStopping(mode=mode)


@pytest.mark.parametrize("patience", [True, np.bool_(False), 1.5, "2", None])
def test_patience_type_validation(patience):
    with pytest.raises(TypeError, match="patience"):
        EarlyStopping(patience=patience)


def test_patience_range_and_numpy_integer_support():
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=-1)
    stopper = EarlyStopping(patience=np.int64(3))
    assert stopper.patience == 3
    assert isinstance(stopper.patience, int)


@pytest.mark.parametrize("min_delta", [True, np.bool_(False), "0.1", None, object()])
def test_min_delta_type_validation(min_delta):
    with pytest.raises(TypeError, match="min_delta"):
        EarlyStopping(min_delta=min_delta)


@pytest.mark.parametrize("min_delta", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_min_delta_range_validation(min_delta):
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(min_delta=min_delta)


def test_conversion_overflowing_min_delta_is_rejected():
    with pytest.raises(ValueError, match="fit in float64"):
        EarlyStopping(min_delta=10**10000)


def test_numpy_real_min_delta_is_normalized():
    stopper = EarlyStopping(min_delta=np.float32(0.125))
    assert stopper.min_delta == 0.125
    assert isinstance(stopper.min_delta, float)


@pytest.mark.parametrize("threshold_mode", [None, 1, "relative", "ABS"])
def test_threshold_mode_validation(threshold_mode):
    expected = TypeError if not isinstance(threshold_mode, str) else ValueError
    with pytest.raises(expected, match="threshold_mode"):
        EarlyStopping(threshold_mode=threshold_mode)


@pytest.mark.parametrize("metric", [True, np.bool_(False), "1.0", None, object()])
def test_metric_type_validation(metric):
    stopper = EarlyStopping()
    with pytest.raises(TypeError, match="metric"):
        stopper.step(metric)
    assert stopper.step_count == 0


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metric_rejected_without_state_change(metric):
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    before = stopper.state_dict()
    with pytest.raises(ValueError, match="metric"):
        stopper.step(metric)
    assert stopper.state_dict() == before


def test_conversion_overflowing_metric_is_rejected_without_state_change():
    stopper = EarlyStopping()
    before = stopper.state_dict()
    with pytest.raises(ValueError, match="fit in float64"):
        stopper.step(10**10000)
    assert stopper.state_dict() == before


def test_extended_precision_metric_outside_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")
    metric = np.longdouble(str(np.finfo(np.float64).max)) * np.longdouble(2)
    stopper = EarlyStopping()
    with pytest.raises(ValueError, match="metric"):
        stopper.step(metric)
    assert stopper.step_count == 0


def test_sticky_stopper_still_validates_new_metrics_before_returning_true():
    stopper = EarlyStopping(patience=0)
    stopper.step(1.0)
    assert stopper.step(2.0) is True
    before = stopper.state_dict()

    with pytest.raises(ValueError, match="metric"):
        stopper.step(float("nan"))
    assert stopper.state_dict() == before


def test_subnormal_metrics_and_delta_are_warning_neutral():
    tiny = np.nextafter(0.0, 1.0)
    stopper = EarlyStopping(mode="max", min_delta=0.0)
    with np.errstate(all="raise"):
        assert stopper.step(0.0) is False
        assert stopper.step(tiny) is False
    assert stopper.best == tiny


def test_exact_relative_threshold_boundary_is_not_improvement():
    stopper = EarlyStopping(
        mode="max", patience=1, min_delta=0.5, threshold_mode="rel"
    )
    assert stopper.step(8.0) is False
    assert stopper.step(12.0) is False
    assert stopper.best == 8.0
    assert stopper.num_bad_epochs == 1


def test_just_beyond_relative_threshold_is_improvement():
    stopper = EarlyStopping(
        mode="max", patience=1, min_delta=0.5, threshold_mode="rel"
    )
    assert stopper.step(8.0) is False
    metric = np.nextafter(12.0, float("inf"))
    assert stopper.step(metric) is False
    assert stopper.best == metric
    assert stopper.num_bad_epochs == 0


def test_empty_mapping_state_and_non_mapping_state_are_rejected_without_mutation():
    stopper = EarlyStopping(mode="max", patience=2)
    stopper.step(3.0)
    before = stopper.state_dict()

    with pytest.raises(ValueError, match="missing required keys"):
        stopper.load_state_dict({})
    assert stopper.state_dict() == before

    with pytest.raises(TypeError, match="mapping"):
        stopper.load_state_dict([])
    assert stopper.state_dict() == before
