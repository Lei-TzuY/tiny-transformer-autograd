import numpy as np
import pytest

from engine.metric_accumulator import WeightedMetricAccumulator


def test_value_validation_rejects_non_real_boolean_and_nonfinite_inputs():
    meter = WeightedMetricAccumulator()

    for value in (True, np.bool_(False), "1", object(), 1 + 2j):
        with pytest.raises(TypeError, match="value must be a real number"):
            meter.update(value)

    for value in (np.nan, np.inf, -np.inf, 10**400, -(10**400)):
        with pytest.raises(ValueError, match="value must be finite"):
            meter.update(value)

    assert meter.state_dict()["observation_count"] == 0


def test_weight_validation_rejects_non_real_boolean_nonfinite_and_nonpositive():
    meter = WeightedMetricAccumulator()

    for weight in (True, np.bool_(False), "1", object(), 1 + 2j):
        with pytest.raises(TypeError, match="weight must be a real number"):
            meter.update(1.0, weight=weight)

    for weight in (np.nan, np.inf, -np.inf, 10**400, -(10**400)):
        with pytest.raises(ValueError, match="weight must be finite"):
            meter.update(1.0, weight=weight)

    for weight in (0, 0.0, -1, -1.0):
        with pytest.raises(ValueError, match="weight must be positive"):
            meter.update(1.0, weight=weight)

    assert meter.state_dict()["observation_count"] == 0


def test_value_is_validated_before_weight_and_failure_is_state_neutral():
    meter = WeightedMetricAccumulator()
    meter.update(2.0)
    before = meter.state_dict()

    with pytest.raises(TypeError, match="value must be a real number"):
        meter.update("bad", weight="also bad")

    assert meter.state_dict() == before


def test_weight_failure_after_valid_value_is_state_neutral():
    meter = WeightedMetricAccumulator()
    meter.update(2.0)
    before = meter.state_dict()

    with pytest.raises(ValueError, match="weight must be positive"):
        meter.update(3.0, weight=0.0)

    assert meter.state_dict() == before


def test_total_weight_overflow_is_rejected_transactionally():
    meter = WeightedMetricAccumulator()
    meter.update(1.0, weight=1.0e308)
    before = meter.state_dict()

    with pytest.raises(ValueError, match="total metric weight must remain finite"):
        meter.update(2.0, weight=1.0e308)

    assert meter.state_dict() == before


def test_merge_total_weight_overflow_is_rejected_transactionally():
    left = WeightedMetricAccumulator()
    right = WeightedMetricAccumulator()
    left.update(1.0, weight=1.0e308)
    right.update(2.0, weight=1.0e308)
    before = left.state_dict()

    with pytest.raises(ValueError, match="total metric weight must remain finite"):
        left.merge(right)

    assert left.state_dict() == before


def test_merge_requires_matching_accumulator_type_before_state_access():
    meter = WeightedMetricAccumulator()

    for other in (None, 1, {}, object()):
        with pytest.raises(TypeError, match="other must be a WeightedMetricAccumulator"):
            meter.merge(other)


def test_finite_extended_precision_outside_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")

    huge = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    assert np.isfinite(huge)

    meter = WeightedMetricAccumulator()
    with pytest.raises(ValueError, match="value must be finite"):
        meter.update(huge)
    with pytest.raises(ValueError, match="weight must be finite"):
        meter.update(1.0, weight=huge)


def test_rejected_updates_do_not_modify_prior_mean_weight_or_count():
    meter = WeightedMetricAccumulator()
    meter.update(5.0, weight=2.0)
    before = meter.state_dict()

    failures = [
        lambda: meter.update(np.nan),
        lambda: meter.update(3.0, weight=-1.0),
        lambda: meter.update(3.0, weight=np.inf),
    ]

    for operation in failures:
        with pytest.raises((TypeError, ValueError)):
            operation()
        assert meter.state_dict() == before


def test_large_but_representable_integer_inputs_are_supported():
    meter = WeightedMetricAccumulator()
    value = 10**300
    weight = 10**200

    result = meter.update(value, weight=weight)

    assert result == float(value)
    assert meter.total_weight == float(weight)


def test_negative_zero_weight_is_rejected():
    meter = WeightedMetricAccumulator()
    with pytest.raises(ValueError, match="weight must be positive"):
        meter.update(1.0, weight=-0.0)


def test_merge_empty_source_still_requires_valid_source_type():
    meter = WeightedMetricAccumulator()
    meter.update(1.0)

    class Lookalike:
        def state_dict(self):
            return WeightedMetricAccumulator().state_dict()

    with pytest.raises(TypeError, match="other must be a WeightedMetricAccumulator"):
        meter.merge(Lookalike())
