import json

import numpy as np
import pytest

from engine.metric_accumulator import WeightedMetricAccumulator


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_weighted_metric_accumulator_tracks_weighted_mean():
    meter = WeightedMetricAccumulator()

    assert meter.mean is None
    assert meter.total_weight == 0.0
    assert meter.observation_count == 0

    assert meter.update(10.0, weight=2.0) == 10.0
    assert meter.update(4.0, weight=1.0) == pytest.approx(8.0)

    assert meter.mean == pytest.approx(8.0)
    assert meter.total_weight == 3.0
    assert meter.observation_count == 2


def test_default_weight_is_one():
    meter = WeightedMetricAccumulator()
    meter.update(2.0)
    meter.update(6.0)
    assert meter.mean == pytest.approx(4.0)
    assert meter.total_weight == 2.0


def test_extreme_same_sign_values_do_not_overflow():
    maximum = np.finfo(np.float64).max
    meter = WeightedMetricAccumulator()

    with np.errstate(all="raise"):
        meter.update(maximum, weight=1.0)
        result = meter.update(maximum, weight=1.0)

    assert result == maximum
    assert meter.mean == maximum


def test_extreme_opposite_sign_values_cancel_without_subtraction_overflow():
    meter = WeightedMetricAccumulator()

    with np.errstate(all="raise"):
        meter.update(1.3e308, weight=1.0)
        result = meter.update(-1.3e308, weight=1.0)

    assert result == 0.0
    assert meter.mean == 0.0


def test_extreme_opposite_sign_weighted_mean_remains_finite():
    meter = WeightedMetricAccumulator()

    with np.errstate(all="raise"):
        meter.update(1.2e308, weight=3.0)
        result = meter.update(-1.2e308, weight=1.0)

    assert result == pytest.approx(6.0e307)
    assert np.isfinite(result)


def test_smallest_subnormal_is_accepted_warning_free():
    tiny = np.nextafter(0.0, 1.0)
    meter = WeightedMetricAccumulator()

    with np.errstate(all="raise"):
        meter.update(tiny)
        result = meter.update(0.0)

    assert np.isfinite(result)
    assert 0.0 <= result <= tiny


def test_numpy_scalar_inputs_are_normalized_to_python_values():
    meter = WeightedMetricAccumulator()
    result = meter.update(np.float32(3.5), weight=np.int64(2))

    assert isinstance(result, float)
    assert isinstance(meter.mean, float)
    assert isinstance(meter.total_weight, float)
    assert isinstance(meter.observation_count, int)


def test_reset_restores_empty_state():
    meter = WeightedMetricAccumulator()
    meter.update(3.0, weight=4.0)

    meter.reset()

    assert meter.mean is None
    assert meter.total_weight == 0.0
    assert meter.observation_count == 0


def test_state_dict_is_strict_json_safe_and_independent():
    meter = WeightedMetricAccumulator()
    meter.update(2.5, weight=4.0)

    state = meter.state_dict()
    encoded = json.dumps(state, allow_nan=False, sort_keys=True)

    assert json.loads(encoded) == state
    state["mean"] = 999.0
    state["total_weight"] = 999.0
    assert meter.mean == 2.5
    assert meter.total_weight == 4.0


def test_merge_matches_sequential_updates_and_preserves_source():
    left = WeightedMetricAccumulator()
    right = WeightedMetricAccumulator()
    sequential = WeightedMetricAccumulator()

    for value, weight in ((10.0, 2.0), (4.0, 1.0)):
        left.update(value, weight=weight)
        sequential.update(value, weight=weight)
    for value, weight in ((-2.0, 3.0), (8.0, 4.0)):
        right.update(value, weight=weight)
        sequential.update(value, weight=weight)

    source_state = right.state_dict()
    result = left.merge(right)

    assert result == pytest.approx(sequential.mean)
    assert left.mean == pytest.approx(sequential.mean)
    assert left.total_weight == sequential.total_weight
    assert left.observation_count == sequential.observation_count
    assert right.state_dict() == source_state


def test_merge_with_empty_source_is_noop():
    target = WeightedMetricAccumulator()
    source = WeightedMetricAccumulator()
    target.update(7.0, weight=2.0)

    assert target.merge(source) == 7.0
    assert target.state_dict()["observation_count"] == 1


def test_merge_into_empty_target_copies_snapshot():
    target = WeightedMetricAccumulator()
    source = WeightedMetricAccumulator()
    source.update(5.0, weight=3.0)
    source.update(1.0, weight=1.0)

    target.merge(source)

    assert target.state_dict() == source.state_dict()


def test_merge_extreme_opposite_sign_means_is_overflow_safe():
    left = WeightedMetricAccumulator()
    right = WeightedMetricAccumulator()
    left.update(1.3e308)
    right.update(-1.3e308)

    with np.errstate(all="raise"):
        result = left.merge(right)

    assert result == 0.0


def test_self_merge_doubles_weight_and_count_without_changing_mean():
    meter = WeightedMetricAccumulator()
    meter.update(3.0, weight=2.0)
    meter.update(5.0, weight=1.0)
    before = meter.mean

    result = meter.merge(meter)

    assert result == pytest.approx(before)
    assert meter.mean == pytest.approx(before)
    assert meter.total_weight == 6.0
    assert meter.observation_count == 4


def test_operations_do_not_consume_numpy_global_rng():
    np.random.seed(12345)
    before = np.random.get_state()

    left = WeightedMetricAccumulator()
    right = WeightedMetricAccumulator()
    left.update(1.0, weight=2.0)
    right.update(3.0, weight=4.0)
    left.merge(right)
    left.state_dict()
    left.reset()

    after = np.random.get_state()
    assert _rng_state_equal(before, after)
