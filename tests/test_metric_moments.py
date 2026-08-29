import json

import numpy as np
import pytest

from engine.metric_moments import WeightedStreamingMoments


def test_weighted_population_mean_variance_and_std():
    moments = WeightedStreamingMoments()
    moments.update(1.0, weight=1.0)
    stats = moments.update(3.0, weight=3.0)

    assert stats["mean"] == pytest.approx(2.5)
    assert stats["variance"] == pytest.approx(0.75)
    assert stats["std"] == pytest.approx(np.sqrt(0.75))
    assert stats["total_weight"] == 4.0
    assert stats["observation_count"] == 2
    assert stats["variance_overflow"] is False
    assert stats["variance_underflow"] is False


def test_default_weights_match_population_statistics():
    moments = WeightedStreamingMoments()
    for value in (1.0, 2.0, 3.0, 4.0):
        moments.update(value)

    assert moments.mean == pytest.approx(2.5)
    assert moments.variance == pytest.approx(1.25)
    assert moments.std == pytest.approx(np.sqrt(1.25))


def test_single_observation_has_zero_variance():
    moments = WeightedStreamingMoments()
    stats = moments.update(-7.5, weight=9.0)

    assert stats["mean"] == -7.5
    assert stats["variance"] == 0.0
    assert stats["std"] == 0.0
    assert stats["variance_overflow"] is False
    assert stats["variance_underflow"] is False


def test_extreme_opposite_values_keep_std_when_variance_overflows():
    maximum = np.finfo(np.float64).max
    moments = WeightedStreamingMoments()

    with np.errstate(all="raise"):
        moments.update(-maximum)
        stats = moments.update(maximum)

    assert stats["mean"] == 0.0
    assert stats["variance"] is None
    assert stats["variance_overflow"] is True
    assert stats["variance_underflow"] is False
    assert stats["std"] == maximum
    assert stats["std_overflow"] is False


def test_extreme_same_values_have_exact_zero_variance():
    maximum = np.finfo(np.float64).max
    moments = WeightedStreamingMoments()

    with np.errstate(all="raise"):
        moments.update(maximum)
        stats = moments.update(maximum)

    assert stats["mean"] == maximum
    assert stats["variance"] == 0.0
    assert stats["std"] == 0.0


def test_scaled_m2_preserves_variance_when_mean_rounds_to_dominant_endpoint():
    maximum = np.finfo(np.float64).max
    tiny_weight = np.nextafter(0.0, 1.0)
    moments = WeightedStreamingMoments()

    moments.update(-maximum, weight=tiny_weight)
    stats = moments.update(maximum, weight=maximum)

    assert stats["mean"] == maximum
    assert stats["variance"] is not None
    assert stats["variance"] > 0.0
    assert stats["std"] is not None
    assert stats["std"] > 0.0


def test_variance_underflow_is_reported_without_losing_representable_std():
    smallest = np.nextafter(0.0, 1.0)
    moments = WeightedStreamingMoments()

    moments.update(0.0)
    stats = moments.update(2.0 * smallest)

    assert stats["mean"] == smallest
    assert stats["variance"] == 0.0
    assert stats["variance_overflow"] is False
    assert stats["variance_underflow"] is True
    assert stats["std"] == smallest
    assert stats["std_underflow"] is False


def test_merge_matches_sequential_weighted_statistics():
    left = WeightedStreamingMoments()
    left.update(-2.0, weight=2.0)
    left.update(5.0, weight=1.0)

    right = WeightedStreamingMoments()
    right.update(4.0, weight=3.0)
    right.update(10.0, weight=2.0)

    sequential = WeightedStreamingMoments()
    for value, weight in ((-2.0, 2.0), (5.0, 1.0), (4.0, 3.0), (10.0, 2.0)):
        sequential.update(value, weight=weight)

    source_before = right.state_dict()
    merged = left.merge(right)

    assert merged["mean"] == pytest.approx(sequential.mean)
    assert merged["variance"] == pytest.approx(sequential.variance)
    assert merged["std"] == pytest.approx(sequential.std)
    assert merged["total_weight"] == sequential.total_weight
    assert merged["observation_count"] == sequential.observation_count
    assert right.state_dict() == source_before


def test_self_merge_doubles_weight_and_count_but_preserves_moments():
    moments = WeightedStreamingMoments()
    moments.update(1.0, weight=2.0)
    moments.update(5.0, weight=3.0)
    before = moments.statistics()

    after = moments.merge(moments)

    assert after["mean"] == before["mean"]
    assert after["variance"] == before["variance"]
    assert after["std"] == before["std"]
    assert after["total_weight"] == before["total_weight"] * 2.0
    assert after["observation_count"] == before["observation_count"] * 2


def test_empty_merge_is_noop_and_empty_target_copies_source():
    empty = WeightedStreamingMoments()
    target = WeightedStreamingMoments()
    target.update(3.0)
    before = target.state_dict()

    target.merge(empty)
    assert target.state_dict() == before

    copied = WeightedStreamingMoments()
    copied.merge(target)
    assert copied.state_dict() == target.state_dict()


def test_reset_returns_to_empty_statistics():
    moments = WeightedStreamingMoments()
    moments.update(1.0)
    moments.update(2.0)
    moments.reset()

    assert moments.statistics() == {
        "mean": None,
        "variance": None,
        "variance_overflow": False,
        "variance_underflow": False,
        "std": None,
        "std_overflow": False,
        "std_underflow": False,
        "total_weight": 0.0,
        "observation_count": 0,
    }


def test_statistics_and_state_are_strict_json_safe():
    moments = WeightedStreamingMoments()
    moments.update(-1.25, weight=2.0)
    moments.update(4.5, weight=3.0)

    json.dumps(moments.statistics(), allow_nan=False)
    json.dumps(moments.state_dict(), allow_nan=False)


def test_numpy_scalar_inputs_normalize_to_plain_python_state():
    moments = WeightedStreamingMoments()
    moments.update(np.float32(2.5), weight=np.float32(4.0))
    state = moments.state_dict()

    assert isinstance(state["mean"], float)
    assert isinstance(state["total_weight"], float)
    assert isinstance(state["observation_count"], int)
    assert isinstance(state["m2_mantissa"], float)
    assert isinstance(state["m2_exponent"], int)


def test_global_numpy_rng_is_unchanged():
    np.random.seed(12345)
    before = np.random.get_state()

    moments = WeightedStreamingMoments()
    moments.update(1.0, weight=2.0)
    moments.update(-3.0, weight=4.0)
    moments.statistics()
    clone = WeightedStreamingMoments()
    clone.merge(moments)
    moments.state_dict()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
