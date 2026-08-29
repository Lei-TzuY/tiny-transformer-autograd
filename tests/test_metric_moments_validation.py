import numpy as np
import pytest

from engine.metric_moments import WeightedStreamingMoments


def test_update_rejects_invalid_values_before_mutation():
    moments = WeightedStreamingMoments()
    moments.update(2.0, weight=3.0)
    before = moments.state_dict()

    for value in (True, np.bool_(False), "2", None, 1 + 2j):
        with pytest.raises(TypeError, match="value must be a real number"):
            moments.update(value)
        assert moments.state_dict() == before

    for value in (np.nan, np.inf, -np.inf, 10**400):
        with pytest.raises(ValueError, match="value must be finite"):
            moments.update(value)
        assert moments.state_dict() == before


def test_update_rejects_invalid_weights_before_mutation():
    moments = WeightedStreamingMoments()
    moments.update(2.0)
    before = moments.state_dict()

    for weight in (True, np.bool_(False), "1", None, 1 + 0j):
        with pytest.raises(TypeError, match="weight must be a real number"):
            moments.update(3.0, weight=weight)
        assert moments.state_dict() == before

    for weight in (0.0, -1.0):
        with pytest.raises(ValueError, match="weight must be positive"):
            moments.update(3.0, weight=weight)
        assert moments.state_dict() == before

    for weight in (np.nan, np.inf, -np.inf, 10**400):
        with pytest.raises(ValueError, match="weight must be finite"):
            moments.update(3.0, weight=weight)
        assert moments.state_dict() == before


def test_value_validation_precedes_weight_validation():
    moments = WeightedStreamingMoments()
    with pytest.raises(TypeError, match="value must be a real number"):
        moments.update("bad", weight="also bad")


def test_total_weight_overflow_is_transactional():
    moments = WeightedStreamingMoments()
    maximum = np.finfo(np.float64).max
    moments.update(1.0, weight=maximum)
    before = moments.state_dict()

    with pytest.raises(ValueError, match="total metric weight must remain finite"):
        moments.update(2.0, weight=maximum)

    assert moments.state_dict() == before


def test_merge_rejects_wrong_type_without_mutation():
    moments = WeightedStreamingMoments()
    moments.update(1.0)
    before = moments.state_dict()

    with pytest.raises(TypeError, match="other must be a WeightedStreamingMoments"):
        moments.merge(object())

    assert moments.state_dict() == before


def test_merge_weight_overflow_is_transactional_for_both_sides():
    maximum = np.finfo(np.float64).max
    left = WeightedStreamingMoments()
    right = WeightedStreamingMoments()
    left.update(-1.0, weight=maximum)
    right.update(1.0, weight=maximum)
    left_before = left.state_dict()
    right_before = right.state_dict()

    with pytest.raises(ValueError, match="total metric weight must remain finite"):
        left.merge(right)

    assert left.state_dict() == left_before
    assert right.state_dict() == right_before


def test_self_merge_weight_overflow_is_transactional():
    maximum = np.finfo(np.float64).max
    moments = WeightedStreamingMoments()
    moments.update(1.0, weight=maximum)
    before = moments.state_dict()

    with pytest.raises(ValueError, match="total metric weight must remain finite"):
        moments.merge(moments)

    assert moments.state_dict() == before


def test_finite_longdouble_inputs_outside_float64_are_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")

    huge = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    moments = WeightedStreamingMoments()

    with pytest.raises(ValueError, match="value must be finite"):
        moments.update(huge)
    with pytest.raises(ValueError, match="weight must be finite"):
        moments.update(1.0, weight=huge)

    assert moments.observation_count == 0


def test_extreme_finite_updates_are_warning_neutral_under_numpy_strict_mode():
    maximum = np.finfo(np.float64).max
    moments = WeightedStreamingMoments()

    with np.errstate(all="raise"):
        moments.update(-maximum, weight=1.0)
        moments.update(maximum, weight=1.0)
        stats = moments.statistics()

    assert stats["variance_overflow"] is True
    assert stats["std"] == maximum


def test_many_tiny_weights_preserve_internal_second_moment_without_raw_products():
    tiny = np.nextafter(0.0, 1.0)
    maximum = np.finfo(np.float64).max
    moments = WeightedStreamingMoments()

    moments.update(-maximum, weight=tiny)
    moments.update(maximum, weight=maximum)
    moments.update(maximum, weight=tiny)

    stats = moments.statistics()
    assert stats["variance"] is not None
    assert stats["variance"] > 0.0
    assert stats["std"] > 0.0
