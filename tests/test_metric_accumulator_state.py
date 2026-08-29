import json

import numpy as np
import pytest

from engine.metric_accumulator import WeightedMetricAccumulator


def _populated_meter():
    meter = WeightedMetricAccumulator()
    meter.update(10.0, weight=2.0)
    meter.update(4.0, weight=1.0)
    return meter


def test_state_round_trip_and_continuation_match_uninterrupted_accumulator():
    original = _populated_meter()
    restored = WeightedMetricAccumulator()
    restored.load_state_dict(original.state_dict())

    uninterrupted = _populated_meter()
    expected = uninterrupted.update(-2.0, weight=3.0)
    actual = restored.update(-2.0, weight=3.0)

    assert actual == pytest.approx(expected)
    assert restored.state_dict() == uninterrupted.state_dict()


def test_empty_state_round_trip():
    source = WeightedMetricAccumulator()
    restored = _populated_meter()

    restored.load_state_dict(source.state_dict())

    assert restored.mean is None
    assert restored.total_weight == 0.0
    assert restored.observation_count == 0


def test_json_round_trip_preserves_state():
    source = _populated_meter()
    encoded = json.dumps(source.state_dict(), allow_nan=False)
    restored = WeightedMetricAccumulator()

    restored.load_state_dict(json.loads(encoded))

    assert restored.state_dict() == source.state_dict()


def test_load_accepts_numpy_integer_metadata_and_real_scalars():
    meter = WeightedMetricAccumulator()
    meter.load_state_dict(
        {
            "version": np.int64(1),
            "type": "WeightedMetricAccumulator",
            "mean": np.float32(2.5),
            "total_weight": np.float32(4.0),
            "observation_count": np.int64(2),
            "future_metadata": {"ignored": True},
        }
    )

    assert meter.mean == 2.5
    assert meter.total_weight == 4.0
    assert meter.observation_count == 2
    state = meter.state_dict()
    assert isinstance(state["version"], int)
    assert isinstance(state["mean"], float)
    assert isinstance(state["total_weight"], float)
    assert isinstance(state["observation_count"], int)


def test_load_requires_mapping():
    meter = WeightedMetricAccumulator()
    with pytest.raises(TypeError, match="metric accumulator state must be a mapping"):
        meter.load_state_dict([])


def test_load_rejects_version_and_type_mismatches():
    meter = WeightedMetricAccumulator()
    valid = meter.state_dict()

    bad_version = dict(valid, version=2)
    with pytest.raises(ValueError, match="unsupported metric accumulator version: 2"):
        meter.load_state_dict(bad_version)

    bad_type = dict(valid, type="Other")
    with pytest.raises(ValueError, match="metric accumulator type must be"):
        meter.load_state_dict(bad_type)


def test_load_rejects_boolean_or_negative_counts():
    meter = WeightedMetricAccumulator()
    state = _populated_meter().state_dict()

    with pytest.raises(TypeError, match="observation_count must be a non-negative integer"):
        meter.load_state_dict(dict(state, observation_count=True))
    with pytest.raises(ValueError, match="observation_count must be a non-negative integer"):
        meter.load_state_dict(dict(state, observation_count=-1))


def test_empty_state_invariants_are_enforced():
    meter = WeightedMetricAccumulator()
    empty = meter.state_dict()

    with pytest.raises(ValueError, match="zero total_weight"):
        meter.load_state_dict(dict(empty, total_weight=1.0))
    with pytest.raises(ValueError, match="mean=None"):
        meter.load_state_dict(dict(empty, mean=0.0))


def test_nonempty_state_requires_positive_weight_and_finite_mean():
    meter = WeightedMetricAccumulator()
    state = _populated_meter().state_dict()

    with pytest.raises(ValueError, match="positive total_weight"):
        meter.load_state_dict(dict(state, total_weight=0.0))
    with pytest.raises(ValueError, match="total_weight must be non-negative"):
        meter.load_state_dict(dict(state, total_weight=-1.0))
    with pytest.raises(ValueError, match="metric accumulator mean must be finite"):
        meter.load_state_dict(dict(state, mean=np.nan))


def test_load_rejects_nonfinite_or_conversion_overflow_weight():
    meter = WeightedMetricAccumulator()
    state = _populated_meter().state_dict()

    for value in (np.inf, -np.inf, np.nan, 10**400):
        with pytest.raises(ValueError, match="metric accumulator total_weight must be finite"):
            meter.load_state_dict(dict(state, total_weight=value))


def test_load_rejection_is_transactional():
    meter = _populated_meter()
    before = meter.state_dict()

    bad_states = [
        dict(before, version=999),
        dict(before, mean=np.inf),
        dict(before, total_weight=0.0),
        dict(before, observation_count=-1),
    ]

    for bad in bad_states:
        with pytest.raises((TypeError, ValueError)):
            meter.load_state_dict(bad)
        assert meter.state_dict() == before


def test_finite_longdouble_mean_outside_float64_is_rejected_transactionally():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")

    meter = _populated_meter()
    before = meter.state_dict()
    huge = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)

    with pytest.raises(ValueError, match="metric accumulator mean must be finite"):
        meter.load_state_dict(dict(before, mean=huge))

    assert meter.state_dict() == before


def test_missing_required_fields_fail_explicitly_and_preserve_state():
    meter = _populated_meter()
    before = meter.state_dict()

    for key in ("version", "type", "mean", "total_weight", "observation_count"):
        malformed = dict(before)
        malformed.pop(key)
        with pytest.raises((TypeError, ValueError)):
            meter.load_state_dict(malformed)
        assert meter.state_dict() == before
