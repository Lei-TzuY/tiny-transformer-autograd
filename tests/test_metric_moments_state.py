import json

import numpy as np
import pytest

from engine.metric_moments import WeightedStreamingMoments


def _source():
    moments = WeightedStreamingMoments()
    moments.update(-4.0, weight=2.0)
    moments.update(3.0, weight=5.0)
    moments.update(9.0, weight=1.0)
    return moments


def test_state_round_trip_and_continuation_match_uninterrupted():
    source = _source()
    restored = WeightedStreamingMoments()
    restored.load_state_dict(source.state_dict())

    expected = source.update(6.0, weight=4.0)
    actual = restored.update(6.0, weight=4.0)

    assert actual["mean"] == expected["mean"]
    assert actual["variance"] == expected["variance"]
    assert actual["std"] == expected["std"]
    assert restored.state_dict() == source.state_dict()


def test_variance_overflow_state_round_trips_and_remains_continuable():
    maximum = np.finfo(np.float64).max
    source = WeightedStreamingMoments()
    source.update(-maximum)
    source.update(maximum)
    assert source.statistics()["variance_overflow"] is True

    restored = WeightedStreamingMoments()
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    assert restored.statistics() == source.statistics()

    expected = source.update(0.0, weight=maximum)
    actual = restored.update(0.0, weight=maximum)
    assert actual == expected


def test_empty_state_round_trip():
    source = WeightedStreamingMoments()
    restored = _source()
    restored.load_state_dict(source.state_dict())

    assert restored.state_dict() == source.state_dict()
    assert restored.statistics()["mean"] is None


def test_state_is_strict_json_round_trip_safe():
    source = _source()
    encoded = json.dumps(source.state_dict(), allow_nan=False)
    restored = WeightedStreamingMoments()
    restored.load_state_dict(json.loads(encoded))

    assert restored.state_dict() == source.state_dict()


def test_load_accepts_numpy_integer_metadata_and_real_scalars():
    source = _source().state_dict()
    state = dict(
        source,
        version=np.int64(source["version"]),
        mean=np.float64(source["mean"]),
        total_weight=np.float32(source["total_weight"]),
        observation_count=np.int64(source["observation_count"]),
        m2_mantissa=np.float64(source["m2_mantissa"]),
        m2_exponent=np.int64(source["m2_exponent"]),
        future_metadata={"ignored": True},
    )
    restored = WeightedStreamingMoments()
    restored.load_state_dict(state)

    emitted = restored.state_dict()
    assert isinstance(emitted["version"], int)
    assert isinstance(emitted["mean"], float)
    assert isinstance(emitted["total_weight"], float)
    assert isinstance(emitted["observation_count"], int)
    assert isinstance(emitted["m2_mantissa"], float)
    assert isinstance(emitted["m2_exponent"], int)


def test_load_requires_mapping_and_supported_metadata():
    moments = WeightedStreamingMoments()

    with pytest.raises(TypeError, match="streaming moments state must be a mapping"):
        moments.load_state_dict([])

    valid = moments.state_dict()
    with pytest.raises(ValueError, match="unsupported streaming moments version"):
        moments.load_state_dict(dict(valid, version=2))
    with pytest.raises(ValueError, match="streaming moments type must be"):
        moments.load_state_dict(dict(valid, type="Other"))


def test_empty_and_nonempty_invariants_are_enforced():
    moments = WeightedStreamingMoments()
    empty = moments.state_dict()

    with pytest.raises(ValueError, match="zero total_weight"):
        moments.load_state_dict(dict(empty, total_weight=1.0))
    with pytest.raises(ValueError, match="mean=None"):
        moments.load_state_dict(dict(empty, mean=0.0))
    with pytest.raises(ValueError, match="zero M2"):
        moments.load_state_dict(dict(empty, m2_mantissa=0.5, m2_exponent=0))

    populated = _source().state_dict()
    with pytest.raises(ValueError, match="positive total_weight"):
        moments.load_state_dict(dict(populated, total_weight=0.0))
    with pytest.raises(ValueError, match="streaming moments mean must be finite"):
        moments.load_state_dict(dict(populated, mean=np.nan))


def test_single_observation_requires_zero_m2():
    one = WeightedStreamingMoments()
    one.update(3.0, weight=2.0)
    state = one.state_dict()

    with pytest.raises(ValueError, match="single-observation.*zero M2"):
        one.load_state_dict(dict(state, m2_mantissa=0.5, m2_exponent=1))


def test_m2_canonical_envelope_is_validated():
    moments = WeightedStreamingMoments()
    state = _source().state_dict()

    for mantissa in (-0.5, 0.25, 1.0, np.nan, np.inf):
        with pytest.raises((TypeError, ValueError)):
            moments.load_state_dict(dict(state, m2_mantissa=mantissa))

    with pytest.raises(ValueError, match="zero streaming moments M2 must use exponent 0"):
        moments.load_state_dict(dict(state, m2_mantissa=0.0, m2_exponent=1))

    for exponent in (-4097, 4097, 10**400):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            moments.load_state_dict(dict(state, m2_exponent=exponent))

    with pytest.raises(TypeError, match="m2_exponent must be an integer"):
        moments.load_state_dict(dict(state, m2_exponent=True))


def test_negative_m2_exponents_are_supported_for_tiny_variance_state():
    smallest = np.nextafter(0.0, 1.0)
    source = WeightedStreamingMoments()
    source.update(0.0)
    source.update(2.0 * smallest)
    state = source.state_dict()

    assert state["m2_exponent"] < 0

    restored = WeightedStreamingMoments()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.statistics() == source.statistics()


def test_rejected_state_load_is_transactional():
    moments = _source()
    before = moments.state_dict()

    bad_states = [
        dict(before, version=999),
        dict(before, mean=np.inf),
        dict(before, total_weight=0.0),
        dict(before, observation_count=-1),
        dict(before, m2_mantissa=0.25),
        dict(before, m2_exponent=5000),
    ]
    for bad in bad_states:
        with pytest.raises((TypeError, ValueError)):
            moments.load_state_dict(bad)
        assert moments.state_dict() == before


def test_missing_required_fields_fail_without_mutation():
    moments = _source()
    before = moments.state_dict()

    for key in (
        "version",
        "type",
        "mean",
        "total_weight",
        "observation_count",
        "m2_mantissa",
        "m2_exponent",
    ):
        malformed = dict(before)
        malformed.pop(key)
        with pytest.raises((TypeError, ValueError)):
            moments.load_state_dict(malformed)
        assert moments.state_dict() == before


def test_finite_longdouble_mean_outside_float64_is_rejected_transactionally():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")

    moments = _source()
    before = moments.state_dict()
    huge = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)

    with pytest.raises(ValueError, match="streaming moments mean must be finite"):
        moments.load_state_dict(dict(before, mean=huge))
    assert moments.state_dict() == before
