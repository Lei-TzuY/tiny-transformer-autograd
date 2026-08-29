import copy
import sys

import numpy as np
import pytest

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


def _populated_estimator():
    p1 = Tensor([0.0, 0.0], requires_grad=True)
    p2 = Tensor(0.0, requires_grad=True)
    estimator = GradientNoiseScaleEstimator([p1, p2], batch_size=4)
    for first, second in [([1.0, 2.0], 3.0), ([5.0, 6.0], -1.0)]:
        p1.grad = np.array(first, dtype=np.float64)
        p2.grad = np.array(second, dtype=np.float64)
        estimator.capture()
    return p1, p2, estimator


def test_empty_state_round_trip():
    p = Tensor([0.0], requires_grad=True)
    source = GradientNoiseScaleEstimator(p, batch_size=3)
    target = GradientNoiseScaleEstimator(p, batch_size=3)

    state = source.state_dict()
    assert state["sample_count"] == 0
    assert state["samples"] == []
    assert target.load_state_dict(state) is target
    assert target.sample_count == 0


def test_populated_state_round_trip_and_resumed_continuation():
    p1, p2, source = _populated_estimator()
    target = GradientNoiseScaleEstimator([p1, p2], batch_size=4)
    target.load_state_dict(source.state_dict())

    assert target.sample_count == source.sample_count
    np.testing.assert_allclose(target.mean_gradients()[0], source.mean_gradients()[0])
    np.testing.assert_allclose(target.mean_gradients()[1], source.mean_gradients()[1])
    assert target.report() == source.report()

    p1.grad = np.array([9.0, 10.0])
    p2.grad = np.array(5.0)
    source.capture()
    target.capture()
    assert target.report() == source.report()


def test_state_dict_arrays_are_independent():
    _, _, estimator = _populated_estimator()
    state = estimator.state_dict()
    state["samples"][0][0][0] = 999.0

    assert estimator.sample_gradients()[0][0][0] == 1.0


def test_float32_state_arrays_normalize_to_float64():
    p = Tensor([0.0, 0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)
    state = {
        "version": 1,
        "type": "GradientNoiseScaleEstimator",
        "batch_size": 2,
        "sample_count": 1,
        "samples": [[np.array([1.5, -2.5], dtype=np.float32)]],
    }

    estimator.load_state_dict(state)
    restored = estimator.sample_gradients()[0][0]
    assert restored.dtype == np.float64
    np.testing.assert_array_equal(restored, [1.5, -2.5])


def test_state_rejects_batch_size_mismatch():
    p = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=4)
    state = estimator.state_dict()
    state["batch_size"] = 8

    with pytest.raises(ValueError, match="batch_size must match"):
        estimator.load_state_dict(state)


def test_malformed_state_is_transactional():
    p1, p2, estimator = _populated_estimator()
    before = estimator.sample_gradients()
    good = estimator.state_dict()

    bad_states = []
    state = copy.deepcopy(good)
    state["version"] = 2
    bad_states.append((state, ValueError, "unsupported"))
    state = copy.deepcopy(good)
    state["type"] = "Other"
    bad_states.append((state, ValueError, "type"))
    state = copy.deepcopy(good)
    state["sample_count"] = 1
    bad_states.append((state, ValueError, "sample_count"))
    state = copy.deepcopy(good)
    state["sample_count"] = True
    bad_states.append((state, TypeError, "non-negative integer"))
    state = copy.deepcopy(good)
    state["sample_count"] = sys.maxsize + 1
    bad_states.append((state, ValueError, "sys.maxsize"))
    state = copy.deepcopy(good)
    state["samples"][0] = [state["samples"][0][0]]
    bad_states.append((state, ValueError, "parameter count mismatch"))
    state = copy.deepcopy(good)
    state["samples"][0][0] = np.array([1.0])
    bad_states.append((state, ValueError, "shape"))
    state = copy.deepcopy(good)
    state["samples"][0][0] = np.array([1, 2], dtype=np.int64)
    bad_states.append((state, TypeError, "floating dtype"))
    state = copy.deepcopy(good)
    state["samples"][0][0] = np.array([np.nan, 2.0])
    bad_states.append((state, ValueError, "finite"))

    for state, error, pattern in bad_states:
        with pytest.raises(error, match=pattern):
            estimator.load_state_dict(state)
        after = estimator.sample_gradients()
        assert len(after) == len(before)
        for after_sample, before_sample in zip(after, before):
            for after_array, before_array in zip(after_sample, before_sample):
                np.testing.assert_array_equal(after_array, before_array)


def test_state_requires_mapping_and_iterable_samples():
    p = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)

    with pytest.raises(TypeError, match="mapping"):
        estimator.load_state_dict([])

    state = estimator.state_dict()
    state["samples"] = "not samples"
    with pytest.raises(TypeError, match="samples must be an iterable"):
        estimator.load_state_dict(state)


def test_extra_state_metadata_is_forward_compatible():
    p = Tensor([0.0], requires_grad=True)
    source = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = np.array([2.0])
    source.capture()
    state = source.state_dict()
    state["future_metadata"] = {"anything": True}

    target = GradientNoiseScaleEstimator(p, batch_size=1)
    target.load_state_dict(state)
    np.testing.assert_array_equal(target.sample_gradients()[0][0], [2.0])


def test_load_validates_live_parameter_binding_before_commit():
    p = Tensor([0.0], requires_grad=True)
    source = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = np.array([2.0])
    source.capture()
    state = source.state_dict()

    target = GradientNoiseScaleEstimator(p, batch_size=1)
    p.data = np.array([[0.0]])
    with pytest.raises(ValueError, match="shape changed"):
        target.load_state_dict(state)
    assert target.sample_count == 0


def test_extended_precision_state_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([0.0], requires_grad=True)
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    state = estimator.state_dict()
    state["sample_count"] = 1
    state["samples"] = [[
        np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    ]]

    with pytest.raises(ValueError, match="fit float64"):
        estimator.load_state_dict(state)
    assert estimator.sample_count == 0
