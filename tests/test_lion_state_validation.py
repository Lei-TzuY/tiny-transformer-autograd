import copy

import numpy as np
import pytest

from engine.lion import Lion
from engine.tensor import Tensor


def _trained_optimizer():
    first = Tensor([1.0, -2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad[...] = [0.5, -0.25]
    second.grad[...] = [1.0]
    optimizer = Lion(
        [first, second], lr=0.02, betas=(0.8, 0.95), weight_decay=0.1
    )
    optimizer.step()
    return optimizer, (first, second)


def test_state_envelope_requires_mapping_and_all_required_keys():
    optimizer, _ = _trained_optimizer()

    with pytest.raises(TypeError, match="state must be a mapping"):
        optimizer.load_state_dict([])

    state = optimizer.state_dict()
    for key in (
        "format_version",
        "optimizer",
        "lr",
        "betas",
        "weight_decay",
        "step_count",
        "steps",
        "momentum",
    ):
        malformed = copy.deepcopy(state)
        del malformed[key]
        with pytest.raises(KeyError, match=key):
            optimizer.load_state_dict(malformed)


def test_state_metadata_validation_is_state_neutral():
    optimizer, _ = _trained_optimizer()
    baseline = optimizer.state_dict()

    cases = [
        ("format_version", 2, ValueError, "format_version"),
        ("format_version", True, TypeError, "non-negative integer"),
        ("optimizer", "AdamW", ValueError, "optimizer must be 'Lion'"),
        ("lr", 0.0, ValueError, "positive"),
        ("lr", np.longdouble(np.inf), ValueError, "finite"),
        ("betas", (1.0, 0.9), ValueError, "less than 1.0"),
        ("betas", (0.9,), ValueError, "two values"),
        ("weight_decay", -0.1, ValueError, "at least 0.0"),
        ("step_count", -1, ValueError, "non-negative integer"),
    ]

    for key, value, error, match in cases:
        state = copy.deepcopy(baseline)
        state[key] = value
        with pytest.raises(error, match=match):
            optimizer.load_state_dict(state)
        current = optimizer.state_dict()
        assert current["lr"] == baseline["lr"]
        assert current["betas"] == baseline["betas"]
        assert current["weight_decay"] == baseline["weight_decay"]
        assert current["step_count"] == baseline["step_count"]
        assert current["steps"] == baseline["steps"]
        for actual, expected in zip(current["momentum"], baseline["momentum"]):
            np.testing.assert_array_equal(actual, expected)


def test_state_steps_validate_container_count_types_and_global_bound():
    optimizer, _ = _trained_optimizer()
    baseline = optimizer.state_dict()

    malformed = copy.deepcopy(baseline)
    malformed["steps"] = np.array([1, 1])
    with pytest.raises(TypeError, match="steps must be a list or tuple"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["steps"] = [1]
    with pytest.raises(ValueError, match="step count mismatch"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["steps"] = [True, 1]
    with pytest.raises(TypeError, match="non-negative integer"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["steps"] = [2, 1]
    with pytest.raises(ValueError, match="cannot exceed step_count"):
        optimizer.load_state_dict(malformed)


def test_state_momentum_validates_container_count_shape_dtype_and_finiteness():
    optimizer, _ = _trained_optimizer()
    baseline = optimizer.state_dict()

    malformed = copy.deepcopy(baseline)
    malformed["momentum"] = np.array([1.0])
    with pytest.raises(TypeError, match="momentum must be a list or tuple"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["momentum"] = malformed["momentum"][:1]
    with pytest.raises(ValueError, match="momentum count mismatch"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["momentum"][0] = np.zeros((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["momentum"][0] = np.array([True, False])
    with pytest.raises(TypeError, match="real numeric dtype"):
        optimizer.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["momentum"][0] = np.array([np.nan, 0.0])
    with pytest.raises(ValueError, match="finite values"):
        optimizer.load_state_dict(malformed)


def test_extended_precision_saved_momentum_must_fit_float64():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")
    optimizer, _ = _trained_optimizer()
    state = optimizer.state_dict()
    state["momentum"][0] = np.array(
        [
            np.longdouble(np.finfo(np.float64).max) * np.longdouble(2.0),
            np.longdouble(0.0),
        ],
        dtype=np.longdouble,
    )

    with pytest.raises(ValueError, match="not representable as float64"):
        optimizer.load_state_dict(state)


def test_float32_saved_momentum_is_normalized_to_float64():
    optimizer, _ = _trained_optimizer()
    state = optimizer.state_dict()
    state["momentum"] = [value.astype(np.float32) for value in state["momentum"]]

    optimizer.load_state_dict(state)

    current = optimizer.state_dict()
    assert all(value.dtype == np.float64 for value in current["momentum"])


def test_read_only_internal_momentum_rejects_load_before_any_state_changes():
    optimizer, _ = _trained_optimizer()
    baseline = optimizer.state_dict()
    state = copy.deepcopy(baseline)
    state["lr"] = 0.5
    optimizer._momentum[1].flags.writeable = False

    with pytest.raises(ValueError, match="momentum\[1\] must be writeable"):
        optimizer.load_state_dict(state)

    assert optimizer.lr == baseline["lr"]
    np.testing.assert_array_equal(optimizer._momentum[0], baseline["momentum"][0])
    np.testing.assert_array_equal(optimizer._momentum[1], baseline["momentum"][1])
    optimizer._momentum[1].flags.writeable = True


def test_state_dict_rejects_corrupted_internal_momentum_without_mutating_parameters():
    optimizer, parameters = _trained_optimizer()
    parameter_values = [parameter.data.copy() for parameter in parameters]
    versions = [parameter._version for parameter in parameters]
    optimizer._momentum[0][...] = np.inf

    with pytest.raises(ValueError, match="momentum\[0\].*finite"):
        optimizer.state_dict()

    for parameter, expected, version in zip(parameters, parameter_values, versions):
        np.testing.assert_array_equal(parameter.data, expected)
        assert parameter._version == version


def test_state_save_and_load_never_write_model_parameters_or_gradients():
    optimizer, parameters = _trained_optimizer()
    state = optimizer.state_dict()
    parameter_values = [parameter.data.copy() for parameter in parameters]
    gradients = [parameter.grad for parameter in parameters]
    gradient_values = [gradient.copy() for gradient in gradients]
    versions = [parameter._version for parameter in parameters]

    optimizer.load_state_dict(state)

    for parameter, expected, gradient, gradient_expected, version in zip(
        parameters, parameter_values, gradients, gradient_values, versions
    ):
        np.testing.assert_array_equal(parameter.data, expected)
        assert parameter.grad is gradient
        np.testing.assert_array_equal(parameter.grad, gradient_expected)
        assert parameter._version == version


def test_unknown_state_keys_are_tolerated_for_forward_metadata_compatibility():
    optimizer, _ = _trained_optimizer()
    state = optimizer.state_dict()
    state["future_metadata"] = {"note": "ignored"}

    optimizer.load_state_dict(state)

    assert optimizer.step_count == 1
