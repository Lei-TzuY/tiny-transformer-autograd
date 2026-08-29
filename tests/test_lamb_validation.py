import numpy as np
import pytest

from engine.lamb import LAMB
from engine.tensor import Tensor


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"lr": True}, TypeError),
        ({"lr": 0.0}, ValueError),
        ({"lr": float("inf")}, ValueError),
        ({"betas": (0.9,)}, ValueError),
        ({"betas": (True, 0.9)}, TypeError),
        ({"betas": (1.0, 0.9)}, ValueError),
        ({"eps": 0.0}, ValueError),
        ({"weight_decay": -1.0}, ValueError),
    ],
)
def test_constructor_option_validation(kwargs, error):
    with pytest.raises(error):
        LAMB([Tensor([1.0], requires_grad=True)], **kwargs)


def test_hyperparameter_overflow_is_normalized_to_value_error():
    huge = 10**400
    with pytest.raises(ValueError, match="fit float64"):
        LAMB([Tensor([1.0], requires_grad=True)], lr=huge)


def test_single_tensor_and_one_shot_generator_are_supported():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    optimizer = LAMB((p for p in (first, second)))
    assert optimizer.steps == (0, 0)


def test_non_tensor_and_duplicate_parameters_are_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        LAMB([parameter, object()])
    with pytest.raises(ValueError, match="duplicate Tensor identities"):
        LAMB([parameter, parameter])


def test_gradient_shape_type_and_finiteness_are_validated_before_state_mutation():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = LAMB(parameter)

    parameter.grad = np.array([1.0])
    with pytest.raises(ValueError, match="shape"):
        optimizer.step()
    assert optimizer.steps == (0,)

    parameter.grad = [1.0, 2.0]
    with pytest.raises(TypeError, match="NumPy array"):
        optimizer.step()
    assert optimizer.steps == (0,)

    parameter.grad = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        optimizer.step()
    assert optimizer.steps == (0,)


def test_float32_gradient_is_normalized_without_rebinding_live_gradient():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    gradient = np.array([1.0, 2.0], dtype=np.float32)
    parameter.grad = gradient
    optimizer = LAMB(parameter, lr=1e-3)

    optimizer.step()

    assert parameter.grad is gradient
    assert parameter.grad.dtype == np.float32
    assert optimizer.state_dict()["states"][0]["m"].dtype == np.float64


def test_extended_precision_gradient_outside_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([np.finfo(np.longdouble).max], dtype=np.longdouble)
    optimizer = LAMB(parameter)

    with pytest.raises(ValueError, match="fit float64"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert optimizer.steps == (0,)


def test_nonfinite_parameter_data_is_rejected_before_any_write():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    second._data[...] = np.nan
    optimizer = LAMB([first, second])
    first_before = first.data.copy()

    with pytest.raises(ValueError, match="parameter 1 data"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, first_before)
    assert optimizer.steps == (0, 0)


def test_shape_drift_is_rejected_before_state_or_data_mutation():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0, 1.0])
    optimizer = LAMB([first, second])
    second.data = np.array([3.0, 4.0])
    first_before = first.data.copy()

    with pytest.raises(ValueError, match="shape changed"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, first_before)
    assert optimizer.steps == (0, 0)


def test_frozen_parameter_with_stale_gradient_is_rejected():
    parameter = Tensor([2.0], requires_grad=True)
    optimizer = LAMB(parameter)
    parameter.requires_grad = False
    parameter.grad = np.array([1.0])

    with pytest.raises(ValueError, match="frozen"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert optimizer.steps == (0,)


def test_malformed_requires_grad_is_rejected():
    parameter = Tensor([2.0], requires_grad=True)
    optimizer = LAMB(parameter)
    parameter.requires_grad = "yes"

    with pytest.raises(TypeError, match="requires_grad"):
        optimizer.step()


def test_malformed_version_metadata_is_rejected_before_earlier_write():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = LAMB([first, second], lr=0.1)
    first_before = first.data.copy()
    second._version = np.int64(0)

    with pytest.raises(TypeError, match="version"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, first_before)
    assert optimizer.steps == (0, 0)


def test_read_only_destination_rejected_only_when_write_is_required():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([1.0])
    optimizer = LAMB(parameter, lr=0.1, weight_decay=0.0)
    parameter.data.flags.writeable = False

    with pytest.raises(ValueError, match="writable"):
        optimizer.step()

    assert optimizer.steps == (0,)


def test_read_only_exact_noop_is_allowed():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    optimizer = LAMB(parameter, lr=0.1, weight_decay=0.0)
    parameter.data.flags.writeable = False

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert optimizer.steps == (1,)


def _overlapping_parameters():
    backing = np.array([2.0, 3.0, 4.0])
    first = Tensor([0.0, 0.0], requires_grad=True)
    second = Tensor([0.0, 0.0], requires_grad=True)
    first._data = backing[:2]
    second._data = backing[1:]
    return first, second


def test_partially_overlapping_parameter_writes_are_rejected():
    first, second = _overlapping_parameters()
    first.grad = np.array([1.0, 1.0])
    second.grad = np.array([1.0, 1.0])
    optimizer = LAMB([first, second], lr=0.1)
    before = np.array([2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="must not overlap"):
        optimizer.step()

    np.testing.assert_array_equal(np.concatenate([first.data[:1], second.data]), before)
    assert optimizer.steps == (0, 0)


def test_write_overlapping_a_noop_bound_parameter_is_still_rejected():
    backing = np.array([2.0, 3.0])
    first = Tensor([0.0, 0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    first._data = backing
    second._data = backing[1:]
    first.grad = np.array([1.0, 1.0])
    second.grad = np.array([0.0])
    optimizer = LAMB([first, second], lr=0.1, weight_decay=0.0)

    with pytest.raises(ValueError, match="must not overlap"):
        optimizer.step()

    np.testing.assert_array_equal(backing, [2.0, 3.0])
    assert optimizer.steps == (0, 0)


def test_disjoint_views_of_one_backing_allocation_are_allowed():
    backing = np.array([2.0, 3.0, 4.0, 5.0])
    first = Tensor([0.0, 0.0], requires_grad=True)
    second = Tensor([0.0, 0.0], requires_grad=True)
    first._data = backing[:2]
    second._data = backing[2:]
    first.grad = np.array([1.0, 1.0])
    second.grad = np.array([1.0, 1.0])
    optimizer = LAMB([first, second], lr=0.01, weight_decay=0.0)

    optimizer.step()

    assert optimizer.steps == (1, 1)
    assert np.all(np.isfinite(backing))


def test_invalid_late_gradient_leaves_earlier_parameter_and_state_unchanged():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([np.inf])
    optimizer = LAMB([first, second], lr=0.1)
    first_before = first.data.copy()

    with pytest.raises(ValueError, match="finite"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, first_before)
    assert optimizer.steps == (0, 0)
