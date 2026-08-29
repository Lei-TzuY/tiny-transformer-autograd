import numpy as np
import pytest

from engine.directional_curvature import directional_curvature
from engine.tensor import Tensor


def test_non_callable_rejected_before_parameter_generator_is_consumed():
    parameter = Tensor([1.0], requires_grad=True)
    consumed = []

    def parameters():
        consumed.append(True)
        yield parameter

    with pytest.raises(TypeError, match="loss_fn must be callable"):
        directional_curvature(None, parameters(), [np.array([1.0])])
    assert consumed == []


@pytest.mark.parametrize("step", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_step_rejected_before_parameter_generator_is_consumed(step):
    parameter = Tensor([1.0], requires_grad=True)
    consumed = []

    def parameters():
        consumed.append(True)
        yield parameter

    with pytest.raises(ValueError):
        directional_curvature(lambda: 0.0, parameters(), [np.array([1.0])], step=step)
    assert consumed == []


@pytest.mark.parametrize("step", [True, np.bool_(False), "0.1"])
def test_non_real_or_boolean_step_is_rejected(step):
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError):
        directional_curvature(lambda: 0.0, parameter, np.array([1.0]), step=step)


def test_numpy_real_step_is_accepted():
    parameter = Tensor([1.0], requires_grad=True)
    report = directional_curvature(
        lambda: parameter.data[0] ** 2,
        parameter,
        np.array([1.0]),
        step=np.float32(0.5),
    )
    assert report["curvature"] == pytest.approx(2.0)


def test_empty_parameter_collection_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        directional_curvature(lambda: 0.0, [], [])


def test_non_tensor_and_duplicate_parameter_entries_are_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError, match="parameter 1"):
        directional_curvature(
            lambda: 0.0,
            [parameter, object()],
            [np.array([1.0]), np.array([0.0])],
        )
    with pytest.raises(ValueError, match="duplicate"):
        directional_curvature(
            lambda: 0.0,
            [parameter, parameter],
            [np.array([1.0]), np.array([0.0])],
        )


def test_non_leaf_parameter_is_rejected():
    leaf = Tensor([2.0], requires_grad=True)
    non_leaf = leaf * leaf
    with pytest.raises(ValueError, match="leaf"):
        directional_curvature(lambda: 0.0, non_leaf, np.array([1.0]))


def test_malformed_parameter_version_is_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    parameter._version = True
    with pytest.raises(TypeError, match="mutation version"):
        directional_curvature(lambda: 0.0, parameter, np.array([1.0]))

    parameter._version = -1
    with pytest.raises(ValueError, match="mutation version"):
        directional_curvature(lambda: 0.0, parameter, np.array([1.0]))


def test_direction_count_shape_type_and_finiteness_are_validated():
    left = Tensor([1.0], requires_grad=True)
    right = Tensor([2.0], requires_grad=True)
    with pytest.raises(ValueError, match="multi-parameter direction"):
        directional_curvature(lambda: 0.0, [left, right], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="contain 2 arrays"):
        directional_curvature(lambda: 0.0, [left, right], [np.array([1.0])])
    with pytest.raises(ValueError, match="shape"):
        directional_curvature(lambda: 0.0, left, np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="NumPy array"):
        directional_curvature(lambda: 0.0, left, [1.0])
    with pytest.raises(ValueError, match="finite"):
        directional_curvature(lambda: 0.0, left, np.array([np.nan]))
    with pytest.raises(TypeError, match="real numeric"):
        directional_curvature(lambda: 0.0, left, np.array([1.0 + 0.0j]))


def test_integer_and_read_only_direction_are_accepted_and_copied():
    parameter = Tensor([2.0], requires_grad=True)
    direction = np.array([1], dtype=np.int64)
    direction.flags.writeable = False
    report = directional_curvature(
        lambda: parameter.data[0] ** 2,
        parameter,
        direction,
        step=0.5,
    )
    assert report["curvature"] == pytest.approx(2.0)
    np.testing.assert_array_equal(direction, [1])
    assert direction.flags.writeable is False


def test_extended_precision_direction_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64")
    parameter = Tensor([1.0], requires_grad=True)
    direction = np.array([np.finfo(np.longdouble).max], dtype=np.longdouble)
    with pytest.raises(ValueError, match="fit float64"):
        directional_curvature(lambda: 0.0, parameter, direction)


def test_all_zero_direction_is_rejected():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="nonzero"):
        directional_curvature(lambda: 0.0, parameter, np.zeros(2))


def test_perturbation_too_small_to_change_parameters_is_rejected_before_callback():
    parameter = Tensor([1.0], requires_grad=True)
    calls = []
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="too small"):
        directional_curvature(
            lambda: calls.append(True) or 0.0,
            parameter,
            np.array([1.0]),
            step=tiny,
        )
    assert calls == []


def test_unrepresentable_perturbation_is_rejected_before_callback():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([maximum], requires_grad=True)
    calls = []
    with pytest.raises(ValueError, match="not representable"):
        directional_curvature(
            lambda: calls.append(True) or 0.0,
            parameter,
            np.array([maximum]),
            step=2.0,
        )
    assert calls == []
    np.testing.assert_array_equal(parameter.data, [maximum])


def test_read_only_parameter_needed_for_write_is_rejected_before_callback():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.data.flags.writeable = False
    calls = []
    with pytest.raises(ValueError, match="read-only"):
        directional_curvature(
            lambda: calls.append(True) or 0.0,
            parameter,
            np.array([1.0]),
            step=0.5,
        )
    assert calls == []
    assert parameter.data.flags.writeable is False


def test_write_destination_overlapping_zero_direction_parameter_is_rejected_preflight():
    left = Tensor([1.0], requires_grad=True)
    right = Tensor([2.0], requires_grad=True)
    right._data = left.data
    calls = []
    with pytest.raises(ValueError, match="overlap"):
        directional_curvature(
            lambda: calls.append(True) or 0.0,
            [left, right],
            [np.array([1.0]), np.array([0.0])],
            step=0.25,
        )
    assert calls == []


@pytest.mark.parametrize(
    "bad_result, expected_exception",
    [
        (True, TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (np.array([1.0]), ValueError),
        (Tensor([1.0], requires_grad=False), ValueError),
        (1.0 + 0.0j, TypeError),
    ],
)
def test_invalid_loss_callback_results_restore_parameters(bad_result, expected_exception):
    parameter = Tensor([2.0], requires_grad=True)
    with pytest.raises(expected_exception):
        directional_curvature(
            lambda: bad_result,
            parameter,
            np.array([1.0]),
            step=0.5,
        )
    np.testing.assert_array_equal(parameter.data, [2.0])


def test_loss_callback_exception_restores_parameter_gradient_and_rng():
    parameter = Tensor([2.0], requires_grad=True)
    gradient = np.array([7.0])
    parameter.grad = gradient
    np.random.seed(77)
    before = np.random.get_state()

    def loss():
        parameter.data[...] = [99.0]
        gradient[...] = [-5.0]
        np.random.random()
        raise LookupError("boom")

    with pytest.raises(LookupError, match="boom"):
        directional_curvature(loss, parameter, np.array([1.0]), step=0.5)

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [7.0])
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_callback_storage_replacement_is_detected_and_original_binding_restored():
    parameter = Tensor([2.0], requires_grad=True)
    original_storage = parameter.data
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls == 2:
            parameter.data = np.array([123.0])
        return parameter.data[0] ** 2

    with pytest.raises(RuntimeError, match="replaced parameter 0 storage"):
        directional_curvature(loss, parameter, np.array([1.0]), step=0.5)

    assert parameter.data is original_storage
    np.testing.assert_array_equal(parameter.data, [2.0])


def test_callback_malformed_trainability_is_detected_and_restored():
    parameter = Tensor([2.0], requires_grad=True)

    def loss():
        parameter.requires_grad = "truthy but invalid"
        return parameter.data[0] ** 2

    with pytest.raises(RuntimeError, match="trainability"):
        directional_curvature(loss, parameter, np.array([1.0]), step=0.5)
    assert parameter.requires_grad is True


def test_callback_gradient_rebinding_is_detected_and_original_binding_restored():
    parameter = Tensor([2.0], requires_grad=True)
    original = np.array([4.0])
    parameter.grad = original

    def loss():
        parameter.grad = np.array([9.0])
        return parameter.data[0] ** 2

    with pytest.raises(RuntimeError, match="replaced parameter 0 gradient"):
        directional_curvature(loss, parameter, np.array([1.0]), step=0.5)
    assert parameter.grad is original
    np.testing.assert_array_equal(parameter.grad, [4.0])
