import copy

import numpy as np
import pytest

from engine.optim import Adam, AdamW, SGD
from engine.optimizer_transaction import optimizer_step_transaction
from engine.tensor import Tensor


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _assert_state_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_state_equal(left_value, right_value)
        return
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert np.array_equal(left, right)
        return
    assert left == right


def _snapshot_state(optimizer):
    return copy.deepcopy(optimizer.state_dict())


def test_successful_sgd_step_commits_without_extra_tensor_mutation():
    parameter = Tensor([2.0, -1.0], requires_grad=True)
    parameter.grad[...] = [0.5, -0.5]
    gradient = parameter.grad
    version = parameter._version
    optimizer = SGD([parameter], lr=0.1)

    np.random.seed(1234)
    rng_before = np.random.get_state()

    with optimizer_step_transaction(optimizer) as yielded:
        assert yielded is optimizer
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.95, -0.95])
    assert parameter._version == version + 1
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [0.5, -0.5])
    assert _rng_state_equal(rng_before, np.random.get_state())


def test_actual_partial_sgd_failure_rolls_back_parameters_and_momentum():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [2.0]
    optimizer = SGD([first, second], lr=0.1, momentum=0.9)

    values_before = (first.data.copy(), second.data.copy())
    state_before = _snapshot_state(optimizer)
    first_version = first._version
    second_version = second._version
    graph = first * 3.0

    second.data.setflags(write=False)
    with pytest.raises(ValueError):
        with optimizer_step_transaction(optimizer):
            optimizer.step()

    np.testing.assert_array_equal(first.data, values_before[0])
    np.testing.assert_array_equal(second.data, values_before[1])
    _assert_state_equal(optimizer.state_dict(), state_before)

    # The first parameter was written and then restored; version history must
    # not be rewound, otherwise a graph built before the failed step looks valid.
    assert first._version == first_version + 2
    assert second._version == second_version
    with pytest.raises(RuntimeError, match="modified after forward"):
        graph.backward()


def test_actual_partial_adam_failure_restores_moments_and_step_counters():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [0.25]
    second.grad[...] = [0.5]
    optimizer = Adam([first, second], lr=0.01)

    values_before = (first.data.copy(), second.data.copy())
    state_before = _snapshot_state(optimizer)
    second.data.setflags(write=False)

    with pytest.raises(ValueError):
        with optimizer_step_transaction(optimizer):
            optimizer.step()

    np.testing.assert_array_equal(first.data, values_before[0])
    np.testing.assert_array_equal(second.data, values_before[1])
    _assert_state_equal(optimizer.state_dict(), state_before)


def test_silent_nonfinite_parameter_is_rejected_and_rolled_back():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1e308]
    optimizer = SGD([parameter], lr=1e308)
    value_before = parameter.data.copy()
    state_before = _snapshot_state(optimizer)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="parameter 0 became non-finite"):
            with optimizer_step_transaction(optimizer):
                optimizer.step()

    np.testing.assert_array_equal(parameter.data, value_before)
    _assert_state_equal(optimizer.state_dict(), state_before)
    assert np.isfinite(parameter.data).all()


def test_silent_nonfinite_adam_state_is_rejected_even_when_parameter_stays_finite():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1e308]
    optimizer = Adam([parameter], lr=1e-3)
    value_before = parameter.data.copy()
    state_before = _snapshot_state(optimizer)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="optimizer transaction state"):
            with optimizer_step_transaction(optimizer):
                optimizer.step()

    np.testing.assert_array_equal(parameter.data, value_before)
    assert np.isfinite(parameter.data).all()
    _assert_state_equal(optimizer.state_dict(), state_before)


def test_body_exception_restores_optimizer_hyperparameters_and_values():
    class Marker(Exception):
        pass

    parameter = Tensor([4.0], requires_grad=True)
    optimizer = AdamW([parameter], lr=0.01, weight_decay=0.1)
    value_before = parameter.data.copy()
    state_before = _snapshot_state(optimizer)

    with pytest.raises(Marker):
        with optimizer_step_transaction(optimizer):
            parameter.data[...] = [9.0]
            optimizer.lr = 0.5
            optimizer.weight_decay = 0.75
            raise Marker("abort")

    np.testing.assert_array_equal(parameter.data, value_before)
    _assert_state_equal(optimizer.state_dict(), state_before)


def test_baseexception_also_rolls_back_before_propagating():
    class Fatal(BaseException):
        pass

    parameter = Tensor([2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    value_before = parameter.data.copy()

    with pytest.raises(Fatal):
        with optimizer_step_transaction(optimizer):
            parameter.data[...] = [7.0]
            raise Fatal("stop")

    np.testing.assert_array_equal(parameter.data, value_before)


def test_shape_drift_on_normal_exit_rolls_back_and_raises():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    value_before = parameter.data.copy()

    with pytest.raises(ValueError, match="parameter shape changed at index 0"):
        with optimizer_step_transaction(optimizer):
            parameter.data = [[5.0, 6.0]]

    assert parameter.shape == (2,)
    np.testing.assert_array_equal(parameter.data, value_before)


def test_requires_grad_drift_on_normal_exit_rolls_back_and_raises():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)

    with pytest.raises(ValueError, match="requires_grad changed at index 0"):
        with optimizer_step_transaction(optimizer):
            parameter.requires_grad = False

    assert parameter.requires_grad is True


def test_optimizer_parameter_collection_drift_is_restored():
    parameter = Tensor([1.0], requires_grad=True)
    extra = Tensor([2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    original_container = optimizer.parameters

    with pytest.raises(RuntimeError, match="parameter collection changed"):
        with optimizer_step_transaction(optimizer):
            optimizer.parameters.append(extra)

    assert optimizer.parameters is original_container
    assert optimizer.parameters == [parameter]


def test_replaced_optimizer_parameter_list_is_restored_on_failure():
    class Marker(Exception):
        pass

    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    original_container = optimizer.parameters

    with pytest.raises(Marker):
        with optimizer_step_transaction(optimizer):
            optimizer.parameters = []
            raise Marker("abort")

    assert optimizer.parameters is original_container
    assert optimizer.parameters == [parameter]


def test_read_only_storage_created_inside_body_uses_replacement_rollback():
    class Marker(Exception):
        pass

    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    value_before = parameter.data.copy()

    with pytest.raises(Marker):
        with optimizer_step_transaction(optimizer):
            parameter.data[...] = [8.0, 9.0]
            parameter.data.setflags(write=False)
            raise Marker("abort")

    np.testing.assert_array_equal(parameter.data, value_before)
    assert parameter.data.flags.writeable


def test_nonfinite_parameter_baseline_is_rejected_before_body():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.data[...] = [np.inf]
    optimizer = SGD([parameter], lr=0.1)
    entered = False

    with pytest.raises(ValueError, match="must contain only finite values"):
        with optimizer_step_transaction(optimizer):
            entered = True

    assert entered is False


def test_nonfinite_optimizer_state_baseline_is_rejected_before_body():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1, momentum=0.9)
    optimizer._v[0][...] = [np.inf]
    entered = False

    with pytest.raises(ValueError, match="optimizer transaction state"):
        with optimizer_step_transaction(optimizer):
            entered = True

    assert entered is False


@pytest.mark.parametrize("factory", [SGD, Adam, AdamW])
def test_empty_builtin_optimizers_are_valid_transactions(factory):
    optimizer = factory([])
    state_before = _snapshot_state(optimizer)

    with optimizer_step_transaction(optimizer):
        optimizer.step()

    _assert_state_equal(optimizer.state_dict(), state_before)


def test_unsupported_optimizer_is_rejected_explicitly():
    class CustomOptimizer:
        parameters = []

    with pytest.raises(
        TypeError,
        match="optimizer must be SGD, Adam, or AdamW",
    ):
        with optimizer_step_transaction(CustomOptimizer()):
            pass


def test_malformed_builtin_parameter_collection_is_rejected_before_body():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    optimizer.parameters.append(object())
    entered = False

    with pytest.raises(TypeError, match="optimizer parameter 1 must be a Tensor"):
        with optimizer_step_transaction(optimizer):
            entered = True

    assert entered is False


def test_duplicate_builtin_parameter_collection_is_rejected_before_body():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    optimizer.parameters.append(parameter)
    entered = False

    with pytest.raises(ValueError, match="must not contain duplicates"):
        with optimizer_step_transaction(optimizer):
            entered = True

    assert entered is False


def test_same_thread_nested_transactions_are_reentrant():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SGD([parameter], lr=0.1)

    with optimizer_step_transaction(optimizer):
        optimizer.step()
        with optimizer_step_transaction(optimizer):
            optimizer.step()

    np.testing.assert_allclose(parameter.data, [0.8], rtol=0.0, atol=1e-15)


def test_rollback_restores_optimizer_state_but_not_gradient_buffers():
    class Marker(Exception):
        pass

    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [3.0]
    gradient = parameter.grad
    optimizer = Adam([parameter], lr=0.01)
    state_before = _snapshot_state(optimizer)

    with pytest.raises(Marker):
        with optimizer_step_transaction(optimizer):
            optimizer.step()
            raise Marker("abort")

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [3.0])
    _assert_state_equal(optimizer.state_dict(), state_before)
