import numpy as np
import pytest

from engine.pcgrad import PCGradBuffer
from engine.tensor import Tensor


def _set_grad(parameter, values):
    parameter.grad = np.asarray(values, dtype=np.float64)


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_two_task_conflict_matches_hand_projection():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer([parameter])

    _set_grad(parameter, [1.0, 0.0])
    assert pcgrad.capture() == 1
    _set_grad(parameter, [-1.0, 1.0])
    assert pcgrad.capture() == 2

    projected = pcgrad.projected_task_gradients()
    np.testing.assert_allclose(projected[0][0], [0.5, 0.5])
    np.testing.assert_allclose(projected[1][0], [0.0, 1.0])

    combined = pcgrad.projected_gradients()
    np.testing.assert_allclose(combined[0], [0.25, 0.75])


def test_nonconflicting_tasks_are_averaged_without_projection():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)

    _set_grad(parameter, [1.0, 2.0])
    pcgrad.capture()
    _set_grad(parameter, [3.0, 4.0])
    pcgrad.capture()

    projected = pcgrad.projected_task_gradients()
    np.testing.assert_array_equal(projected[0][0], [1.0, 2.0])
    np.testing.assert_array_equal(projected[1][0], [3.0, 4.0])
    np.testing.assert_array_equal(pcgrad.projected_gradients()[0], [2.0, 3.0])


def test_missing_gradient_is_zero_task_contribution():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer([parameter])

    parameter.grad = None
    pcgrad.capture()
    _set_grad(parameter, [2.0, -4.0])
    pcgrad.capture()

    tasks = pcgrad.task_gradients()
    np.testing.assert_array_equal(tasks[0][0], [0.0, 0.0])
    np.testing.assert_array_equal(pcgrad.projected_gradients()[0], [1.0, -2.0])


def test_single_task_is_returned_exactly():
    p1 = Tensor([[0.0, 0.0]], requires_grad=True)
    p2 = Tensor(0.0, requires_grad=True)
    pcgrad = PCGradBuffer([p1, p2])

    _set_grad(p1, [[2.0, -3.0]])
    p2.grad = np.asarray(4.0)
    pcgrad.capture()

    projected = pcgrad.projected_gradients()
    np.testing.assert_array_equal(projected[0], [[2.0, -3.0]])
    np.testing.assert_array_equal(projected[1], 4.0)


def test_capture_does_not_modify_live_gradient_objects_or_values():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    gradient = np.asarray([3.0, -2.0], dtype=np.float64)
    parameter.grad = gradient
    pcgrad = PCGradBuffer([parameter])

    pcgrad.capture()

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [3.0, -2.0])


def test_task_snapshots_and_projected_results_are_independent():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer([parameter])
    _set_grad(parameter, [1.0, 2.0])
    pcgrad.capture()

    first = pcgrad.task_gradients()
    first[0][0][0] = 999.0
    second = pcgrad.task_gradients()
    np.testing.assert_array_equal(second[0][0], [1.0, 2.0])

    projected = pcgrad.projected_gradients()
    projected[0][1] = 999.0
    np.testing.assert_array_equal(pcgrad.projected_gradients()[0], [1.0, 2.0])


def test_copy_to_grads_installs_independent_float64_arrays():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer([parameter])
    parameter.grad = np.asarray([1.0, 2.0], dtype=np.float32)
    pcgrad.capture()

    result = pcgrad.projected_gradients()[0]
    assert pcgrad.copy_to_grads() is pcgrad

    assert parameter.grad.dtype == np.float64
    np.testing.assert_array_equal(parameter.grad, [1.0, 2.0])
    assert not np.shares_memory(parameter.grad, result)


def test_reset_discards_tasks_without_touching_live_gradients():
    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    gradient = np.asarray([2.0])
    parameter.grad = gradient
    pcgrad.capture()

    assert pcgrad.reset() is pcgrad
    assert pcgrad.task_count == 0
    assert parameter.grad is gradient
    with pytest.raises(RuntimeError, match="no task gradients"):
        pcgrad.projected_gradients()


def test_seeded_projection_is_reproducible_and_global_rng_neutral():
    parameter = Tensor([0.0, 0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    for gradient in ([1.0, -2.0, 0.5], [-2.0, 1.0, 0.25], [0.5, 0.5, -1.0]):
        _set_grad(parameter, gradient)
        pcgrad.capture()

    np.random.seed(12345)
    before = np.random.get_state()
    first = pcgrad.projected_gradients(seed=77)
    after = np.random.get_state()
    second = pcgrad.projected_gradients(seed=np.int64(77))

    _assert_rng_state_equal(before, after)
    np.testing.assert_array_equal(first[0], second[0])


def test_empty_parameter_collection_supports_task_bookkeeping():
    pcgrad = PCGradBuffer([])
    assert pcgrad.capture() == 1
    assert pcgrad.task_count == 1
    assert pcgrad.projected_task_gradients() == ((),)
    assert pcgrad.projected_gradients() == ()
    assert pcgrad.copy_to_grads() is pcgrad
