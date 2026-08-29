import numpy as np
import pytest

from engine.pcgrad import PCGradBuffer
from engine.tensor import Tensor


def test_constructor_rejects_non_iterable_non_tensor():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        PCGradBuffer(123)


def test_constructor_rejects_non_tensor_entries_and_duplicates():
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError, match="only Tensors"):
        PCGradBuffer([parameter, object()])
    with pytest.raises(ValueError, match="duplicate"):
        PCGradBuffer([parameter, parameter])


def test_constructor_rejects_frozen_parameters():
    with pytest.raises(ValueError, match="require gradients"):
        PCGradBuffer([Tensor([1.0], requires_grad=False)])


def test_seed_validation_precedes_projection():
    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    parameter.grad = np.asarray([1.0])
    pcgrad.capture()

    for bad in (True, 1.5, "7", object()):
        with pytest.raises(TypeError, match="seed"):
            pcgrad.projected_gradients(seed=bad)
    for bad in (-1, 2**64):
        with pytest.raises(ValueError, match="seed"):
            pcgrad.projected_gradients(seed=bad)


def test_empty_projection_rejects_after_valid_seed():
    pcgrad = PCGradBuffer([])
    with pytest.raises(RuntimeError, match="no task gradients"):
        pcgrad.projected_task_gradients(seed=0)


def test_capture_rejects_gradient_shape_mismatch_transactionally():
    p1 = Tensor([0.0], requires_grad=True)
    p2 = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer([p1, p2])
    p1.grad = np.asarray([1.0])
    p2.grad = np.asarray([2.0])

    with pytest.raises(ValueError, match=r"gradient 1 shape mismatch"):
        pcgrad.capture()
    assert pcgrad.task_count == 0


def test_capture_rejects_nonfloating_and_nonfinite_gradients():
    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)

    parameter.grad = np.asarray([1], dtype=np.int64)
    with pytest.raises(TypeError, match="floating-point"):
        pcgrad.capture()

    parameter.grad = np.asarray([np.nan])
    with pytest.raises(ValueError, match="finite"):
        pcgrad.capture()

    parameter.grad = np.asarray([np.inf])
    with pytest.raises(ValueError, match="finite"):
        pcgrad.capture()

    assert pcgrad.task_count == 0


def test_capture_rejects_non_array_gradient():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = [1.0]
    pcgrad = PCGradBuffer(parameter)
    with pytest.raises(TypeError, match="NumPy array"):
        pcgrad.capture()


def test_capture_normalizes_float32_to_float64_snapshot():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.asarray([1.25, -2.5], dtype=np.float32)
    pcgrad = PCGradBuffer(parameter)
    pcgrad.capture()

    task = pcgrad.task_gradients()[0][0]
    assert task.dtype == np.float64
    np.testing.assert_array_equal(task, [1.25, -2.5])


def test_extended_precision_gradient_outside_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")

    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    parameter.grad = np.asarray([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    assert np.isfinite(parameter.grad).all()

    with pytest.raises(ValueError, match="fit in float64"):
        pcgrad.capture()
    assert pcgrad.task_count == 0


def test_parameter_shape_drift_is_rejected_before_capture_or_projection():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    parameter.grad = np.asarray([1.0, 2.0])
    pcgrad.capture()
    parameter.data = np.asarray([[0.0, 0.0]])

    with pytest.raises(ValueError, match="shape changed"):
        pcgrad.capture()
    with pytest.raises(ValueError, match="shape changed"):
        pcgrad.projected_gradients()


def test_parameter_trainability_drift_is_rejected():
    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    parameter.grad = np.asarray([1.0])
    pcgrad.capture()
    parameter.requires_grad = False

    with pytest.raises(ValueError, match="no longer requires"):
        pcgrad.projected_gradients()


def test_extreme_finite_conflict_projects_without_overflow():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)

    parameter.grad = np.asarray([1.3e308, 0.0])
    pcgrad.capture()
    parameter.grad = np.asarray([-1.3e308, 1.3e308])
    pcgrad.capture()

    with np.errstate(all="raise"):
        projected = pcgrad.projected_task_gradients()
        combined = pcgrad.projected_gradients()

    np.testing.assert_allclose(projected[0][0], [6.5e307, 6.5e307], rtol=1e-15)
    np.testing.assert_allclose(projected[1][0], [0.0, 1.3e308], rtol=1e-15)
    np.testing.assert_allclose(combined[0], [3.25e307, 9.75e307], rtol=1e-15)


def test_opposite_maximum_finite_tasks_do_not_leak_runtime_warnings():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer(parameter)
    parameter.grad = np.asarray([maximum])
    pcgrad.capture()
    parameter.grad = np.asarray([-maximum])
    pcgrad.capture()

    with np.errstate(all="raise"):
        result = pcgrad.projected_gradients()
    np.testing.assert_array_equal(result[0], [0.0])
