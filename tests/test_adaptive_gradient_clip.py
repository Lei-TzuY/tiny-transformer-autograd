import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def test_scalar_gradient_is_clipped_relative_to_scalar_parameter_norm():
    parameter = Tensor(2.0, requires_grad=True)
    parameter.grad = np.array(10.0)
    gradient_ref = parameter.grad
    version = parameter._version

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-6)

    assert changed == 1
    assert parameter.grad is gradient_ref
    assert parameter.grad.shape == ()
    assert parameter.grad.item() == pytest.approx(0.2)
    assert parameter.data.item() == 2.0
    assert parameter._version == version


def test_vector_is_one_unit():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0])

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-6)

    assert changed == 1
    np.testing.assert_allclose(parameter.grad, [0.3, 0.4])


def test_matrix_clips_each_last_axis_row_independently():
    parameter = Tensor([[3.0, 4.0], [0.0, 0.0]], requires_grad=True)
    parameter.grad = np.array([[6.0, 8.0], [3.0, 4.0]])

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=2.0)

    assert changed == 2
    np.testing.assert_allclose(parameter.grad[0], [0.3, 0.4])
    np.testing.assert_allclose(parameter.grad[1], [0.12, 0.16])


def test_matrix_can_clip_one_row_and_leave_another_exactly_unchanged():
    parameter = Tensor([[3.0, 4.0], [30.0, 40.0]], requires_grad=True)
    parameter.grad = np.array([[6.0, 8.0], [0.3, 0.4]])
    second_before = parameter.grad[1].copy()

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-6)

    assert changed == 1
    np.testing.assert_allclose(parameter.grad[0], [0.3, 0.4])
    np.testing.assert_array_equal(parameter.grad[1], second_before)


def test_rank_three_tensor_uses_last_axis_units():
    parameter = Tensor(
        [
            [[3.0, 4.0], [0.0, 5.0]],
            [[6.0, 8.0], [5.0, 12.0]],
        ],
        requires_grad=True,
    )
    parameter.grad = np.array(
        [
            [[6.0, 8.0], [0.0, 10.0]],
            [[0.6, 0.8], [5.0, 12.0]],
        ]
    )

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-6)

    assert changed == 3
    np.testing.assert_allclose(parameter.grad[0, 0], [0.3, 0.4])
    np.testing.assert_allclose(parameter.grad[0, 1], [0.0, 0.5])
    np.testing.assert_array_equal(parameter.grad[1, 0], [0.6, 0.8])
    np.testing.assert_allclose(parameter.grad[1, 1], [0.5, 1.2])


def test_grad_none_is_ignored_and_zero_gradient_is_exact_noop():
    missing = Tensor([2.0, 3.0], requires_grad=True)
    missing.grad = None
    zero = Tensor([2.0, 3.0], requires_grad=True)
    zero.grad = np.zeros(2)
    zero_ref = zero.grad

    assert adaptive_clip_grad_([missing, zero], clip_factor=0.1) == 0
    assert missing.grad is None
    assert zero.grad is zero_ref
    np.testing.assert_array_equal(zero.grad, [0.0, 0.0])


def test_noop_preserves_read_only_gradient_and_identity():
    parameter = Tensor([30.0, 40.0], requires_grad=True)
    parameter.grad = np.array([0.3, 0.4])
    parameter.grad.flags.writeable = False
    gradient_ref = parameter.grad

    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 0
    assert parameter.grad is gradient_ref
    assert parameter.grad.flags.writeable is False


def test_float32_gradient_preserves_dtype_and_object_identity():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0], dtype=np.float32)
    gradient_ref = parameter.grad

    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 1
    assert parameter.grad is gradient_ref
    assert parameter.grad.dtype == np.float32
    np.testing.assert_allclose(parameter.grad, np.array([0.3, 0.4], dtype=np.float32))


def test_true_l2_overflow_for_parameter_and_gradient_still_clips_correctly():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([maximum, maximum], requires_grad=True)
    parameter.grad = np.array([maximum, -maximum])

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.5, eps=1e-300)

    assert changed == 1
    np.testing.assert_array_equal(parameter.grad, [maximum * 0.5, -maximum * 0.5])


def test_extreme_parameter_norm_can_make_extreme_gradient_a_noop():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([maximum, maximum], requires_grad=True)
    parameter.grad = np.array([maximum * 0.25, -maximum * 0.25])
    before = parameter.grad.copy()

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.5, eps=1e-300)

    assert changed == 0
    np.testing.assert_array_equal(parameter.grad, before)


def test_zero_parameter_uses_eps_floor_even_against_float64_max_gradient():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([maximum])

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.25, eps=4e-200)

    assert changed == 1
    assert parameter.grad[0] == pytest.approx(1e-200)


def test_smallest_subnormal_gradient_is_warning_free():
    tiny = np.nextafter(0.0, 1.0)
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([tiny])

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.01, eps=tiny)

    assert changed == 0
    assert parameter.grad[0] == tiny


def test_empty_parameter_collection_and_empty_vector_are_noops():
    assert adaptive_clip_grad_([], clip_factor=0.1) == 0

    parameter = Tensor(np.empty((0,)), requires_grad=True)
    parameter.grad = np.empty((0,))
    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 0


def test_numpy_global_rng_and_parameter_values_are_neutral():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0])
    data_before = parameter.data.copy()
    version = parameter._version
    rng_before = np.random.get_state()

    adaptive_clip_grad_(parameter, clip_factor=0.1)

    rng_after = np.random.get_state()
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter._version == version
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
