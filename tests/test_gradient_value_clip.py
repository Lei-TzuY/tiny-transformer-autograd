import numpy as np
import pytest

from engine.gradient_value_clip import clip_grad_value_
from engine.tensor import Tensor


class _MutateThenRaise(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.failures_remaining = 1
        return array

    def __array_finalize__(self, source):
        self.failures_remaining = getattr(source, "failures_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected gradient write failure")


def _tensor(values):
    return Tensor(np.asarray(values, dtype=np.float64), requires_grad=True)


def test_clips_present_gradients_in_place_and_returns_changed_buffer_count():
    first = _tensor([1.0, 2.0, 3.0])
    second = _tensor([-4.0, 0.5])
    first.grad[...] = np.array([0.25, -3.0, 2.0])
    second.grad[...] = np.array([-9.0, 0.5])
    first_grad = first.grad
    second_grad = second.grad

    changed = clip_grad_value_([first, second], 1.5)

    assert changed == 2
    assert first.grad is first_grad
    assert second.grad is second_grad
    np.testing.assert_array_equal(first.grad, np.array([0.25, -1.5, 1.5]))
    np.testing.assert_array_equal(second.grad, np.array([-1.5, 0.5]))


def test_missing_gradient_is_ignored_and_single_tensor_input_is_supported():
    parameter = _tensor([1.0, 2.0])
    parameter.grad = None

    assert clip_grad_value_(parameter, 1.0) == 0
    assert parameter.grad is None


def test_noop_preserves_gradient_values_identity_and_read_only_storage():
    parameter = _tensor([1.0, 2.0])
    parameter.grad[...] = np.array([-0.5, 0.75])
    parameter.grad.flags.writeable = False
    gradient = parameter.grad

    try:
        assert clip_grad_value_([parameter], 1.0) == 0
        assert parameter.grad is gradient
        np.testing.assert_array_equal(parameter.grad, np.array([-0.5, 0.75]))
        assert not parameter.grad.flags.writeable
    finally:
        parameter.grad.flags.writeable = True


def test_late_read_only_gradient_is_rejected_before_any_write():
    first = _tensor([1.0])
    second = _tensor([1.0])
    first.grad[...] = 9.0
    second.grad[...] = -8.0
    second.grad.flags.writeable = False

    try:
        with pytest.raises(ValueError, match=r"parameters\[1\].*writeable"):
            clip_grad_value_([first, second], 2.0)
        np.testing.assert_array_equal(first.grad, np.array([9.0]))
        np.testing.assert_array_equal(second.grad, np.array([-8.0]))
    finally:
        second.grad.flags.writeable = True


def test_commit_failure_rolls_back_every_attempted_gradient():
    first = _tensor([1.0, 2.0])
    second = _tensor([3.0, 4.0])
    first.grad[...] = np.array([7.0, -8.0])
    second.grad = _MutateThenRaise([9.0, -10.0])
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        clip_grad_value_([first, second], 1.0)

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_float32_gradient_keeps_dtype_and_identity():
    parameter = _tensor([1.0, 2.0, 3.0])
    parameter.grad = np.array([3.0, -0.5, -4.0], dtype=np.float32)
    gradient = parameter.grad

    assert clip_grad_value_([parameter], np.float32(2.0)) == 1
    assert parameter.grad is gradient
    assert parameter.grad.dtype == np.float32
    np.testing.assert_array_equal(parameter.grad, np.array([2.0, -0.5, -2.0], np.float32))


def test_extreme_finite_gradient_clips_without_floating_warning():
    parameter = _tensor([1.0, 2.0])
    maximum = np.finfo(np.float64).max
    parameter.grad[...] = np.array([maximum, -maximum])

    with np.errstate(all="raise"):
        assert clip_grad_value_([parameter], 1e300) == 1

    np.testing.assert_array_equal(parameter.grad, np.array([1e300, -1e300]))


def test_generator_is_materialized_once_and_order_is_preserved():
    first = _tensor([1.0])
    second = _tensor([1.0])
    first.grad[...] = 3.0
    second.grad[...] = -4.0
    visits = []

    def parameters():
        visits.append("start")
        yield first
        visits.append("middle")
        yield second
        visits.append("end")

    assert clip_grad_value_(parameters(), 2.0) == 2
    assert visits == ["start", "middle", "end"]
    np.testing.assert_array_equal(first.grad, np.array([2.0]))
    np.testing.assert_array_equal(second.grad, np.array([-2.0]))


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "clip_value must be a real number"),
        ("1", TypeError, "clip_value must be a real number"),
        (0.0, ValueError, "clip_value must be positive"),
        (-1.0, ValueError, "clip_value must be positive"),
        (np.inf, ValueError, "clip_value must be finite"),
        (np.nan, ValueError, "clip_value must be finite"),
        (10**400, ValueError, "clip_value must be finite"),
    ],
)
def test_clip_value_validation_happens_before_parameter_iteration(value, error, message):
    consumed = False

    def parameters():
        nonlocal consumed
        consumed = True
        yield _tensor([1.0])

    with pytest.raises(error, match=message):
        clip_grad_value_(parameters(), value)
    assert not consumed


def test_parameter_collection_validation_is_explicit():
    parameter = _tensor([1.0])

    with pytest.raises(TypeError, match="Tensor or iterable"):
        clip_grad_value_(object(), 1.0)
    with pytest.raises(TypeError, match=r"parameters\[1\] must be a Tensor"):
        clip_grad_value_([parameter, object()], 1.0)
    with pytest.raises(ValueError, match="duplicate Tensor references"):
        clip_grad_value_([parameter, parameter], 1.0)


@pytest.mark.parametrize("kind", ["type", "dtype", "shape", "nonfinite"])
def test_gradient_validation_is_transactional(kind):
    first = _tensor([1.0, 2.0])
    second = _tensor([3.0, 4.0])
    first.grad[...] = np.array([8.0, -9.0])
    first_before = first.grad.copy()

    if kind == "type":
        second.grad = [1.0, 2.0]
        expected = TypeError
    elif kind == "dtype":
        second.grad = np.array([1, 2], dtype=np.int64)
        expected = TypeError
    elif kind == "shape":
        second.grad = np.array([1.0])
        expected = ValueError
    else:
        second.grad = np.array([1.0, np.inf])
        expected = ValueError

    with pytest.raises(expected):
        clip_grad_value_([first, second], 1.0)
    np.testing.assert_array_equal(first.grad, first_before)


def test_clipping_is_parameter_data_version_and_rng_neutral():
    parameter = _tensor([1.0, -2.0])
    parameter.grad[...] = np.array([5.0, -6.0])
    data_before = parameter.data.copy()
    version_before = parameter._version

    np.random.seed(314159)
    rng_before = np.random.get_state()
    assert clip_grad_value_([parameter], 2.0) == 1
    rng_after = np.random.get_state()

    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter._version == version_before
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
