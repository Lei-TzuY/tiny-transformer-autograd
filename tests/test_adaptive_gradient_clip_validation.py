import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def test_options_validate_before_parameter_generator_is_consumed():
    events = []

    def parameters():
        events.append("consumed")
        yield Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="clip_factor"):
        adaptive_clip_grad_(parameters(), clip_factor=True)
    assert events == []

    with pytest.raises(ValueError, match="clip_factor"):
        adaptive_clip_grad_(parameters(), clip_factor=0.0)
    assert events == []

    with pytest.raises(TypeError, match="eps"):
        adaptive_clip_grad_(parameters(), eps=False)
    assert events == []

    with pytest.raises(ValueError, match="eps"):
        adaptive_clip_grad_(parameters(), eps=0.0)
    assert events == []


def test_numpy_real_options_are_normalized_and_conversion_overflow_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    assert adaptive_clip_grad_(parameter, clip_factor=np.float32(0.1), eps=np.float64(1e-3)) == 0

    with pytest.raises(ValueError, match="fit float64"):
        adaptive_clip_grad_(parameter, clip_factor=10**400)


def test_parameter_collection_rejects_non_iterable_non_tensor_and_duplicates():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        adaptive_clip_grad_(123)
    with pytest.raises(TypeError, match="parameter 0"):
        adaptive_clip_grad_([object()])

    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="duplicate"):
        adaptive_clip_grad_([parameter, parameter])


def test_parameter_generator_is_materialized_once():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = np.array([0.0, 0.0])
    events = []

    def parameters():
        events.append("start")
        yield first
        yield second
        events.append("end")

    assert adaptive_clip_grad_(parameters(), clip_factor=0.1) == 1
    assert events == ["start", "end"]


def test_gradient_must_be_numpy_floating_exact_shape_and_finite():
    parameter = Tensor([1.0, 2.0], requires_grad=True)

    parameter.grad = [3.0, 4.0]
    with pytest.raises(TypeError, match="NumPy array"):
        adaptive_clip_grad_(parameter)

    parameter.grad = np.array([3, 4], dtype=np.int64)
    with pytest.raises(TypeError, match="floating dtype"):
        adaptive_clip_grad_(parameter)

    parameter.grad = np.array([[3.0, 4.0]])
    with pytest.raises(ValueError, match="shape mismatch"):
        adaptive_clip_grad_(parameter)

    parameter.grad = np.array([np.nan, 4.0])
    with pytest.raises(ValueError, match="finite"):
        adaptive_clip_grad_(parameter)

    parameter.grad = np.array([np.inf, 4.0])
    with pytest.raises(ValueError, match="finite"):
        adaptive_clip_grad_(parameter)


def test_active_parameter_data_must_be_real_finite_and_fit_float64():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([3.0, 4.0])
    parameter.data[...] = [np.inf, 2.0]
    with pytest.raises(ValueError, match="parameter 0 data.*finite"):
        adaptive_clip_grad_(parameter)


def test_frozen_parameter_with_stale_gradient_is_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.requires_grad = False
    parameter.grad = np.array([2.0])

    with pytest.raises(ValueError, match="frozen"):
        adaptive_clip_grad_(parameter)


def test_float32_gradients_are_supported_but_unrepresentable_longdouble_is_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2.0], dtype=np.float32)
    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 1

    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    parameter.grad = np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    with pytest.raises(ValueError, match="fit float64"):
        adaptive_clip_grad_(parameter)


def test_late_invalid_gradient_is_transactional_before_earlier_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([1.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = np.array([np.nan])
    first_before = first.grad.copy()

    with pytest.raises(ValueError, match="parameter 1"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    np.testing.assert_array_equal(first.grad, first_before)


def test_late_read_only_gradient_that_needs_write_is_preflighted_before_any_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = np.array([6.0, 8.0])
    second.grad.flags.writeable = False
    first_before = first.grad.copy()

    with pytest.raises(ValueError, match="parameter 1.*writable"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    np.testing.assert_array_equal(first.grad, first_before)


def test_exact_shared_gradient_storage_is_rejected_when_a_write_is_required():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([30.0, 40.0], requires_grad=True)
    shared = np.array([6.0, 8.0])
    first.grad = shared
    second.grad = shared

    with pytest.raises(ValueError, match="gradient storage must not overlap"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)
    np.testing.assert_array_equal(shared, [6.0, 8.0])


def test_overlapping_gradient_views_are_rejected_but_disjoint_views_are_allowed():
    backing = np.array([6.0, 8.0, 6.0, 8.0, 6.0, 8.0])
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = backing[:2]
    second.grad = backing[1:3]

    with pytest.raises(ValueError, match="gradient storage must not overlap"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    disjoint = np.array([6.0, 8.0, 99.0, 99.0, 6.0, 8.0])
    first.grad = disjoint[:2]
    second.grad = disjoint[4:]
    assert adaptive_clip_grad_([first, second], clip_factor=0.1) == 2
    np.testing.assert_allclose(disjoint, [0.3, 0.4, 99.0, 99.0, 0.3, 0.4])


def test_gradient_parameter_data_alias_is_rejected_before_any_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([6.0, 8.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = first.data
    first_before = first.grad.copy()
    data_before = first.data.copy()

    with pytest.raises(ValueError, match="overlap parameter 0 data"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(first.data, data_before)


def test_noop_gradient_parameter_alias_is_allowed_because_no_write_occurs():
    parameter = Tensor([30.0, 40.0], requires_grad=True)
    parameter.grad = parameter.data
    version = parameter._version

    assert adaptive_clip_grad_(parameter, clip_factor=1.0) == 0
    assert parameter.grad is parameter.data
    assert parameter._version == version
