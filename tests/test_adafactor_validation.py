import numpy as np
import pytest

from engine.adafactor import Adafactor
from engine.tensor import Tensor


_TINY = np.nextafter(0.0, 1.0)
_MAX = np.finfo(np.float64).max


def test_constructor_validates_parameter_collection_and_hyperparameters():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        Adafactor(123)
    with pytest.raises(TypeError, match="parameter 0"):
        Adafactor([object()])

    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="duplicate"):
        Adafactor([p, p])

    for name, kwargs in [
        ("lr", {"lr": 0.0}),
        ("lr", {"lr": np.inf}),
        ("beta2", {"beta2": -0.1}),
        ("beta2", {"beta2": 1.0}),
        ("eps", {"eps": 0.0}),
        ("clip_threshold", {"clip_threshold": 0.0}),
    ]:
        with pytest.raises((TypeError, ValueError), match=name):
            Adafactor(p, **kwargs)

    for name, kwargs in [
        ("lr", {"lr": True}),
        ("beta2", {"beta2": False}),
        ("eps", {"eps": True}),
        ("clip_threshold", {"clip_threshold": np.bool_(True)}),
    ]:
        with pytest.raises(TypeError, match=name):
            Adafactor(p, **kwargs)


def test_huge_python_integer_hyperparameter_conversion_is_rejected():
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="fit float64"):
        Adafactor(p, lr=10**400)


def test_step_rejects_shape_drift_before_mutating_state_or_data():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = Adafactor([first, second], beta2=0.0, eps=_TINY)
    before_state = optimizer.state_dict()
    first_version = first._version

    second.data = np.array([[2.0]])
    second.grad = np.array([[1.0]])

    with pytest.raises(ValueError, match="parameter 1 shape changed"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    assert first._version == first_version
    assert optimizer.state_dict()["states"][0]["step"] == before_state["states"][0]["step"]


def test_late_malformed_gradient_is_transactional():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([np.nan])
    optimizer = Adafactor([first, second], beta2=0.0, eps=_TINY)
    first_version = first._version

    with pytest.raises(ValueError, match="gradient for parameter 1"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    assert first._version == first_version
    assert optimizer.steps == (0, 0)


def test_gradient_type_dtype_and_shape_validation():
    p = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adafactor(p)

    p.grad = [1.0, 2.0]
    with pytest.raises(TypeError, match="NumPy array"):
        optimizer.step()

    p.grad = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    with pytest.raises(TypeError, match="real numeric"):
        optimizer.step()

    p.grad = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="shape"):
        optimizer.step()


def test_float32_gradient_is_normalized_without_modifying_original():
    p = Tensor([1.0, 2.0], requires_grad=True)
    gradient = np.array([3.0, -4.0], dtype=np.float32)
    p.grad = gradient
    optimizer = Adafactor(p, beta2=0.0, eps=_TINY, clip_threshold=10.0)

    optimizer.step()

    assert p.grad is gradient
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, [3.0, -4.0])


def test_extended_precision_gradient_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([1.0], requires_grad=True)
    gradient = np.array([np.finfo(np.float64).max], dtype=np.longdouble)
    gradient *= np.longdouble(2)
    p.grad = gradient
    optimizer = Adafactor(p)

    with pytest.raises(ValueError, match="fit float64"):
        optimizer.step()
    assert optimizer.steps == (0,)


def test_frozen_parameter_with_stale_gradient_is_rejected():
    p = Tensor([1.0], requires_grad=True)
    p.requires_grad = False
    p.grad = np.array([1.0])
    optimizer = Adafactor(p)

    with pytest.raises(ValueError, match="frozen"):
        optimizer.step()


def test_frozen_parameter_without_gradient_is_safely_skipped():
    p = Tensor([1.0], requires_grad=True)
    optimizer = Adafactor(p)
    p.requires_grad = False
    p.grad = None
    version = p._version

    optimizer.step()

    np.testing.assert_array_equal(p.data, [1.0])
    assert p._version == version
    assert optimizer.steps == (0,)


def test_nonfinite_active_parameter_is_rejected_before_state_mutation():
    p = Tensor([1.0], requires_grad=True)
    p.grad = np.array([1.0])
    optimizer = Adafactor(p)
    p.data[...] = np.inf
    version = p._version

    with pytest.raises(ValueError, match="parameter 0"):
        optimizer.step()

    assert optimizer.steps == (0,)
    assert p._version == version


def test_float64_max_scalar_gradient_is_warning_free():
    p = Tensor(0.0, requires_grad=True)
    p.grad = np.array(_MAX)
    optimizer = Adafactor(
        p, lr=0.25, beta2=0.0, eps=_TINY, clip_threshold=10.0
    )

    with np.errstate(all="raise"):
        optimizer.step()

    assert p.data.item() == pytest.approx(-0.25)
    state = optimizer.state_dict()["states"][0]
    assert state["scale"] == _MAX
    assert state["v"].item() == pytest.approx(1.0)


def test_opposite_float64_max_vector_components_are_warning_free():
    p = Tensor([0.0, 0.0], requires_grad=True)
    p.grad = np.array([_MAX, -_MAX])
    optimizer = Adafactor(
        p, lr=0.1, beta2=0.0, eps=_TINY, clip_threshold=10.0
    )

    with np.errstate(all="raise"):
        optimizer.step()

    np.testing.assert_allclose(p.data, [-0.1, 0.1], rtol=0.0, atol=1e-15)
    assert np.all(np.isfinite(p.data))


def test_float64_max_factored_matrix_is_warning_free():
    p = Tensor(np.zeros((2, 2)), requires_grad=True)
    p.grad = np.array([[_MAX, -_MAX], [_MAX, -_MAX]])
    optimizer = Adafactor(
        p, lr=0.1, beta2=0.0, eps=_TINY, clip_threshold=10.0
    )

    with np.errstate(all="raise"):
        optimizer.step()

    np.testing.assert_allclose(
        p.data,
        [[-0.1, 0.1], [-0.1, 0.1]],
        rtol=0.0,
        atol=1e-15,
    )
    state = optimizer.state_dict()["states"][0]
    assert state["scale"] == _MAX
    assert np.all(np.isfinite(state["row"]))
    assert np.all(np.isfinite(state["col"]))


def test_smallest_subnormal_scalar_gradient_remains_accepted():
    p = Tensor(0.0, requires_grad=True)
    p.grad = np.array(_TINY)
    optimizer = Adafactor(
        p,
        lr=1.0,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    with np.errstate(all="raise"):
        optimizer.step()

    assert np.isfinite(p.data.item())
    assert optimizer.steps == (1,)


def test_late_unrepresentable_candidate_leaves_earlier_parameter_and_state_unchanged():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([_MAX], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([-1.0])
    optimizer = Adafactor(
        [first, second],
        lr=_MAX,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )
    first_version = first._version

    with pytest.raises(ValueError, match="parameter 1"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    assert first._version == first_version
    assert optimizer.steps == (0, 0)


def test_read_only_late_destination_is_preflighted_before_earlier_write():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = Adafactor(
        [first, second], beta2=0.0, eps=_TINY, clip_threshold=10.0
    )
    second.data.flags.writeable = False
    first_version = first._version

    with pytest.raises(ValueError, match="parameter 1 data must be writable"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    assert first._version == first_version
    assert optimizer.steps == (0, 0)


def test_shared_active_parameter_storage_is_rejected_before_write():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    second._data = first.data
    first.grad = np.array([1.0, 1.0])
    second.grad = np.array([1.0, 1.0])
    optimizer = Adafactor(
        [first, second], beta2=0.0, eps=_TINY, clip_threshold=10.0
    )
    before = first.data.copy()

    with pytest.raises(ValueError, match="storage must not overlap"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, before)
    assert optimizer.steps == (0, 0)


def test_disjoint_views_into_one_backing_allocation_are_allowed():
    backing = np.array([1.0, 2.0, 3.0, 4.0])
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first._data = backing[:2]
    second._data = backing[2:]
    first.grad = np.array([1.0, 1.0])
    second.grad = np.array([1.0, 1.0])
    optimizer = Adafactor(
        [first, second],
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )

    optimizer.step()

    np.testing.assert_allclose(backing, [0.9, 1.9, 2.9, 3.9])
    assert optimizer.steps == (1, 1)
