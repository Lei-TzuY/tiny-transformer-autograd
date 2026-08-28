import copy
import threading

import numpy as np
import pytest

from engine.lookahead import Lookahead
from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
        return
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        np.testing.assert_array_equal(left, right)
        return
    assert left == right


def test_sgd_fast_steps_and_scheduled_sync_have_exact_arithmetic():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    inner = SGD([parameter], lr=0.1)
    optimizer = Lookahead(inner, sync_period=2, alpha=0.5)
    version = parameter._version

    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.9], rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [1.0])
    assert optimizer.step_count == 1
    assert parameter._version == version + 1

    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.9], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(optimizer.slow_weights()[0], [0.9], rtol=0.0, atol=1e-15)
    assert optimizer.step_count == 2
    # The second inner step writes 0.8, then Lookahead writes the synchronized 0.9.
    assert parameter._version == version + 3

    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.8], rtol=0.0, atol=1e-15)
    assert optimizer.step_count == 3


@pytest.mark.parametrize("factory", [Adam, AdamW])
def test_adam_family_inner_state_advances_through_wrapper(factory):
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [0.25]
    inner = factory([parameter], lr=0.01)
    optimizer = Lookahead(inner, sync_period=3, alpha=0.5)

    optimizer.step()
    optimizer.step()

    assert inner.t == 2
    assert inner._steps == [2]
    assert optimizer.step_count == 2
    assert optimizer.pending_sync is False
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [1.0])


def test_alpha_one_updates_slow_without_extra_fast_tensor_write():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lookahead(SGD([parameter], lr=0.1), sync_period=1, alpha=1.0)
    version = parameter._version

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [0.9], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(optimizer.slow_weights()[0], [0.9], rtol=0.0, atol=1e-15)
    assert parameter._version == version + 1


def test_alpha_zero_restores_slow_weight_at_each_sync():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lookahead(SGD([parameter], lr=0.25), sync_period=1, alpha=0.0)

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [2.0])


def test_manual_sync_does_not_change_step_count_or_automatic_cadence():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lookahead(SGD([parameter], lr=0.1), sync_period=3, alpha=0.5)

    optimizer.step()
    optimizer.sync()
    assert optimizer.step_count == 1
    np.testing.assert_allclose(parameter.data, [0.95], rtol=0.0, atol=1e-15)

    optimizer.step()
    assert optimizer.step_count == 2
    np.testing.assert_allclose(parameter.data, [0.85], rtol=0.0, atol=1e-15)

    optimizer.step()
    assert optimizer.step_count == 3
    # Slow was 0.95 after the manual sync; fast reaches 0.75 before auto sync.
    np.testing.assert_allclose(parameter.data, [0.85], rtol=0.0, atol=1e-15)


def test_opposite_extreme_finite_weights_sync_without_subtraction_overflow():
    parameter = Tensor([1.3e308], requires_grad=True)
    parameter.grad = None
    optimizer = Lookahead(SGD([parameter], lr=0.1), sync_period=9, alpha=0.5)
    parameter.data[...] = [-1.3e308]

    with np.errstate(all="raise"):
        optimizer.sync()

    np.testing.assert_array_equal(parameter.data, [0.0])
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [0.0])


def test_zero_grad_is_forwarded_and_preserves_gradient_object_when_requested():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [3.0]
    gradient = parameter.grad
    optimizer = Lookahead(SGD([parameter]), sync_period=2)

    optimizer.zero_grad()
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [0.0])

    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None


def test_scheduled_sync_failure_becomes_pending_and_blocks_more_fast_steps():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = None
    inner = SGD([parameter], lr=0.1)
    optimizer = Lookahead(inner, sync_period=1, alpha=0.5)

    parameter.data[...] = [3.0]
    parameter.data.setflags(write=False)

    with pytest.raises(ValueError, match="must be writeable for sync"):
        optimizer.step()

    assert optimizer.step_count == 1
    assert optimizer.pending_sync is True
    np.testing.assert_array_equal(parameter.data, [3.0])
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [1.0])

    with pytest.raises(RuntimeError, match="pending synchronization"):
        optimizer.step()
    assert optimizer.step_count == 1

    parameter.data.setflags(write=True)
    assert optimizer.sync() is optimizer
    assert optimizer.pending_sync is False
    np.testing.assert_array_equal(parameter.data, [2.0])
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [2.0])


def test_nonfinite_fast_value_rejects_sync_without_advancing_slow():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = None
    optimizer = Lookahead(SGD([parameter]), sync_period=2)
    slow_before = optimizer.slow_weights()[0]
    parameter.data[...] = [np.inf]

    with pytest.raises(ValueError, match="must contain only finite values"):
        optimizer.sync()

    np.testing.assert_array_equal(optimizer.slow_weights()[0], slow_before)


def test_shape_drift_is_rejected_before_inner_step_advances():
    parameter = Tensor([1.0], requires_grad=True)
    inner = Adam([parameter], lr=0.01)
    optimizer = Lookahead(inner, sync_period=2)
    parameter.data = [1.0, 2.0]

    with pytest.raises(ValueError, match="parameter shape changed at index 0"):
        optimizer.step()

    assert inner.t == 0
    assert optimizer.step_count == 0


def test_parameter_collection_replacement_and_reordering_are_rejected():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    inner = SGD([first, second])
    optimizer = Lookahead(inner)

    inner.parameters = [first, second]
    with pytest.raises(RuntimeError, match="parameter collection changed"):
        optimizer.step()

    inner.parameters = optimizer._parameter_container
    inner.parameters[:] = [second, first]
    with pytest.raises(RuntimeError, match="parameter collection changed"):
        optimizer.slow_weights()


def test_inner_step_failure_does_not_advance_lookahead_state():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad[...] = [1.0]
    second.grad[...] = [1.0]
    inner = SGD([first, second], lr=0.1, momentum=0.9)
    optimizer = Lookahead(inner, sync_period=2)
    slow_before = optimizer.slow_weights()

    second.data.setflags(write=False)
    with pytest.raises(ValueError):
        optimizer.step()

    # The inner optimizer owns its own failure semantics; Lookahead itself does
    # not pretend that the partially failed fast step was successful.
    assert optimizer.step_count == 0
    assert optimizer.pending_sync is False
    for before, after in zip(slow_before, optimizer.slow_weights()):
        np.testing.assert_array_equal(before, after)


def test_state_dict_round_trip_restores_wrapper_and_inner_state_not_fast_values():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [0.5]
    inner = Adam([parameter], lr=0.01)
    optimizer = Lookahead(inner, sync_period=2, alpha=0.25)
    optimizer.step()
    saved = optimizer.state_dict()
    saved_fast = parameter.data.copy()

    optimizer.alpha = 0.75
    optimizer.sync_period = 5
    optimizer.step()
    parameter.data[...] = [9.0]

    assert optimizer.load_state_dict(saved) is optimizer

    assert optimizer.alpha == 0.25
    assert optimizer.sync_period == 2
    assert optimizer.step_count == 1
    assert optimizer.pending_sync is False
    assert inner.t == 1
    assert inner._steps == [1]
    # Model parameters are checkpointed by the model, not optimizer state.
    np.testing.assert_array_equal(parameter.data, [9.0])
    assert not np.array_equal(parameter.data, saved_fast)
    np.testing.assert_array_equal(optimizer.slow_weights()[0], [1.0])


def test_state_dict_returns_independent_nested_copies():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lookahead(Adam([parameter]), sync_period=2)

    state = optimizer.state_dict()
    state["slow_weights"][0][...] = [7.0]
    state["optimizer"]["m"][0][...] = [8.0]

    np.testing.assert_array_equal(optimizer.slow_weights()[0], [1.0])
    np.testing.assert_array_equal(optimizer.optimizer._m[0], [0.0])


def test_malformed_inner_state_load_is_transactional_for_wrapper_and_inner():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [0.25]
    inner = Adam([parameter], lr=0.01)
    optimizer = Lookahead(inner, sync_period=3, alpha=0.4)
    optimizer.step()

    wrapper_before = optimizer.state_dict()
    malformed = copy.deepcopy(wrapper_before)
    malformed["alpha"] = 0.9
    malformed["sync_period"] = 7
    malformed["optimizer"]["m"][0] = np.zeros((2,), dtype=np.float64)

    with pytest.raises(ValueError, match="shape mismatch"):
        optimizer.load_state_dict(malformed)

    wrapper_after = optimizer.state_dict()
    _assert_nested_equal(wrapper_after, wrapper_before)


def test_malformed_slow_weight_is_rejected_before_inner_state_changes():
    parameter = Tensor([1.0], requires_grad=True)
    inner = SGD([parameter], lr=0.1, momentum=0.9)
    optimizer = Lookahead(inner, sync_period=2)
    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["optimizer"]["lr"] = 0.25
    malformed["slow_weights"][0] = np.array([np.inf])

    with pytest.raises(ValueError, match="must contain only finite values"):
        optimizer.load_state_dict(malformed)

    _assert_nested_equal(optimizer.state_dict(), before)


def test_pending_sync_round_trips_and_blocks_step_after_restore():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = None
    optimizer = Lookahead(SGD([parameter]), sync_period=1, alpha=0.5)
    parameter.data[...] = [3.0]
    parameter.data.setflags(write=False)
    with pytest.raises(ValueError):
        optimizer.step()
    state = optimizer.state_dict()

    parameter.data.setflags(write=True)
    optimizer.sync()
    assert optimizer.pending_sync is False

    optimizer.load_state_dict(state)
    assert optimizer.pending_sync is True
    with pytest.raises(RuntimeError, match="pending synchronization"):
        optimizer.step()


def test_dynamic_alpha_and_sync_period_are_validated():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lookahead(SGD([parameter]))

    optimizer.alpha = np.float32(0.25)
    optimizer.sync_period = np.int64(3)
    assert optimizer.alpha == 0.25
    assert optimizer.sync_period == 3

    for bad in (True, "0.5", 1 + 0j):
        with pytest.raises(TypeError):
            optimizer.alpha = bad
    for bad in (np.inf, -0.1, 1.1, 10**400):
        with pytest.raises(ValueError):
            optimizer.alpha = bad

    for bad in (True, 1.5, "2"):
        with pytest.raises(TypeError):
            optimizer.sync_period = bad
    for bad in (0, -1):
        with pytest.raises(ValueError):
            optimizer.sync_period = bad


def test_empty_built_in_optimizer_is_supported():
    inner = Adam([])
    optimizer = Lookahead(inner, sync_period=1)

    optimizer.step()

    assert inner.t == 1
    assert optimizer.step_count == 1
    assert optimizer.pending_sync is False
    assert optimizer.slow_weights() == ()


def test_constructor_rejects_unsupported_malformed_duplicate_and_nonfinite_parameters():
    class Unsupported:
        parameters = []

    with pytest.raises(TypeError, match="must be SGD, Adam, or AdamW"):
        Lookahead(Unsupported())

    parameter = Tensor([1.0], requires_grad=True)
    inner = SGD([parameter])
    inner.parameters.append(object())
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        Lookahead(inner)

    inner = SGD([parameter])
    inner.parameters.append(parameter)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        Lookahead(inner)

    nonfinite = Tensor([1.0], requires_grad=True)
    nonfinite.data[...] = [np.inf]
    with pytest.raises(ValueError, match="must contain only finite values"):
        Lookahead(SGD([nonfinite]))


def test_read_only_fast_parameter_needing_no_alpha_one_write_is_allowed():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = None
    optimizer = Lookahead(SGD([parameter]), sync_period=1, alpha=1.0)
    parameter.data.setflags(write=False)
    version = parameter._version

    optimizer.step()

    assert optimizer.pending_sync is False
    assert optimizer.step_count == 1
    assert parameter._version == version


def test_same_wrapper_serializes_overlapping_steps_across_threads():
    class BlockingSGD(SGD):
        def __init__(self, parameters):
            super().__init__(parameters, lr=0.1)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def step(self):
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
                assert self.release.wait(timeout=5.0)
            return super().step()

    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    inner = BlockingSGD([parameter])
    optimizer = Lookahead(inner, sync_period=100)
    errors = []

    def worker():
        try:
            optimizer.step()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    assert inner.entered.wait(timeout=5.0)
    second.start()

    # The second thread cannot enter the wrapped optimizer while the first owns
    # the Lookahead lock.
    assert inner.calls == 1
    inner.release.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert inner.calls == 2
    assert optimizer.step_count == 2
    np.testing.assert_allclose(parameter.data, [0.8], rtol=0.0, atol=1e-15)
