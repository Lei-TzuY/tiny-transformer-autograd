"""
test_grad_mode.py — no_grad / enable_grad graph-suppression semantics.

Covers what the flag must and must not do:
  - op results created while recording is off are detached (no parents, no
    gradient buffer, no backward closure) and drop their intermediates
  - explicitly created leaves keep requires_grad, so building a model inside
    no_grad() does not silently freeze it
  - blocks nest, restore on exception, work as decorators, and are per-thread
  - inference values are identical with and without recording
"""

import gc
import os
import sys
import threading
import weakref

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.grad_mode import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from engine.tensor import Tensor, _no_backward
from nn.transformer import GPT


def _model():
    return GPT(
        vocab_size=8,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )


class TestGraphSuppression:
    def test_recording_is_enabled_by_default(self):
        assert is_grad_enabled()
        x = Tensor([1.0, 2.0], requires_grad=True)
        out = x * x
        assert out.requires_grad
        assert out._children == {x}
        assert out._backward is not _no_backward

    def test_no_grad_detaches_op_results(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            out = ops.sum(x * x)

        assert not out.requires_grad
        assert out.grad is None
        assert out._children == set()
        assert out._backward is _no_backward
        np.testing.assert_allclose(out.data, 5.0)

    def test_no_grad_leaves_existing_tensors_untouched(self):
        x = Tensor([1.0], requires_grad=True)
        with no_grad():
            pass
        assert x.requires_grad
        assert x.grad is not None

    def test_no_grad_preserves_explicit_leaf_requires_grad(self):
        with no_grad():
            leaf = Tensor([1.0, 2.0], requires_grad=True)
            model = _model()

        assert leaf.requires_grad
        assert model.param_count() > 0
        # A model built under no_grad is still trainable afterwards.
        idx = np.zeros((1, 3), dtype=np.int64)
        loss = ops.cross_entropy(model(idx), idx)
        loss.backward()
        assert any(np.any(p.grad != 0) for p in model.parameters())

    def test_no_grad_releases_forward_intermediates(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            intermediate = x * x
            reference = weakref.ref(intermediate)
            out = intermediate + x
            del intermediate
            gc.collect()
            assert reference() is None, "detached result still holds its parent"
        assert not out.requires_grad

        # With recording on, the same intermediate must stay reachable.
        kept = x * x
        kept_reference = weakref.ref(kept)
        out = kept + x
        del kept
        gc.collect()
        assert kept_reference() is not None
        assert out._children

    def test_gradients_are_not_accumulated_under_no_grad(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        with no_grad():
            ops.sum(x * x)
        np.testing.assert_array_equal(x.grad, np.zeros(2))

    def test_detached_result_can_seed_a_new_graph(self):
        x = Tensor([2.0], requires_grad=True)
        with no_grad():
            constant = x * x
        assert not constant.requires_grad

        out = x * constant  # 2 * 4 = 8, constant contributes no gradient
        out.backward()
        np.testing.assert_allclose(out.data, [8.0])
        np.testing.assert_allclose(x.grad, [4.0])


class TestScopeBehaviour:
    def test_enable_grad_reenables_inside_no_grad(self):
        x = Tensor([1.0], requires_grad=True)
        with no_grad():
            assert not is_grad_enabled()
            with enable_grad():
                assert is_grad_enabled()
                tracked = x * x
            assert not is_grad_enabled()
        assert is_grad_enabled()
        assert tracked.requires_grad

    def test_nested_no_grad_restores_outer_mode(self):
        with no_grad():
            with no_grad():
                assert not is_grad_enabled()
            assert not is_grad_enabled()
        assert is_grad_enabled()

    def test_reused_instance_is_reentrant(self):
        guard = no_grad()
        with guard:
            with guard:
                assert not is_grad_enabled()
            assert not is_grad_enabled()
        assert is_grad_enabled()

    def test_mode_is_restored_after_exception(self):
        with pytest.raises(RuntimeError):
            with no_grad():
                raise RuntimeError("boom")
        assert is_grad_enabled()

    def test_decorator_form_disables_recording(self):
        @no_grad()
        def detached_square(tensor):
            assert not is_grad_enabled()
            return tensor * tensor

        x = Tensor([3.0], requires_grad=True)
        out = detached_square(x)
        assert not out.requires_grad
        assert is_grad_enabled()

    def test_decorator_rejects_deferred_function_bodies(self):
        async def coroutine_function():
            return None

        def generator_function():
            yield None

        async def async_generator_function():
            yield None

        for function in (
            coroutine_function,
            generator_function,
            async_generator_function,
        ):
            with pytest.raises(TypeError, match="only support synchronous"):
                no_grad()(function)

    def test_set_grad_enabled_accepts_both_modes(self):
        with set_grad_enabled(False):
            assert not is_grad_enabled()
            with set_grad_enabled(True):
                assert is_grad_enabled()
        assert is_grad_enabled()

    def test_set_grad_enabled_rejects_non_boolean(self):
        with pytest.raises(TypeError, match="bool"):
            set_grad_enabled(0)

    def test_mode_is_thread_local(self):
        observed = {}

        def worker():
            observed["enabled"] = is_grad_enabled()

        with no_grad():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert not is_grad_enabled()

        assert observed["enabled"] is True

    def test_reused_guard_restores_each_threads_own_mode(self):
        guard = set_grad_enabled(True)
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_exited = threading.Event()
        observed = {}
        errors = []

        def first_worker():
            try:
                with no_grad():
                    observed["first_before"] = is_grad_enabled()
                    with guard:
                        observed["first_inside"] = is_grad_enabled()
                        first_entered.set()
                        assert second_entered.wait(2.0)
                    observed["first_after"] = is_grad_enabled()
                    first_exited.set()
            except BaseException as error:
                errors.append(error)
                first_entered.set()
                first_exited.set()

        def second_worker():
            try:
                assert first_entered.wait(2.0)
                observed["second_before"] = is_grad_enabled()
                with guard:
                    observed["second_inside"] = is_grad_enabled()
                    second_entered.set()
                    assert first_exited.wait(2.0)
                observed["second_after"] = is_grad_enabled()
            except BaseException as error:
                errors.append(error)
                second_entered.set()

        first = threading.Thread(target=first_worker)
        second = threading.Thread(target=second_worker)
        first.start()
        second.start()
        first.join(3.0)
        second.join(3.0)

        assert not first.is_alive() and not second.is_alive()
        assert errors == []
        assert observed == {
            "first_before": False,
            "first_inside": True,
            "first_after": False,
            "second_before": True,
            "second_inside": True,
            "second_after": True,
        }


class TestBackwardGuard:
    def test_backward_on_detached_tensor_under_no_grad_raises(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            loss = ops.sum(x * x)
            with pytest.raises(RuntimeError, match="no_grad"):
                loss.backward()

    def test_backward_on_suppressed_tensor_raises_after_leaving_no_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            loss = ops.sum(x * x)

        with pytest.raises(RuntimeError, match="detached by no_grad"):
            loss.backward()

    def test_suppression_provenance_survives_a_detached_chain(self):
        x = Tensor([2.0], requires_grad=True)
        with no_grad():
            detached = x * x
        derived = detached + 1.0

        with pytest.raises(RuntimeError, match="detached by no_grad"):
            derived.backward()

    def test_backward_of_outer_graph_is_allowed_inside_no_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        loss = ops.sum(x * x)
        with no_grad():
            loss.backward()
        np.testing.assert_allclose(x.grad, [2.0, 4.0])

    def test_constant_backward_outside_no_grad_stays_a_no_op(self):
        constant = Tensor([1.0, 2.0])
        constant.backward()
        assert constant.grad is None

    def test_explicit_constant_backward_inside_no_grad_stays_a_no_op(self):
        constant = Tensor([1.0, 2.0])
        with no_grad():
            constant.backward()
        assert constant.grad is None

    def test_constant_only_op_backward_inside_no_grad_stays_a_no_op(self):
        with no_grad():
            constant = Tensor([2.0]) * Tensor([3.0])
            constant.backward()

        assert constant.grad is None
        assert constant._children == set()


class TestGraphPruning:
    def test_constant_only_result_does_not_retain_its_parents(self):
        parent = Tensor([2.0])
        reference = weakref.ref(parent)
        result = parent * parent
        del parent
        gc.collect()

        assert not result.requires_grad
        assert result._children == set()
        assert result._backward is _no_backward
        assert reference() is None

    def test_frozen_branch_value_can_feed_a_trainable_graph(self):
        frozen = Tensor([2.0])
        frozen_branch = frozen * frozen
        trainable = Tensor([3.0], requires_grad=True)

        loss = trainable * frozen_branch
        loss.backward()

        assert frozen_branch._children == set()
        assert frozen.grad is None
        np.testing.assert_allclose(trainable.grad, [4.0])


class TestModelIntegration:
    def test_forward_values_match_with_and_without_recording(self):
        model = _model().eval()
        idx = np.array([[1, 2, 3, 4]], dtype=np.int64) % 8

        tracked = model(idx)
        with no_grad():
            detached = model(idx)

        np.testing.assert_array_equal(tracked.data, detached.data)
        assert tracked.requires_grad
        assert not detached.requires_grad

    def test_no_grad_forward_leaves_parameter_gradients_untouched(self):
        model = _model().eval()
        model.zero_grad()
        idx = np.array([[0, 1, 2, 3]], dtype=np.int64)

        with no_grad():
            logits = model(idx)
            loss = ops.cross_entropy(logits, idx)

        assert not loss.requires_grad
        for parameter in model.parameters():
            np.testing.assert_array_equal(
                parameter.grad, np.zeros_like(parameter.data)
            )

    def test_training_evaluate_runs_without_building_a_graph(self):
        import train

        model = _model()
        data = np.arange(40, dtype=np.int64) % 8
        model.zero_grad()

        result = train.evaluate(model, data, model.context_len, 2, 3)

        assert result is not None
        mean_loss, perplexity = result
        assert np.isfinite(mean_loss) and perplexity > 0
        assert model.training is True
        for parameter in model.parameters():
            np.testing.assert_array_equal(
                parameter.grad, np.zeros_like(parameter.data)
            )
