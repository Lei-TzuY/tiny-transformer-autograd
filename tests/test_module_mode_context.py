"""Temporary module-mode contexts must restore exact per-module state."""

import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.grad_mode import enable_grad, is_grad_enabled, no_grad
from engine.module_mode import evaluating, inference, training
from engine.tensor import Tensor
from nn.module import Module


class Leaf(Module):
    def __init__(self, training=None):
        if training is not None:
            self.training = training


class Tree(Module):
    def __init__(self):
        self.left = Leaf(True)
        self.right = Leaf(False)


class PartiallyFailingTree(Tree):
    def train(self, mode=True):
        super().train(mode)
        raise RuntimeError("mode install failed")


class GuardedLeaf(Leaf):
    def __init__(self, training):
        self.reject_training_writes = False
        super().__init__(training)

    def __setattr__(self, name, value):
        if name == "training" and vars(self).get("reject_training_writes", False):
            raise RuntimeError("training write rejected")
        super().__setattr__(name, value)


def _assert_rng_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_evaluating_sets_all_existing_modules_false_and_restores_mixed_state():
    model = Tree()
    assert "training" not in vars(model)

    with evaluating(model) as returned:
        assert returned is model
        assert model.training is False
        assert model.left.training is False
        assert model.right.training is False

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_training_sets_all_existing_modules_true_and_restores_mixed_state():
    model = Tree()
    assert "training" not in vars(model)

    with training(model) as returned:
        assert returned is model
        assert model.training is True
        assert model.left.training is True
        assert model.right.training is True

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_evaluating_restores_after_body_exception():
    model = Tree()

    with pytest.raises(RuntimeError, match="boom"):
        with evaluating(model):
            model.train(True)
            raise RuntimeError("boom")

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_training_restores_after_body_exception():
    model = Tree()

    with pytest.raises(RuntimeError, match="boom"):
        with training(model):
            model.eval()
            raise RuntimeError("boom")

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


@pytest.mark.parametrize("helper", [evaluating, training, inference])
def test_mode_helpers_restore_when_recursive_mode_installation_raises(helper):
    model = PartiallyFailingTree()
    np.random.seed(2468)
    before_rng = np.random.get_state()
    assert is_grad_enabled()

    with pytest.raises(RuntimeError, match="mode install failed"):
        with helper(model):
            raise AssertionError("context body must not run")

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False
    assert is_grad_enabled()
    _assert_rng_equal(before_rng, np.random.get_state())


def test_mode_restore_bypasses_custom_training_setattr_after_entry():
    model = Module()
    model.child = GuardedLeaf(True)

    with evaluating(model):
        assert model.child.training is False
        model.child.reject_training_writes = True

    assert "training" not in vars(model)
    assert model.child.training is True
    assert model.child.reject_training_writes is True


def test_evaluating_is_reentrant_and_preserves_outer_eval_state():
    model = Tree()

    with evaluating(model):
        assert model.left.training is False
        with evaluating(model):
            model.train(True)
            assert model.left.training is True
        assert model.training is False
        assert model.left.training is False
        assert model.right.training is False

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_training_and_evaluating_nest_and_restore_each_scope():
    model = Tree()

    with training(model):
        assert model.training is True
        assert model.left.training is True
        assert model.right.training is True
        with evaluating(model):
            assert model.training is False
            assert model.left.training is False
            assert model.right.training is False
        assert model.training is True
        assert model.left.training is True
        assert model.right.training is True

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        (evaluating, "evaluating module must be an nn.Module"),
        (training, "training module must be an nn.Module"),
        (inference, "inference module must be an nn.Module"),
    ],
)
def test_mode_helpers_reject_non_module_before_touching_rng(helper, message):
    np.random.seed(123)
    before = np.random.get_state()

    with pytest.raises(TypeError, match=message):
        with helper(object()):
            raise AssertionError("unreachable")

    _assert_rng_equal(before, np.random.get_state())


@pytest.mark.parametrize("helper", [evaluating, training, inference])
def test_mode_helpers_do_not_touch_numpy_rng(helper):
    model = Tree()
    np.random.seed(456)
    before = np.random.get_state()

    with helper(model):
        pass

    _assert_rng_equal(before, np.random.get_state())


def test_overlapping_mixed_helper_contexts_are_serialized():
    model = Tree()
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
    second_saw_training = []
    errors = []

    def first():
        try:
            with evaluating(model):
                entered_first.set()
                release_first.wait(timeout=2.0)
        except BaseException as exc:  # pragma: no cover - failure reporting path
            errors.append(exc)

    def second():
        try:
            entered_first.wait(timeout=2.0)
            with training(model):
                second_saw_training.append(model.right.training)
                entered_second.set()
        except BaseException as exc:  # pragma: no cover - failure reporting path
            errors.append(exc)

    thread_a = threading.Thread(target=first)
    thread_b = threading.Thread(target=second)
    thread_a.start()
    thread_b.start()

    assert entered_first.wait(timeout=2.0)
    time.sleep(0.05)
    assert not entered_second.is_set()

    release_first.set()
    thread_a.join(timeout=2.0)
    thread_b.join(timeout=2.0)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert not errors
    assert entered_second.is_set()
    assert second_saw_training == [True]
    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_inference_combines_eval_mode_and_no_grad_then_restores_both():
    model = Tree()
    x = Tensor([2.0], requires_grad=True)
    assert is_grad_enabled()

    with inference(model) as returned:
        assert returned is model
        assert model.training is False
        assert model.left.training is False
        assert model.right.training is False
        assert not is_grad_enabled()

        y = x * 3.0
        assert y.requires_grad is False
        assert not y._children

    assert is_grad_enabled()
    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_inference_restores_outer_disabled_grad_mode_after_exception():
    model = Tree()

    with no_grad():
        assert not is_grad_enabled()
        with pytest.raises(RuntimeError, match="boom"):
            with inference(model):
                assert not is_grad_enabled()
                model.train(True)
                raise RuntimeError("boom")
        assert not is_grad_enabled()

    assert is_grad_enabled()
    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


def test_inference_allows_explicit_inner_enable_grad_without_leaking_it():
    model = Tree()
    x = Tensor([2.0], requires_grad=True)

    with inference(model):
        assert not is_grad_enabled()
        with enable_grad():
            assert is_grad_enabled()
            y = x * 3.0
            assert y.requires_grad is True
        assert not is_grad_enabled()

    assert is_grad_enabled()
