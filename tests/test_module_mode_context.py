"""Temporary evaluation mode must restore exact per-module state."""

import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.module_mode import evaluating
from nn.module import Module


class Leaf(Module):
    def __init__(self, training=None):
        if training is not None:
            self.training = training


class Tree(Module):
    def __init__(self):
        self.left = Leaf(True)
        self.right = Leaf(False)


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


def test_evaluating_restores_after_body_exception():
    model = Tree()

    with pytest.raises(RuntimeError, match="boom"):
        with evaluating(model):
            model.train(True)
            raise RuntimeError("boom")

    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False


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


def test_evaluating_rejects_non_module_before_touching_rng():
    np.random.seed(123)
    before = np.random.get_state()

    with pytest.raises(TypeError, match="evaluating module must be an nn.Module"):
        with evaluating(object()):
            raise AssertionError("unreachable")

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_evaluating_does_not_touch_numpy_rng():
    model = Tree()
    np.random.seed(456)
    before = np.random.get_state()

    with evaluating(model):
        pass

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_overlapping_helper_contexts_are_serialized():
    model = Tree()
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
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
            with evaluating(model):
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
    assert "training" not in vars(model)
    assert model.left.training is True
    assert model.right.training is False
