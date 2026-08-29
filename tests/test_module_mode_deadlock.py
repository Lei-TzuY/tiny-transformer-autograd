"""Mode-tree reservations must fail closed instead of forming nested deadlocks."""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.module_mode import evaluating, training
from nn.module import Module


class Leaf(Module):
    def __init__(self, training):
        self.training = training


class Tree(Module):
    def __init__(self):
        self.left = Leaf(True)
        self.right = Leaf(False)


def test_nested_cross_thread_reservation_fails_instead_of_waiting():
    first_model = Tree()
    second_model = Tree()
    worker_has_first = threading.Event()
    attempt_nested = threading.Event()
    nested_finished = threading.Event()
    errors = []

    def worker():
        try:
            with evaluating(first_model):
                worker_has_first.set()
                assert attempt_nested.wait(timeout=2.0)
                with pytest.raises(
                    RuntimeError,
                    match="nested module mode context cannot wait for another thread",
                ):
                    with evaluating(second_model):
                        raise AssertionError("blocked nested context must not enter")
                nested_finished.set()
        except BaseException as exc:  # pragma: no cover - failure reporting path
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_has_first.wait(timeout=2.0)

    # The main thread owns the second tree while the worker already owns the first.
    # Waiting here would permit an AB/BA deadlock if this thread later nested into the
    # first tree. The nested worker request must therefore fail without blocking.
    with training(second_model):
        attempt_nested.set()
        assert nested_finished.wait(timeout=2.0)
        assert second_model.left.training is True
        assert second_model.right.training is True

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not errors

    assert "training" not in vars(first_model)
    assert first_model.left.training is True
    assert first_model.right.training is False
    assert "training" not in vars(second_model)
    assert second_model.left.training is True
    assert second_model.right.training is False
