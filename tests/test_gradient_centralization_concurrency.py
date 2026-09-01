"""Concurrency regressions for gradient centralization transactions."""

import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_centralization import centralize_gradients_
from engine.tensor import Tensor


class _BlockingWrite(np.ndarray):
    def __new__(cls, values, entered, release):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._entered = entered
        array._release = release
        return array

    def __array_finalize__(self, obj):
        self._entered = getattr(obj, "_entered", None)
        self._release = getattr(obj, "_release", None)

    def __setitem__(self, key, value):
        self._entered.set()
        if not self._release.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release gradient write")
        np.ndarray.__setitem__(self, key, value)


class _SignalWrite(np.ndarray):
    def __new__(cls, values, entered):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._entered = entered
        return array

    def __array_finalize__(self, obj):
        self._entered = getattr(obj, "_entered", None)

    def __setitem__(self, key, value):
        self._entered.set()
        np.ndarray.__setitem__(self, key, value)


def _run_centralization(parameter, errors):
    try:
        centralize_gradients_([parameter])
    except BaseException as exc:
        errors.append(exc)


def test_independent_calls_do_not_overlap_transaction_commits():
    first_entered = threading.Event()
    first_release = threading.Event()
    second_entered = threading.Event()
    errors = []

    first = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    first.grad = _BlockingWrite([[1.0, 3.0]], first_entered, first_release)
    second = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    second.grad = _SignalWrite([[2.0, 6.0]], second_entered)

    first_thread = threading.Thread(
        target=_run_centralization, args=(first, errors), daemon=True
    )
    second_thread = threading.Thread(
        target=_run_centralization, args=(second, errors), daemon=True
    )

    first_thread.start()
    assert first_entered.wait(timeout=2.0)
    second_thread.start()
    try:
        assert not second_entered.wait(timeout=0.2)
    finally:
        first_release.set()

    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert second_entered.is_set()
    np.testing.assert_array_equal(first.grad, [[-1.0, 1.0]])
    np.testing.assert_array_equal(second.grad, [[-2.0, 2.0]])
