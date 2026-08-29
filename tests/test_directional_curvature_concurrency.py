import threading
import time

import numpy as np
import pytest

from engine.directional_curvature import directional_curvature
from engine.tensor import Tensor


def test_complete_probes_are_serialized_across_threads():
    parameter = Tensor([2.0], requires_grad=True)
    entered = threading.Event()
    release = threading.Event()
    contender_started = threading.Event()
    contender_callback = threading.Event()
    results = []
    failures = []
    first_calls = 0

    def first_loss():
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("timed out waiting for release")
        return parameter.data[0] ** 2

    def first_worker():
        try:
            results.append(
                ("first", directional_curvature(first_loss, parameter, np.array([1.0]), step=0.5))
            )
        except BaseException as exc:
            failures.append(exc)

    def second_worker():
        try:
            contender_started.set()

            def second_loss():
                contender_callback.set()
                return parameter.data[0] ** 2

            results.append(
                ("second", directional_curvature(second_loss, parameter, np.array([2.0]), step=0.25))
            )
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert contender_started.wait(timeout=5)
    time.sleep(0.05)
    assert contender_callback.is_set() is False

    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert contender_callback.is_set() is True
    assert [name for name, _ in results] == ["first", "second"]
    assert results[0][1]["curvature"] == pytest.approx(2.0)
    assert results[1][1]["curvature"] == pytest.approx(8.0)
    np.testing.assert_array_equal(parameter.data, [2.0])


def test_same_thread_nested_probe_on_disjoint_parameter_is_reentrant():
    outer = Tensor([2.0], requires_grad=True)
    inner = Tensor([3.0], requires_grad=True)
    nested_reports = []
    outer_calls = 0

    def outer_loss():
        nonlocal outer_calls
        outer_calls += 1
        if outer_calls == 1:
            nested_reports.append(
                directional_curvature(
                    lambda: inner.data[0] ** 2,
                    inner,
                    np.array([1.0]),
                    step=0.5,
                )
            )
        return outer.data[0] ** 2

    report = directional_curvature(
        outer_loss,
        outer,
        np.array([1.0]),
        step=0.5,
    )

    assert report["curvature"] == pytest.approx(2.0)
    assert nested_reports[0]["curvature"] == pytest.approx(2.0)
    np.testing.assert_array_equal(outer.data, [2.0])
    np.testing.assert_array_equal(inner.data, [3.0])


def test_same_parameter_nested_probe_fails_loudly_instead_of_hiding_version_change():
    parameter = Tensor([2.0], requires_grad=True)
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls == 1:
            directional_curvature(
                lambda: parameter.data[0] ** 2,
                parameter,
                np.array([1.0]),
                step=0.25,
            )
        return parameter.data[0] ** 2

    with pytest.raises(RuntimeError, match="version"):
        directional_curvature(loss, parameter, np.array([1.0]), step=0.5)
    np.testing.assert_array_equal(parameter.data, [2.0])
