"""Grad-mode decorators must reject every deferred awaitable result."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.grad_mode import enable_grad, is_grad_enabled, no_grad


class _CustomAwaitable:
    """Awaitable whose deferred body is observable without creating a coroutine."""

    def __init__(self):
        self.started = False

    def __await__(self):
        self.started = True
        if False:
            yield None
        return "done"


class _OrdinaryIterable:
    def __iter__(self):
        return iter((1, 2, 3))


def test_no_grad_decorator_rejects_custom_awaitable_result_before_scope_exit():
    deferred = _CustomAwaitable()

    @no_grad()
    def forwarder():
        assert not is_grad_enabled()
        return deferred

    with pytest.raises(TypeError, match="only support synchronous"):
        forwarder()

    assert deferred.started is False
    assert is_grad_enabled()


def test_enable_grad_decorator_restores_outer_disabled_mode_after_rejection():
    deferred = _CustomAwaitable()

    @enable_grad()
    def forwarder():
        assert is_grad_enabled()
        return deferred

    with no_grad():
        assert not is_grad_enabled()
        with pytest.raises(TypeError, match="only support synchronous"):
            forwarder()
        assert not is_grad_enabled()

    assert deferred.started is False
    assert is_grad_enabled()


def test_ordinary_iterable_result_remains_supported():
    value = _OrdinaryIterable()

    @no_grad()
    def forwarder():
        return value

    assert forwarder() is value
    assert list(value) == [1, 2, 3]
    assert is_grad_enabled()
