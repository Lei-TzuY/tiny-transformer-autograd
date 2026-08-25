"""Regression tests for callable-object grad-mode decorators."""

import functools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.grad_mode import is_grad_enabled, no_grad, set_grad_enabled


class _AsyncCallable:
    async def __call__(self):
        return is_grad_enabled()


class _GeneratorCallable:
    def __call__(self):
        yield is_grad_enabled()


class _AsyncGeneratorCallable:
    async def __call__(self):
        yield is_grad_enabled()


class _SyncCallable:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return is_grad_enabled()


@pytest.mark.parametrize(
    "function",
    [_AsyncCallable(), _GeneratorCallable(), _AsyncGeneratorCallable()],
)
def test_no_grad_rejects_deferred_callable_objects(function):
    assert is_grad_enabled()

    with pytest.raises(TypeError, match="only support synchronous"):
        no_grad()(function)

    assert is_grad_enabled()


@pytest.mark.parametrize(
    "function",
    [
        functools.partial(_AsyncCallable()),
        functools.partial(functools.partial(_GeneratorCallable())),
        functools.partial(_AsyncGeneratorCallable()),
    ],
)
def test_no_grad_rejects_partial_wrapped_deferred_callables(function):
    with pytest.raises(TypeError, match="only support synchronous"):
        no_grad()(function)

    assert is_grad_enabled()


def test_synchronous_callable_object_remains_supported():
    function = _SyncCallable()
    wrapped = no_grad()(function)

    assert wrapped() is False
    assert function.calls == 1
    assert is_grad_enabled()


def test_callable_object_respects_enable_mode_too():
    function = _SyncCallable()
    wrapped = set_grad_enabled(True)(function)

    with no_grad():
        assert wrapped() is True
        assert not is_grad_enabled()

    assert function.calls == 1
    assert is_grad_enabled()


@pytest.mark.parametrize("value", [None, 1, object(), "not-callable"])
def test_decorator_rejects_non_callables_at_decoration_time(value):
    assert is_grad_enabled()

    with pytest.raises(TypeError, match="require a callable"):
        no_grad()(value)

    assert is_grad_enabled()
