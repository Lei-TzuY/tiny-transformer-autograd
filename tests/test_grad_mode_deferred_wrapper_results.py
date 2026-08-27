"""Grad-mode decorators must not leak through hidden deferred call results."""

import functools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.grad_mode import is_grad_enabled, no_grad, set_grad_enabled


def _opaque_forwarder(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _metadata_forwarder(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def _generator_factory(events):
    def deferred():
        events.append(is_grad_enabled())
        yield "generator body ran"

    return deferred


def _coroutine_factory(events):
    async def deferred():
        events.append(is_grad_enabled())
        return "coroutine body ran"

    return deferred


def _async_generator_factory(events):
    async def deferred():
        events.append(is_grad_enabled())
        yield "async generator body ran"

    return deferred


@pytest.mark.parametrize("forwarder", [_opaque_forwarder, _metadata_forwarder])
@pytest.mark.parametrize(
    "factory",
    [_generator_factory, _coroutine_factory, _async_generator_factory],
)
def test_no_grad_rejects_deferred_results_hidden_by_sync_wrapper(forwarder, factory):
    events = []
    hidden = forwarder(factory(events))
    wrapped = no_grad()(hidden)

    with pytest.raises(TypeError, match="only support synchronous"):
        wrapped()

    # Creating a generator/coroutine/async-generator object does not execute its
    # body. Rejection must happen before a caller can resume it under the
    # restored outer mode.
    assert events == []
    assert is_grad_enabled()


def test_runtime_rejection_restores_surrounding_grad_mode():
    events = []
    wrapped = set_grad_enabled(True)(_opaque_forwarder(_generator_factory(events)))

    with no_grad():
        assert not is_grad_enabled()
        with pytest.raises(TypeError, match="only support synchronous"):
            wrapped()
        assert not is_grad_enabled()

    assert events == []
    assert is_grad_enabled()


def test_generator_metadata_does_not_reject_an_eager_sync_wrapper():
    def generator():
        yield is_grad_enabled()

    @functools.wraps(generator)
    def consume_eagerly():
        return list(generator())

    wrapped = no_grad()(consume_eagerly)

    assert wrapped() == [False]
    assert is_grad_enabled()
