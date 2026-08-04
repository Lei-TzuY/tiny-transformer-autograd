"""
grad_mode.py — Global switch that turns computational-graph recording off.

Why this exists
---------------
Every op in ``ops.py`` builds a node: it stores the parents in ``_children``
and keeps a ``_backward`` closure alive.  That closure captures the forward
intermediates it needs (``softmax``'s probabilities, ``sigmoid``'s output, …),
so a forward pass costs roughly as much memory as the graph it records.

During validation, benchmarking, or generation no one will ever call
``backward()``, so all of that bookkeeping is pure waste.  Wrapping the work in
``no_grad()`` makes each op return a plain, detached result:

    with no_grad():
        logits = model(x)            # no graph, no closures, no .grad buffers

Semantics (deliberately the same as PyTorch's ``torch.no_grad``)
---------------------------------------------------------------
- Op *outputs* created while recording is off have ``requires_grad=False``,
  no ``_children``, and no backward closure.
- Explicitly created leaves are untouched: ``Tensor(x, requires_grad=True)``
  inside a ``no_grad()`` block is still trainable, so constructing a model
  under ``no_grad()`` does not silently freeze it.
- The flag never changes an existing tensor; it only affects tensors created
  while it is off.

The state is thread-local, so a worker thread cannot disable recording for the
thread that is training.  Blocks nest, and the previous mode is restored even
if the body raises.
"""

import functools
import inspect
import threading


class _GradState(threading.local):
    """Per-thread recording flag (defaults to enabled in every new thread)."""

    def __init__(self):
        self.enabled = True


_state = _GradState()


class _ModeStack(threading.local):
    """Per-thread entry stack for a reusable grad-mode guard."""

    def __init__(self):
        self.values = []


def is_grad_enabled() -> bool:
    """Return True when ops record a computational graph in this thread."""
    return _state.enabled


class set_grad_enabled:
    """
    Context manager and decorator that sets graph recording for a scope.

    Usable in either form::

        with set_grad_enabled(False):
            ...

        @set_grad_enabled(False)
        def evaluate(...):
            ...

    Instances are reentrant: the mode in effect on entry is pushed onto a
    stack, so the same object can be nested or used recursively.
    """

    def __init__(self, mode: bool):
        if not isinstance(mode, bool):
            raise TypeError("grad mode must be a bool")
        self.mode = mode
        # A guard may be shared by worker threads. The recording flag itself is
        # thread-local, so its restoration stack must be thread-local too.
        self._previous = _ModeStack()

    def __enter__(self):
        self._previous.values.append(_state.enabled)
        _state.enabled = self.mode
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _state.enabled = self._previous.values.pop()
        return False

    def __call__(self, function):
        if (
            inspect.iscoroutinefunction(function)
            or inspect.isgeneratorfunction(function)
            or inspect.isasyncgenfunction(function)
        ):
            raise TypeError(
                "grad mode decorators only support synchronous, "
                "non-generator functions"
            )
        mode = self.mode

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # A fresh guard per call keeps recursive calls independent of the
            # decorated instance (whose subclasses take no constructor args).
            with set_grad_enabled(mode):
                return function(*args, **kwargs)

        return wrapper


class no_grad(set_grad_enabled):
    """Disable graph recording inside the block (``with no_grad(): ...``)."""

    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    """Re-enable graph recording inside a surrounding ``no_grad()`` block."""

    def __init__(self):
        super().__init__(True)
