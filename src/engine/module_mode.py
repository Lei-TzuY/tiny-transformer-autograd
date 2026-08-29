"""Exception-safe temporary mode contexts for ``nn.Module`` trees.

``evaluating(module)`` and ``training(module)`` temporarily place the module tree in
one mode and then restore the exact per-module training-mode state that existed on
entry. ``inference(module)`` composes evaluation mode with ``no_grad()`` for callers
that want both inference behaviors in one explicit scope.
"""

from contextlib import contextmanager
import threading

from engine.grad_mode import no_grad
from nn.module import Module


_MODE_CONTEXT_CONDITION = threading.Condition(threading.RLock())
_ACTIVE_MODE_MODULES = {}
_MISSING = object()


def _validate_module(module, helper_name):
    if not isinstance(module, Module):
        raise TypeError(f"{helper_name} module must be an nn.Module")


def _restore_mode_states(states):
    for child, state in states:
        namespace = vars(child)
        if state is _MISSING:
            namespace.pop("training", None)
        else:
            # Restore the exact snapshotted local state without invoking a custom
            # ``__setattr__`` that may itself reject the rollback operation.
            namespace["training"] = state


def _tree_is_available(markers, owner):
    return all(
        marker not in _ACTIVE_MODE_MODULES
        or _ACTIVE_MODE_MODULES[marker][0] == owner
        for marker in markers
    )


def _owner_has_reservation(owner):
    return any(active_owner == owner for active_owner, _ in _ACTIVE_MODE_MODULES.values())


@contextmanager
def _reserve_mode_tree(modules):
    """Reserve one entry module tree while allowing disjoint trees to proceed."""
    owner = threading.get_ident()
    markers = tuple(id(module) for module in modules)

    with _MODE_CONTEXT_CONDITION:
        while not _tree_is_available(markers, owner):
            # Waiting while already holding another tree permits the classic AB/BA
            # deadlock: two threads can each hold one tree and nest into the other.
            # Top-level callers may wait, but nested cross-owner acquisition fails
            # explicitly so the outer reservation can unwind and release its tree.
            if _owner_has_reservation(owner):
                raise RuntimeError(
                    "nested module mode context cannot wait for another thread"
                )
            _MODE_CONTEXT_CONDITION.wait()

        for marker in markers:
            active = _ACTIVE_MODE_MODULES.get(marker)
            if active is None:
                _ACTIVE_MODE_MODULES[marker] = (owner, 1)
            else:
                _ACTIVE_MODE_MODULES[marker] = (owner, active[1] + 1)

    try:
        yield
    finally:
        with _MODE_CONTEXT_CONDITION:
            for marker in markers:
                active_owner, count = _ACTIVE_MODE_MODULES[marker]
                if active_owner != owner:
                    raise RuntimeError("module mode reservation ownership changed")
                if count == 1:
                    del _ACTIVE_MODE_MODULES[marker]
                else:
                    _ACTIVE_MODE_MODULES[marker] = (owner, count - 1)
            _MODE_CONTEXT_CONDITION.notify_all()


@contextmanager
def _temporary_mode(module, mode):
    modules = tuple(module.modules())
    with _reserve_mode_tree(modules):
        states = tuple(
            (child, vars(child).get("training", _MISSING)) for child in modules
        )

        # Installation is part of the transaction too. A custom ``train()`` may
        # mutate part of a tree and then raise before the context body is entered.
        try:
            module.train(mode)
            yield module
        finally:
            _restore_mode_states(states)


@contextmanager
def evaluating(module):
    """Temporarily put ``module`` and its current descendants in evaluation mode.

    Existing child modules may have different training flags. The context snapshots
    those flags individually, calls the normal recursive ``eval()`` path, and restores
    the exact entry state in ``finally``. A module that did not have a local
    ``training`` attribute before entry has that temporary attribute removed again.

    Helper-managed contexts that share any entry Module are serialized because their
    restores could otherwise run out of order. Contexts over disjoint module trees may
    run concurrently. Direct ``train()``/``eval()`` calls made outside these helpers are
    not synchronized.
    """
    _validate_module(module, "evaluating")
    with _temporary_mode(module, False) as active:
        yield active


@contextmanager
def training(module):
    """Temporarily put ``module`` and its current descendants in training mode.

    The complete entry mode of every existing module is restored on exit, including
    mixed train/eval subtrees and modules that originally lacked a local ``training``
    attribute. This is useful for controlled training-mode forwards without making a
    permanent recursive ``train()`` change.
    """
    _validate_module(module, "training")
    with _temporary_mode(module, True) as active:
        yield active


@contextmanager
def inference(module):
    """Temporarily enable evaluation mode and disable autograd graph recording.

    Module training flags are restored with the same exact per-module semantics as
    ``evaluating()``. The ``no_grad()`` scope covers mode installation as well as the
    caller body, so custom ``train(False)`` hooks cannot accidentally record a graph
    while entering inference. Grad mode is restored exactly even when setup or the
    body raises.
    """
    _validate_module(module, "inference")
    # Enter no-grad before installing evaluation mode. Custom ``train()`` overrides
    # may compute/cache Tensors, and inference should not record those setup graphs.
    with no_grad():
        with _temporary_mode(module, False) as active:
            yield active
