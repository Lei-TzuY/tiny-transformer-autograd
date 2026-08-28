"""Exception-safe temporary mode contexts for ``nn.Module`` trees.

``evaluating(module)`` and ``training(module)`` temporarily place the module tree in
one mode and then restore the exact per-module training-mode state that existed on
entry.  These helpers are intentionally separate from autograd grad mode: module mode
controls behavior such as dropout, while ``no_grad()`` controls graph recording.
"""

from contextlib import contextmanager
import threading

from nn.module import Module


_MODE_CONTEXT_LOCK = threading.RLock()
_MISSING = object()


def _validate_module(module, helper_name):
    if not isinstance(module, Module):
        raise TypeError(f"{helper_name} module must be an nn.Module")


@contextmanager
def _temporary_mode(module, mode):
    with _MODE_CONTEXT_LOCK:
        modules = tuple(module.modules())
        states = tuple(
            (child, vars(child).get("training", _MISSING)) for child in modules
        )

        module.train(mode)
        try:
            yield module
        finally:
            for child, state in states:
                if state is _MISSING:
                    vars(child).pop("training", None)
                else:
                    child.training = state


@contextmanager
def evaluating(module):
    """Temporarily put ``module`` and its current descendants in evaluation mode.

    Existing child modules may have different training flags.  The context snapshots
    those flags individually, calls the normal recursive ``eval()`` path, and restores
    the exact entry state in ``finally``.  A module that did not have a local
    ``training`` attribute before entry has that temporary attribute removed again.

    Helper-managed contexts are serialized because overlapping contexts can otherwise
    restore shared module state out of order.  Direct ``train()``/``eval()`` calls made
    outside these helpers are not synchronized.
    """
    _validate_module(module, "evaluating")
    with _temporary_mode(module, False) as active:
        yield active


@contextmanager
def training(module):
    """Temporarily put ``module`` and its current descendants in training mode.

    The complete entry mode of every existing module is restored on exit, including
    mixed train/eval subtrees and modules that originally lacked a local ``training``
    attribute.  This is useful for controlled training-mode forwards without making a
    permanent recursive ``train()`` change.
    """
    _validate_module(module, "training")
    with _temporary_mode(module, True) as active:
        yield active
