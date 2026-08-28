"""Exception-safe temporary evaluation mode for ``nn.Module`` trees.

``evaluating(module)`` temporarily places the module tree in evaluation mode and
then restores the exact per-module training-mode state that existed on entry.
This is intentionally separate from autograd grad mode: evaluation mode controls
module behavior such as dropout, while ``no_grad()`` controls graph recording.
"""

from contextlib import contextmanager
import threading

from nn.module import Module


_MODE_CONTEXT_LOCK = threading.RLock()
_MISSING = object()


@contextmanager
def evaluating(module):
    """Temporarily put ``module`` and its current descendants in evaluation mode.

    Existing child modules may have different training flags.  The context snapshots
    those flags individually, calls the normal recursive ``eval()`` path, and restores
    the exact entry state in ``finally``.  A module that did not have a local
    ``training`` attribute before entry has that temporary attribute removed again.

    Helper-managed contexts are serialized because overlapping contexts can otherwise
    restore shared module state out of order.  Direct ``train()``/``eval()`` calls made
    outside this helper are not synchronized.
    """
    if not isinstance(module, Module):
        raise TypeError("evaluating module must be an nn.Module")

    with _MODE_CONTEXT_LOCK:
        modules = tuple(module.modules())
        states = tuple(
            (child, vars(child).get("training", _MISSING)) for child in modules
        )

        module.eval()
        try:
            yield module
        finally:
            for child, state in states:
                if state is _MISSING:
                    vars(child).pop("training", None)
                else:
                    child.training = state
