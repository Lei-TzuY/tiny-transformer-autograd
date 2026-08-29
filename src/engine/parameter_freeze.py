"""Exception-safe temporary freezing for leaf ``Tensor`` collections.

``temporarily_frozen(parameters)`` disables ``requires_grad`` only for the duration
of one context and restores every entry flag exactly afterwards. Gradient buffers,
Tensor data, mutation versions, and NumPy RNG state are never modified.
"""

from contextlib import contextmanager
import threading

from engine.tensor import Tensor


_FREEZE_CONDITION = threading.Condition(threading.RLock())
_ACTIVE_TENSORS = {}


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        items = (parameters,)
    else:
        try:
            items = tuple(parameters)
        except TypeError as exc:
            raise TypeError(
                "temporarily_frozen parameters must be a Tensor or iterable of Tensors"
            ) from exc

    seen = set()
    for index, parameter in enumerate(items):
        if not isinstance(parameter, Tensor):
            raise TypeError(
                f"temporarily_frozen parameter {index} must be a Tensor"
            )
        marker = id(parameter)
        if marker in seen:
            raise ValueError("temporarily_frozen parameters must not contain duplicates")
        seen.add(marker)
        if parameter._children:
            raise ValueError(
                f"temporarily_frozen parameter {index} must be a leaf Tensor"
            )
    return items


def _available(markers, owner):
    return all(
        marker not in _ACTIVE_TENSORS or _ACTIVE_TENSORS[marker][0] == owner
        for marker in markers
    )


def _owner_has_reservation(owner):
    return any(active_owner == owner for active_owner, _ in _ACTIVE_TENSORS.values())


@contextmanager
def _reserve(parameters):
    owner = threading.get_ident()
    markers = tuple(id(parameter) for parameter in parameters)

    with _FREEZE_CONDITION:
        while not _available(markers, owner):
            if _owner_has_reservation(owner):
                raise RuntimeError(
                    "nested temporary parameter freeze cannot wait for another thread"
                )
            _FREEZE_CONDITION.wait()

        for marker in markers:
            active = _ACTIVE_TENSORS.get(marker)
            if active is None:
                _ACTIVE_TENSORS[marker] = (owner, 1)
            else:
                _ACTIVE_TENSORS[marker] = (owner, active[1] + 1)

    try:
        yield
    finally:
        with _FREEZE_CONDITION:
            for marker in markers:
                active_owner, count = _ACTIVE_TENSORS[marker]
                if active_owner != owner:
                    raise RuntimeError("temporary parameter freeze ownership changed")
                if count == 1:
                    del _ACTIVE_TENSORS[marker]
                else:
                    _ACTIVE_TENSORS[marker] = (owner, count - 1)
            _FREEZE_CONDITION.notify_all()


@contextmanager
def temporarily_frozen(parameters):
    """Temporarily disable gradient participation for leaf Tensors.

    ``parameters`` may be one Tensor or an iterable materialized exactly once. Mixed
    trainable/frozen input is supported: every entry is forced to ``requires_grad=False``
    inside the context and its exact entry flag is restored on exit, including when the
    body raises or changes the flag itself.

    Existing gradient buffers are deliberately preserved. The helper changes whether
    future operations build graph edges through these leaves; it does not clear stale
    gradients or make an optimizer step safe while parameters are frozen.

    Helper-managed contexts sharing any Tensor are serialized across threads. Disjoint
    Tensor collections may run concurrently, and same-thread nesting is reentrant.
    """
    items = _materialize_parameters(parameters)
    with _reserve(items):
        states = tuple(parameter.requires_grad for parameter in items)
        try:
            for parameter in items:
                vars(parameter)["requires_grad"] = False
            yield items
        finally:
            for parameter, state in zip(items, states):
                vars(parameter)["requires_grad"] = state
