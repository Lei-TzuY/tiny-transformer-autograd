"""Function transforms that evaluate a scalar value and its reverse-mode gradients."""

from functools import wraps
from numbers import Integral

import numpy as np

from .autograd import grad
from .tensor import Tensor


def value_and_grad(function, *, argnums=0, has_aux=False):
    """Return a callable that evaluates ``function`` and selected gradients.

    Parameters
    ----------
    function : callable
        Function whose scalar :class:`Tensor` result should be differentiated.
    argnums : int or tuple[int, ...], optional
        Positional argument index or indices to differentiate with respect to.
        Negative indices follow normal Python positional-index semantics.
    has_aux : bool, optional
        When true, ``function`` must return ``(value, aux)``. Only ``value`` is
        differentiated; ``aux`` is returned unchanged.

    Returns
    -------
    callable
        A wrapped function. For one integer ``argnums`` it returns
        ``(value, gradient)``. For tuple ``argnums`` it returns
        ``(value, gradients)`` where ``gradients`` is a tuple in the requested
        order. With ``has_aux=True``, the first item is ``(value, aux)``.

    Notes
    -----
    The wrapped function runs exactly once. Gradient evaluation delegates to
    :func:`engine.autograd.grad`, so persistent ``Tensor.grad`` buffer objects
    and values are restored after both successful and failed differentiation.

    A scalar Tensor output is required. Arbitrary non-scalar vector-Jacobian
    products remain available directly through :func:`engine.autograd.grad`.
    """
    if not callable(function):
        raise TypeError("value_and_grad function must be callable")

    indices, single = _normalize_argnums(argnums)
    if not isinstance(has_aux, (bool, np.bool_)):
        raise TypeError("value_and_grad has_aux must be a boolean")
    has_aux = bool(has_aux)

    @wraps(function)
    def wrapped(*args, **kwargs):
        resolved = _resolve_argnums(indices, len(args))
        selected = tuple(args[index] for index in resolved)

        for index, value in zip(resolved, selected):
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"value_and_grad positional argument {index} must be a Tensor"
                )
            if not value.requires_grad:
                raise ValueError(
                    f"value_and_grad positional argument {index} must require gradients"
                )

        identities = [id(value) for value in selected]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "value_and_grad selected argnums must refer to distinct Tensor objects"
            )

        result = function(*args, **kwargs)
        if has_aux:
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "value_and_grad function must return a (value, aux) tuple "
                    "when has_aux=True"
                )
            value, aux = result
        else:
            value = result
            aux = None

        if not isinstance(value, Tensor):
            raise TypeError("value_and_grad function value must be a Tensor")
        if value.shape != ():
            raise ValueError("value_and_grad function value must be a scalar Tensor")

        gradients = grad(value, selected)
        gradient_result = gradients[0] if single else gradients
        if has_aux:
            return (value, aux), gradient_result
        return value, gradient_result

    return wrapped


def _normalize_argnums(argnums):
    if isinstance(argnums, (bool, np.bool_)):
        raise TypeError(
            "value_and_grad argnums must be an integer or non-empty tuple of integers"
        )

    if isinstance(argnums, Integral):
        return (int(argnums),), True

    if not isinstance(argnums, tuple) or not argnums:
        raise TypeError(
            "value_and_grad argnums must be an integer or non-empty tuple of integers"
        )

    normalized = []
    for value in argnums:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(
                "value_and_grad argnums must be an integer or non-empty tuple of integers"
            )
        normalized.append(int(value))

    if len(set(normalized)) != len(normalized):
        raise ValueError("value_and_grad argnums must not contain duplicate indices")
    return tuple(normalized), False


def _resolve_argnums(indices, positional_count):
    resolved = []
    for index in indices:
        normalized = index
        if normalized < 0:
            normalized += positional_count
        if normalized < 0 or normalized >= positional_count:
            raise ValueError(
                "value_and_grad argnums index is out of range for positional arguments"
            )
        resolved.append(normalized)

    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "value_and_grad argnums resolve to duplicate positional arguments"
        )
    return tuple(resolved)
