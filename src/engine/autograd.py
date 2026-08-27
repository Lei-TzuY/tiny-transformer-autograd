"""Functional reverse-mode automatic differentiation helpers."""

from collections.abc import Iterable
from fractions import Fraction

import numpy as np

from .tensor import Tensor


def _topological_order(output):
    """Return graph nodes in post-order without recursive Python calls."""
    topo = []
    visited = set()
    stack = [(output, False)]
    while stack:
        node, expanded = stack.pop()
        node_id = id(node)
        if expanded:
            topo.append(node)
            continue
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.append((node, True))
        for child in node._children:
            if id(child) not in visited:
                stack.append((child, False))
    return topo


def _validate_grad_request(output, inputs):
    """Normalise and validate one functional reverse-mode differentiation request."""
    if not isinstance(output, Tensor):
        raise TypeError("output must be a Tensor")
    if not output.requires_grad:
        raise ValueError("output must require gradients")

    if isinstance(inputs, Tensor):
        requested = (inputs,)
    else:
        if not isinstance(inputs, Iterable):
            raise TypeError("inputs must be a Tensor or iterable of Tensors")
        requested = tuple(inputs)

    if not requested:
        raise ValueError("inputs must contain at least one Tensor")
    if any(not isinstance(value, Tensor) for value in requested):
        raise TypeError("inputs must contain only Tensors")
    if any(not value.requires_grad for value in requested):
        raise ValueError("all requested inputs must require gradients")

    topo = _topological_order(output)
    reachable = {id(node) for node in topo}
    if any(id(value) not in reachable for value in requested):
        raise ValueError("all requested inputs must be reachable from output")
    return requested, topo


def _validate_tangents(requested, tangents, *, single_input):
    """Return finite float64 tangent snapshots aligned with requested inputs."""
    if single_input:
        supplied = (tangents,)
    else:
        if isinstance(tangents, (str, bytes)) or not isinstance(tangents, Iterable):
            raise TypeError("tangents must be an iterable with one value per input")
        supplied = tuple(tangents)
        if len(supplied) != len(requested):
            raise ValueError(
                "tangents must contain exactly one value per requested input"
            )

    validated = []
    for index, (value, tangent) in enumerate(zip(requested, supplied)):
        raw = np.asarray(tangent)
        is_integer = np.issubdtype(raw.dtype, np.integer)
        is_floating = np.issubdtype(raw.dtype, np.floating)
        if np.issubdtype(raw.dtype, np.bool_) or not (is_integer or is_floating):
            raise TypeError(f"tangent {index} must contain real numeric values")
        tangent_array = np.array(raw, dtype=np.float64, copy=True)
        if tangent_array.shape != value.shape:
            raise ValueError(
                f"tangent {index} shape mismatch: expected {value.shape}, "
                f"got {tangent_array.shape}"
            )
        if not np.isfinite(tangent_array).all():
            raise ValueError(f"tangent {index} must contain only finite values")
        validated.append(tangent_array)
    return tuple(validated)


def _exact_directional_sum(derivatives, tangents):
    """Exactly accumulate finite derivative*tangent terms, then round once."""
    exact = Fraction()
    for derivative, tangent in zip(derivatives, tangents):
        for left, right in zip(derivative.flat, tangent.flat):
            exact += Fraction.from_float(float(left)) * Fraction.from_float(float(right))
    try:
        return float(exact)
    except OverflowError:
        return -np.inf if exact < 0 else np.inf


def _directional_sum(derivatives, tangents):
    """Preserve ordinary NumPy contraction while recovering finite cancellation."""
    with np.errstate(over="ignore", invalid="ignore"):
        terms = [
            np.sum(derivative * tangent)
            for derivative, tangent in zip(derivatives, tangents)
        ]
        historical = np.sum(np.asarray(terms, dtype=np.float64))
    if np.isfinite(historical):
        return float(historical)

    source_finite = all(np.isfinite(value).all() for value in derivatives)
    if not source_finite:
        return float(historical)
    return _exact_directional_sum(derivatives, tangents)


def grad(output, inputs, grad_output=None):
    """Compute a vector-Jacobian product without modifying persistent ``.grad``.

    Parameters
    ----------
    output : Tensor
        Graph output whose vector-Jacobian product should be evaluated.
    inputs : Tensor or iterable[Tensor]
        Reachable trainable tensors to differentiate with respect to.
    grad_output : array-like or None
        Optional VJP seed. It follows :meth:`Tensor.backward` validation and
        defaults to ones with ``output.shape``.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        One independent gradient array for every requested input, in order.

    Notes
    -----
    This helper deliberately reuses ``Tensor.backward`` as the single source of
    truth for reverse-mode rules. All graph gradient buffer references are
    preserved, temporarily replaced to isolate this VJP, and restored even if
    backward fails. Leaf accumulation semantics of ``Tensor.backward`` are
    therefore unchanged for normal training code.
    """
    requested, topo = _validate_grad_request(output, inputs)

    snapshots = {}
    for node in topo:
        if node.requires_grad:
            snapshots[id(node)] = (node, node.grad)

    try:
        for node, _ in snapshots.values():
            node.grad = np.zeros(node.shape, dtype=np.float64)
        output.backward(grad_output)
        return tuple(np.array(value.grad, copy=True) for value in requested)
    finally:
        for node, previous in snapshots.values():
            node.grad = previous


def jacobian(output, inputs):
    """Return full Jacobians without modifying persistent ``.grad`` buffers.

    Parameters
    ----------
    output : Tensor
        Differentiable graph output. The leading axes of every returned
        Jacobian match ``output.shape``.
    inputs : Tensor or iterable[Tensor]
        Reachable trainable tensors to differentiate with respect to.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        For each requested input ``x``, returns an independent float64 array
        with shape ``output.shape + x.shape``. Entry ``J[o + i]`` is the
        derivative of output element ``o`` with respect to input element ``i``.

    Notes
    -----
    The engine is reverse-mode only, so a dense Jacobian requires one VJP per
    output element. This helper intentionally delegates every row to
    :func:`grad`, keeping graph validation, mutation detection, numerical VJPs,
    and gradient-buffer isolation in one implementation. It is therefore aimed
    at small tensors, diagnostics, and educational use rather than large model
    training loops.
    """
    requested, _ = _validate_grad_request(output, inputs)
    jacobians = [
        np.empty(output.shape + value.shape, dtype=np.float64)
        for value in requested
    ]
    output_size = int(output.data.size)

    # There is no basis vector to replay for an empty output, but still execute
    # one zero-seeded VJP so stale-graph checks and the public validation
    # contract remain identical to non-empty Jacobian requests.
    if output_size == 0:
        grad(output, requested, np.zeros_like(output.data, dtype=np.float64))
        return tuple(jacobians)

    flat_views = [
        value.reshape((output_size,) + requested_input.shape)
        for value, requested_input in zip(jacobians, requested)
    ]
    seed = np.zeros(output.shape, dtype=np.float64)
    flat_seed = seed.reshape(-1)

    for output_index in range(output_size):
        flat_seed.fill(0.0)
        flat_seed[output_index] = 1.0
        derivatives = grad(output, requested, seed)
        for view, derivative in zip(flat_views, derivatives):
            view[output_index] = derivative

    return tuple(jacobians)


def jvp(output, inputs, tangents):
    """Compute a Jacobian-vector product from an already-built reverse graph.

    Parameters
    ----------
    output : Tensor
        Differentiable graph output.
    inputs : Tensor or iterable[Tensor]
        Reachable trainable tensors whose input-space direction is specified.
    tangents : array-like or iterable[array-like]
        Tangent structure mirrors ``inputs``. If ``inputs`` itself is one
        Tensor, pass one array-like tangent directly. If ``inputs`` is an
        iterable, pass an iterable with exactly one tangent for every element,
        including when that iterable contains only one Tensor.

    Returns
    -------
    numpy.ndarray
        Independent float64 array with shape ``output.shape`` containing
        ``J @ v`` for the requested joint input direction.

    Notes
    -----
    This engine records reverse-mode VJPs, not forward-mode dual numbers. A JVP
    therefore requires one reverse pass per output element. Unlike
    :func:`jacobian`, this helper contracts each row immediately and never
    materialises the full dense Jacobian, reducing peak diagnostic memory when
    only a directional derivative is needed.
    """
    single_input = isinstance(inputs, Tensor)
    requested, _ = _validate_grad_request(output, inputs)
    tangent_arrays = _validate_tangents(
        requested,
        tangents,
        single_input=single_input,
    )
    result = np.empty(output.shape, dtype=np.float64)
    output_size = int(output.data.size)

    # Match jacobian(): even a zero-element result must validate the tape so a
    # stale graph cannot hide behind an otherwise allocation-only fast path.
    if output_size == 0:
        grad(output, requested, np.zeros_like(output.data, dtype=np.float64))
        return result

    seed = np.zeros(output.shape, dtype=np.float64)
    flat_seed = seed.reshape(-1)
    flat_result = result.reshape(-1)
    for output_index in range(output_size):
        flat_seed.fill(0.0)
        flat_seed[output_index] = 1.0
        derivatives = grad(output, requested, seed)
        flat_result[output_index] = _directional_sum(derivatives, tangent_arrays)
    return result
