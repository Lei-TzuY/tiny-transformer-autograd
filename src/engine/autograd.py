"""Functional reverse-mode automatic differentiation helpers."""

from collections.abc import Iterable

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
    truth for reverse-mode rules. All graph gradient buffers are snapshotted,
    temporarily cleared to isolate this VJP, and restored even if backward
    fails. Leaf accumulation semantics of ``Tensor.backward`` are therefore
    unchanged for normal training code.
    """
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

    snapshots = {}
    for node in topo:
        if node.requires_grad:
            snapshots[id(node)] = (
                node,
                None if node.grad is None else np.array(node.grad, copy=True),
            )

    try:
        for node, _ in snapshots.values():
            node.grad = np.zeros(node.shape, dtype=np.float64)
        output.backward(grad_output)
        return tuple(np.array(value.grad, copy=True) for value in requested)
    finally:
        for node, previous in snapshots.values():
            node.grad = previous
