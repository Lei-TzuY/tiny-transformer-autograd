"""Read-only inspection helpers for recorded autograd graphs.

The core engine intentionally keeps graph representation minimal: each ``Tensor``
records its parent tensors, the operation label that produced it, and a backward
closure.  This module turns that internal structure into stable diagnostics without
running backward, mutating gradient buffers, or requiring Graphviz at runtime.

``graph_stats()`` returns a JSON-friendly summary that is useful for tests,
benchmarks, and memory investigations. ``graph_to_dot()`` emits deterministic DOT
text that can be rendered by Graphviz or inspected directly.

The byte counts are the logical NumPy payloads owned by Tensor ``data`` and ``grad``
arrays. They deliberately do not claim to measure Python object, closure, allocator,
or externally captured-array overhead.
"""

from collections import Counter
from collections.abc import Iterable

from .tensor import Tensor


def _normalize_outputs(outputs):
    """Materialize and validate one Tensor output or an iterable of outputs."""
    if isinstance(outputs, Tensor):
        roots = (outputs,)
    else:
        if not isinstance(outputs, Iterable):
            raise TypeError("graph outputs must be a Tensor or iterable of Tensors")
        roots = tuple(outputs)

    if not roots:
        raise ValueError("graph outputs must contain at least one Tensor")
    if any(not isinstance(value, Tensor) for value in roots):
        raise TypeError("graph outputs must contain only Tensors")

    seen = set()
    for value in roots:
        value_id = id(value)
        if value_id in seen:
            raise ValueError("graph outputs must not contain duplicate Tensor references")
        seen.add(value_id)
    return roots


def _collect_graph(roots):
    """Return deterministic discovery/post-order traversals and reject corruption."""
    # state: 1 = active on the DFS stack, 2 = fully processed.  Explicit stack
    # traversal keeps inspection safe for the same deep graphs that backward()
    # supports without relying on Python recursion depth.
    state = {}
    discovery = []
    postorder = []

    for root in roots:
        if state.get(id(root), 0) == 2:
            continue
        stack = [(root, 0)]
        while stack:
            node, next_child = stack[-1]
            node_id = id(node)
            if state.get(node_id, 0) == 0:
                state[node_id] = 1
                discovery.append(node)

            children = node._children
            if next_child < len(children):
                child = children[next_child]
                stack[-1] = (node, next_child + 1)
                if not isinstance(child, Tensor):
                    raise RuntimeError("autograd graph contains a non-Tensor parent")

                child_state = state.get(id(child), 0)
                if child_state == 0:
                    stack.append((child, 0))
                elif child_state == 1:
                    raise RuntimeError("autograd graph contains a cycle")
                continue

            state[node_id] = 2
            postorder.append(node)
            stack.pop()

    return discovery, postorder


def graph_stats(outputs):
    """Return deterministic, JSON-friendly statistics for an autograd graph.

    ``outputs`` may be one Tensor or a non-empty iterable of distinct Tensor
    outputs. Shared ancestors are counted once. Edges count stored parent
    references in the recorded graph, matching the engine's identity-deduplicated
    ``Tensor._children`` representation rather than mathematical operand
    multiplicity (for example ``x * x`` stores one parent Tensor).

    The function is observational only: it does not validate graph versions, run
    backward closures, allocate/replace gradient buffers, or consume NumPy RNG.
    """
    roots = _normalize_outputs(outputs)
    nodes, postorder = _collect_graph(roots)

    consumer_counts = {id(node): 0 for node in nodes}
    edge_count = 0
    for node in nodes:
        edge_count += len(node._children)
        for child in node._children:
            consumer_counts[id(child)] += 1

    depth = {}
    for node in postorder:
        if node._children:
            depth[id(node)] = 1 + max(depth[id(child)] for child in node._children)
        else:
            depth[id(node)] = 0

    op_counts = Counter((node._op or "<leaf>") for node in nodes)
    tensor_data_bytes = sum(int(node.data.nbytes) for node in nodes)
    gradient_buffer_bytes = sum(
        int(node.grad.nbytes) for node in nodes if node.grad is not None
    )

    leaf_nodes = [node for node in nodes if not node._children]
    return {
        "output_count": len(roots),
        "node_count": len(nodes),
        "edge_count": edge_count,
        "leaf_count": len(leaf_nodes),
        "trainable_leaf_count": sum(node.requires_grad for node in leaf_nodes),
        "requires_grad_node_count": sum(node.requires_grad for node in nodes),
        "detached_node_count": sum(
            bool(getattr(node, "_detached_by_no_grad", False)) for node in nodes
        ),
        "shared_node_count": sum(count > 1 for count in consumer_counts.values()),
        "max_depth": max(depth[id(root)] for root in roots),
        "tensor_data_bytes": tensor_data_bytes,
        "gradient_buffer_bytes": gradient_buffer_bytes,
        "total_array_bytes": tensor_data_bytes + gradient_buffer_bytes,
        "op_counts": dict(sorted(op_counts.items())),
    }


def _dot_escape(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def graph_to_dot(outputs):
    """Return a deterministic Graphviz DOT representation of an autograd graph.

    Parent tensors point toward the result tensors that consume them, matching
    forward data flow. Output nodes use a double border. Node identifiers are
    traversal-local (``n0``, ``n1``, ...) rather than memory addresses, so two
    structurally identical graphs produce stable text across processes.

    Graphviz itself is not a dependency; this function only returns DOT text.
    """
    roots = _normalize_outputs(outputs)
    nodes, _ = _collect_graph(roots)
    identifiers = {id(node): f"n{index}" for index, node in enumerate(nodes)}
    output_indices = {id(node): index for index, node in enumerate(roots)}

    lines = ["digraph autograd {", "  rankdir=LR;"]
    for node in nodes:
        label_parts = []
        output_index = output_indices.get(id(node))
        if output_index is not None:
            label_parts.append(f"output[{output_index}]")
        label_parts.extend(
            [
                f"op={node._op or '<leaf>'}",
                f"shape={node.shape}",
                f"requires_grad={bool(node.requires_grad)}",
            ]
        )
        if getattr(node, "_detached_by_no_grad", False):
            label_parts.append("detached_by_no_grad=True")

        attrs = [f'label="{_dot_escape(chr(10).join(label_parts))}"']
        if output_index is not None:
            attrs.append("peripheries=2")
        lines.append(f"  {identifiers[id(node)]} [{', '.join(attrs)}];")

    for node in nodes:
        for child in node._children:
            lines.append(f"  {identifiers[id(child)]} -> {identifiers[id(node)]};")

    lines.append("}")
    return "\n".join(lines) + "\n"
