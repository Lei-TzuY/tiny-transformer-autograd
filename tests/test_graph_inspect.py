import numpy as np
import pytest

import engine.ops as ops
from engine.grad_mode import no_grad
from engine.graph_inspect import graph_stats, graph_to_dot
from engine.tensor import Tensor


def _rng_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_graph_stats_counts_shared_graph_and_array_payloads():
    x = Tensor([2.0, 3.0], requires_grad=True)
    square = ops.mul(x, x)
    output = ops.add(square, x)

    assert graph_stats(output) == {
        "output_count": 1,
        "node_count": 3,
        "edge_count": 3,
        "leaf_count": 1,
        "trainable_leaf_count": 1,
        "requires_grad_node_count": 3,
        "detached_node_count": 0,
        "shared_node_count": 1,
        "max_depth": 2,
        "tensor_data_bytes": 48,
        "gradient_buffer_bytes": 48,
        "total_array_bytes": 96,
        "op_counts": {"<leaf>": 1, "add": 1, "mul": 1},
    }


def test_graph_stats_deduplicates_shared_ancestors_across_multiple_outputs():
    x = Tensor([2.0, 3.0], requires_grad=True)
    square = ops.mul(x, x)
    output = ops.add(square, x)

    stats = graph_stats((square, output))

    assert stats["output_count"] == 2
    assert stats["node_count"] == 3
    assert stats["edge_count"] == 3
    assert stats["shared_node_count"] == 1
    assert stats["max_depth"] == 2
    assert stats["op_counts"] == {"<leaf>": 1, "add": 1, "mul": 1}


def test_graph_stats_reports_no_grad_result_as_detached_single_node():
    x = Tensor([2.0, 3.0], requires_grad=True)
    with no_grad():
        detached = ops.mul(x, x)

    assert graph_stats(detached) == {
        "output_count": 1,
        "node_count": 1,
        "edge_count": 0,
        "leaf_count": 1,
        "trainable_leaf_count": 0,
        "requires_grad_node_count": 0,
        "detached_node_count": 1,
        "shared_node_count": 0,
        "max_depth": 0,
        "tensor_data_bytes": 16,
        "gradient_buffer_bytes": 0,
        "total_array_bytes": 16,
        "op_counts": {"mul": 1},
    }


def test_inspection_is_gradient_and_rng_neutral():
    np.random.seed(123)
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.mul(x, x)
    x.grad[...] = [7.0, 11.0]
    grad_object = x.grad
    grad_value = x.grad.copy()
    rng_before = np.random.get_state()

    graph_stats(output)
    graph_to_dot(output)

    assert x.grad is grad_object
    assert np.array_equal(x.grad, grad_value)
    assert _rng_state_equal(np.random.get_state(), rng_before)


def test_graph_to_dot_is_deterministic_and_uses_forward_edge_direction():
    x = Tensor([2.0, 3.0], requires_grad=True)
    square = ops.mul(x, x)
    output = ops.add(square, x)

    expected = """digraph autograd {
  rankdir=LR;
  n0 [label="output[0]\\nop=add\\nshape=(2,)\\nrequires_grad=True", peripheries=2];
  n1 [label="op=mul\\nshape=(2,)\\nrequires_grad=True"];
  n2 [label="op=<leaf>\\nshape=(2,)\\nrequires_grad=True"];
  n1 -> n0;
  n2 -> n0;
  n2 -> n1;
}
"""

    assert graph_to_dot(output) == expected
    assert graph_to_dot(output) == expected


def test_graph_to_dot_marks_detached_no_grad_outputs():
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        output = ops.mul(x, x)

    dot = graph_to_dot(output)

    assert "detached_by_no_grad=True" in dot
    assert "requires_grad=False" in dot
    assert "->" not in dot


def test_graph_output_validation_is_explicit():
    x = Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="graph outputs must be a Tensor or iterable"):
        graph_stats(object())
    with pytest.raises(ValueError, match="at least one Tensor"):
        graph_stats([])
    with pytest.raises(TypeError, match="contain only Tensors"):
        graph_stats([x, object()])
    with pytest.raises(ValueError, match="duplicate Tensor references"):
        graph_stats([x, x])


def test_graph_output_generator_is_materialized_once():
    x = Tensor([1.0], requires_grad=True)
    output = ops.mul(x, x)
    iterations = []

    def outputs():
        iterations.append("started")
        yield output

    stats = graph_stats(outputs())

    assert stats["output_count"] == 1
    assert iterations == ["started"]


def test_corrupted_graph_cycle_is_rejected_instead_of_looping():
    x = Tensor([1.0], requires_grad=True)
    x._children = (x,)

    with pytest.raises(RuntimeError, match="autograd graph contains a cycle"):
        graph_stats(x)


def test_corrupted_graph_non_tensor_parent_is_rejected():
    x = Tensor([1.0], requires_grad=True)
    x._children = (object(),)

    with pytest.raises(RuntimeError, match="non-Tensor parent"):
        graph_to_dot(x)
