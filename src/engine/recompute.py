"""
recompute.py — Gradient checkpointing (activation recomputation).

The idea
--------
A backward pass needs the forward intermediates of every op. Keeping them is
what makes training memory scale with depth. Gradient checkpointing trades that
memory for time: run a section of the model *without* recording, keep only its
input and output values, and when the backward pass reaches it, run the section
a second time — this time recording — and differentiate the replay.

Cost: one extra forward pass over the wrapped section.
Saving: none of that section's intermediates are retained between the forward
and backward passes.

    from engine.recompute import recompute

    x = recompute(lambda inp: block(inp, mask), x)   # instead of block(x, mask)

A section may return one Tensor or a tuple of Tensors. Tuple outputs are packed
behind one tiny synthetic graph node, so all output cotangents reach one replay;
using two outputs does not replay the section twice.

Why the replay uses detached copies
-----------------------------------
The replay's leaves must be private to this closure. If it differentiated into
the *original* input tensor, ``Tensor.backward`` would see a node that has
parents in the outer graph and reset its gradient — discarding contributions
that other consumers of the same tensor (a residual connection, for instance)
had already accumulated. So the replay runs on copies and its input gradients
are added into the originals afterwards.

Dropout and the RNG
-------------------
A replay that draws different dropout masks would differentiate a different
function than the one the forward pass computed. The NumPy RNG state is
therefore captured before the recorded-free forward and restored for the replay,
so both see identical masks. The state in effect when the backward pass started
is put back afterwards, leaving the surrounding training loop's random stream
untouched — a run with checkpointing enabled follows the same trajectory as one
without it.

Replay consistency
------------------
Shape-compatible output is not enough: a closure may read parameters or other
state that changed between the original forward and backward. The replay is
therefore compared against a private snapshot of every forward output before
any replay gradient is propagated. Observable value or dtype drift fails
closed instead of differentiating a different forward pass silently.
"""

import numpy as np

import engine.ops as ops
from .grad_mode import enable_grad, is_grad_enabled, no_grad
from .tensor import Tensor


def _normalize_outputs(output):
    """Return ``(outputs, is_tuple)`` and validate the public output contract."""
    if isinstance(output, Tensor):
        return (output,), False
    if isinstance(output, tuple):
        if not output:
            raise ValueError("recompute output tuple must not be empty")
        if not all(isinstance(value, Tensor) for value in output):
            raise TypeError("recompute output tuple must contain only Tensors")
        return output, True
    raise TypeError("recompute expects function to return a Tensor or tuple of Tensors")


def _snapshot_outputs(outputs):
    """Copy forward values so later closure/output mutation cannot rewrite history."""
    return tuple(np.array(value.data, copy=True) for value in outputs)


def _validate_replay(output, expected_is_tuple, expected_shapes, expected_values):
    """Require replay to reproduce the forward output structure, shape and values."""
    try:
        outputs, is_tuple = _normalize_outputs(output)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "recompute function returned a different output structure during replay"
        ) from exc
    if is_tuple != expected_is_tuple or len(outputs) != len(expected_shapes):
        raise RuntimeError(
            "recompute function returned a different output structure during replay"
        )
    for index, (value, shape, expected) in enumerate(
        zip(outputs, expected_shapes, expected_values)
    ):
        if value.shape != shape:
            raise RuntimeError(
                f"recompute output {index} changed shape during replay: "
                f"expected {shape}, got {value.shape}"
            )
        if value.data.dtype != expected.dtype:
            raise RuntimeError(
                f"recompute output {index} changed dtype during replay: "
                f"expected {expected.dtype}, got {value.data.dtype}"
            )
        if not np.array_equal(value.data, expected, equal_nan=True):
            raise RuntimeError(
                f"recompute output {index} changed values during replay"
            )
    return outputs


def _replay_inputs(inputs):
    return [
        Tensor(value.data, requires_grad=value.requires_grad)
        for value in inputs
    ]


def _accumulate_input_grads(inputs, replay_inputs):
    for original, replayed_input in zip(inputs, replay_inputs):
        if original.requires_grad and replayed_input.grad is not None:
            original._ensure_grad()
            original.grad += replayed_input.grad


def _flatten_outputs(outputs):
    """Join replay outputs into one graph Tensor without changing their values."""
    flattened = [value.reshape((-1,)) for value in outputs]
    if len(flattened) == 1:
        return flattened[0]
    return ops.concat(flattened, axis=0)


def recompute(function, *inputs):
    """
    Run ``function(*inputs)`` without recording, replaying it in the backward pass.

    Parameters
    ----------
    function : callable
        Takes the given tensors and returns a Tensor or a non-empty tuple of
        Tensors. It must be replayable: the same inputs and parameters must
        reproduce the same output structure, shapes, dtypes, and values.
        Parameter gradients are accumulated by the replay itself, so any module
        the function closes over is trained normally.
    *inputs : Tensor
        Tensors whose gradients must flow back through the section.

    Returns
    -------
    Tensor or tuple[Tensor, ...]
        The section output with the same outer structure as ``function``. While
        recording is enabled each returned Tensor participates in a tiny wrapper
        graph; the function's internal graph is discarded until backward. Use
        ``no_grad()`` if you want no graph at all — inside it ``recompute`` is a
        plain call.
    """
    if not callable(function):
        raise TypeError("recompute function must be callable")
    if not inputs:
        raise ValueError("recompute requires at least one Tensor input")
    for value in inputs:
        if not isinstance(value, Tensor):
            raise TypeError("recompute inputs must be Tensors")

    if not is_grad_enabled():
        # Nothing will be differentiated, so there is nothing to trade. Keep
        # the public output contract identical to the recording path while
        # returning the function's original Tensor objects unchanged.
        output = function(*inputs)
        _normalize_outputs(output)
        return output

    forward_rng_state = np.random.get_state()
    with no_grad():
        output = function(*inputs)
    outputs, is_tuple = _normalize_outputs(output)
    expected_shapes = tuple(value.shape for value in outputs)
    expected_values = _snapshot_outputs(outputs)

    if not is_tuple:
        out = Tensor(
            outputs[0].data,
            requires_grad=True,
            _children=inputs,
            _op="recompute",
        )

        def _backward():
            replay_inputs = _replay_inputs(inputs)
            backward_rng_state = np.random.get_state()
            np.random.set_state(forward_rng_state)
            try:
                with enable_grad():
                    replayed = function(*replay_inputs)
                    replayed_outputs = _validate_replay(
                        replayed, False, expected_shapes, expected_values
                    )
                    replayed_outputs[0].backward(out.grad)
            finally:
                # Leave the caller's random stream exactly where it was.
                np.random.set_state(backward_rng_state)

            _accumulate_input_grads(inputs, replay_inputs)

        out._backward = _backward
        return out

    packed_data = np.concatenate(
        [value.data.reshape(-1) for value in outputs], axis=0
    )
    packed = Tensor(
        packed_data,
        requires_grad=True,
        _children=inputs,
        _op="recompute",
    )

    result = []
    start = 0
    for value in outputs:
        stop = start + value.data.size
        result.append(packed[start:stop].reshape(value.shape))
        start = stop

    def _backward_multi():
        replay_inputs = _replay_inputs(inputs)
        backward_rng_state = np.random.get_state()
        np.random.set_state(forward_rng_state)
        try:
            with enable_grad():
                replayed = function(*replay_inputs)
                replayed_outputs = _validate_replay(
                    replayed, True, expected_shapes, expected_values
                )
                _flatten_outputs(replayed_outputs).backward(packed.grad)
        finally:
            # One replay consumes exactly the same random draws as the forward,
            # while the caller's stream remains untouched by backward.
            np.random.set_state(backward_rng_state)

        _accumulate_input_grads(inputs, replay_inputs)

    packed._backward = _backward_multi
    return tuple(result)
