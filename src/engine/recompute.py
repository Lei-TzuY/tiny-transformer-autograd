"""
recompute.py — Gradient checkpointing (activation recomputation).

The idea
--------
A backward pass needs the forward intermediates of every op.  Keeping them is
what makes training memory scale with depth.  Gradient checkpointing trades that
memory for time: run a section of the model *without* recording, keep only its
input and output, and when the backward pass reaches it, run the section a
second time — this time recording — and differentiate the replay.

Cost: one extra forward pass over the wrapped section.
Saving: none of that section's intermediates are retained between the forward
and backward passes.

    from engine.recompute import recompute

    x = recompute(lambda inp: block(inp, mask), x)   # instead of block(x, mask)

Why the replay uses detached copies
-----------------------------------
The replay's leaves must be private to this closure.  If it differentiated into
the *original* input tensor, ``Tensor.backward`` would see a node that has
parents in the outer graph and reset its gradient — discarding contributions
that other consumers of the same tensor (a residual connection, for instance)
had already accumulated.  So the replay runs on copies and its input gradients
are added into the originals afterwards.

Dropout and the RNG
-------------------
A replay that draws different dropout masks would differentiate a different
function than the one the forward pass computed.  The NumPy RNG state is
therefore captured before the recorded-free forward and restored for the replay,
so both see identical masks.  The state in effect when the backward pass started
is put back afterwards, leaving the surrounding training loop's random stream
untouched — a run with checkpointing enabled follows the same trajectory as one
without it.
"""

import numpy as np

from .grad_mode import enable_grad, is_grad_enabled, no_grad
from .tensor import Tensor


def recompute(function, *inputs) -> Tensor:
    """
    Run ``function(*inputs)`` without recording, replaying it in the backward pass.

    Parameters
    ----------
    function : callable
        Takes the given tensors and returns a single Tensor. It must be
        replayable: same inputs and parameters must give the same output.
        Parameter gradients are accumulated by the replay itself, so any module
        the function closes over is trained normally.
    *inputs : Tensor
        Tensors whose gradients must flow back through the section.

    Returns
    -------
    Tensor
        The section's output. While recording is enabled this is always a graph
        node — the function's internals may need gradients even when none of
        ``inputs`` do. Use ``no_grad()`` if you want no graph at all; inside it
        ``recompute`` is a plain call.
    """
    if not inputs:
        raise ValueError("recompute requires at least one Tensor input")
    for value in inputs:
        if not isinstance(value, Tensor):
            raise TypeError("recompute inputs must be Tensors")

    if not is_grad_enabled():
        # Nothing will be differentiated, so there is nothing to trade.
        return function(*inputs)

    forward_rng_state = np.random.get_state()
    with no_grad():
        output = function(*inputs)
    if not isinstance(output, Tensor):
        raise TypeError("recompute expects function to return a single Tensor")

    out = Tensor(
        output.data,
        requires_grad=True,
        _children=inputs,
        _op="recompute",
    )

    def _backward():
        replay_inputs = [
            Tensor(value.data, requires_grad=value.requires_grad)
            for value in inputs
        ]
        backward_rng_state = np.random.get_state()
        np.random.set_state(forward_rng_state)
        try:
            with enable_grad():
                replayed = function(*replay_inputs)
                replayed.backward(out.grad)
        finally:
            # Leave the caller's random stream exactly where it was.
            np.random.set_state(backward_rng_state)

        for original, replayed_input in zip(inputs, replay_inputs):
            if original.requires_grad:
                original._ensure_grad()
                original.grad += replayed_input.grad

    out._backward = _backward
    return out
