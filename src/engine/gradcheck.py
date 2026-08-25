"""Public finite-difference gradcheck with caller-state normalization."""

import numpy as np

from . import _gradcheck_impl


def gradcheck(
    function,
    *inputs,
    parameters=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-4,
):
    """Run the finite-difference checker without changing caller grad state.

    A trainable parameter may legitimately have ``grad is None`` before the
    check. The underlying analytical pass expects a gradient buffer when it
    snapshots results; an unused parameter would otherwise stay ``None`` and
    fail while trying to copy that result. Treat such missing buffers as zero
    for the duration of the check, then restore the exact ``None`` state.
    """
    parameter_items = _gradcheck_impl._normalise_parameters(parameters)
    normalised_parameters = [
        parameter if name is None else (name, parameter)
        for name, parameter in parameter_items
    ]
    missing_grads = [
        parameter
        for _, parameter in parameter_items
        if parameter.grad is None
    ]
    for parameter in missing_grads:
        parameter.grad = np.zeros(parameter.data.shape, dtype=np.float64)

    try:
        return _gradcheck_impl.gradcheck(
            function,
            *inputs,
            parameters=normalised_parameters,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
    finally:
        for parameter in missing_grads:
            parameter.grad = None
