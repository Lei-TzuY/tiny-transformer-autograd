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
    """Run the finite-difference checker without changing caller parameter state.

    A trainable parameter may legitimately have ``grad is None`` before the
    check. The underlying analytical pass expects a gradient buffer when it
    snapshots results; an unused parameter would otherwise stay ``None`` and
    fail while trying to copy that result. Treat such missing buffers as zero
    for the duration of the check, then restore the caller's exact gradient
    representation, mutation version, and existing gradient-buffer identity.
    """
    # Reject an invalid public target before inspecting ``parameters``. Besides
    # keeping the documented function error deterministic, this avoids consuming
    # a caller-owned parameter iterator when no check can be performed.
    if not callable(function):
        raise TypeError("gradcheck function must be callable")

    parameter_items = _gradcheck_impl._normalise_parameters(parameters)
    normalised_parameters = [
        parameter if name is None else (name, parameter)
        for name, parameter in parameter_items
    ]
    caller_state = [
        (
            parameter,
            parameter._version,
            parameter.grad,
            None if parameter.grad is None else parameter.grad.copy(),
        )
        for _, parameter in parameter_items
    ]

    for parameter, _, grad_buffer, _ in caller_state:
        if grad_buffer is None:
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
        # The implementation perturbs parameter.data in place, which correctly
        # increments Tensor's mutation version during the check. It also resets
        # gradients for the analytical pass. Both are private checker details:
        # once data values have been restored, put the caller-visible state back
        # exactly so a graph built before gradcheck remains valid afterward.
        for parameter, version, grad_buffer, grad_values in caller_state:
            parameter._version = version
            if grad_buffer is None:
                parameter.grad = None
            else:
                grad_buffer[...] = grad_values
                parameter.grad = grad_buffer
