"""Finite-difference checks for reverse-mode gradients.

``gradcheck`` compares VJPs produced by ``Tensor.backward`` with central
finite-difference approximations. It is intentionally small and slow: this is
a correctness tool for tiny inputs, parameters, and new primitive ops, not a
training-time facility.

For non-scalar outputs the checker contracts the output with a deterministic,
non-uniform cotangent before comparing derivatives. That exercises a real VJP
instead of silently checking only an all-ones reduction.
"""

import numpy as np

from .grad_mode import no_grad
from .tensor import Tensor


def gradcheck(
    function,
    *inputs,
    parameters=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-4,
):
    """Return ``True`` when analytical and finite-difference gradients agree.

    Parameters
    ----------
    function : callable
        Function from zero or more ``Tensor`` inputs to one ``Tensor`` output.
    *inputs : Tensor
        Input points at which to check the VJP. Originals are never mutated and
        need not have ``requires_grad=True``; private trainable copies are used.
    parameters : iterable, optional
        Trainable ``Tensor`` objects used by ``function`` or ``(name, Tensor)``
        pairs such as ``module.named_parameters()``. Parameter data and existing
        gradients are restored exactly before returning, including on failure.
    eps : float
        Central-difference perturbation size.
    atol, rtol : float
        Absolute and relative tolerances used by ``numpy.isclose``.

    Notes
    -----
    Every forward evaluation starts from the same NumPy RNG state and the
    caller's state is restored before returning. Randomized functions are thus
    checkable without consuming the surrounding random stream.

    At least one input or parameter must be supplied. Parameter checking is
    deliberately generic rather than depending on ``nn.Module`` so the autograd
    engine remains usable on its own.

    Raises
    ------
    AssertionError
        If any analytical gradient disagrees with its numerical estimate.
    TypeError, ValueError
        For invalid arguments or a function whose output contract changes
        during finite-difference evaluation.
    """
    parameter_items = _normalise_parameters(parameters)
    _validate_arguments(function, inputs, parameter_items, eps, atol, rtol)
    rng_state = np.random.get_state()
    parameter_state = _snapshot_parameters(parameter_items)

    try:
        analytical_inputs = [
            Tensor(value.data.copy(), requires_grad=True) for value in inputs
        ]
        for _, parameter in parameter_items:
            parameter.zero_grad()

        np.random.set_state(rng_state)
        output = function(*analytical_inputs)
        _validate_output(output, "function")
        if not np.isfinite(output.data).all():
            raise ValueError("function output must contain only finite values")

        cotangent = _cotangent(output.shape)
        output.backward(cotangent)
        analytical_inputs_grad = [value.grad.copy() for value in analytical_inputs]
        analytical_parameters = [
            parameter.grad.copy() for _, parameter in parameter_items
        ]

        for input_number, source in enumerate(inputs):
            numerical = np.zeros_like(source.data, dtype=np.float64)
            for index in np.ndindex(source.shape):
                plus = _plain_inputs(inputs)
                minus = _plain_inputs(inputs)
                plus[input_number].data[index] += eps
                minus[input_number].data[index] -= eps

                plus_value = _objective(
                    function, plus, cotangent, output.shape, rng_state
                )
                minus_value = _objective(
                    function, minus, cotangent, output.shape, rng_state
                )
                numerical[index] = (plus_value - minus_value) / (2.0 * eps)

            _assert_close(
                f"input {input_number}",
                analytical_inputs_grad[input_number],
                numerical,
                atol=atol,
                rtol=rtol,
            )

        for parameter_number, (name, parameter) in enumerate(parameter_items):
            numerical = np.zeros_like(parameter.data, dtype=np.float64)
            original = parameter.data.copy()
            for index in np.ndindex(parameter.shape):
                parameter.data[index] = original[index] + eps
                plus_value = _objective(
                    function,
                    _plain_inputs(inputs),
                    cotangent,
                    output.shape,
                    rng_state,
                )

                parameter.data[index] = original[index] - eps
                minus_value = _objective(
                    function,
                    _plain_inputs(inputs),
                    cotangent,
                    output.shape,
                    rng_state,
                )
                parameter.data[index] = original[index]
                numerical[index] = (plus_value - minus_value) / (2.0 * eps)

            label = (
                f"parameter {parameter_number}"
                if name is None
                else f"parameter {name!r}"
            )
            _assert_close(
                label,
                analytical_parameters[parameter_number],
                numerical,
                atol=atol,
                rtol=rtol,
            )
        return True
    finally:
        _restore_parameters(parameter_items, parameter_state)
        np.random.set_state(rng_state)


def _objective(function, inputs, cotangent, expected_shape, rng_state):
    """Evaluate the scalar projection used by one finite-difference sample."""
    np.random.set_state(rng_state)
    with no_grad():
        output = function(*inputs)
    _validate_output(output, "function")
    if output.shape != expected_shape:
        raise ValueError(
            "function output shape changed during gradcheck: expected "
            f"{expected_shape}, got {output.shape}"
        )
    if not np.isfinite(output.data).all():
        raise ValueError("function output must contain only finite values")
    return float(np.sum(output.data * cotangent))


def _plain_inputs(inputs):
    return [Tensor(value.data.copy()) for value in inputs]


def _cotangent(shape):
    """Build a deterministic non-uniform VJP seed without touching the RNG."""
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if size == 1:
        return np.ones(shape or (), dtype=np.float64)
    return np.linspace(0.5, 1.5, size, dtype=np.float64).reshape(shape)


def _assert_close(label, analytical, numerical, *, atol, rtol):
    close = np.isclose(analytical, numerical, atol=atol, rtol=rtol)
    if np.all(close):
        return

    error = np.abs(analytical - numerical)
    allowance = atol + rtol * np.abs(numerical)
    excess = error - allowance
    flat_index = int(np.argmax(excess))
    index = np.unravel_index(flat_index, analytical.shape)
    actual = float(analytical[index])
    expected = float(numerical[index])
    absolute_error = float(error[index])
    tolerance = float(allowance[index])
    raise AssertionError(
        f"gradcheck failed for {label} at index {index}: "
        f"analytical={actual:.12g}, numerical={expected:.12g}, "
        f"abs_error={absolute_error:.3g}, tolerance={tolerance:.3g}"
    )


def _normalise_parameters(parameters):
    if parameters is None:
        return []
    if isinstance(parameters, Tensor):
        raw_items = [parameters]
    else:
        try:
            raw_items = list(parameters)
        except TypeError as exc:
            raise TypeError("gradcheck parameters must be an iterable") from exc

    items = []
    seen = set()
    for number, item in enumerate(raw_items):
        name = None
        parameter = item
        if isinstance(item, tuple):
            if len(item) != 2 or not isinstance(item[0], str):
                raise TypeError(
                    "gradcheck named parameters must be (name, Tensor) pairs"
                )
            name, parameter = item
        if not isinstance(parameter, Tensor):
            raise TypeError(f"gradcheck parameter {number} must be a Tensor")
        if id(parameter) in seen:
            raise ValueError("gradcheck parameters must not contain duplicates")
        seen.add(id(parameter))
        items.append((name, parameter))
    return items


def _snapshot_parameters(parameter_items):
    return [
        (
            parameter.data.copy(),
            None if parameter.grad is None else parameter.grad.copy(),
        )
        for _, parameter in parameter_items
    ]


def _restore_parameters(parameter_items, states):
    for (_, parameter), (data, grad) in zip(parameter_items, states):
        parameter.data[...] = data
        parameter.grad = None if grad is None else grad.copy()


def _validate_arguments(function, inputs, parameter_items, eps, atol, rtol):
    if not callable(function):
        raise TypeError("gradcheck function must be callable")
    if not inputs and not parameter_items:
        raise ValueError("gradcheck requires at least one Tensor input or parameter")
    for number, value in enumerate(inputs):
        if not isinstance(value, Tensor):
            raise TypeError(f"gradcheck input {number} must be a Tensor")
        _validate_tensor_value(value, f"input {number}")

    for number, (_, parameter) in enumerate(parameter_items):
        if not parameter.requires_grad:
            raise ValueError(f"gradcheck parameter {number} must require gradients")
        _validate_tensor_value(parameter, f"parameter {number}")

    _validate_tolerance("eps", eps, strictly_positive=True)
    _validate_tolerance("atol", atol, strictly_positive=False)
    _validate_tolerance("rtol", rtol, strictly_positive=False)


def _validate_tensor_value(value, label):
    if value.data.size == 0:
        raise ValueError(f"gradcheck {label} must not be empty")
    if not np.isfinite(value.data).all():
        raise ValueError(f"gradcheck {label} must contain only finite values")


def _validate_tolerance(name, value, *, strictly_positive):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"gradcheck {name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"gradcheck {name} must be finite")
    if strictly_positive and value <= 0:
        raise ValueError(f"gradcheck {name} must be positive")
    if not strictly_positive and value < 0:
        raise ValueError(f"gradcheck {name} must be non-negative")


def _validate_output(output, source):
    if not isinstance(output, Tensor):
        raise TypeError(f"gradcheck {source} must return a Tensor")
