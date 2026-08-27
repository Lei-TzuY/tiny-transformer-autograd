"""Finite-difference validation for functional Jacobian-vector products."""

from collections.abc import Iterable

import numpy as np

from .autograd import jvp
from .grad_mode import no_grad
from .tensor import Tensor


def jvpcheck(function, inputs, tangents, eps=1e-6, atol=1e-5, rtol=1e-4):
    """Check one input-space directional derivative with central differences.

    Parameters
    ----------
    function : callable
        Function from the supplied Tensor inputs to one Tensor output.
    inputs : Tensor or iterable[Tensor]
        Finite, non-empty input points. Caller-owned tensors are never mutated.
    tangents : array-like or iterable[array-like]
        Input-space direction with the same container structure as ``inputs``.
    eps : float
        Positive central-difference step size.
    atol, rtol : float
        Non-negative absolute and relative tolerances.

    Returns
    -------
    bool
        ``True`` when the analytical JVP agrees with the central difference.

    Notes
    -----
    The analytical pass uses private trainable input copies and the public
    :func:`jvp` implementation. Numerical plus/minus evaluations start from the
    same NumPy RNG state and run under ``no_grad()``, so randomized functions
    see matching randomness and the caller's random stream is not consumed.
    """
    if not callable(function):
        raise TypeError("jvpcheck function must be callable")
    sources, single_input = _normalise_inputs(inputs)
    tangent_arrays = _normalise_tangents(
        sources,
        tangents,
        single_input=single_input,
    )
    eps = _validate_tolerance("eps", eps, strictly_positive=True)
    atol = _validate_tolerance("atol", atol, strictly_positive=False)
    rtol = _validate_tolerance("rtol", rtol, strictly_positive=False)

    rng_state = np.random.get_state()
    try:
        analytical_inputs = tuple(
            Tensor(value.data.copy(), requires_grad=True) for value in sources
        )
        np.random.set_state(rng_state)
        output = function(*analytical_inputs)
        _validate_output(output, "analytical")
        if output.data.size == 0:
            raise ValueError("jvpcheck function output must not be empty")
        if not np.isfinite(output.data).all():
            raise ValueError("jvpcheck function output must contain only finite values")

        analytical_structure = (
            analytical_inputs[0] if single_input else analytical_inputs
        )
        tangent_structure = tangent_arrays[0] if single_input else tangent_arrays
        analytical = jvp(output, analytical_structure, tangent_structure)
        if not np.isfinite(analytical).all():
            raise ValueError("jvpcheck analytical JVP must contain only finite values")

        plus_inputs, minus_inputs = _perturbed_inputs(sources, tangent_arrays, eps)
        plus = _evaluate(function, plus_inputs, output.shape, rng_state, "plus")
        minus = _evaluate(function, minus_inputs, output.shape, rng_state, "minus")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            numerical = (plus - minus) / (2.0 * eps)
        if not np.isfinite(numerical).all():
            raise ValueError(
                "jvpcheck finite-difference JVP must contain only finite values"
            )

        _assert_close(analytical, numerical, atol=atol, rtol=rtol)
        return True
    finally:
        np.random.set_state(rng_state)


def _normalise_inputs(inputs):
    single_input = isinstance(inputs, Tensor)
    if single_input:
        sources = (inputs,)
    else:
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Iterable):
            raise TypeError("jvpcheck inputs must be a Tensor or iterable of Tensors")
        sources = tuple(inputs)

    if not sources:
        raise ValueError("jvpcheck inputs must contain at least one Tensor")
    for index, value in enumerate(sources):
        if not isinstance(value, Tensor):
            raise TypeError(f"jvpcheck input {index} must be a Tensor")
        if value.data.size == 0:
            raise ValueError(f"jvpcheck input {index} must not be empty")
        if not np.isfinite(value.data).all():
            raise ValueError(
                f"jvpcheck input {index} must contain only finite values"
            )
    return sources, single_input


def _normalise_tangents(sources, tangents, *, single_input):
    if single_input:
        supplied = (tangents,)
    else:
        if isinstance(tangents, (str, bytes)) or not isinstance(tangents, Iterable):
            raise TypeError(
                "jvpcheck tangents must be an iterable with one value per input"
            )
        supplied = tuple(tangents)
        if len(supplied) != len(sources):
            raise ValueError(
                "jvpcheck tangents must contain exactly one value per input"
            )

    validated = []
    for index, (source, tangent) in enumerate(zip(sources, supplied)):
        raw = np.asarray(tangent)
        is_integer = np.issubdtype(raw.dtype, np.integer)
        is_floating = np.issubdtype(raw.dtype, np.floating)
        if np.issubdtype(raw.dtype, np.bool_) or not (is_integer or is_floating):
            raise TypeError(
                f"jvpcheck tangent {index} must contain real numeric values"
            )
        value = np.array(raw, dtype=np.float64, copy=True)
        if value.shape != source.shape:
            raise ValueError(
                f"jvpcheck tangent {index} shape mismatch: expected {source.shape}, "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"jvpcheck tangent {index} must contain only finite values"
            )
        validated.append(value)
    return tuple(validated)


def _validate_tolerance(name, value, *, strictly_positive):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"jvpcheck {name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"jvpcheck {name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"jvpcheck {name} must be finite")
    if strictly_positive and value <= 0.0:
        raise ValueError(f"jvpcheck {name} must be positive")
    if not strictly_positive and value < 0.0:
        raise ValueError(f"jvpcheck {name} must be non-negative")
    return value


def _perturbed_inputs(sources, tangents, eps):
    plus = []
    minus = []
    for index, (source, tangent) in enumerate(zip(sources, tangents)):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            plus_data = np.asarray(source.data) + eps * tangent
            minus_data = np.asarray(source.data) - eps * tangent
        if not np.isfinite(plus_data).all() or not np.isfinite(minus_data).all():
            raise ValueError(
                f"jvpcheck perturbation for input {index} must remain finite"
            )
        plus.append(Tensor(plus_data))
        minus.append(Tensor(minus_data))
    return tuple(plus), tuple(minus)


def _evaluate(function, inputs, expected_shape, rng_state, label):
    np.random.set_state(rng_state)
    with no_grad():
        output = function(*inputs)
    _validate_output(output, label)
    if output.shape != expected_shape:
        raise ValueError(
            "jvpcheck function output shape changed during finite differences: "
            f"expected {expected_shape}, got {output.shape}"
        )
    if not np.isfinite(output.data).all():
        raise ValueError(
            f"jvpcheck {label} output must contain only finite values"
        )
    return np.array(output.data, dtype=np.float64, copy=True)


def _validate_output(output, label):
    if not isinstance(output, Tensor):
        raise TypeError(f"jvpcheck {label} function evaluation must return a Tensor")


def _assert_close(analytical, numerical, *, atol, rtol):
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
        f"jvpcheck failed at output index {index}: analytical={actual:.12g}, "
        f"numerical={expected:.12g}, abs_error={absolute_error:.3g}, "
        f"tolerance={tolerance:.3g}"
    )
