"""Gradient noise-scale diagnostics from repeated micro-batch gradients."""

from collections.abc import Iterable, Mapping
import math
from numbers import Integral
import sys
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "GradientNoiseScaleEstimator"


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if normalized > sys.maxsize:
        raise ValueError(f"{name} must be at most sys.maxsize")
    return normalized


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if normalized > sys.maxsize:
        raise ValueError(f"{name} must be at most sys.maxsize")
    return normalized


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        materialized = (parameters,)
    else:
        try:
            materialized = tuple(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be a Tensor or iterable of Tensors") from exc

    seen = set()
    for index, parameter in enumerate(materialized):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(marker)
        if parameter.requires_grad is not True:
            raise ValueError(f"parameter {index} must require gradients")
    return materialized


def _normalize_gradient(gradient, shape, index):
    if gradient is None:
        return np.zeros(shape, dtype=np.float64)
    if not isinstance(gradient, np.ndarray):
        raise TypeError(f"gradient {index} must be a NumPy array or None")
    if gradient.shape != shape:
        raise ValueError(f"gradient {index} shape must be {shape}, got {gradient.shape}")
    if not np.issubdtype(gradient.dtype, np.floating):
        raise TypeError(f"gradient {index} must have floating dtype")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"gradient {index} must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            normalized = np.asarray(gradient, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"gradient {index} must fit float64") from exc
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"gradient {index} must fit float64")
    return normalized


def _stable_equal_mean(arrays):
    if not arrays:
        raise ValueError("cannot average an empty array sequence")
    shape = arrays[0].shape
    mean = arrays[0].reshape(-1).copy()
    count = 1
    for current_array in arrays[1:]:
        current = current_array.reshape(-1)
        total = count + 1
        old_fraction = count / total
        new_fraction = 1.0 / total
        same_sign = ((mean >= 0.0) & (current >= 0.0)) | (
            (mean <= 0.0) & (current <= 0.0)
        )
        candidate = np.empty_like(mean)
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                if np.any(same_sign):
                    candidate[same_sign] = mean[same_sign] + new_fraction * (
                        current[same_sign] - mean[same_sign]
                    )
                opposite = ~same_sign
                if np.any(opposite):
                    candidate[opposite] = (
                        mean[opposite] * old_fraction
                        + current[opposite] * new_fraction
                    )
        except FloatingPointError as exc:
            raise ValueError("mean gradient is not representable as float64") from exc
        if not np.all(np.isfinite(candidate)):
            raise ValueError("mean gradient is not representable as float64")
        mean = candidate
        count = total
    return mean.reshape(shape)


# A scaled non-negative value is either None for exact zero or (mantissa, exponent)
# with value = mantissa * 2**exponent and 0.5 <= mantissa < 1.
def _scaled_square(scale, normalized_squared_sum):
    if scale == 0.0 or normalized_squared_sum == 0.0:
        return None
    scale_mantissa, scale_exponent = math.frexp(scale)
    sum_mantissa, sum_exponent = math.frexp(normalized_squared_sum)
    product = scale_mantissa * scale_mantissa * sum_mantissa
    mantissa, exponent = math.frexp(product)
    return mantissa, exponent + 2 * scale_exponent + sum_exponent


def _scaled_add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    left_mantissa, left_exponent = left
    right_mantissa, right_exponent = right
    if right_exponent > left_exponent:
        left_mantissa, right_mantissa = right_mantissa, left_mantissa
        left_exponent, right_exponent = right_exponent, left_exponent
    aligned = math.ldexp(right_mantissa, right_exponent - left_exponent)
    mantissa, exponent = math.frexp(left_mantissa + aligned)
    return mantissa, exponent + left_exponent


def _scaled_multiply_float(value, factor):
    if value is None or factor == 0.0:
        return None
    if not math.isfinite(factor) or factor < 0.0:
        raise ValueError("scaled factor must be finite and non-negative")
    factor_mantissa, factor_exponent = math.frexp(factor)
    mantissa, exponent = math.frexp(value[0] * factor_mantissa)
    return mantissa, exponent + value[1] + factor_exponent


def _scaled_ratio(numerator, denominator, factor=1.0):
    if numerator is None:
        return None
    if denominator is None:
        raise ZeroDivisionError("scaled denominator is zero")
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("scaled ratio factor must be positive and finite")
    factor_mantissa, factor_exponent = math.frexp(factor)
    raw = (numerator[0] / denominator[0]) * factor_mantissa
    mantissa, exponent = math.frexp(raw)
    return (
        mantissa,
        exponent + numerator[1] - denominator[1] + factor_exponent,
    )


def _scaled_to_float(value):
    if value is None:
        return 0.0, False, False
    try:
        result = math.ldexp(value[0], value[1])
    except OverflowError:
        return None, True, False
    if not math.isfinite(result):
        return None, True, False
    if result == 0.0:
        return 0.0, False, True
    return result, False, False


def _scaled_sqrt_to_float(value):
    if value is None:
        return 0.0, False, False
    mantissa, exponent = value
    if exponent % 2:
        mantissa *= 2.0
        exponent -= 1
    root = math.sqrt(mantissa)
    root_mantissa, root_exponent = math.frexp(root)
    return _scaled_to_float((root_mantissa, root_exponent + exponent // 2))


def _max_abs(array):
    if not array.size:
        return 0.0
    return float(np.max(np.abs(array)))


def _parameter_signal_and_m2(arrays, mean):
    # Signal and noise intentionally use different scales. A huge fluctuating
    # component can cancel out of the mean while a tiny stable component remains;
    # normalizing that mean by the fluctuation scale could underflow real signal
    # to a false exact zero.
    signal_scale = _max_abs(mean)
    if signal_scale == 0.0:
        signal = None
    else:
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"
            ):
                mean_signal_normalized = mean / signal_scale
                signal_normalized = float(
                    np.sum(
                        mean_signal_normalized * mean_signal_normalized,
                        dtype=np.float64,
                    )
                )
        except FloatingPointError as exc:
            raise ValueError("gradient signal statistics are not representable") from exc
        signal = _scaled_square(signal_scale, signal_normalized)

    noise_scale = 0.0
    for array in arrays:
        noise_scale = max(noise_scale, _max_abs(array))
    if noise_scale == 0.0:
        return signal, None

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            mean_noise_normalized = mean / noise_scale
            m2_normalized_parts = []
            for array in arrays:
                residual = array / noise_scale - mean_noise_normalized
                m2_normalized_parts.append(
                    float(np.sum(residual * residual, dtype=np.float64))
                )
    except FloatingPointError as exc:
        raise ValueError("gradient noise statistics are not representable") from exc

    m2 = _scaled_square(noise_scale, math.fsum(m2_normalized_parts))
    return signal, m2


class GradientNoiseScaleEstimator:
    """Estimate gradient signal/noise from repeated equal-size batch gradients.

    Each capture is assumed to be the mean gradient from ``batch_size`` examples.
    For at least two captures, the reported noise scale is

        batch_size * tr(sample_covariance) / ||mean_gradient||^2

    using the unbiased sample covariance across captured batch gradients.
    """

    def __init__(self, parameters, batch_size):
        self._batch_size = _positive_int("batch_size", batch_size)
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        self._samples = []
        self._lock = threading.RLock()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_count(self):
        with self._lock:
            return len(self._samples)

    @property
    def parameter_count(self):
        return len(self._parameters)

    def _validate_binding_locked(self):
        for index, (parameter, shape) in enumerate(zip(self._parameters, self._shapes)):
            if parameter.shape != shape:
                raise ValueError(
                    f"parameter {index} shape changed from {shape} to {parameter.shape}"
                )
            if parameter.requires_grad is not True:
                raise ValueError(f"parameter {index} must continue to require gradients")

    def capture(self):
        """Capture the current live gradients as one independent batch sample."""
        with self._lock:
            self._validate_binding_locked()
            sample = tuple(
                _normalize_gradient(parameter.grad, shape, index)
                for index, (parameter, shape) in enumerate(
                    zip(self._parameters, self._shapes)
                )
            )
            if len(self._samples) >= sys.maxsize:
                raise OverflowError("gradient noise sample count exceeds sys.maxsize")
            self._samples.append(sample)
            return len(self._samples)

    def sample_gradients(self):
        """Return deep independent copies of every captured gradient sample."""
        with self._lock:
            return tuple(
                tuple(array.copy() for array in sample) for sample in self._samples
            )

    def mean_gradients(self):
        """Return the stable equal-weight mean gradient for each bound parameter."""
        with self._lock:
            if not self._samples:
                raise RuntimeError("gradient noise estimator has no samples")
            return tuple(
                _stable_equal_mean([sample[index] for sample in self._samples])
                for index in range(len(self._parameters))
            )

    def report(self):
        """Return a deterministic strict-JSON-safe gradient noise report."""
        with self._lock:
            sample_count = len(self._samples)
            base = {
                "batch_size": self._batch_size,
                "sample_count": sample_count,
                "parameter_count": len(self._parameters),
                "mean_gradient_l2": None,
                "mean_gradient_l2_overflow": False,
                "mean_gradient_l2_underflow": False,
                "gradient_variance_trace": None,
                "gradient_variance_trace_overflow": False,
                "gradient_variance_trace_underflow": False,
                "noise_scale": None,
                "noise_scale_defined": False,
                "noise_scale_infinite": False,
                "noise_scale_overflow": False,
                "noise_scale_underflow": False,
                "noise_scale_reason": "empty",
            }
            if sample_count == 0:
                return base

            means = tuple(
                _stable_equal_mean([sample[index] for sample in self._samples])
                for index in range(len(self._parameters))
            )
            signal = None
            m2 = None
            for index, mean in enumerate(means):
                parameter_samples = [sample[index] for sample in self._samples]
                parameter_signal, parameter_m2 = _parameter_signal_and_m2(
                    parameter_samples, mean
                )
                signal = _scaled_add(signal, parameter_signal)
                m2 = _scaled_add(m2, parameter_m2)

            mean_l2, mean_overflow, mean_underflow = _scaled_sqrt_to_float(signal)
            base["mean_gradient_l2"] = mean_l2
            base["mean_gradient_l2_overflow"] = mean_overflow
            base["mean_gradient_l2_underflow"] = mean_underflow

            if sample_count < 2:
                base["noise_scale_reason"] = "insufficient_samples"
                return base

            covariance_trace_scaled = _scaled_multiply_float(
                m2, 1.0 / (sample_count - 1)
            )
            variance, variance_overflow, variance_underflow = _scaled_to_float(
                covariance_trace_scaled
            )
            base["gradient_variance_trace"] = variance
            base["gradient_variance_trace_overflow"] = variance_overflow
            base["gradient_variance_trace_underflow"] = variance_underflow

            if signal is None:
                if m2 is None:
                    base["noise_scale_reason"] = "zero_signal_and_noise"
                else:
                    base["noise_scale_reason"] = "zero_signal"
                    base["noise_scale_infinite"] = True
                return base

            if m2 is None:
                base["noise_scale"] = 0.0
                base["noise_scale_defined"] = True
                base["noise_scale_reason"] = "ok"
                return base

            factor = self._batch_size / (sample_count - 1)
            noise_scaled = _scaled_ratio(m2, signal, factor=factor)
            noise, noise_overflow, noise_underflow = _scaled_to_float(noise_scaled)
            base["noise_scale"] = noise
            base["noise_scale_overflow"] = noise_overflow
            base["noise_scale_underflow"] = noise_underflow
            base["noise_scale_defined"] = not noise_overflow
            base["noise_scale_reason"] = "overflow" if noise_overflow else "ok"
            return base

    def reset(self):
        """Forget all captured samples without touching live gradients."""
        with self._lock:
            self._samples = []
            return self

    def state_dict(self):
        """Return independent checkpoint state for the captured samples."""
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "batch_size": self._batch_size,
                "sample_count": len(self._samples),
                "samples": [
                    [array.copy() for array in sample] for sample in self._samples
                ],
            }

    def load_state_dict(self, state):
        """Validate and transactionally restore captured gradient samples."""
        if not isinstance(state, Mapping):
            raise TypeError("gradient noise state must be a mapping")
        version = _nonnegative_int("gradient noise version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported gradient noise version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"gradient noise type must be {_STATE_TYPE!r}")
        batch_size = _positive_int("gradient noise batch_size", state.get("batch_size"))
        if batch_size != self._batch_size:
            raise ValueError(
                f"gradient noise batch_size must match bound value {self._batch_size}"
            )
        sample_count = _nonnegative_int(
            "gradient noise sample_count", state.get("sample_count")
        )

        raw_samples = state.get("samples")
        if isinstance(raw_samples, (str, bytes)) or not isinstance(raw_samples, Iterable):
            raise TypeError("gradient noise samples must be an iterable")
        raw_samples = tuple(raw_samples)
        if len(raw_samples) != sample_count:
            raise ValueError("gradient noise sample_count must match samples")

        normalized_samples = []
        for sample_index, raw_sample in enumerate(raw_samples):
            if isinstance(raw_sample, (str, bytes)) or not isinstance(raw_sample, Iterable):
                raise TypeError(f"gradient noise sample {sample_index} must be an iterable")
            raw_sample = tuple(raw_sample)
            if len(raw_sample) != len(self._parameters):
                raise ValueError(
                    f"gradient noise sample {sample_index} parameter count mismatch"
                )
            normalized_sample = []
            for parameter_index, (raw, shape) in enumerate(
                zip(raw_sample, self._shapes)
            ):
                if not isinstance(raw, np.ndarray):
                    raise TypeError(
                        f"gradient noise sample {sample_index} gradient "
                        f"{parameter_index} must be a NumPy array"
                    )
                normalized_sample.append(
                    _normalize_gradient(raw, shape, parameter_index)
                )
            normalized_samples.append(tuple(normalized_sample))

        with self._lock:
            self._validate_binding_locked()
            self._samples = normalized_samples
        return self
