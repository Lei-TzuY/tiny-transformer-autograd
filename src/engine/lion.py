"""Lion optimizer with transactional NumPy updates.

Lion (Evolved Sign Momentum) uses the sign of a momentum/gradient blend for
parameter updates while maintaining a second momentum blend for future steps.
This module is intentionally standalone so experiments can opt into Lion
without changing the repository's existing optimizer module or trainer CLI.
"""

from collections.abc import Mapping
from numbers import Integral, Real
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1


def _real_scalar(name, value, *, positive=False, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and value >= upper:
        raise ValueError(f"{name} must be less than {upper}")
    return value


def _validate_betas(value, name="Lion betas"):
    try:
        betas = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real numbers") from exc
    if len(betas) != 2:
        raise ValueError(f"{name} must contain two values")
    return tuple(
        _real_scalar(
            f"{name}[{index}]", beta, lower=0.0, upper=1.0
        )
        for index, beta in enumerate(betas)
    )


def _bool_flag(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _nonnegative_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _as_float64_array(value, *, label, shape=None):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if shape is not None and value.shape != shape:
        raise ValueError(
            f"{label} shape mismatch: expected {shape}, got {value.shape}"
        )
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise TypeError(f"{label} must have a real numeric dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must contain only finite values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = np.asarray(value, dtype=np.float64)
    if not np.isfinite(narrowed).all():
        raise ValueError(f"{label} contains values not representable as float64")
    return np.array(narrowed, dtype=np.float64, copy=True, subok=False)


def _same_array(left, right):
    return left.shape == right.shape and np.array_equal(left, right)


def _decay_factor(lr, weight_decay):
    if weight_decay == 0.0:
        return 1.0
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            product = np.multiply(np.float64(lr), np.float64(weight_decay))
            factor = np.subtract(np.float64(1.0), product)
    except FloatingPointError as exc:
        raise ValueError("Lion lr * weight_decay must remain finite") from exc
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("Lion weight decay factor must be finite")
    return factor


def _lion_candidates(parameter, momentum, gradient, *, lr, beta1, beta2, decay):
    """Return candidate parameter and momentum arrays without mutating state."""
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            update_blend = beta1 * momentum + (1.0 - beta1) * gradient
            direction = np.sign(update_blend)
            next_momentum = beta2 * momentum + (1.0 - beta2) * gradient
            next_parameter = parameter * decay - lr * direction
    except FloatingPointError as exc:
        raise ValueError("Lion step produced an unrepresentable finite update") from exc

    if not np.isfinite(next_momentum).all():
        raise ValueError("Lion momentum update must remain finite")
    if not np.isfinite(next_parameter).all():
        raise ValueError("Lion parameter update must remain finite")
    return (
        np.array(next_parameter, dtype=np.float64, copy=True, subok=False),
        np.array(next_momentum, dtype=np.float64, copy=True, subok=False),
    )


class Lion:
    """Evolved Sign Momentum optimizer with decoupled weight decay.

    For an active gradient ``g`` and momentum ``m`` the update is::

        direction = sign(beta1 * m + (1 - beta1) * g)
        parameter = parameter * (1 - lr * weight_decay) - lr * direction
        m = beta2 * m + (1 - beta2) * g

    Candidate values for every active parameter are computed and validated before
    the first live write. A late malformed gradient, shape drift, read-only
    destination, or numerical overflow therefore cannot partially advance an
    earlier parameter.
    """

    def __init__(
        self,
        parameters,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    ):
        self._lock = threading.RLock()
        materialized = list(parameters)
        seen = set()
        for index, parameter in enumerate(materialized):
            if not isinstance(parameter, Tensor):
                raise TypeError(f"Lion parameter {index} must be a Tensor")
            marker = id(parameter)
            if marker in seen:
                raise ValueError(
                    "Lion parameters must not contain duplicate references: "
                    f"duplicate at index {index}"
                )
            seen.add(marker)

        self.parameters = materialized
        self._parameter_container = self.parameters
        self._parameter_ids = tuple(id(parameter) for parameter in materialized)
        self._parameter_shapes = tuple(parameter.data.shape for parameter in materialized)

        self._lr = _real_scalar("Lion lr", lr, positive=True)
        self._beta1, self._beta2 = _validate_betas(betas)
        self._weight_decay = _real_scalar(
            "Lion weight_decay", weight_decay, lower=0.0
        )
        self.step_count = 0
        self._steps = [0 for _ in materialized]
        self._momentum = [
            np.zeros(parameter.data.shape, dtype=np.float64) for parameter in materialized
        ]

    @property
    def lr(self):
        with self._lock:
            return self._lr

    @lr.setter
    def lr(self, value):
        value = _real_scalar("Lion lr", value, positive=True)
        with self._lock:
            self._lr = value

    @property
    def betas(self):
        with self._lock:
            return (self._beta1, self._beta2)

    @betas.setter
    def betas(self, value):
        beta1, beta2 = _validate_betas(value)
        with self._lock:
            self._beta1, self._beta2 = beta1, beta2

    @property
    def beta1(self):
        with self._lock:
            return self._beta1

    @property
    def beta2(self):
        with self._lock:
            return self._beta2

    @property
    def weight_decay(self):
        with self._lock:
            return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, value):
        value = _real_scalar("Lion weight_decay", value, lower=0.0)
        with self._lock:
            self._weight_decay = value

    def _validate_binding(self):
        if self.parameters is not self._parameter_container:
            raise RuntimeError("Lion parameter list was replaced after construction")
        if len(self.parameters) != len(self._parameter_ids):
            raise RuntimeError("Lion parameter count changed after construction")
        for index, parameter in enumerate(self.parameters):
            if not isinstance(parameter, Tensor):
                raise TypeError(f"Lion parameter {index} must remain a Tensor")
            if id(parameter) != self._parameter_ids[index]:
                raise RuntimeError(
                    f"Lion parameter identity/order changed at index {index}"
                )
            if parameter.data.shape != self._parameter_shapes[index]:
                raise ValueError(
                    f"Lion parameter shape changed at index {index}: expected "
                    f"{self._parameter_shapes[index]}, got {parameter.data.shape}"
                )
        if len(self._momentum) != len(self.parameters):
            raise RuntimeError("Lion momentum count does not match parameter count")
        if len(self._steps) != len(self.parameters):
            raise RuntimeError("Lion parameter step count does not match parameter count")

    def _validated_active_gradients(self):
        gradients = []
        for index, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            if gradient is None:
                gradients.append(None)
                continue
            if not np.isfinite(np.asarray(parameter.data)).all():
                raise ValueError(
                    f"Lion parameter {index} must contain only finite values before step()"
                )
            gradient = _as_float64_array(
                gradient,
                label=f"Lion gradient for parameter {index}",
                shape=parameter.data.shape,
            )
            momentum = self._momentum[index]
            if not isinstance(momentum, np.ndarray):
                raise TypeError(f"Lion momentum[{index}] must be a NumPy array")
            if momentum.shape != parameter.data.shape:
                raise ValueError(
                    f"Lion momentum shape mismatch at index {index}: expected "
                    f"{parameter.data.shape}, got {momentum.shape}"
                )
            if momentum.dtype != np.float64:
                raise TypeError(f"Lion momentum[{index}] must have dtype float64")
            if not np.isfinite(momentum).all():
                raise ValueError(f"Lion momentum[{index}] must contain only finite values")
            gradients.append(gradient)
        return gradients

    def step(self):
        """Apply one transactionally precomputed Lion update."""
        with self._lock:
            self._validate_binding()
            gradients = self._validated_active_gradients()
            decay = _decay_factor(self._lr, self._weight_decay)

            candidates = []
            for index, (parameter, gradient) in enumerate(
                zip(self.parameters, gradients)
            ):
                if gradient is None:
                    continue
                next_parameter, next_momentum = _lion_candidates(
                    np.asarray(parameter.data),
                    self._momentum[index],
                    gradient,
                    lr=self._lr,
                    beta1=self._beta1,
                    beta2=self._beta2,
                    decay=decay,
                )
                parameter_changed = not _same_array(
                    np.asarray(parameter.data), next_parameter
                )
                momentum_changed = not _same_array(
                    self._momentum[index], next_momentum
                )
                if parameter_changed and not parameter.data.flags.writeable:
                    raise ValueError(
                        f"Lion parameter {index} storage must be writeable for step()"
                    )
                if momentum_changed and not self._momentum[index].flags.writeable:
                    raise ValueError(
                        f"Lion momentum[{index}] must be writeable for step()"
                    )
                candidates.append(
                    (
                        index,
                        next_parameter,
                        next_momentum,
                        parameter_changed,
                        momentum_changed,
                    )
                )

            parameter_snapshots = {
                index: self.parameters[index].data.copy() for index, *_ in candidates
            }
            momentum_snapshots = {
                index: self._momentum[index].copy() for index, *_ in candidates
            }

            try:
                for (
                    index,
                    next_parameter,
                    next_momentum,
                    parameter_changed,
                    momentum_changed,
                ) in candidates:
                    if momentum_changed:
                        self._momentum[index][...] = next_momentum
                    if parameter_changed:
                        self.parameters[index].data[...] = next_parameter
            except BaseException:
                for index, saved in momentum_snapshots.items():
                    try:
                        self._momentum[index][...] = saved
                    except BaseException:
                        self._momentum[index] = saved.copy()
                for index, saved in parameter_snapshots.items():
                    parameter = self.parameters[index]
                    if _same_array(np.asarray(parameter.data), saved):
                        continue
                    try:
                        parameter.data[...] = saved
                    except BaseException:
                        parameter.data = saved
                raise

            self.step_count += 1
            for index, *_ in candidates:
                self._steps[index] += 1
            return self

    def zero_grad(self, set_to_none=False):
        """Clear gradients with full preflight for the in-place zeroing path."""
        set_to_none = _bool_flag("set_to_none", set_to_none)
        with self._lock:
            self._validate_binding()
            if not set_to_none:
                for index, parameter in enumerate(self.parameters):
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if not isinstance(gradient, np.ndarray):
                        raise TypeError(
                            f"Lion gradient for parameter {index} must be a NumPy array"
                        )
                    if gradient.shape != parameter.data.shape:
                        raise ValueError(
                            f"Lion gradient shape mismatch for parameter {index}: expected "
                            f"{parameter.data.shape}, got {gradient.shape}"
                        )
                    if not gradient.flags.writeable:
                        raise ValueError(
                            f"Lion gradient for parameter {index} must be writeable to zero"
                        )
            for parameter in self.parameters:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad[...] = 0.0
            return self

    def state_dict(self):
        """Return an independent serializable snapshot of Lion optimizer state."""
        with self._lock:
            self._validate_binding()
            momenta = []
            for index, (parameter, momentum) in enumerate(
                zip(self.parameters, self._momentum)
            ):
                validated = _as_float64_array(
                    momentum,
                    label=f"Lion momentum[{index}]",
                    shape=parameter.data.shape,
                )
                momenta.append(validated)
            return {
                "format_version": _STATE_VERSION,
                "optimizer": "Lion",
                "lr": self._lr,
                "betas": (self._beta1, self._beta2),
                "weight_decay": self._weight_decay,
                "step_count": self.step_count,
                "steps": list(self._steps),
                "momentum": momenta,
            }

    def load_state_dict(self, state):
        """Validate the complete state before committing any Lion field."""
        with self._lock:
            self._validate_binding()
            if not isinstance(state, Mapping):
                raise TypeError("Lion state must be a mapping")

            required = (
                "format_version",
                "optimizer",
                "lr",
                "betas",
                "weight_decay",
                "step_count",
                "steps",
                "momentum",
            )
            for key in required:
                if key not in state:
                    raise KeyError(f"Lion state missing required key: {key}")

            version = _nonnegative_integer(
                "Lion format_version", state["format_version"]
            )
            if version != _STATE_VERSION:
                raise ValueError(
                    f"unsupported Lion state format_version {version}; "
                    f"expected {_STATE_VERSION}"
                )
            if state["optimizer"] != "Lion":
                raise ValueError("Lion state optimizer must be 'Lion'")

            lr = _real_scalar("Lion state lr", state["lr"], positive=True)
            beta1, beta2 = _validate_betas(state["betas"], "Lion state betas")
            weight_decay = _real_scalar(
                "Lion state weight_decay", state["weight_decay"], lower=0.0
            )
            step_count = _nonnegative_integer(
                "Lion state step_count", state["step_count"]
            )

            saved_steps = state["steps"]
            if not isinstance(saved_steps, (list, tuple)):
                raise TypeError("Lion state steps must be a list or tuple")
            if len(saved_steps) != len(self.parameters):
                raise ValueError(
                    "Lion state step count mismatch: expected "
                    f"{len(self.parameters)}, got {len(saved_steps)}"
                )
            steps = []
            for index, value in enumerate(saved_steps):
                parameter_step = _nonnegative_integer(
                    f"Lion state steps[{index}]", value
                )
                if parameter_step > step_count:
                    raise ValueError(
                        f"Lion state steps[{index}] cannot exceed step_count "
                        f"{step_count}"
                    )
                steps.append(parameter_step)

            saved_momenta = state["momentum"]
            if not isinstance(saved_momenta, (list, tuple)):
                raise TypeError("Lion state momentum must be a list or tuple")
            if len(saved_momenta) != len(self.parameters):
                raise ValueError(
                    "Lion state momentum count mismatch: expected "
                    f"{len(self.parameters)}, got {len(saved_momenta)}"
                )
            momenta = []
            for index, (parameter, saved) in enumerate(
                zip(self.parameters, saved_momenta)
            ):
                momenta.append(
                    _as_float64_array(
                        saved,
                        label=f"Lion state momentum[{index}]",
                        shape=parameter.data.shape,
                    )
                )
                if not self._momentum[index].flags.writeable:
                    raise ValueError(
                        f"Lion momentum[{index}] must be writeable to load state"
                    )

            old = {
                "lr": self._lr,
                "beta1": self._beta1,
                "beta2": self._beta2,
                "weight_decay": self._weight_decay,
                "step_count": self.step_count,
                "steps": list(self._steps),
                "momentum": [value.copy() for value in self._momentum],
            }
            try:
                for current, saved in zip(self._momentum, momenta):
                    current[...] = saved
                self._lr = lr
                self._beta1, self._beta2 = beta1, beta2
                self._weight_decay = weight_decay
                self.step_count = step_count
                self._steps = steps
            except BaseException:
                self._lr = old["lr"]
                self._beta1, self._beta2 = old["beta1"], old["beta2"]
                self._weight_decay = old["weight_decay"]
                self.step_count = old["step_count"]
                self._steps = old["steps"]
                for index, saved in enumerate(old["momentum"]):
                    try:
                        self._momentum[index][...] = saved
                    except BaseException:
                        self._momentum[index] = saved.copy()
                raise
            return self
