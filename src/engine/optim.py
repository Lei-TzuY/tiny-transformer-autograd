"""
optim.py — Parameter optimizers: SGD, Adam, and AdamW.

All work with lists of Tensor objects that have requires_grad=True.
The update rules operate directly on tensor.data (in-place NumPy), so
they are not part of the computational graph themselves.
"""

from numbers import Integral, Real

import numpy as np


class SGD:
    """Stochastic Gradient Descent (with optional momentum)."""

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.parameters = _unique_parameters(parameters)
        self.lr = _real_scalar("lr", lr, positive=True)
        self.momentum = _real_scalar(
            "momentum", momentum, lower=0.0, upper=1.0, upper_inclusive=False
        )
        self.weight_decay = _real_scalar(
            "weight_decay", weight_decay, lower=0.0
        )
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        _validate_step_inputs(self.parameters)
        for p, v in zip(self.parameters, self._v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                v[:] = self.momentum * v + g
                p.data -= self.lr * v
            else:
                p.data -= self.lr * g

    def zero_grad(self, set_to_none=False):
        set_to_none = _bool_flag("set_to_none", set_to_none)
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad[:] = 0.0

    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "v": [value.copy() for value in self._v],
        }

    def load_state_dict(self, state):
        lr = _real_scalar("SGD lr", state["lr"], positive=True)
        momentum = _real_scalar(
            "SGD momentum",
            state["momentum"],
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        weight_decay = _real_scalar(
            "SGD weight_decay", state["weight_decay"], lower=0.0
        )
        saved_v = _snapshot_buffers(self._v, state["v"], "SGD velocity")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        _copy_buffers(self._v, saved_v)


class Adam:
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        self.parameters = _unique_parameters(parameters)
        self.lr = _real_scalar("lr", lr, positive=True)
        self.beta1, self.beta2 = _validate_betas(betas, "betas")
        self.eps = _real_scalar("eps", eps, positive=True)
        self.weight_decay = _real_scalar(
            "weight_decay", weight_decay, lower=0.0
        )
        # Keep the historical global call counter for compatibility, while
        # bias correction uses each parameter's actual number of moment updates.
        self.t = 0
        self._steps = [0 for _ in self.parameters]
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        _validate_step_inputs(self.parameters)
        self.t += 1

        for index, (p, m, v) in enumerate(zip(self.parameters, self._m, self._v)):
            if p.grad is None:
                continue
            self._steps[index] += 1
            parameter_step = self._steps[index]
            bc1 = 1.0 - self.beta1 ** parameter_step
            bc2 = 1.0 - self.beta2 ** parameter_step

            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * g * g
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, set_to_none=False):
        set_to_none = _bool_flag("set_to_none", set_to_none)
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad[:] = 0.0

    def state_dict(self):
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "steps": list(self._steps),
            "m": [value.copy() for value in self._m],
            "v": [value.copy() for value in self._v],
        }

    def load_state_dict(self, state):
        lr = _real_scalar("Adam lr", state["lr"], positive=True)
        beta1, beta2 = _validate_betas(state["betas"], "Adam betas")
        eps = _real_scalar("Adam eps", state["eps"], positive=True)
        weight_decay = _real_scalar(
            "Adam weight_decay", state["weight_decay"], lower=0.0
        )
        step = _nonnegative_step(state["t"], "Adam step")
        parameter_steps = _parameter_steps_from_state(
            state.get("steps"), step, len(self.parameters)
        )
        saved_m = _snapshot_buffers(
            self._m, state["m"], "Adam first moment"
        )
        saved_v = _snapshot_buffers(
            self._v, state["v"], "Adam second moment"
        )

        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = step
        self._steps = parameter_steps
        _copy_buffers(self._m, saved_m)
        _copy_buffers(self._v, saved_v)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def step(self):
        _validate_step_inputs(self.parameters)
        self.t += 1

        for index, (p, m, v) in enumerate(zip(self.parameters, self._m, self._v)):
            if p.grad is None:
                continue
            self._steps[index] += 1
            parameter_step = self._steps[index]
            bc1 = 1.0 - self.beta1 ** parameter_step
            bc2 = 1.0 - self.beta2 ** parameter_step

            g = p.grad
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * g * g
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


def _unique_parameters(parameters):
    """Materialize parameters and reject duplicate Tensor references."""
    materialized = list(parameters)
    seen = set()
    for index, parameter in enumerate(materialized):
        marker = id(parameter)
        if marker in seen:
            raise ValueError(
                "optimizer parameters must not contain duplicate references: "
                f"duplicate at index {index}"
            )
        seen.add(marker)
    return materialized


def _bool_flag(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _real_scalar(
    name,
    value,
    *,
    positive=False,
    lower=None,
    upper=None,
    upper_inclusive=True,
):
    """Validate one finite real optimizer hyperparameter and return float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None:
        invalid = value > upper if upper_inclusive else value >= upper
        if invalid:
            relation = "at most" if upper_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {upper}")
    return value


def _validate_betas(betas, name):
    try:
        values = tuple(betas)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real numbers") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    return tuple(
        _real_scalar(
            f"{name}[{index}]",
            beta,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        for index, beta in enumerate(values)
    )


def _nonnegative_step(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _parameter_steps_from_state(saved_steps, total_step, parameter_count):
    """Load per-parameter Adam steps, migrating scalar-only legacy states."""
    if saved_steps is None:
        return [total_step for _ in range(parameter_count)]
    if not isinstance(saved_steps, (list, tuple)):
        raise TypeError("Adam parameter steps must be a list or tuple")
    if len(saved_steps) != parameter_count:
        raise ValueError(
            "Adam parameter step count mismatch: "
            f"expected {parameter_count}, got {len(saved_steps)}"
        )

    steps = []
    for index, value in enumerate(saved_steps):
        parameter_step = _nonnegative_step(value, f"Adam parameter step[{index}]")
        if parameter_step > total_step:
            raise ValueError(
                f"Adam parameter step[{index}] cannot exceed optimizer step {total_step}"
            )
        steps.append(parameter_step)
    return steps


def _validate_step_inputs(parameters):
    """Validate every active gradient before an optimizer mutates any state."""
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            continue
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"gradient for parameter {index} must be a NumPy array")
        if gradient.shape != parameter.data.shape:
            raise ValueError(
                f"gradient shape mismatch for parameter {index}: expected "
                f"{parameter.data.shape}, got {gradient.shape}"
            )
        if (
            not np.issubdtype(gradient.dtype, np.number)
            or np.issubdtype(gradient.dtype, np.complexfloating)
        ):
            raise TypeError(
                f"gradient for parameter {index} must have a real numeric dtype"
            )
        if not np.isfinite(gradient).all():
            raise ValueError(
                f"gradient for parameter {index} must contain only finite values"
            )
        if not np.isfinite(parameter.data).all():
            raise ValueError(
                f"parameter {index} must contain only finite values before step()"
            )


def _snapshot_buffers(destination, source, label):
    """Validate and detach saved buffers before any optimizer state mutation."""
    _validate_buffers(destination, source, label)
    return [np.array(saved, copy=True) for saved in source]


def _validate_buffers(destination, source, label):
    if not isinstance(source, (list, tuple)):
        raise TypeError(f"{label} buffers must be a list or tuple")
    if len(destination) != len(source):
        raise ValueError(
            f"{label} count mismatch: expected {len(destination)}, got {len(source)}"
        )
    for index, (current, saved) in enumerate(zip(destination, source)):
        if not isinstance(saved, np.ndarray):
            raise TypeError(f"{label}[{index}] must be a NumPy array")
        if saved.shape != current.shape:
            raise ValueError(
                f"{label} shape mismatch at index {index}: expected "
                f"{current.shape}, got {saved.shape}"
            )
        if not np.issubdtype(saved.dtype, np.number) or np.issubdtype(
            saved.dtype, np.complexfloating
        ):
            raise TypeError(f"{label}[{index}] must have a real numeric dtype")
        if not np.isfinite(saved).all():
            raise ValueError(f"{label}[{index}] must contain only finite values")


def _copy_buffers(destination, source):
    for current, saved in zip(destination, source):
        current[:] = saved
