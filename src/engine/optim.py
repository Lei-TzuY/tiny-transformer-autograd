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
        self.parameters = list(parameters)
        self.lr = _real_scalar("lr", lr, positive=True)
        self.momentum = _real_scalar(
            "momentum", momentum, lower=0.0, upper=1.0, upper_inclusive=False
        )
        self.weight_decay = _real_scalar(
            "weight_decay", weight_decay, lower=0.0
        )
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
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

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
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
        saved_v = state["v"]
        _validate_buffers(self._v, saved_v, "SGD velocity")

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
        self.parameters = list(parameters)
        self.lr = _real_scalar("lr", lr, positive=True)
        self.beta1, self.beta2 = _validate_betas(betas, "betas")
        self.eps = _real_scalar("eps", eps, positive=True)
        self.weight_decay = _real_scalar(
            "weight_decay", weight_decay, lower=0.0
        )
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for p, m, v in zip(self.parameters, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * g * g
            m_hat = m / bc1
            v_hat = v / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad[:] = 0.0

    def state_dict(self):
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
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
        saved_m = state["m"]
        saved_v = state["v"]
        _validate_buffers(self._m, saved_m, "Adam first moment")
        _validate_buffers(self._v, saved_v, "Adam second moment")

        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = step
        _copy_buffers(self._m, saved_m)
        _copy_buffers(self._v, saved_v)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for p, m, v in zip(self.parameters, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * g * g
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


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
