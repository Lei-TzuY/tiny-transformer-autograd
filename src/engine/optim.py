"""
optim.py — Parameter optimizers: SGD and Adam.

Both work with lists of Tensor objects that have requires_grad=True.
The update rules operate directly on tensor.data (in-place NumPy), so
they are not part of the computational graph themselves.

SGD update rule
---------------
    p ← p - lr * p.grad

Adam update rule  (Kingma & Ba, 2015)
--------------------------------------
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
    m̂_t = m_t / (1 - β₁ᵗ)   ← bias correction
    v̂_t = v_t / (1 - β₂ᵗ)
    p ← p - lr · m̂_t / (√v̂_t + ε)
"""

import numpy as np


class SGD:
    """Stochastic Gradient Descent (with optional momentum)."""

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        Parameters
        ----------
        parameters : list[Tensor]
        lr         : learning rate
        momentum   : momentum coefficient (0 = plain SGD)
        weight_decay : L2 regularization coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Velocity buffers (only used when momentum > 0)
        self._v = [np.zeros_like(p.data) for p in parameters]

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
        self.lr = state["lr"]
        self.momentum = state["momentum"]
        self.weight_decay = state["weight_decay"]
        _load_buffers(self._v, state["v"], "SGD velocity")


class Adam:
    """
    Adam optimizer with bias correction.
    Default hyper-params from the original paper: lr=1e-3, β=(0.9,0.999), ε=1e-8.
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # time step

        self._m = [np.zeros_like(p.data) for p in parameters]  # 1st moment
        self._v = [np.zeros_like(p.data) for p in parameters]  # 2nd moment

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t  # bias correction denominators
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
        self.lr = state["lr"]
        self.beta1, self.beta2 = state["betas"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]
        self.t = state["t"]
        _load_buffers(self._m, state["m"], "Adam first moment")
        _load_buffers(self._v, state["v"], "Adam second moment")


def _load_buffers(destination, source, label):
    if len(destination) != len(source):
        raise ValueError(
            f"{label} count mismatch: expected {len(destination)}, got {len(source)}"
        )
    for current, saved in zip(destination, source):
        if current.shape != saved.shape:
            raise ValueError(
                f"{label} shape mismatch: expected {current.shape}, got {saved.shape}"
            )
        current[:] = saved
