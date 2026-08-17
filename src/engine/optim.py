"""
optim.py — Parameter optimizers: SGD, Adam, and AdamW.

All work with lists of Tensor objects that have requires_grad=True.
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

AdamW  (Loshchilov & Hutter, 2019)
-----------------------------------
Same moments as Adam, but weight decay is *decoupled*: it never enters
m/v, so the adaptive scaling does not distort the regularisation:
    p ← p - lr·λ·p - lr · m̂_t / (√v̂_t + ε)
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
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Velocity buffers (only used when momentum > 0)
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
        lr = state["lr"]
        momentum = state["momentum"]
        weight_decay = state["weight_decay"]
        saved_v = state["v"]
        if lr < 0 or not 0.0 <= momentum < 1.0 or weight_decay < 0:
            raise ValueError("invalid SGD optimizer state")
        _validate_buffers(self._v, saved_v, "SGD velocity")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        _copy_buffers(self._v, saved_v)


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
        if lr <= 0:
            raise ValueError("lr must be positive")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # time step

        self._m = [np.zeros_like(p.data) for p in self.parameters]  # 1st moment
        self._v = [np.zeros_like(p.data) for p in self.parameters]  # 2nd moment

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
        lr = state["lr"]
        betas = state["betas"]
        eps = state["eps"]
        weight_decay = state["weight_decay"]
        step = state["t"]
        saved_m = state["m"]
        saved_v = state["v"]
        if (
            lr < 0
            or len(betas) != 2
            or not all(0.0 <= beta < 1.0 for beta in betas)
            or eps <= 0
            or weight_decay < 0
            or step < 0
        ):
            raise ValueError("invalid Adam optimizer state")
        _validate_buffers(self._m, saved_m, "Adam first moment")
        _validate_buffers(self._v, saved_v, "Adam second moment")

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = step
        _copy_buffers(self._m, saved_m)
        _copy_buffers(self._v, saved_v)


class AdamW(Adam):
    """
    Adam with decoupled weight decay.

    In Adam, weight_decay is classic L2: λ·p is added to the gradient and
    therefore flows through the m/v moment estimates, where the adaptive
    denominator shrinks it for parameters with large gradient variance.
    AdamW instead decays the weights directly, keeping regularisation
    strength independent of gradient statistics:

        p ← p − lr·λ·p − lr · m̂ / (√v̂ + ε)
    """

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


def _validate_buffers(destination, source, label):
    if len(destination) != len(source):
        raise ValueError(
            f"{label} count mismatch: expected {len(destination)}, got {len(source)}"
        )
    for current, saved in zip(destination, source):
        if current.shape != saved.shape:
            raise ValueError(
                f"{label} shape mismatch: expected {current.shape}, got {saved.shape}"
            )


def _copy_buffers(destination, source):
    for current, saved in zip(destination, source):
        current[:] = saved
