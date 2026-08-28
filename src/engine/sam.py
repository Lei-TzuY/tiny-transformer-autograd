"""Sharpness-Aware Minimization (SAM) wrapper for built-in optimizers.

SAM performs one training update in two explicit phases. ``first_step()`` uses
current gradients to move live parameters to a fixed-radius neighbourhood
point. Callers then rebuild the forward graph and backward pass at that point.
``second_step()`` restores the original parameters and lets the wrapped
optimizer apply the neighbourhood gradients.
"""

from copy import deepcopy
from numbers import Integral, Real
import threading

import numpy as np

from .optim import Adam, AdamW, SGD
from .tensor import Tensor


_SUPPORTED_OPTIMIZERS = (SGD, Adam, AdamW)
_STATE_VERSION = 1


def _nonnegative_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _nonnegative_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _snapshot_parameters(optimizer):
    if not isinstance(optimizer, _SUPPORTED_OPTIMIZERS):
        raise TypeError("SAM optimizer must be SGD, Adam, or AdamW")
    container = getattr(optimizer, "parameters", None)
    if not isinstance(container, list):
        raise TypeError("SAM optimizer parameters must be stored in a list")

    parameters = []
    seen = set()
    for index, parameter in enumerate(container):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"SAM optimizer parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("SAM optimizer parameters must not contain duplicates")
        seen.add(marker)
        if not np.isfinite(np.asarray(parameter.data)).all():
            raise ValueError(
                f"SAM optimizer parameter {index} must contain only finite values"
            )
        parameters.append(parameter)
    return container, tuple(parameters)


def _validate_gradients(parameters, label):
    gradients = []
    active = 0
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            gradients.append(None)
            continue
        active += 1
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"{label} gradient for parameter {index} must be a NumPy array")
        if gradient.shape != parameter.data.shape:
            raise ValueError(
                f"{label} gradient shape mismatch for parameter {index}: expected "
                f"{parameter.data.shape}, got {gradient.shape}"
            )
        if (
            not np.issubdtype(gradient.dtype, np.number)
            or np.issubdtype(gradient.dtype, np.complexfloating)
        ):
            raise TypeError(
                f"{label} gradient for parameter {index} must have a real numeric dtype"
            )
        if not np.isfinite(gradient).all():
            raise ValueError(
                f"{label} gradient for parameter {index} must contain only finite values"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.array(
                gradient, dtype=np.float64, copy=True, subok=False
            )
        if not np.isfinite(converted).all():
            raise ValueError(
                f"{label} gradient for parameter {index} must fit in float64"
            )
        gradients.append(converted)
    if active == 0:
        raise ValueError(f"SAM {label} requires at least one gradient")
    return gradients


def _normalized_perturbations(gradients, rho):
    """Compute ``rho * g / ||g||`` without materializing an overflowing norm."""
    maximum = 0.0
    for gradient in gradients:
        if gradient is None or gradient.size == 0:
            continue
        magnitude = float(np.max(np.abs(gradient)))
        maximum = max(maximum, magnitude)

    if maximum == 0.0 or rho == 0.0:
        return [
            None if gradient is None else np.zeros_like(gradient, dtype=np.float64)
            for gradient in gradients
        ]

    scaled_square_sum = 0.0
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        for gradient in gradients:
            if gradient is None or gradient.size == 0:
                continue
            scaled = gradient / maximum
            scaled_square_sum += float(np.sum(scaled * scaled, dtype=np.float64))
        denominator = float(np.sqrt(scaled_square_sum))
        radius_scale = rho / denominator

        perturbations = []
        for gradient in gradients:
            if gradient is None:
                perturbations.append(None)
                continue
            perturbation = (gradient / maximum) * radius_scale
            if not np.isfinite(perturbation).all():
                raise ValueError("SAM perturbation must contain only finite values")
            perturbations.append(np.array(perturbation, copy=True, subok=False))
    return perturbations


def _state_is_finite(value):
    if isinstance(value, dict):
        return all(_state_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_is_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.isfinite(value).all())
        return True
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        try:
            return bool(np.isfinite(float(value)))
        except OverflowError:
            return False
    return True


class SAM:
    """Two-phase Sharpness-Aware Minimization wrapper.

    Typical use::

        optimizer = SAM(AdamW(model.parameters(), lr=1e-3), rho=0.05)

        loss = forward()
        loss.backward()
        optimizer.first_step()
        optimizer.zero_grad()

        neighbourhood_loss = forward()
        neighbourhood_loss.backward()
        optimizer.second_step()
        optimizer.zero_grad()

    The two forward/backward passes remain explicit so callers retain control of
    data sampling, RNG, evaluation modes, and loss construction.
    """

    def __init__(self, optimizer, *, rho=0.05):
        rho = _nonnegative_real("SAM rho", rho)
        container, parameters = _snapshot_parameters(optimizer)

        self.optimizer = optimizer
        self._parameter_container = container
        self._parameters = parameters
        self._rho = rho
        self._step_count = 0
        self._phase = "ready"
        self._base_values = None
        self._perturbed_values = None
        self._owner_thread = None
        self._lock = threading.RLock()

    @property
    def parameters(self):
        return self.optimizer.parameters

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        value = _nonnegative_real("SAM rho", value)
        with self._lock:
            if self._phase != "ready":
                raise RuntimeError("SAM rho cannot change while parameters are perturbed")
            self._rho = value

    @property
    def phase(self):
        return self._phase

    @property
    def step_count(self):
        return self._step_count

    def _validate_binding(self):
        current = getattr(self.optimizer, "parameters", None)
        if current is not self._parameter_container:
            raise RuntimeError("SAM optimizer parameter collection changed")
        if len(current) != len(self._parameters) or any(
            live is not saved for live, saved in zip(current, self._parameters)
        ):
            raise RuntimeError("SAM optimizer parameter collection changed")

    def _require_phase(self, expected, operation):
        if self._phase != expected:
            raise RuntimeError(
                f"SAM {operation} requires phase '{expected}', current phase is '{self._phase}'"
            )

    def _require_owner(self, operation):
        if self._owner_thread != threading.get_ident():
            raise RuntimeError(
                f"SAM {operation} must run on the thread that called first_step()"
            )

    def first_step(self):
        """Move parameters to the SAM neighbourhood point using current gradients."""
        with self._lock:
            self._require_phase("ready", "first_step")
            self._validate_binding()
            gradients = _validate_gradients(self._parameters, "first_step")

            base_values = []
            perturbed_values = []
            perturbations = _normalized_perturbations(gradients, self._rho)
            write_required = []

            for index, (parameter, perturbation) in enumerate(
                zip(self._parameters, perturbations)
            ):
                base = np.array(parameter.data, dtype=np.float64, copy=True, subok=False)
                if not np.isfinite(base).all():
                    raise ValueError(
                        f"SAM parameter {index} must contain only finite values before first_step()"
                    )
                if perturbation is None:
                    perturbed = base.copy()
                else:
                    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                        perturbed = base + perturbation
                    if not np.isfinite(perturbed).all():
                        raise ValueError(
                            f"SAM perturbation would make parameter {index} non-finite"
                        )
                needs_write = not np.array_equal(base, perturbed)
                if needs_write and base.size and not parameter.data.flags.writeable:
                    raise ValueError(
                        f"SAM parameter {index} must be writeable for first_step()"
                    )
                base_values.append(base)
                perturbed_values.append(np.array(perturbed, copy=True, subok=False))
                write_required.append(needs_write)

            written = []
            try:
                for index, (parameter, value, needs_write) in enumerate(
                    zip(self._parameters, perturbed_values, write_required)
                ):
                    if not needs_write:
                        continue
                    parameter.data[...] = value
                    written.append(index)
            except BaseException:
                try:
                    for index in written:
                        parameter = self._parameters[index]
                        previous = base_values[index]
                        if (
                            parameter.data.shape == previous.shape
                            and parameter.data.flags.writeable
                        ):
                            parameter.data[...] = previous
                        else:
                            parameter.data = previous
                except BaseException as rollback_error:
                    raise RuntimeError("SAM first_step rollback failed") from rollback_error
                raise

            self._base_values = base_values
            self._perturbed_values = perturbed_values
            self._phase = "perturbed"
            self._owner_thread = threading.get_ident()
        return self

    def _restore_values(self, values):
        for parameter, value in zip(self._parameters, values):
            current = np.asarray(parameter.data)
            if current.shape == value.shape and np.array_equal(current, value):
                continue
            if current.shape == value.shape and parameter.data.flags.writeable:
                parameter.data[...] = value
            else:
                parameter.data = value

    def restore(self):
        """Cancel an in-progress SAM step and restore the pre-perturbation weights."""
        with self._lock:
            self._require_phase("perturbed", "restore")
            self._require_owner("restore")
            self._validate_binding()
            self._restore_values(self._base_values)
            self._base_values = None
            self._perturbed_values = None
            self._phase = "ready"
            self._owner_thread = None
        return self

    def second_step(self):
        """Restore base weights and apply neighbourhood gradients with the inner optimizer."""
        with self._lock:
            self._require_phase("perturbed", "second_step")
            self._require_owner("second_step")
            self._validate_binding()

            for index, (parameter, expected) in enumerate(
                zip(self._parameters, self._perturbed_values)
            ):
                current = np.asarray(parameter.data)
                if current.shape != expected.shape or not np.array_equal(current, expected):
                    raise RuntimeError(
                        f"SAM perturbed parameter {index} changed before second_step(); "
                        "call restore() before restarting the SAM step"
                    )

            _validate_gradients(self._parameters, "second_step")
            for index, (parameter, base, perturbed) in enumerate(
                zip(self._parameters, self._base_values, self._perturbed_values)
            ):
                if (
                    not np.array_equal(base, perturbed)
                    and base.size
                    and not parameter.data.flags.writeable
                ):
                    raise ValueError(
                        f"SAM parameter {index} must be writeable for second_step()"
                    )

            inner_before = deepcopy(self.optimizer.state_dict())
            try:
                for parameter, base, perturbed in zip(
                    self._parameters, self._base_values, self._perturbed_values
                ):
                    if np.array_equal(base, perturbed):
                        continue
                    parameter.data[...] = base

                result = self.optimizer.step()
                for index, (parameter, base) in enumerate(
                    zip(self._parameters, self._base_values)
                ):
                    if parameter.data.shape != base.shape:
                        raise ValueError(
                            f"SAM inner optimizer changed parameter shape at index {index}"
                        )
                    if not np.isfinite(np.asarray(parameter.data)).all():
                        raise ValueError(
                            f"SAM inner optimizer produced a non-finite parameter at index {index}"
                        )
                if not _state_is_finite(self.optimizer.state_dict()):
                    raise ValueError("SAM inner optimizer produced non-finite state")
            except BaseException:
                try:
                    self.optimizer.load_state_dict(inner_before)
                    self._restore_values(self._perturbed_values)
                except BaseException as rollback_error:
                    raise RuntimeError("SAM second_step rollback failed") from rollback_error
                raise

            self._base_values = None
            self._perturbed_values = None
            self._phase = "ready"
            self._owner_thread = None
            self._step_count += 1
            return result

    def zero_grad(self, set_to_none=False):
        """Forward gradient clearing to the wrapped optimizer."""
        with self._lock:
            if self._phase == "perturbed":
                self._require_owner("zero_grad")
            return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return independent SAM metadata and wrapped-optimizer state.

        Saving while parameters are perturbed is rejected because model state at
        that instant is intentionally temporary and should not be checkpointed.
        """
        with self._lock:
            self._require_phase("ready", "state_dict")
            self._validate_binding()
            return {
                "version": _STATE_VERSION,
                "optimizer_type": type(self.optimizer).__name__,
                "rho": self._rho,
                "step_count": self._step_count,
                "optimizer": deepcopy(self.optimizer.state_dict()),
            }

    def load_state_dict(self, state):
        """Transactionally restore SAM metadata and wrapped-optimizer state."""
        if not isinstance(state, dict):
            raise TypeError("SAM state must be a dictionary")
        with self._lock:
            self._require_phase("ready", "load_state_dict")
            self._validate_binding()

            version = _nonnegative_integer("SAM state version", state["version"])
            if version != _STATE_VERSION:
                raise ValueError(
                    f"unsupported SAM state version {version}; expected {_STATE_VERSION}"
                )
            optimizer_type = state["optimizer_type"]
            if not isinstance(optimizer_type, str):
                raise TypeError("SAM optimizer_type must be a string")
            expected_type = type(self.optimizer).__name__
            if optimizer_type != expected_type:
                raise ValueError(
                    f"SAM optimizer type mismatch: expected {expected_type}, got {optimizer_type}"
                )
            rho = _nonnegative_real("SAM rho", state["rho"])
            step_count = _nonnegative_integer("SAM step_count", state["step_count"])
            if "optimizer" not in state:
                raise KeyError("optimizer")
            inner_state = deepcopy(state["optimizer"])
            inner_before = deepcopy(self.optimizer.state_dict())

            try:
                self.optimizer.load_state_dict(inner_state)
                if not _state_is_finite(self.optimizer.state_dict()):
                    raise ValueError("SAM restored optimizer state must be finite")
            except BaseException:
                try:
                    self.optimizer.load_state_dict(inner_before)
                except BaseException as rollback_error:
                    raise RuntimeError("SAM optimizer state rollback failed") from rollback_error
                raise

            self._rho = rho
            self._step_count = step_count
        return self
