import numpy as np
import pytest

from engine.optim import SGD
from engine.plateau import ReduceLROnPlateau
from engine.tensor import Tensor


def _optimizer(lr=1.0):
    return SGD([Tensor([1.0], requires_grad=True)], lr=lr)


def test_constructor_rejects_optimizer_without_lr_attribute():
    with pytest.raises(TypeError, match="optimizer must expose an lr attribute"):
        ReduceLROnPlateau(object())


def test_constructor_rejects_invalid_optimizer_lr():
    class Optimizer:
        pass

    optimizer = Optimizer()
    for value in (0.0, -1.0, np.nan, np.inf, True, "1.0", 10**400):
        optimizer.lr = value
        with pytest.raises((TypeError, ValueError)):
            ReduceLROnPlateau(optimizer)


def test_constructor_validates_mode_and_threshold_mode():
    for value in (None, 1, True):
        with pytest.raises(TypeError, match="mode must be a string"):
            ReduceLROnPlateau(_optimizer(), mode=value)
    with pytest.raises(ValueError, match="mode must be one of"):
        ReduceLROnPlateau(_optimizer(), mode="median")

    for value in (None, 1, True):
        with pytest.raises(TypeError, match="threshold_mode must be a string"):
            ReduceLROnPlateau(_optimizer(), threshold_mode=value)
    with pytest.raises(ValueError, match="threshold_mode must be one of"):
        ReduceLROnPlateau(_optimizer(), threshold_mode="percent")


def test_constructor_rejects_invalid_factor_values():
    for value in (0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf, True, "0.5", 10**400):
        with pytest.raises((TypeError, ValueError)):
            ReduceLROnPlateau(_optimizer(), factor=value)


def test_constructor_rejects_invalid_integer_controls():
    for name in ("patience", "cooldown"):
        for value in (-1, 1.5, True, np.nan, "1"):
            kwargs = {name: value}
            with pytest.raises((TypeError, ValueError)):
                ReduceLROnPlateau(_optimizer(), **kwargs)


def test_constructor_rejects_invalid_nonnegative_reals():
    for name in ("threshold", "min_lr", "eps"):
        for value in (-1.0, np.nan, np.inf, -np.inf, True, "0.1", 10**400):
            kwargs = {name: value}
            with pytest.raises((TypeError, ValueError)):
                ReduceLROnPlateau(_optimizer(), **kwargs)


def test_min_lr_cannot_start_above_optimizer_lr():
    with pytest.raises(ValueError, match="must not exceed optimizer.lr"):
        ReduceLROnPlateau(_optimizer(0.1), min_lr=0.2)


def test_numpy_scalar_configuration_is_normalized():
    scheduler = ReduceLROnPlateau(
        _optimizer(0.8),
        factor=np.float32(0.5),
        patience=np.int64(2),
        threshold=np.float32(0.2),
        cooldown=np.int64(3),
        min_lr=np.float32(0.1),
        eps=np.float32(1e-6),
    )

    assert type(scheduler.factor) is float
    assert type(scheduler.patience) is int
    assert type(scheduler.threshold) is float
    assert type(scheduler.cooldown) is int
    assert type(scheduler.min_lr) is float
    assert type(scheduler.eps) is float


def test_get_lr_rejects_external_nonfinite_or_nonpositive_lr():
    optimizer = _optimizer(0.8)
    scheduler = ReduceLROnPlateau(optimizer)

    for value in (0.0, -1.0, np.nan, np.inf):
        optimizer.lr = value
        with pytest.raises(ValueError):
            scheduler.get_lr()


def test_step_reads_live_optimizer_lr_and_reduces_from_external_valid_value():
    optimizer = _optimizer(1.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    scheduler.step(1.0)
    optimizer.lr = 0.8

    assert scheduler.step(2.0) == 0.4
    assert optimizer.lr == 0.4


def test_in_cooldown_and_get_lr_are_observational():
    parameter = Tensor([2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=1.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        cooldown=2,
        eps=0.0,
    )
    version = parameter._version

    scheduler.step(1.0)
    scheduler.step(2.0)

    assert scheduler.in_cooldown is True
    assert scheduler.get_lr() == 0.5
    assert parameter._version == version
