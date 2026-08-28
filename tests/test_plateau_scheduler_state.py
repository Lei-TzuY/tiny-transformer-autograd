import copy

import numpy as np
import pytest

from engine.optim import SGD
from engine.plateau import ReduceLROnPlateau
from engine.tensor import Tensor


def _optimizer(lr=0.8):
    return SGD([Tensor([1.0], requires_grad=True)], lr=lr)


def _progressed_scheduler():
    optimizer = _optimizer()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=0.05,
        threshold_mode="abs",
        cooldown=2,
        min_lr=0.1,
        eps=1e-12,
    )
    scheduler.step(1.0)
    scheduler.step(1.1)
    scheduler.step(1.2)
    return scheduler, optimizer


def _assert_same_state(actual, expected):
    assert actual == expected


def test_state_roundtrip_restores_configuration_runtime_state_and_optimizer_lr():
    source, source_optimizer = _progressed_scheduler()
    state = source.state_dict()

    target_optimizer = _optimizer(0.3)
    target = ReduceLROnPlateau(
        target_optimizer,
        mode="max",
        factor=0.2,
        patience=9,
        threshold=9.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.0,
        eps=1.0,
    )

    assert target.load_state_dict(state) is target

    _assert_same_state(target.state_dict(), state)
    assert target_optimizer.lr == source_optimizer.lr


def test_resume_reproduces_the_next_plateau_decision_exactly():
    uninterrupted, _ = _progressed_scheduler()
    state = uninterrupted.state_dict()

    resumed_optimizer = _optimizer(0.7)
    resumed = ReduceLROnPlateau(resumed_optimizer)
    resumed.load_state_dict(state)

    for metric in (1.3, 0.8, 0.82, 0.83, 0.84):
        left = uninterrupted.step(metric)
        right = resumed.step(metric)
        assert right == left
        _assert_same_state(resumed.state_dict(), uninterrupted.state_dict())


def test_load_requires_mapping_and_complete_envelope():
    scheduler, _ = _progressed_scheduler()

    with pytest.raises(TypeError, match="state must be a mapping"):
        scheduler.load_state_dict([])

    state = scheduler.state_dict()
    for key in tuple(state):
        malformed = copy.deepcopy(state)
        del malformed[key]
        with pytest.raises(ValueError, match="missing keys"):
            scheduler.load_state_dict(malformed)


def test_rejected_state_is_scheduler_and_optimizer_neutral():
    scheduler, optimizer = _progressed_scheduler()
    baseline = scheduler.state_dict()

    cases = [
        ("format_version", 2, ValueError),
        ("format_version", True, TypeError),
        ("scheduler", "Other", ValueError),
        ("mode", "median", ValueError),
        ("factor", 1.0, ValueError),
        ("factor", 0.0, ValueError),
        ("factor", 10**400, ValueError),
        ("patience", -1, ValueError),
        ("patience", True, TypeError),
        ("threshold", -1.0, ValueError),
        ("threshold", np.inf, ValueError),
        ("threshold_mode", "fraction", ValueError),
        ("cooldown", -1, ValueError),
        ("min_lr", -1.0, ValueError),
        ("eps", -1.0, ValueError),
        ("best", np.nan, ValueError),
        ("num_bad_epochs", -1, ValueError),
        ("cooldown_counter", -1, ValueError),
        ("step_count", -1, ValueError),
        ("reductions", -1, ValueError),
        ("current_lr", 0.0, ValueError),
    ]

    for key, value, error in cases:
        malformed = copy.deepcopy(baseline)
        malformed[key] = value
        with pytest.raises(error):
            scheduler.load_state_dict(malformed)
        _assert_same_state(scheduler.state_dict(), baseline)
        assert optimizer.lr == baseline["current_lr"]


def test_state_counter_invariants_are_validated_before_restore():
    scheduler, _ = _progressed_scheduler()
    baseline = scheduler.state_dict()

    malformed = copy.deepcopy(baseline)
    malformed["num_bad_epochs"] = malformed["patience"] + 1
    with pytest.raises(ValueError, match="num_bad_epochs"):
        scheduler.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["cooldown_counter"] = malformed["cooldown"] + 1
    with pytest.raises(ValueError, match="cooldown_counter"):
        scheduler.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["reductions"] = malformed["step_count"] + 1
    with pytest.raises(ValueError, match="reductions"):
        scheduler.load_state_dict(malformed)

    malformed = copy.deepcopy(baseline)
    malformed["reductions"] = 0
    malformed["cooldown_counter"] = 1
    with pytest.raises(ValueError, match="cooldown"):
        scheduler.load_state_dict(malformed)


def test_empty_state_invariants_reject_impossible_best_and_counters():
    scheduler = ReduceLROnPlateau(_optimizer())
    empty = scheduler.state_dict()

    malformed = copy.deepcopy(empty)
    malformed["step_count"] = 1
    with pytest.raises(ValueError, match="best may be None"):
        scheduler.load_state_dict(malformed)

    malformed = copy.deepcopy(empty)
    malformed["best"] = 1.0
    with pytest.raises(ValueError, match="positive step_count"):
        scheduler.load_state_dict(malformed)

    malformed = copy.deepcopy(empty)
    malformed["num_bad_epochs"] = 1
    malformed["patience"] = 1
    with pytest.raises(ValueError, match="zero counters"):
        scheduler.load_state_dict(malformed)


def test_loaded_current_lr_must_respect_loaded_floor():
    scheduler, _ = _progressed_scheduler()
    state = scheduler.state_dict()
    state["min_lr"] = 0.4
    state["current_lr"] = 0.2

    with pytest.raises(ValueError, match="at least min_lr"):
        scheduler.load_state_dict(state)


def test_unknown_state_metadata_is_tolerated():
    scheduler, _ = _progressed_scheduler()
    state = scheduler.state_dict()
    state["future"] = {"schema": 2}

    scheduler.load_state_dict(state)

    assert scheduler.step_count == state["step_count"]
    assert scheduler.best == state["best"]


def test_load_normalizes_numpy_scalars_to_builtin_state_values():
    scheduler = ReduceLROnPlateau(_optimizer())
    state = scheduler.state_dict()
    state.update(
        {
            "factor": np.float32(0.5),
            "patience": np.int64(2),
            "threshold": np.float32(0.25),
            "cooldown": np.int64(1),
            "min_lr": np.float32(0.1),
            "eps": np.float32(1e-6),
            "best": np.float32(3.0),
            "num_bad_epochs": np.int64(1),
            "cooldown_counter": np.int64(0),
            "step_count": np.int64(4),
            "reductions": np.int64(1),
            "current_lr": np.float32(0.4),
        }
    )

    scheduler.load_state_dict(state)
    restored = scheduler.state_dict()

    assert type(restored["factor"]) is float
    assert type(restored["patience"]) is int
    assert type(restored["threshold"]) is float
    assert type(restored["step_count"]) is int
    assert type(restored["best"]) is float


def test_state_save_does_not_touch_parameter_or_gradient_buffers():
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    parameter.grad[...] = [3.0, 4.0]
    optimizer = SGD([parameter], lr=0.5)
    scheduler = ReduceLROnPlateau(optimizer)
    values = parameter.data.copy()
    gradient = parameter.grad
    gradient_values = gradient.copy()
    version = parameter._version

    scheduler.state_dict()

    np.testing.assert_array_equal(parameter.data, values)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_values)
    assert parameter._version == version
