import json

import numpy as np
import pytest

from engine.optim import AdamW, SGD
from engine.plateau import ReduceLROnPlateau
from engine.tensor import Tensor


def _sgd(lr=1.0):
    return SGD([Tensor([1.0], requires_grad=True)], lr=lr)


def test_min_mode_patience_and_cooldown_have_explicit_sequence_semantics():
    optimizer = _sgd(1.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1,
        cooldown=2,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )

    assert scheduler.step(1.0) == 1.0
    assert scheduler.best == 1.0
    assert scheduler.num_bad_epochs == 0

    assert scheduler.step(1.1) == 1.0
    assert scheduler.num_bad_epochs == 1

    assert scheduler.step(1.2) == 0.5
    assert optimizer.lr == 0.5
    assert scheduler.reductions == 1
    assert scheduler.num_bad_epochs == 0
    assert scheduler.cooldown_counter == 2

    assert scheduler.step(1.3) == 0.5
    assert scheduler.cooldown_counter == 1
    assert scheduler.num_bad_epochs == 0

    assert scheduler.step(1.4) == 0.5
    assert scheduler.cooldown_counter == 0
    assert scheduler.num_bad_epochs == 0

    assert scheduler.step(1.5) == 0.5
    assert scheduler.num_bad_epochs == 1
    assert scheduler.step(1.6) == 0.25
    assert scheduler.reductions == 2
    assert scheduler.step_count == 7


def test_improvement_resets_bad_epochs_and_max_mode_tracks_larger_values():
    optimizer = _sgd(0.8)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )

    scheduler.step(10.0)
    scheduler.step(9.0)
    assert scheduler.num_bad_epochs == 1

    scheduler.step(11.0)
    assert scheduler.best == 11.0
    assert scheduler.num_bad_epochs == 0
    assert optimizer.lr == 0.8

    scheduler.step(10.5)
    scheduler.step(10.0)
    assert optimizer.lr == 0.4
    assert scheduler.reductions == 1


def test_absolute_threshold_requires_improvement_strictly_beyond_margin():
    scheduler = ReduceLROnPlateau(
        _sgd(),
        patience=0,
        threshold=0.1,
        threshold_mode="abs",
        factor=0.5,
        eps=0.0,
    )

    scheduler.step(1.0)
    # Exactly the configured margin is not a qualifying improvement.
    assert scheduler.step(0.9) == 0.5
    assert scheduler.best == 1.0


def test_relative_threshold_uses_abs_best_and_handles_extreme_opposite_signs():
    optimizer = _sgd()
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        threshold=1.0,
        threshold_mode="rel",
        factor=0.5,
        eps=0.0,
    )

    scheduler.step(1.3e308)
    with np.errstate(all="raise"):
        lr = scheduler.step(-1.3e308)

    assert lr == 1.0
    assert scheduler.best == -1.3e308
    assert scheduler.reductions == 0


def test_relative_threshold_is_sign_symmetric_for_negative_best_in_max_mode():
    optimizer = _sgd()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=0,
        threshold=0.1,
        threshold_mode="rel",
        factor=0.5,
        eps=0.0,
    )

    scheduler.step(-100.0)
    # Improvement is 9, but relative margin is 10.
    assert scheduler.step(-91.0) == 0.5
    assert scheduler.best == -100.0


def test_cooldown_still_updates_best_without_accumulating_bad_epochs():
    optimizer = _sgd()
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        cooldown=2,
        threshold=0.0,
        threshold_mode="abs",
        factor=0.5,
        eps=0.0,
    )

    scheduler.step(1.0)
    scheduler.step(2.0)
    assert optimizer.lr == 0.5
    assert scheduler.cooldown_counter == 2

    scheduler.step(0.8)
    assert scheduler.best == 0.8
    assert scheduler.cooldown_counter == 1
    assert scheduler.num_bad_epochs == 0

    scheduler.step(0.9)
    assert scheduler.cooldown_counter == 0
    assert scheduler.num_bad_epochs == 0
    assert optimizer.lr == 0.5


def test_min_lr_clamps_reduction_and_eps_can_suppress_small_change():
    optimizer = _sgd(0.1)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.08,
        eps=0.03,
    )

    scheduler.step(1.0)
    assert scheduler.step(2.0) == 0.1
    assert optimizer.lr == 0.1
    assert scheduler.reductions == 0
    assert scheduler.num_bad_epochs == 0

    optimizer = _sgd(0.1)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.08,
        eps=0.0,
    )
    scheduler.step(1.0)
    assert scheduler.step(2.0) == pytest.approx(0.08)
    assert scheduler.step(3.0) == pytest.approx(0.08)
    assert scheduler.reductions == 1


def test_subnormal_reduction_is_warning_neutral():
    tiny = np.nextafter(0.0, 1.0)
    optimizer = _sgd(tiny * 4.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    scheduler.step(1.0)

    with np.errstate(all="raise"):
        lr = scheduler.step(2.0)

    assert lr >= 0.0
    assert np.isfinite(lr)


def test_step_accepts_numpy_real_metrics_and_rejects_bad_metrics_before_state_change():
    optimizer = _sgd()
    scheduler = ReduceLROnPlateau(optimizer)

    scheduler.step(np.float32(1.25))
    baseline = scheduler.state_dict()

    for bad in (True, "1.0", np.nan, np.inf, -np.inf, 10**400):
        with pytest.raises((TypeError, ValueError)):
            scheduler.step(bad)
        assert scheduler.state_dict() == baseline


def test_external_lr_below_floor_is_rejected_without_scheduler_mutation():
    optimizer = _sgd(0.5)
    scheduler = ReduceLROnPlateau(optimizer, min_lr=0.2)
    scheduler.step(1.0)
    baseline = scheduler.state_dict()

    optimizer.lr = 0.1
    with pytest.raises(ValueError, match="below min_lr"):
        scheduler.step(2.0)

    assert scheduler.best == baseline["best"]
    assert scheduler.step_count == baseline["step_count"]
    assert scheduler.reductions == baseline["reductions"]


def test_state_dict_is_json_safe_and_independent_scalar_snapshot():
    optimizer = _sgd(0.5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        cooldown=1,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    scheduler.step(1.0)
    scheduler.step(2.0)

    state = scheduler.state_dict()
    encoded = json.dumps(state, allow_nan=False, sort_keys=True)

    assert '"scheduler": "ReduceLROnPlateau"' in encoded
    state["best"] = 999.0
    state["current_lr"] = 999.0
    assert scheduler.best == 1.0
    assert optimizer.lr == 0.05


def test_scheduler_works_with_adamw_lr_field_without_touching_optimizer_state():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = AdamW([parameter], lr=0.2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        factor=0.5,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    before = optimizer.state_dict()

    scheduler.step(1.0)
    scheduler.step(2.0)

    after = optimizer.state_dict()
    assert optimizer.lr == 0.1
    assert after["t"] == before["t"]
    assert after["steps"] == before["steps"]
    np.testing.assert_array_equal(after["m"][0], before["m"][0])
    np.testing.assert_array_equal(after["v"][0], before["v"][0])


def test_scheduler_operations_do_not_consume_numpy_global_rng():
    optimizer = _sgd()
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    np.random.seed(12345)
    before = np.random.get_state()

    scheduler.step(1.0)
    scheduler.step(2.0)
    scheduler.get_lr()
    scheduler.state_dict()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
