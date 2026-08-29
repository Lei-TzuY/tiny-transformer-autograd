import numpy as np
import pytest

from engine.swa import StochasticWeightAverage
from engine.tensor import Tensor


def test_first_update_captures_checkpoint_and_later_updates_are_equal_weighted():
    p = Tensor([1.0, 3.0], requires_grad=True)
    swa = StochasticWeightAverage([p])

    assert swa.num_averaged == 0
    with pytest.raises(RuntimeError, match="no averaged checkpoints"):
        swa.averages()

    assert swa.update() == 1
    p.data[...] = [3.0, 7.0]
    assert swa.update() == 2
    p.data[...] = [8.0, -1.0]
    assert swa.update() == 3

    np.testing.assert_allclose(swa.averages()[0], [4.0, 3.0])


def test_scalar_tensor_and_extreme_opposite_sign_values_are_safe():
    limit = np.finfo(np.float64).max
    p = Tensor(limit, requires_grad=True)
    swa = StochasticWeightAverage(p)

    with np.errstate(all="raise"):
        swa.update()
        p.data[...] = -limit
        swa.update()
        assert swa.averages()[0].shape == ()
        assert swa.averages()[0].item() == 0.0
        p.data[...] = limit
        swa.update()

    assert swa.averages()[0].item() == pytest.approx(limit / 3.0)


def test_same_sign_float64_maxima_do_not_overflow():
    limit = np.finfo(np.float64).max
    p = Tensor([limit, limit], requires_grad=True)
    swa = StochasticWeightAverage(p)

    with np.errstate(all="raise"):
        swa.update()
        swa.update()

    np.testing.assert_array_equal(swa.averages()[0], [limit, limit])


def test_update_only_reads_live_parameter_grad_and_version_state():
    p = Tensor([2.0, -5.0], requires_grad=True)
    p.grad[...] = [7.0, 11.0]
    grad_ref = p.grad
    version = p._version
    data = p.data.copy()
    rng = np.random.get_state()
    swa = StochasticWeightAverage(p)

    swa.update()

    np.testing.assert_array_equal(p.data, data)
    assert p.grad is grad_ref
    np.testing.assert_array_equal(p.grad, [7.0, 11.0])
    assert p._version == version
    after = np.random.get_state()
    assert rng[0] == after[0]
    np.testing.assert_array_equal(rng[1], after[1])
    assert rng[2:] == after[2:]


def test_average_snapshots_are_independent():
    p = Tensor([1.0, 2.0], requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()

    first = swa.averages()
    first[0][0] = 999.0

    np.testing.assert_array_equal(swa.averages()[0], [1.0, 2.0])


def test_copy_to_parameters_uses_tracked_writes_and_skips_equal_values():
    p = Tensor([1.0, 2.0], requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [5.0, 6.0]
    version_before_copy = p._version
    grad_ref = p.grad

    changed = swa.copy_to_parameters()

    assert changed == 1
    np.testing.assert_array_equal(p.data, [1.0, 2.0])
    assert p._version == version_before_copy + 1
    assert p.grad is grad_ref

    version_after = p._version
    assert swa.copy_to_parameters() == 0
    assert p._version == version_after


def test_copy_to_parameters_invalidates_graph_built_from_live_weights():
    p = Tensor(1.0, requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = 2.0
    loss = p * p

    swa.copy_to_parameters()

    with pytest.raises(RuntimeError, match="tensor data was modified after forward"):
        loss.backward()


def test_average_parameters_restores_entry_values_after_body_and_exception():
    p = Tensor([1.0, 2.0], requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [9.0, 10.0]
    entry = p.data.copy()

    with swa.average_parameters():
        np.testing.assert_array_equal(p.data, [1.0, 2.0])
        p.data[...] = [4.0, 5.0]
    np.testing.assert_array_equal(p.data, entry)

    with pytest.raises(RuntimeError, match="body failure"):
        with swa.average_parameters():
            raise RuntimeError("body failure")
    np.testing.assert_array_equal(p.data, entry)


def test_average_parameters_restores_after_body_replaces_shape_and_storage():
    p = Tensor([1.0, 2.0], requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [7.0, 8.0]
    entry = p.data.copy()

    with swa.average_parameters():
        p.data = np.array([[99.0]])
        assert p.shape == (1, 1)

    assert p.shape == (2,)
    np.testing.assert_array_equal(p.data, entry)


def test_reset_forgets_average_without_touching_live_parameters():
    p = Tensor([3.0], requires_grad=True)
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [8.0]
    version = p._version

    assert swa.reset() is swa
    assert swa.num_averaged == 0
    np.testing.assert_array_equal(p.data, [8.0])
    assert p._version == version


def test_empty_parameter_collection_supports_checkpoint_bookkeeping():
    swa = StochasticWeightAverage([])
    assert swa.update() == 1
    assert swa.averages() == ()
    assert swa.copy_to_parameters() == 0
    with swa.average_parameters():
        assert swa.num_averaged == 1
