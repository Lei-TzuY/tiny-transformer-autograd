import numpy as np
import pytest

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_constructor_captures_live_values_without_mutating_model_state():
    p = Tensor([1.0, -2.0], requires_grad=True)
    p.grad[...] = [3.0, 4.0]
    grad_ref = p.grad
    version = p._version
    rng = np.random.get_state()

    snapshot = ParameterSnapshot(p)

    np.testing.assert_array_equal(snapshot.values()[0], [1.0, -2.0])
    np.testing.assert_array_equal(p.data, [1.0, -2.0])
    assert p.grad is grad_ref
    np.testing.assert_array_equal(p.grad, [3.0, 4.0])
    assert p._version == version
    after = np.random.get_state()
    assert rng[0] == after[0]
    np.testing.assert_array_equal(rng[1], after[1])
    assert rng[2:] == after[2:]


def test_explicit_values_create_offline_snapshot_without_installing_them():
    p = Tensor([10.0, 20.0], requires_grad=True)
    candidate = np.array([1.0, 2.0])

    snapshot = ParameterSnapshot(p, values=candidate)
    candidate[...] = 99.0

    np.testing.assert_array_equal(p.data, [10.0, 20.0])
    np.testing.assert_array_equal(snapshot.values()[0], [1.0, 2.0])


def test_values_returns_independent_copies():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)

    exported = snapshot.values()
    exported[0][0] = 99.0

    np.testing.assert_array_equal(snapshot.values()[0], [1.0, 2.0])
    np.testing.assert_array_equal(p.data, [1.0, 2.0])


def test_capture_replaces_snapshot_with_current_live_values_only():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    p.data[...] = [5.0, 6.0]
    version = p._version
    grad_ref = p.grad

    assert snapshot.capture() is snapshot

    np.testing.assert_array_equal(snapshot.values()[0], [5.0, 6.0])
    assert p._version == version
    assert p.grad is grad_ref


def test_restore_installs_snapshot_and_preserves_gradient_reference():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    p.data[...] = [8.0, 9.0]
    p.grad[...] = [4.0, 5.0]
    grad_ref = p.grad
    before_version = p._version

    changed = snapshot.restore()

    assert changed == 1
    np.testing.assert_array_equal(p.data, [1.0, 2.0])
    assert p.grad is grad_ref
    np.testing.assert_array_equal(p.grad, [4.0, 5.0])
    assert p._version == before_version + 1


def test_noop_restore_skips_write_and_version_increment():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    version = p._version

    assert snapshot.restore() == 0
    assert p._version == version


def test_scalar_tensor_snapshot_and_restore_are_supported():
    p = Tensor(3.0, requires_grad=True)
    snapshot = ParameterSnapshot(p)
    p.data[...] = -7.0

    assert snapshot.restore() == 1
    assert p.data.shape == ()
    assert p.data.item() == 3.0


def test_restore_invalidates_graph_built_from_different_live_values():
    p = Tensor(2.0, requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array(1.0))
    output = p * p

    snapshot.restore()

    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()


def test_noop_restore_does_not_invalidate_existing_graph():
    p = Tensor(2.0, requires_grad=True)
    snapshot = ParameterSnapshot(p)
    output = p * p

    assert snapshot.restore() == 0
    output.backward()
    assert p.grad.item() == 4.0


def test_installed_temporarily_swaps_snapshot_and_restores_entry_values():
    p = Tensor([10.0, 20.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([1.0, 2.0]))
    entry_grad = p.grad

    with snapshot.installed() as active:
        assert active is snapshot
        np.testing.assert_array_equal(p.data, [1.0, 2.0])
        assert p.grad is entry_grad

    np.testing.assert_array_equal(p.data, [10.0, 20.0])
    assert p.grad is entry_grad


def test_installed_restores_entry_values_after_body_exception():
    p = Tensor([10.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([1.0]))

    with pytest.raises(RuntimeError, match="body failed"):
        with snapshot.installed():
            np.testing.assert_array_equal(p.data, [1.0])
            raise RuntimeError("body failed")

    np.testing.assert_array_equal(p.data, [10.0])


def test_installed_restores_after_body_replaces_storage_and_shape():
    p = Tensor([10.0, 20.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([1.0, 2.0]))

    with snapshot.installed():
        p.data = np.array([[7.0, 8.0]])
        assert p.shape == (1, 2)

    assert p.shape == (2,)
    np.testing.assert_array_equal(p.data, [10.0, 20.0])


def test_nested_installed_scopes_restore_each_entry_state_in_order():
    p = Tensor([3.0], requires_grad=True)
    first = ParameterSnapshot(p, values=np.array([1.0]))
    second = ParameterSnapshot(p, values=np.array([2.0]))

    with first.installed():
        np.testing.assert_array_equal(p.data, [1.0])
        with second.installed():
            np.testing.assert_array_equal(p.data, [2.0])
        np.testing.assert_array_equal(p.data, [1.0])
    np.testing.assert_array_equal(p.data, [3.0])


def test_empty_parameter_collection_is_a_valid_noop_snapshot():
    snapshot = ParameterSnapshot([])
    assert snapshot.parameter_count == 0
    assert snapshot.values() == ()
    assert snapshot.restore() == 0
    assert snapshot.capture() is snapshot
    with snapshot.installed():
        pass


def test_parameters_property_preserves_bound_identity_and_order():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=False)
    snapshot = ParameterSnapshot([first, second])

    assert snapshot.parameters == (first, second)
    assert snapshot.parameter_count == 2
