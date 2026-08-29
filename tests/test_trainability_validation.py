import numpy as np
import pytest

from engine.trainability import freeze_, set_trainable_, unfreeze_
from engine.tensor import Tensor


def test_enabled_must_be_boolean_before_parameter_iterable_is_consumed():
    consumed = False

    def values():
        nonlocal consumed
        consumed = True
        yield Tensor(1.0, requires_grad=True)

    with pytest.raises(TypeError, match="enabled must be boolean"):
        set_trainable_(values(), 1)
    assert consumed is False


def test_numpy_boolean_enabled_is_accepted():
    parameter = Tensor(1.0, requires_grad=True)
    assert set_trainable_(parameter, np.bool_(False)) == 1
    assert parameter.requires_grad is False


def test_non_iterable_parameter_collection_is_rejected():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        freeze_(123)


def test_collection_must_contain_only_tensors():
    parameter = Tensor(1.0, requires_grad=True)
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        freeze_([parameter, object()])
    assert parameter.requires_grad is True


def test_duplicate_tensor_references_are_rejected_before_mutation():
    parameter = Tensor(1.0, requires_grad=True)
    version = parameter._version
    grad = parameter.grad

    with pytest.raises(ValueError, match="duplicate at index 1"):
        freeze_([parameter, parameter])

    assert parameter.requires_grad is True
    assert parameter.grad is grad
    assert parameter._version == version


def test_non_leaf_tensor_is_rejected_before_any_earlier_parameter_changes():
    first = Tensor(1.0, requires_grad=True)
    source = Tensor(2.0, requires_grad=True)
    non_leaf = source * 3.0
    first_grad = first.grad
    first_version = first._version

    with pytest.raises(ValueError, match="parameter 1 must be a leaf Tensor"):
        freeze_([first, non_leaf])

    assert first.requires_grad is True
    assert first.grad is first_grad
    assert first._version == first_version


def test_malformed_late_requires_grad_is_transactional():
    first = Tensor(1.0, requires_grad=True)
    bad = Tensor(2.0, requires_grad=True)
    bad.requires_grad = "yes"
    first_grad = first.grad
    first_version = first._version

    with pytest.raises(TypeError, match="parameter 1 requires_grad must be boolean"):
        freeze_([first, bad])

    assert first.requires_grad is True
    assert first.grad is first_grad
    assert first._version == first_version


def test_negative_mutation_version_is_rejected_before_mutation():
    first = Tensor(1.0, requires_grad=True)
    bad = Tensor(2.0, requires_grad=True)
    bad._version = -1
    first_grad = first.grad

    with pytest.raises(ValueError, match="mutation version must be non-negative"):
        freeze_([first, bad])

    assert first.requires_grad is True
    assert first.grad is first_grad
    assert first._version == 0


def test_non_integer_mutation_version_is_rejected_before_mutation():
    parameter = Tensor(1.0, requires_grad=True)
    parameter._version = 1.5

    with pytest.raises(TypeError, match="mutation version must be an integer"):
        freeze_(parameter)

    assert parameter.requires_grad is True
    assert parameter.grad is not None


def test_boolean_mutation_version_is_rejected():
    parameter = Tensor(1.0, requires_grad=True)
    parameter._version = True

    with pytest.raises(TypeError, match="mutation version must be an integer"):
        freeze_(parameter)


def test_external_numpy_boolean_requires_grad_is_normalized_on_transition():
    parameter = Tensor(1.0, requires_grad=True)
    parameter.requires_grad = np.bool_(True)

    assert freeze_(parameter) == 1
    assert parameter.requires_grad is False


def test_freezing_discards_arbitrary_stale_gradient_object_without_reading_it():
    parameter = Tensor(1.0, requires_grad=True)
    sentinel = object()
    parameter.grad = sentinel

    assert freeze_(parameter) == 1
    assert parameter.grad is None


def test_unfreezing_discards_gradient_left_on_frozen_tensor():
    parameter = Tensor(1.0, requires_grad=False)
    parameter.grad = np.array(99.0)

    assert unfreeze_(parameter) == 1
    assert parameter.requires_grad is True
    assert parameter.grad is None


def test_noop_unfreeze_does_not_validate_or_replace_live_gradient_payload():
    parameter = Tensor(1.0, requires_grad=True)
    sentinel = object()
    parameter.grad = sentinel
    version = parameter._version

    assert unfreeze_(parameter) == 0
    assert parameter.grad is sentinel
    assert parameter._version == version


def test_parameter_data_shape_and_values_are_unchanged():
    parameter = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    before = parameter.data.copy()
    shape = parameter.shape

    freeze_(parameter)
    unfreeze_(parameter)

    assert parameter.shape == shape
    np.testing.assert_array_equal(parameter.data, before)


def test_transition_updates_grad_shape_metadata_to_current_shape():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter._grad_shape = (999,)

    freeze_(parameter)
    assert parameter._grad_shape == parameter.shape

    unfreeze_(parameter)
    assert parameter._grad_shape == parameter.shape
