import weakref

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def test_parameter_storage_rejects_owner_weakref_with_callback_before_write():
    parameter = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    gradient = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float64)
    parameter.grad = gradient
    entry_gradient = gradient.copy()

    def callback(_reference):
        raise AssertionError("parameter storage owner weakrefs must not run callbacks")

    parameter.data._owner_ref = weakref.ref(parameter, callback)

    with pytest.raises(
        TypeError,
        match="parameter 0 data ownership metadata must be a callback-free weak reference",
    ):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, entry_gradient)
    assert parameter.data._owner_ref() is parameter
    assert parameter.data._owner_ref.__callback__ is callback
