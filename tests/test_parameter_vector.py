"""Regression tests for flattening and restoring Tensor parameter vectors."""

import numpy as np
import pytest

import engine.ops as ops
from engine.parameter_vector import parameters_to_vector, vector_to_parameters_
from engine.tensor import Tensor


def test_parameters_to_vector_preserves_order_shapes_and_snapshot_independence():
    scalar = Tensor(np.array(3.0), requires_grad=True)
    matrix = Tensor(np.array([[1.0, 2.0], [4.0, 5.0]]), requires_grad=True)
    empty = Tensor(np.empty((0, 2)), requires_grad=True)

    vector = parameters_to_vector(value for value in (scalar, matrix, empty))

    assert type(vector) is np.ndarray
    assert vector.dtype == np.float64
    assert vector.shape == (5,)
    np.testing.assert_array_equal(vector, [3.0, 1.0, 2.0, 4.0, 5.0])

    vector[0] = -99.0
    assert scalar.data == pytest.approx(3.0)


def test_vector_to_parameters_restores_heterogeneous_shapes_and_tracks_mutation():
    scalar = Tensor(np.array(0.0), requires_grad=True)
    matrix = Tensor(np.zeros((2, 2)), requires_grad=True)
    empty = Tensor(np.empty((0, 3)), requires_grad=True)
    versions = [value._version for value in (scalar, matrix, empty)]

    vector_to_parameters_([9, 1, 2, 3, 4], (scalar, matrix, empty))

    assert scalar.data == pytest.approx(9.0)
    np.testing.assert_array_equal(matrix.data, [[1.0, 2.0], [3.0, 4.0]])
    assert empty.shape == (0, 3)
    assert scalar._version == versions[0] + 1
    assert matrix._version == versions[1] + 1
    assert empty._version == versions[2]


def test_parameter_vector_round_trip_is_exact():
    first = Tensor(np.array([[1.25, -2.5], [3.75, 4.5]]), requires_grad=True)
    second = Tensor(np.array([-8.0, 13.0]), requires_grad=False)
    before_first = first.data.copy()
    before_second = second.data.copy()

    vector = parameters_to_vector([first, second])
    first.data.fill(0.0)
    second.data.fill(0.0)

    vector_to_parameters_(vector, [first, second])

    np.testing.assert_array_equal(first.data, before_first)
    np.testing.assert_array_equal(second.data, before_second)


def test_empty_collection_round_trip_and_nonempty_length_mismatch():
    vector = parameters_to_vector([])

    assert vector.shape == (0,)
    assert vector.dtype == np.float64
    assert vector_to_parameters_(vector, []) is None

    with pytest.raises(ValueError, match="vector length mismatch: expected 0, got 1"):
        vector_to_parameters_([1.0], [])


def test_duplicate_parameter_references_are_rejected_before_mutation():
    parameter = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    before = parameter.data.copy()
    version = parameter._version

    with pytest.raises(ValueError, match="duplicate Tensor references"):
        vector_to_parameters_([4.0, 5.0, 4.0, 5.0], [parameter, parameter])

    np.testing.assert_array_equal(parameter.data, before)
    assert parameter._version == version

    with pytest.raises(ValueError, match="duplicate Tensor references"):
        parameters_to_vector([parameter, parameter])


def test_non_tensor_parameter_is_rejected_before_any_destination_write():
    first = Tensor(np.array([1.0]), requires_grad=True)
    before = first.data.copy()
    version = first._version

    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        vector_to_parameters_([8.0], [first, object()])

    np.testing.assert_array_equal(first.data, before)
    assert first._version == version


@pytest.mark.parametrize(
    "bad_vector, error, message",
    [
        (np.array(1.0), ValueError, "one-dimensional"),
        (np.ones((1, 1)), ValueError, "one-dimensional"),
        ([True], TypeError, "real numeric"),
        ([1.0 + 2.0j], TypeError, "real numeric"),
        (["1.0"], TypeError, "real numeric"),
        ([np.nan], ValueError, "only finite"),
        ([np.inf], ValueError, "only finite"),
    ],
)
def test_invalid_vectors_fail_before_mutating_parameters(bad_vector, error, message):
    parameter = Tensor(np.array([2.0]), requires_grad=True)
    before = parameter.data.copy()
    version = parameter._version

    with pytest.raises(error, match=message):
        vector_to_parameters_(bad_vector, [parameter])

    np.testing.assert_array_equal(parameter.data, before)
    assert parameter._version == version


def test_vector_length_mismatch_is_transactional():
    first = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    second = Tensor(np.array([3.0]), requires_grad=True)
    first_before = first.data.copy()
    second_before = second.data.copy()
    versions = (first._version, second._version)

    with pytest.raises(ValueError, match="vector length mismatch: expected 3, got 2"):
        vector_to_parameters_([7.0, 8.0], [first, second])

    np.testing.assert_array_equal(first.data, first_before)
    np.testing.assert_array_equal(second.data, second_before)
    assert (first._version, second._version) == versions


def test_read_only_late_parameter_is_rejected_before_earlier_write():
    first = Tensor(np.array([1.0]), requires_grad=True)
    second = Tensor(np.array([2.0]), requires_grad=True)
    first_before = first.data.copy()
    second_before = second.data.copy()
    versions = (first._version, second._version)
    second.data.flags.writeable = False

    with pytest.raises(ValueError, match="parameter 1 data must be writeable"):
        vector_to_parameters_([7.0, 8.0], [first, second])

    np.testing.assert_array_equal(first.data, first_before)
    np.testing.assert_array_equal(second.data, second_before)
    assert (first._version, second._version) == versions


def test_read_only_empty_parameter_requires_no_write():
    empty = Tensor(np.empty((0, 2)), requires_grad=True)
    empty.data.flags.writeable = False
    version = empty._version

    vector_to_parameters_(np.empty(0), [empty])

    assert empty._version == version


def test_float32_vector_is_converted_before_write_without_changing_parameter_dtype():
    parameter = Tensor(np.zeros(3), requires_grad=True)

    vector_to_parameters_(np.array([1.5, -2.25, 3.75], dtype=np.float32), [parameter])

    assert parameter.data.dtype == np.float64
    np.testing.assert_allclose(parameter.data, [1.5, -2.25, 3.75], rtol=0.0, atol=0.0)


def test_wider_finite_vector_cast_overflow_is_rejected_transactionally():
    longdouble_info = np.finfo(np.longdouble)
    float64_info = np.finfo(np.float64)
    if longdouble_info.max <= float64_info.max:
        pytest.skip("platform longdouble has no wider finite range than float64")

    parameter = Tensor(np.array([1.0]), requires_grad=True)
    before = parameter.data.copy()
    version = parameter._version
    oversized = np.array([longdouble_info.max / np.longdouble(2.0)], dtype=np.longdouble)

    with pytest.raises(ValueError, match="representable as finite Tensor data"):
        vector_to_parameters_(oversized, [parameter])

    np.testing.assert_array_equal(parameter.data, before)
    assert parameter._version == version


def test_successful_parameter_restore_invalidates_existing_forward_graph():
    parameter = Tensor(np.array([2.0]), requires_grad=True)
    output = ops.mul(parameter, parameter)

    vector_to_parameters_([3.0], [parameter])

    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()
