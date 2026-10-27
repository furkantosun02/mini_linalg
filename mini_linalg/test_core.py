
import pytest

from mini_linalg.core import (
    Vector,
    Matrix,
    batch_add_vectors,
    batch_add_matrices,
    batch_scale_vectors,
    batch_scale_matrices,
)


def test_vector_creation():
    v = Vector([1, 2, 3])
    assert v.data == [1, 2, 3]


def test_matrix_creation():
    m = Matrix([[1, 2], [3, 4]])
    assert m.data == [[1, 2], [3, 4]]


def test_matrix_vector_product():
    matrix = Matrix([[1, 2], [3, 4]])
    vector = Vector([5, 6])

    assert matrix.matvec(vector).data == [17, 39]


def test_matrix_matrix_product():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    assert a.matmul(b).data == [[19, 22], [43, 50]]


def test_batch_add_vectors():
    a = [Vector([1, 2]), Vector([3, 4])]
    b = [Vector([10, 20]), Vector([30, 40])]

    result = batch_add_vectors(a, b)

    assert [v.data for v in result] == [
        [11, 22],
        [33, 44],
    ]


def test_batch_add_vectors_dimension_mismatch():
    a = [Vector([1, 2])]
    b = [Vector([3, 4, 5])]

    with pytest.raises(ValueError):
        batch_add_vectors(a, b)


def test_batch_add_vectors_batch_size_mismatch():
    a = [Vector([1, 2])]
    b = [Vector([3, 4]), Vector([5, 6])]

    with pytest.raises(ValueError):
        batch_add_vectors(a, b)


def test_batch_add_matrices():
    a = [Matrix([[1, 2], [3, 4]])]
    b = [Matrix([[10, 20], [30, 40]])]

    result = batch_add_matrices(a, b)

    assert result[0].data == [[11, 22], [33, 44]]


def test_batch_add_matrices_dimension_mismatch():
    a = [Matrix([[1, 2], [3, 4]])]
    b = [Matrix([[1, 2, 3], [4, 5, 6]])]

    with pytest.raises(ValueError):
        batch_add_matrices(a, b)


def test_batch_scale_vectors():
    vectors = [Vector([1, 2]), Vector([3, 4])]
    scalars = [2, 3]

    result = batch_scale_vectors(vectors, scalars)

    assert [v.data for v in result] == [
        [2, 4],
        [9, 12],
    ]


def test_batch_scale_matrices():
    matrices = [
        Matrix([[1, 2], [3, 4]]),
        Matrix([[5, 6], [7, 8]]),
    ]
    scalars = [2, 3]

    result = batch_scale_matrices(matrices, scalars)

    assert [m.data for m in result] == [
        [[2, 4], [6, 8]],
        [[15, 18], [21, 24]],
    ]


def test_batch_scale_batch_size_mismatch():
    with pytest.raises(ValueError):
        batch_scale_vectors([Vector([1, 2])], [])
