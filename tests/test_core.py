import pytest
from core import Vector, Matrix


def test_vector_creation():
    v = Vector([1, 2, 3])
    assert v.data == [1, 2, 3]


def test_matrix_creation():
    m = Matrix([[1, 2], [3, 4]])
    assert m.data == [[1, 2], [3, 4]]


def test_vector_representation():
    v = Vector([1, 2, 3])
    assert repr(v) == "Vector([1, 2, 3])"


def test_matrix_representation():
    m = Matrix([[1, 2], [3, 4]])
    assert repr(m) == "Matrix([[1, 2], [3, 4]])"


def test_vector_data_copy():
    data = [1, 2, 3]
    v = Vector(data)

    data[0] = 100

    assert v.data == [1, 2, 3]


def test_matrix_data_copy():
    data = [[1, 2], [3, 4]]
    m = Matrix(data)

    data[0][0] = 100

    assert m.data == [[1, 2], [3, 4]] 
    


def test_vector_empty():
    v = Vector([])
    assert v.data == []


def test_matrix_empty():
    m = Matrix([])
    assert m.data == []


def test_vector_negative_numbers():
    v = Vector([-1, -2, -3])
    assert v.data == [-1, -2, -3]


def test_matrix_negative_numbers():
    m = Matrix([[-1, -2], [-3, -4]])
    assert m.data == [[-1, -2], [-3, -4]]


def test_vector_float_numbers():
    v = Vector([1.5, 2.5, 3.5])
    assert v.data == [1.5, 2.5, 3.5]


def test_matrix_float_numbers():
    m = Matrix([[1.5, 2.5], [3.5, 4.5]])
    assert m.data == [[1.5, 2.5], [3.5, 4.5]]


def test_matvec_correct():
    m = Matrix([[1, 2], [3, 4]])
    v = Vector([5, 6])

    result = m.matvec(v)

    assert result.data == [17, 39]


def test_matvec_dimension_mismatch():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 2])

    with pytest.raises(ValueError):
        m.matvec(v)


def test_matvec_empty_matrix():
    m = Matrix([])
    v = Vector([1, 2])

    with pytest.raises(ValueError):
        m.matvec(v)


def test_matmul_correct():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    result = a.matmul(b)

    assert result.data == [[19, 22], [43, 50]]


def test_matmul_dimension_mismatch():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_empty_matrix():
    a = Matrix([])
    b = Matrix([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        a.matmul(b) 