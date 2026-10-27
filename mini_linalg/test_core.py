from mini_linalg.core import Vector, Matrix


def test_vector_creation():
    assert Vector([1, 2, 3]).data == [1, 2, 3]


def test_matrix_creation():
    assert Matrix([[1, 2], [3, 4]]).data == [[1, 2], [3, 4]]
