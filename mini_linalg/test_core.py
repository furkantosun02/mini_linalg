from mini_linalg.core import Vector, Matrix


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
