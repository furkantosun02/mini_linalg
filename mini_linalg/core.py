class Vector :
    def __init__ ( self , data : list [ float ]) :
        self . data = list ( data )
    def __repr__ ( self ) -> str :
        return f" Vector ({ self . data })"

class Matrix :
    def __init__ ( self , data : list [ list [ float ]]) :
        self . data = [ list (r) for r in data ]
    def __repr__ ( self ) -> str :
        return f" Matrix ({ self . data })"

def batch_add_vectors(
    vectors_a: list[Vector],
    vectors_b: list[Vector],
) -> list[Vector]:
    if len(vectors_a) != len(vectors_b):
        raise ValueError("Batch sizes must match.")

    results = []

    for a, b in zip(vectors_a, vectors_b):
        if len(a.data) != len(b.data):
            raise ValueError("Vector dimensions must match.")

        result = [x + y for x, y in zip(a.data, b.data)]
        results.append(Vector(result))

    return results


def batch_add_matrices(
    matrices_a: list[Matrix],
    matrices_b: list[Matrix],
) -> list[Matrix]:
    if len(matrices_a) != len(matrices_b):
        raise ValueError("Batch sizes must match.")

    results = []

    for a, b in zip(matrices_a, matrices_b):
        if len(a.data) != len(b.data):
            raise ValueError("Matrix dimensions must match.")

        if any(
            len(row_a) != len(row_b)
            for row_a, row_b in zip(a.data, b.data)
        ):
            raise ValueError("Matrix dimensions must match.")

        result = [
            [x + y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(a.data, b.data)
        ]
        results.append(Matrix(result))

    return results


def batch_scale_vectors(
    vectors: list[Vector],
    scalars: list[float],
) -> list[Vector]:
    if len(vectors) != len(scalars):
        raise ValueError("Batch sizes must match.")

    return [
        Vector([scalar * x for x in vector.data])
        for vector, scalar in zip(vectors, scalars)
    ]


def batch_scale_matrices(
    matrices: list[Matrix],
    scalars: list[float],
) -> list[Matrix]:
    if len(matrices) != len(scalars):
        raise ValueError("Batch sizes must match.")

    return [
        Matrix([
            [scalar * x for x in row]
            for row in matrix.data
        ])
        for matrix, scalar in zip(matrices, scalars)
    ]

