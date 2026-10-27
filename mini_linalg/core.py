class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = [list(row) for row in data]

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def matvec(self, vector: Vector) -> Vector:
        if not self.data or len(self.data[0]) != len(vector.data):
            raise ValueError("Incompatible dimensions")

        result = [
            sum(a * b for a, b in zip(row, vector.data))
            for row in self.data
        ]
        return Vector(result)

    def matmul(self, other: "Matrix") -> "Matrix":
        if not self.data or not other.data:
            raise ValueError("Empty matrices are not supported")

        if len(self.data[0]) != len(other.data):
            raise ValueError("Incompatible dimensions")

        result = [
            [
                sum(a * b for a, b in zip(row, col))
                for col in zip(*other.data)
            ]
            for row in self.data
        ]
        return Matrix(result)
