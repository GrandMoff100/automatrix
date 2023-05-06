"""A module to interact with matrices."""
from __future__ import annotations

import copy
import itertools
from fractions import Fraction
from typing import Generator


class Matrix:
    """A math matrix class."""

    def __init__(self, body: list[list[Fraction]]) -> None:
        self.body = body

    @classmethod
    def from_string(cls, content: str) -> "Matrix":
        """Create a matrix from a string in the format."""
        content = content.strip()
        return Matrix(
            [[Fraction(num) for num in row.split("&")] for row in content.split("\\")]
        )

    def determinant(self) -> Fraction:
        """Calculate the determinant of a matrix using cofactor expansion and the 2x2 formula."""
        if self.shape == (1, 1):
            return self.body[0][0]
        elif self.shape == (2, 2):
            (a, b), (c, d) = self.body  # pylint: disable=invalid-name
            return a * d - b * c
        else:
            return sum(
                [
                    (-1) ** i * self.body[0][i] * self.crop(i, 0).determinant()
                    for i in range(self.columns)
                ]
            )

    def crop(self, column: int, row: int) -> "Matrix":
        """Crop a matrix by removing a row and column."""
        new = Matrix(copy.deepcopy(self.body))
        new.body.pop(row)
        for row in new.body:
            row.pop(column)
        return new

    @property
    def rows(self) -> int:
        """Return the number of rows in the matrix."""
        return len(self.body)

    @property
    def columns(self) -> int:
        """Return the number of columns in the matrix."""
        return len(self.body[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the matrix as a tuple."""
        return (self.rows, self.columns)

    @classmethod
    def identity(cls, size: int) -> "Matrix":
        """Return the identity matrix of any size."""
        return Matrix([[1 if i == j else 0 for i in range(size)] for j in range(size)])

    def __matmul__(self, matrix) -> "Matrix":
        if self.columns != matrix.rows:
            raise ValueError("Matrices are not compatible")
        return Matrix(
            [
                [
                    sum([a * b for a, b in zip(row, column)])
                    for column in zip(*matrix.body)
                ]
                for row in self.body
            ]
        )

    def patterns(
        self,
    ) -> Generator[
        tuple[
            list[tuple[int, int]],
            list[
                tuple[
                    tuple[int, int],
                    tuple[int, int],
                ]
            ],
        ],
        None,
        None,
    ]:
        """
        Generate the "patterns" and the "inversions" of each "pattern" of a matrix
        i.e. the permutations of its columns/rows.
        """
        for row_permutation in itertools.permutations(range(self.rows)):
            pattern = [
                (i, j)
                for j, row in enumerate(self.body)
                for i, _ in enumerate(row)
                if (i, j) in enumerate(row_permutation)
            ]
            inversions = [
                ((x1, y1), (x2, y2))
                for x2, y2 in pattern
                for x1, y1 in pattern
                if y1 > y2 and x1 < x2
            ]
            yield pattern, inversions


class AugmentedMatrix:
    """An augmented matrix of two smaller `Matrix`'s."""

    def __init__(self, left: Matrix, right: Matrix) -> None:
        self.left = left
        self.right = right

    def row_operation(self, row_coefficients: list[int], target_row: int) -> None:
        """Perform a row operation on the matrix."""
        self.left.body[target_row] = self.row_combination(self.left, row_coefficients)
        self.right.body[target_row] = self.row_combination(self.right, row_coefficients)

    @staticmethod
    def row_combination(matrix: Matrix, row_coefficients: list[int]) -> list[Fraction]:
        """Linearly combine rows of a matrix."""
        return [
            sum(column)
            for column in zip(
                *[
                    [coeff * elem for elem in row]
                    for coeff, row in zip(row_coefficients, matrix.body)
                ]
            )
        ]

    def is_identity(self) -> bool:
        """Check if the left matrix is an identity matrix."""
        return all(
            [
                self.left.body[i][j] == 1 if i == j else self.left.body[i][j] == 0
                for i, j in itertools.product(
                    range(self.left.rows), range(self.left.columns)
                )
            ]
        )

    def find_pivot_row(self, column: int) -> int | None:
        """Find the first non-zero element in a column."""
        for i, row in enumerate(self.left.body):
            if row[column] != 0 and i >= column:
                return i
        return None

    def move_row(self, row: int, target: int) -> None:
        """Move a row to a new position."""
        self.left.body.insert(target, self.left.body.pop(row))
        self.right.body.insert(target, self.right.body.pop(row))

    def scale_row(self, coeff: Fraction, row: int) -> None:
        """Scale a row by a coefficient."""
        coeffs = [0] * self.left.rows
        coeffs[row] = coeff
        self.row_operation(coeffs, row)

    def add_to_row(self, coeff: Fraction, source: int, target: int) -> None:
        """Add a linear combination of rows to another row."""
        coeffs = [0] * self.left.rows
        coeffs[source] = coeff
        coeffs[target] = 1
        self.row_operation(coeffs, target)
