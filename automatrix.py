from __future__ import annotations

import pathlib
import sys
import json
import traceback
import itertools
from fractions import Fraction


class Matrix:
    def __init__(self, body: list[list[Fraction]]) -> None:
        self.body = body

    @classmethod
    def from_string(cls, content: str):
        content = content.strip()
        return Matrix(
            [[Fraction(num) for num in row.split("&")] for row in content.split("\\")]
        )

    def determinant(self):
        (a, b), (c, d) = self.body
        return a * d - b * c

    @property
    def rows(self):
        return len(self.body)

    @property
    def columns(self):
        return len(self.body[0])

    @property
    def shape(self):
        return (self.rows, self.columns)

    @classmethod
    def identity(cls, n: int):
        return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])


class AugmentedMatrix:
    def __init__(self, left: Matrix, right: Matrix):
        self.left = left
        self.right = right

    def row_operation(self, row_coefficients: list[int], target_row: int):
        self.left.body[target_row] = self.row_combination(self.left, row_coefficients)
        self.right.body[target_row] = self.row_combination(self.right, row_coefficients)

    @staticmethod
    def row_combination(matrix: Matrix, row_coefficients: list[int]) -> list[Fraction]:
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
        return all(
            [
                self.left.body[i][j] == 1 if i == j else self.left.body[i][j] == 0
                for i, j in itertools.product(
                    range(self.left.rows), range(self.left.columns)
                )
            ]
        )

    def find_pivot_row(self, column: int) -> int | None:
        for i, row in enumerate(self.left.body):
            if row[column] != 0 and i >= column:
                return i
        return None

    def move_row(self, row: int, target: int) -> None:
        self.left.body.insert(target, self.left.body.pop(row))
        self.right.body.insert(target, self.right.body.pop(row))

    def scale_row(self, coeff: Fraction, row: int) -> None:
        coeffs = [0] * self.left.rows
        coeffs[row] = coeff
        self.row_operation(coeffs, row)

    def add_to_row(self, coeff: Fraction, source: int, target: int) -> None:
        coeffs = [0] * self.left.rows
        coeffs[source] = coeff
        coeffs[target] = 1
        self.row_operation(coeffs, target)


class LatexInterface:
    def __init__(self, matrix_class: str):
        self.matrix_class = matrix_class

    def render_matrix(self, matrix: Matrix, matrix_class: str | None = None) -> str:
        if matrix_class is None:
            matrix_class = self.matrix_class
        body = "\\\\".join(
            [" & ".join([str(element) for element in row]) for row in matrix.body]
        )
        return f"\\begin{{{matrix_class}}}{body}\\end{{{matrix_class}}}"

    def render_augmented(self, matrix: AugmentedMatrix):
        return f'\\augmentedmatrix{{{self.render_matrix(matrix.left, "matrix")}}}{{{self.render_matrix(matrix.right, "matrix")}}}'

    def output_debug(self, text: str) -> None:
        print("\\begin{verbatim}" + text + "\\end{verbatim}")

    def output(self, text: str) -> None:
        print(text)

    def step(self, line: str, newline: str = "\\\\", prefix: str = "&= ") -> None:
        self.output(f"{prefix}{line}{newline}")


class Engine:
    commands = {}

    def __init__(
        self, command: str, arguments: list[str], debug: bool, matrix_class: str
    ):
        self.command = command
        self.arguments = arguments
        self.debug = debug
        self.interface = LatexInterface(matrix_class)

    @classmethod
    def command(cls, name: str):
        def wrapper(func):
            cls.commands[name] = func
            return func

        return wrapper

    def default(self, *_):
        self.interface.output("Hello World")

    def run(self):
        self.commands.get(self.command, self.default)(self, *self.arguments)


@Engine.command("inverse-formula")
def inverse_by_formula(engine: Engine, matrix: str):
    matrix = Matrix.from_string(matrix)
    (a, b), (c, d) = matrix.body
    engine.interface.step(
        f"\\frac{{1}}{{\\left({a}\\right)\\left({d}\\right)-\\left({b}\\right)\\left({c}\\right)}}"
        + engine.interface.render_matrix(Matrix([[d, -b], [-c, a]]))
    )
    det = matrix.determinant()
    engine.interface.step(
        engine.interface.render_matrix(
            Matrix([[d / det, -b / det], [-c / det, a / det]])
        )
    )


@Engine.command("inverse-rref")
def inverse_by_rref(engine: Engine, matrix: str):
    matrix = Matrix.from_string(matrix)
    augmented = AugmentedMatrix(matrix, matrix.identity(matrix.rows))
    engine.interface.output(engine.interface.render_augmented(augmented))
    # Find a pivot for the first column
    for column in range(augmented.left.columns):
        pivot_row = augmented.find_pivot_row(column)
        if pivot_row != column:
            augmented.move_row(pivot_row, column)
            engine.interface.step(
                engine.interface.render_augmented(augmented), prefix="&\\implies "
            )
        # Make the pivot
        if 1 / augmented.left.body[column][column] != 1:
            augmented.scale_row(1 / augmented.left.body[column][column], column)
            engine.interface.step(
                engine.interface.render_augmented(augmented), prefix="&\\implies "
            )
        # Make the rest of the column 0
        for row in range(augmented.left.rows):
            if row != column:
                augmented.add_to_row(-augmented.left.body[row][column], column, row)
        engine.interface.step(
            engine.interface.render_augmented(augmented), prefix="&\\implies "
        )


def main():
    with pathlib.Path(sys.argv[1]).open("r") as file:
        config = json.loads(file.read())
    Engine(**config).run()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print(f"\\typeout{{{traceback.format_exc()}}}")
        exit(1)
