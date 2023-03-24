from __future__ import annotations

import pathlib
import sys
import json
import traceback
import itertools
import copy
from fractions import Fraction
from typing import Any, Generator


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

    def crop(self, x: int, y: int) -> Matrix:
        newmat = Matrix(copy.deepcopy(self.body))
        newmat.body.pop(y)
        for row in newmat.body:
            row.pop(x)
        return newmat

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

    def __matmul__(self, matrix):
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

    def patterns(self) -> Generator[list[Fraction], None, None]:
        """Generate the "patterns" of a matrix, i.e. the permutations of its rows"""
        if self.rows == 1:
            yield [self.body[0][0]]
            return
        for i in range(self.rows):
            cropped_mat = self.crop(i, 0)
            for others in cropped_mat.patterns():
                yield [self.body[0][i], *others]


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

    def render_matrix_body(self, matrix_body: list[list[Any]]) -> str:
        return "\\\\".join(
            [" & ".join([str(element) for element in row]) for row in matrix_body]
        )

    def render_matrix(self, matrix: Matrix, matrix_class: str | None = None) -> str:
        if matrix_class is None:
            matrix_class = self.matrix_class
        body = self.render_matrix_body(matrix.body)
        return f"\\begin{{{matrix_class}}}{body}\\end{{{matrix_class}}}"

    def render_matrices(self, matrices: list[Matrix], matrix_class: str | None = None):
        return "".join(
            [self.render_matrix(matrix, matrix_class) for matrix in matrices]
        )

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
        if pivot_row is None:
            continue
        elif pivot_row != column:
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


@Engine.command("matrix-multiply")
def matrix_multiply(engine: Engine, matrices: str):
    matrices = [Matrix.from_string(matrix) for matrix in matrices.split(",")]
    engine.interface.output(engine.interface.render_matrices(matrices))
    while len(matrices) > 1:
        *others, left, right = matrices
        matrices = others + [left @ right]
        left.body = [
            [
                "+".join(
                    [
                        f"\\left({a}\\right)\\left({b}\\right)"
                        for a, b in zip(row, column)
                    ]
                )
                for column in zip(*right.body)
            ]
            for row in left.body
        ]
        engine.interface.step(engine.interface.render_matrices(others + [left]))
        engine.interface.step(engine.interface.render_matrices(matrices))


@Engine.command("determinant-by-pattern")
def determinant_pattern(engine: Engine, matrix: str):
    """
    Determine the determinant using the pattern method.
    Best method to calculate determinants, if you use
    anything else then you are lame.
    """

    matrix = Matrix.from_string(matrix)
    patterns = list(addcoords(matrix).patterns())
    inversion_list = []
    for pat in patterns:
        inversions = 0
        for pair in itertools.combinations(pat, 2):
            a = pair[0][1]
            b = pair[1][1]
            if (a[0] < b[0] and a[1] > b[1]) or (b[0] < a[0] and b[1] > a[1]):
                inversions += 1
        inversion_list.append(inversions)
    total = 0
    products = []
    lr = 0
    next_out = None
    engine.interface.output(
        "\\begin{mdframed} $\\det\\left("
        + engine.interface.render_matrix(matrix)
        + "\\right)$ by pattern method.\\\\\\vspace{2mm}\\line(1,0){\\columnwidth} \\\\\\vspace{2mm}\\\\"
    )
    for pat, invs in zip(patterns, inversion_list):
        circled = copy.deepcopy(matrix)
        for num in pat:
            # This is fine
            circled.body[num[1][0]][num[1][1]] = "\\circled{%s}" % str(
                circled.body[num[1][0]][num[1][1]]
            )
        engine.interface.output(
            ("\\hfill" if lr % 2 == 1 else "")
            + "$"
            + engine.interface.render_matrix(circled)
            + "$"
        )
        sign = "+" if invs % 2 == 0 else "-"
        lr += 1
        prod = 1
        for i in pat:
            prod *= i[0]
        if sign == "+":
            total += prod
        else:
            total -= prod
        if lr % 2 == 0:
            pats = "\\cdot".join([str(i[0]) for i in pat])
            engine.interface.output(f"\\\\${sign}({pats}) \\Rightarrow {sign}$")
            engine.interface.output(prod)
            if next_out is not None:
                engine.interface.output("\\hfill" + next_out[0])
                engine.interface.output(str(next_out[1]) + "\\\\")
        else:
            pats = "\\cdot".join([str(i[0]) for i in pat])
            next_out = (f"${sign}({pats}) \\Rightarrow {sign}$", prod)
        products.append(sign + str(prod))
    engine.interface.output(
        "\\\\\\vspace{2mm}\\line(1,0){\\columnwidth} \\\\\\vspace{2mm}\\\\"
        "$\\det\\left("
        f"{engine.interface.render_matrix(matrix)}"
        f"\\right) = {''.join(products)} = {total}"
        "$\n\\end{mdframed}"
    )


def addcoords(matrix: Matrix):
    """
    Add coords to a matrix
    ~~Not at all violating typehints~~
    """
    matrix = copy.deepcopy(matrix)
    for i, row in enumerate(matrix.body):
        for j, val in enumerate(row):
            matrix.body[i][j] = (val, (i, j))
    return matrix


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
