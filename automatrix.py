from __future__ import annotations

import copy
import itertools
import json
import pathlib
import sys
import traceback
from fractions import Fraction
from typing import Any, Generator, Iterable


def intersperse(
    _iterable: Iterable[Any], delimiters: Iterable[Any]
) -> Generator[Any, None, None]:
    """
    Intersperse elements in a delimiter iterable in between elements in an iterable.
    Note: cuts off iterable with delimiter size.
    """
    iterable = iter(_iterable)
    yield next(iterable)
    for delimiter, element in zip(delimiters, iterable):
        yield delimiter
        yield element


def rectangularize(_iterable: Iterable[Any], width: int) -> list[list[Any]]:
    """Reshape a 1D iterable into a 2D list."""
    elements = list(_iterable)
    return [
        elements[i * width : (i + 1) * width] for i in range(len(elements) // width + 1)
    ]


class Matrix:
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
        """Calculate the determinant of a 2x2 matrix."""
        (a, b), (c, d) = self.body
        return a * d - b * c

    def crop(self, x: int, y: int) -> "Matrix":
        new = Matrix(copy.deepcopy(self.body))
        new.body.pop(y)
        for row in new.body:
            row.pop(x)
        return new

    @property
    def rows(self) -> int:
        return len(self.body)

    @property
    def columns(self) -> int:
        return len(self.body[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.columns)

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])

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
        """Generate the "patterns" and the "inversions" of each "pattern" of a matrix, i.e. the permutations of its columns/rows"""
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
    def __init__(self, matrix_class: str, pattern_grid_width: int) -> None:
        self.matrix_class = matrix_class
        self.pattern_grid_width = pattern_grid_width

    def wrap_matrix_with_nodes(
        self, body: list[list[str]], node_name_prefix: str = ""
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Wrap a matrix's element as tikz nodes. Return the matrix and the node names"""
        return [
            [
                self.render_command("rn", f"{node_name_prefix}{i}-{j}", element)
                for i, element in enumerate(row)
            ]
            for j, row in enumerate(body)
        ], [
            [f"{node_name_prefix}{i}-{j}" for i, _ in enumerate(row)]
            for j, row in enumerate(body)
        ]

    def embed_pattern(
        self, matrix: Matrix, pattern: list[tuple[int, int]]
    ) -> list[list[str]]:
        """Embed a pattern into a matrix, i.e. circle the elements of the matrix that are in the pattern"""
        return [
            [
                self.render_command("circled", element)
                if (i, j) in pattern
                else str(element)
                for i, element in enumerate(row)
            ]
            for j, row in enumerate(matrix.body)
        ]

    def draw_arrows(self, arrows: list[tuple[str, str]]) -> list[list[str]]:
        """Draw arrows between the tikz nodes in a matrix"""
        return "\n".join(f"\\draw [->] ({start}) -- ({end});" for start, end in arrows)

    def render_grid(
        self, matrix_body: list[list[Any]], matrix_class: str | None = None
    ) -> str:
        if matrix_class is None:
            matrix_class = "matrix"
        return self.render_environment(
            matrix_class,
            "\\\\".join(
                [" & ".join([str(element) for element in row]) for row in matrix_body]
            ),
        )

    def render_matrix(self, matrix: Matrix, matrix_class: str | None = None) -> str:
        if matrix_class is None:
            matrix_class = self.matrix_class
        return self.render_grid(matrix.body, matrix_class)

    def render_matrices(self, matrices: list[Matrix], matrix_class: str | None = None):
        return "".join(
            [self.render_matrix(matrix, matrix_class) for matrix in matrices]
        )

    def render_augmented(self, matrix: AugmentedMatrix):
        return self.render_command(
            "augmentedmatrix",
            self.render_matrices([matrix.left, matrix.right], "matrix"),
        )

    def render_environment(self, environment: str, content: str, options: str = ""):
        return f"\\begin{{{environment}}}{options}\n{content}\n\\end{{{environment}}}"

    def render_command(self, command: str, *args: str):
        return f"\\{command}" + "".join([f"{{{arg}}}" for arg in args])

    def output_debug(self, text: str) -> None:
        self.output(self.render_environment("verbatim", text))

    def output(self, text: str) -> None:
        print(text)

    def step(self, line: str, newline: str = "\\\\", prefix: str = "&= ") -> None:
        self.output(f"{prefix}{line}{newline}")


class Engine:
    commands = {}

    def __init__(
        self,
        command: str,
        arguments: list[str],
        debug: bool,
        matrix_class: str,
        pattern_grid_width: int,
    ):
        self.command = command
        self.arguments = arguments
        self.debug = debug
        self.interface = LatexInterface(matrix_class, pattern_grid_width)

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
def matrix_multiply(engine: Engine, matrix_strings: str):
    matrices = [Matrix.from_string(matrix) for matrix in matrix_strings.split(",")]
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
def determinant_pattern(engine: Engine, matrix_string: str) -> None:
    """
    Determine the determinant using the pattern method.
    Best method to calculate determinants, if you use
    anything else then you are lame.
    -- Zeb --
    """
    matrix = Matrix.from_string(matrix_string)
    rendered_patterns = []
    rendered_inversion_counts = []
    rendered_inversion_arrows = []
    for i, (pattern, inversions) in enumerate(matrix.patterns()):
        rendered_pattern, node_names = engine.interface.wrap_matrix_with_nodes(
            engine.interface.embed_pattern(matrix, pattern),
            node_name_prefix=f"pattern{i}-",
        )
        rendered_patterns.append(
            engine.interface.render_grid(rendered_pattern, "bmatrix")
        )
        if inversions:
            rendered_inversion_arrows.append(
                engine.interface.draw_arrows(
                    [
                        (node_names[j1][i1], node_names[j2][i2])
                        for (i1, j1), (i2, j2) in inversions
                    ],
                )
            )
        rendered_inversion_counts.append(
            f"{len(inversions)}"
            + engine.interface.render_command(
                "text",
                " inversion" + "s" * int(len(inversions) != 1),
            )
        )
    display_matrix = intersperse(
        rectangularize(rendered_patterns, engine.interface.pattern_grid_width),
        rectangularize(rendered_inversion_counts, engine.interface.pattern_grid_width),
    )
    engine.interface.output(f"\\[{engine.interface.render_grid(display_matrix)}\\]")
    engine.interface.output(
        engine.interface.render_environment(
            "tikzpicture",
            "\n".join(rendered_inversion_arrows),
            options="[remember picture,overlay]",
        )
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
