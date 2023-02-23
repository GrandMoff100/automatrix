from __future__ import annotations

import pathlib
import sys
import json
import traceback
from fractions import Fraction

class Matrix:
    def __init__(self, body: list[list[Fraction]]) -> None:
        self.body = body

    @classmethod
    def from_string(cls, content: str):
        content = content.strip()
        return Matrix([[Fraction(num) for num in row.split("&")] for row in content.split("\\")])

    def determinant(self):
        (a, b), (c, d) = self.body
        return a * d - b * c

    @classmethod
    def identity(cls, n: int):
        return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])


class AugmentedMatrix:
    def __init__(self, left: Matrix, right: Matrix):
        self.left = left
        self.right = right

    def row_operation(self):
        pass


class LatexInterface:
    def __init__(self, matrix_class: str):
        self.matrix_class = matrix_class

    def render(self, matrix: list[list[str]]) -> str:
        body = "\\\\".join([" & ".join([str(element) for element in row]) for row in matrix])
        return f"\\begin{{{self.matrix_class}}}{body}\\end{{{self.matrix_class}}}"

    def render_augmented(self, left: Matrix, right: Matrix):
        pass

    def output_debug(self, text: str) -> None:
        print("\\begin{verbatim}" + text + "\\end{verbatim}")

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
        matrix_class: str
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
    (a,b), (c,d) = matrix.body
    engine.interface.step(
        f"\\frac{{1}}{{\\left({a}\\right)\\left({d}\\right)-\\left({b}\\right)\\left({c}\\right)}}" + 
        engine.interface.render([[d, -b],[-c, a]])
    )
    det = matrix.determinant()
    engine.interface.step(
        engine.interface.render([[d / det, -b / det],[-c / det, a / det]])
    )


@Engine.command("inverse-rref")
def inverse_by_rref(engine: Engine, matrix: str):
    matrix = Matrix.from_string(matrix)
    augmented = AugmentedMatrix(matrix, matrix.identity())


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