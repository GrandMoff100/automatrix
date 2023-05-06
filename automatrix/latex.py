"""A module for raw-LaTeX-generating classes and functions."""
from __future__ import annotations

import contextlib
from typing import Any

from automatrix.matrix import AugmentedMatrix, Matrix


class LatexInterface:
    """A class to render raw LaTeX."""

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
        """
        Embed a pattern into a matrix
        i.e. circle the elements of the matrix that are in the pattern
        """
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
        return "\n".join(
            f"\\draw [->] ({start}) to[bend left=30] ({end});" for start, end in arrows
        )

    def render_grid(
        self, matrix_body: list[list[Any]], matrix_class: str | None = None
    ) -> str:
        """Render elements in an invisible LaTeX matrix"""
        if matrix_class is None:
            matrix_class = "matrix"
        return self.render_environment(
            matrix_class,
            "\\\\".join(
                [" & ".join([str(element) for element in row]) for row in matrix_body]
            ),
        )

    def render_matrix(self, matrix: Matrix, matrix_class: str | None = None) -> str:
        """Render a matrix as a LaTeX matrix"""
        if matrix_class is None:
            matrix_class = self.matrix_class
        return self.render_grid(matrix.body, matrix_class)

    def render_matrices(self, matrices: list[Matrix], matrix_class: str | None = None):
        """Render multiple matrices as LaTeX matrices"""
        return "".join(
            [self.render_matrix(matrix, matrix_class) for matrix in matrices]
        )

    def render_augmented(self, matrix: AugmentedMatrix):
        """Render an augmented matrix as a LaTeX matrix"""
        return self.render_command(
            "augmentedmatrix",
            self.render_matrices([matrix.left, matrix.right], "matrix"),
        )

    def render_environment(self, environment: str, content: str, options: str = ""):
        """Render a LaTeX environment"""
        return f"\\begin{{{environment}}}{options}\n{content}\n\\end{{{environment}}}"

    def render_command(self, command: str, *args: str):
        """Render a LaTeX command"""
        return f"\\{command}" + "".join([f"{{{arg}}}" for arg in args])

    def output_debug(self, text: str) -> None:
        """Output debug information"""
        self.output(self.render_environment("verbatim", text))

    def output(self, text: str) -> None:
        """Output raw LaTeX to stdout"""
        print(text)

    def step(self, line: str, newline: str = "\\\\", prefix: str = "&= ") -> None:
        """Output a step in a proof"""
        self.output(f"{prefix}{line}{newline}")

    @contextlib.contextmanager
    def output_environment(self, name: str) -> None:
        """Output within LaTeX environment"""
        self.output(f"\\begin{{{name}}}")
        yield
        self.output(f"\\end{{{name}}}")
