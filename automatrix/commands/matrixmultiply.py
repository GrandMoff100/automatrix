"""Module for multiplying matrices in LaTeX.""" ""
from __future__ import annotations

from automatrix.dispatcher import Dispatcher
from automatrix.matrix import Matrix


@Dispatcher.command("matrix-multiply")
def matrix_multiply(dispatcher: Dispatcher, matrix_strings: str):
    """Multiply a series of matrices together"""
    matrices = [Matrix.from_string(matrix) for matrix in matrix_strings.split(",")]
    dispatcher.interface.output(dispatcher.interface.render_matrices(matrices))
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
        dispatcher.interface.step(dispatcher.interface.render_matrices(others + [left]))
        dispatcher.interface.step(dispatcher.interface.render_matrices(matrices))
