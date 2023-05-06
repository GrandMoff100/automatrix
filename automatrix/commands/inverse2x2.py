"""Command for calculating the inverse of a 2x2 matrix"""
from __future__ import annotations

from automatrix.dispatcher import Dispatcher
from automatrix.matrix import Matrix


@Dispatcher.command("inverse-2x2-formula")
def inverse_by_formula(dispatcher: Dispatcher, matrix: str):
    """Calculate the inverse of a 2x2 matrix using the formula"""
    matrix = Matrix.from_string(matrix)
    (a, b), (c, d) = matrix.body  # pylint: disable=invalid-name
    dispatcher.interface.step(
        f"\\frac{{1}}{{\\left({a}\\right)\\left({d}\\right)-\\left({b}\\right)\\left({c}\\right)}}"
        + dispatcher.interface.render_matrix(Matrix([[d, -b], [-c, a]]))
    )
    det = matrix.determinant()
    dispatcher.interface.step(
        dispatcher.interface.render_matrix(
            Matrix([[d / det, -b / det], [-c / det, a / det]])
        )
    )
