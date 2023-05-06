"""Calculate the inverse of a matrix using row reduction"""
from __future__ import annotations

from automatrix.dispatcher import Dispatcher
from automatrix.matrix import AugmentedMatrix, Matrix


@Dispatcher.command("inverse-rref")
def inverse_by_rref(dispatcher: Dispatcher, matrix: str):
    """Calculate the inverse of a matrix using row reduction"""
    matrix = Matrix.from_string(matrix)
    augmented = AugmentedMatrix(matrix, matrix.identity(matrix.rows))
    dispatcher.interface.output(dispatcher.interface.render_augmented(augmented))
    # Find a pivot for the first column
    for column in range(augmented.left.columns):
        pivot_row = augmented.find_pivot_row(column)
        if pivot_row is None:
            continue
        elif pivot_row != column:
            augmented.move_row(pivot_row, column)
            dispatcher.interface.step(
                dispatcher.interface.render_augmented(augmented), prefix="&\\implies "
            )
        # Make the pivot
        if 1 / augmented.left.body[column][column] != 1:
            augmented.scale_row(1 / augmented.left.body[column][column], column)
            dispatcher.interface.step(
                dispatcher.interface.render_augmented(augmented), prefix="&\\implies "
            )
        # Make the rest of the column 0
        for row in range(augmented.left.rows):
            if row != column:
                augmented.add_to_row(-augmented.left.body[row][column], column, row)
        dispatcher.interface.step(
            dispatcher.interface.render_augmented(augmented), prefix="&\\implies "
        )
