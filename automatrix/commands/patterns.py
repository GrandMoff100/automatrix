"""Commands for working with patterns."""
from __future__ import annotations

import operator
from fractions import Fraction
from functools import reduce

from automatrix.dispatcher import Dispatcher
from automatrix.matrix import Matrix
from automatrix.utils import intersperse, rectangularize, remove_prefix


@Dispatcher.command("list-patterns")
def list_patterns(dispatcher: Dispatcher, matrix_string: str) -> None:
    """List all the patterns in a matrix and their inversion counts."""
    matrix = Matrix.from_string(matrix_string)
    rendered_patterns = []
    rendered_inversion_counts = []
    rendered_inversion_arrows = []
    for i, (pattern, inversions) in enumerate(matrix.patterns()):
        rendered_pattern, node_names = dispatcher.interface.wrap_matrix_with_nodes(
            dispatcher.interface.embed_pattern(matrix, pattern),
            node_name_prefix=f"pattern{i}-",
        )
        rendered_patterns.append(
            dispatcher.interface.render_grid(rendered_pattern, "bmatrix")
        )
        if inversions:
            rendered_inversion_arrows.append(
                dispatcher.interface.draw_arrows(
                    [
                        (node_names[j1][i1], node_names[j2][i2])
                        for (i1, j1), (i2, j2) in inversions
                    ],
                )
            )
        rendered_inversion_counts.append(
            f"{len(inversions)}"
            + dispatcher.interface.render_command(
                "text",
                " inversion" + "s" * int(len(inversions) != 1),
            )
        )
    display_matrix = intersperse(
        rectangularize(rendered_patterns, dispatcher.interface.pattern_grid_width),
        rectangularize(
            rendered_inversion_counts, dispatcher.interface.pattern_grid_width
        ),
    )
    dispatcher.interface.output(
        f"\\[{dispatcher.interface.render_grid(display_matrix)}\\]"
    )
    dispatcher.interface.output(
        dispatcher.interface.render_environment(
            "tikzpicture",
            "\n".join(rendered_inversion_arrows),
            options="[remember picture,overlay]",
        )
    )


@Dispatcher.command("determinant-by-pattern")
def determinant_by_pattern(dispatcher: Dispatcher, matrix_string: str) -> None:
    """
    Determine the determinant using the pattern method.
    Best method to calculate determinants, if you use
    anything else then you are lame.
    -- Zeb --
    """
    matrix = Matrix.from_string(matrix_string)
    with dispatcher.interface.output_environment("align*"):
        dispatcher.interface.output(
            dispatcher.interface.render_matrix(matrix, "vmatrix")
        )
        dispatcher.interface.step(
            "+".join(
                f"(-1)^{len(inversions)}"
                + "".join(f"\\left({matrix.body[j][i]}\\right)" for i, j in pattern)
                for pattern, inversions in matrix.patterns()
                if 0 not in (matrix.body[j][i] for i, j in pattern)
            )
        )
        dispatcher.interface.step(
            remove_prefix(
                "".join(
                    "+" * int(product >= 0) + str(product)
                    for product in (
                        Fraction(
                            (-1) ** len(inversions)
                            * reduce(
                                operator.mul, (matrix.body[j][i] for i, j in pattern), 1
                            )
                        )
                        for pattern, inversions in matrix.patterns()
                        if 0 not in (matrix.body[j][i] for i, j in pattern)
                    )
                ),
                "+",
            )
        )
        dispatcher.interface.step(
            str(
                sum(
                    Fraction(
                        (-1) ** len(inversions)
                        * reduce(
                            operator.mul, (matrix.body[j][i] for i, j in pattern), 1
                        )
                    )
                    for pattern, inversions in matrix.patterns()
                )
            ),
            newline="",
        )
