"""A module for utility functions."""
from __future__ import annotations

from typing import Any, Generator, Iterable


def remove_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from a string."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def intersperse(
    _iterable: Iterable[Any],
    delimiters: Iterable[Any],
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
