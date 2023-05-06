"""The entry module for the automatrix package."""
from __future__ import annotations

import json
import pathlib
import sys
import traceback

from automatrix.dispatcher import Dispatcher


def main():
    """The entrypoint for the automatrix package."""
    try:
        with pathlib.Path(sys.argv[1]).open("r", encoding="utf-8") as file:
            config = json.loads(file.read())
    except IndexError:
        print("Usage: automatrix <config.json>")
        exit(1)
    try:
        Dispatcher(**config).run()
    except BaseException:  # pylint: disable=broad-except
        print(f"\\typeout{{{traceback.format_exc()}}}")
        exit(1)


if __name__ == "__main__":
    main()
