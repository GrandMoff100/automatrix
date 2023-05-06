"""A module for the dispatcher class."""
from __future__ import annotations

from automatrix.latex import LatexInterface


class Dispatcher:
    """A class to dispatch commands to functions"""

    commands = {}

    def __init__(
        self,
        command: str,
        arguments: list[str],
        debug: bool,
        matrix_class: str,
        pattern_grid_width: int,
    ):
        self._command = command
        self.arguments = arguments
        self.debug = debug
        self.interface = LatexInterface(matrix_class, pattern_grid_width)

    @classmethod
    def command(cls, name: str):
        """A decorator to register a command"""

        def wrapper(func):
            cls.commands[name] = func
            return func

        return wrapper

    def default(self, *_):
        """The default command"""
        self.interface.output("Hello World")

    def run(self):
        """Run the command"""
        self.commands.get(self._command, self.default)(self, *self.arguments)
