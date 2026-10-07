"""Click classes that print a command's examples verbatim.

Click re-wraps every help paragraph, so an "Examples:" block in a docstring
merges into one line with its heading and long commands break mid-argument.
These classes cut the block off the help text and print it after the
options, one line per docstring line.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any

import click

_EXAMPLES = re.compile(r"^Examples:[ \t]*$", re.MULTILINE)


def split_examples(help_text: str | None) -> tuple[str | None, str | None]:
    """Split a docstring into its help text and its dedented examples block."""
    if not help_text:
        return help_text, None
    text = inspect.cleandoc(help_text)
    parts = _EXAMPLES.split(text, maxsplit=1)
    if len(parts) == 1:
        return text, None
    examples = textwrap.dedent(parts[1]).strip("\n")
    return parts[0].rstrip(), examples or None


class _ExamplesMixin:
    examples: str | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.help, self.examples = split_examples(self.help)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        super().format_help(ctx, formatter)  # type: ignore[misc]
        if not self.examples:
            return
        with formatter.section("Examples"):
            indent = " " * formatter.current_indent
            for line in self.examples.splitlines():
                formatter.write(f"{indent}{line}\n" if line.strip() else "\n")


class ExamplesCommand(_ExamplesMixin, click.Command):
    """A command whose docstring examples are printed verbatim."""


class ExamplesGroup(_ExamplesMixin, click.Group):
    """A group whose commands and subgroups print examples verbatim."""

    command_class = ExamplesCommand
    group_class = type  # subgroups use this class too
