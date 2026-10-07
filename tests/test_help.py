"""Tests for the verbatim examples in CLI help."""

from click.testing import CliRunner

from pyzotero._help import split_examples
from pyzotero.cli import main


def test_split_examples():
    help_text, examples = split_examples(
        """Do a thing.

        Examples:
            pyzotero thing A

            pyzotero thing B --long
        """
    )
    assert help_text == "Do a thing."
    assert examples == "pyzotero thing A\n\npyzotero thing B --long"
    assert split_examples("No examples.") == ("No examples.", None)


def test_examples_are_printed_verbatim():
    out = CliRunner().invoke(main, ["highlight", "--help"]).output
    assert "Examples:\n" in out
    assert (
        "  pyzotero highlight ABC12345 --rect 1 72 688 300 701 --rect 1 72 674 200 687\n"
        in out
    )
    assert "Examples:  " not in out


def test_subgroup_commands_use_examples_class():
    out = CliRunner().invoke(main, ["note", "add", "--help"]).output
    assert "\nExamples:\n" in out
