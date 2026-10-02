"""Tests for Zotero notes: the Markdown converters, the operations, CLI and MCP."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pyzotero import cli, mcp_server, notes
from pyzotero import errors as ze


def note_item(key="NOTE0001", html="<p>Body</p>", version=7, **data):
    data = {
        "key": key,
        "version": version,
        "itemType": "note",
        "note": html,
        "dateModified": "2026-01-02T03:04:05Z",
        **data,
    }
    return {"key": key, "version": version, "data": data}


@pytest.fixture
def zot():
    mock = MagicMock()
    mock.create_items.return_value = {"success": {"0": "NEWNOTE"}, "failed": {}}
    mock.everything.side_effect = lambda query: query
    mock.children.return_value = []
    mock.item.return_value = note_item()
    return mock


class TestMarkdownToHtml:
    @pytest.mark.parametrize(
        ("markdown", "expected"),
        [
            ("# Title", "<h1>Title</h1>"),
            ("### Sub ###", "<h3>Sub</h3>"),
            ("one\ntwo\n\nthree", "<p>one two</p>\n<p>three</p>"),
            (
                "a **b** __c__ *d* _e_",
                "<p>a <strong>b</strong> <strong>c</strong> <em>d</em> <em>e</em></p>",
            ),
            ("use `a < b` here", "<p>use <code>a &lt; b</code> here</p>"),
            ("snake_case_name and 2 * 3 * 4", "<p>snake_case_name and 2 * 3 * 4</p>"),
            (r"\*not italic\*", "<p>*not italic*</p>"),
            ("<script>x</script> & y", "<p>&lt;script&gt;x&lt;/script&gt; &amp; y</p>"),
            (
                "[x](https://a.org/p?q=1&r=2)",
                '<p><a href="https://a.org/p?q=1&amp;r=2">x</a></p>',
            ),
            (
                "[w](https://en.wikipedia.org/wiki/A_(b))",
                '<p><a href="https://en.wikipedia.org/wiki/A_(b)">w</a></p>',
            ),
            ("[bad](javascript:alert(1))", "<p>bad</p>"),
            ("> quoted\n> more", "<blockquote><p>quoted more</p></blockquote>"),
            ("---", "<hr>"),
            (
                "```py\nx = 1 < 2\n\ny = **3**\n```",
                "<pre><code>x = 1 &lt; 2\n\ny = **3**</code></pre>",
            ),
            ("```\nunclosed", "<pre><code>unclosed</code></pre>"),
        ],
    )
    def test_cases(self, markdown, expected):
        assert notes.markdown_to_html(markdown) == expected

    def test_bullet_list(self):
        assert (
            notes.markdown_to_html("- a\n* b\n+ c")
            == "<ul><li>a</li><li>b</li><li>c</li></ul>"
        )

    def test_numbered_list(self):
        assert notes.markdown_to_html("1. a\n2) b") == "<ol><li>a</li><li>b</li></ol>"

    def test_nested_list_and_continuation(self):
        md = "- one\n- two\n  - deep\n  - deeper\n- three\n  more"
        assert notes.markdown_to_html(md) == (
            "<ul><li>one</li><li>two<ul><li>deep</li><li>deeper</li></ul></li>"
            "<li>three more</li></ul>"
        )

    def test_list_interrupts_paragraph_and_blank_lines_keep_one_list(self):
        assert notes.markdown_to_html("Results:\n- a\n\n- b") == (
            "<p>Results:</p>\n<ul><li>a</li><li>b</li></ul>"
        )

    def test_mixed_list_types_become_two_lists(self):
        assert notes.markdown_to_html("- a\n1. b") == (
            "<ul><li>a</li></ul>\n<ol><li>b</li></ol>"
        )

    def test_crlf(self):
        assert notes.markdown_to_html("# A\r\n\r\ntext") == "<h1>A</h1>\n<p>text</p>"

    def test_rule_is_not_a_list(self):
        assert notes.markdown_to_html("- - -") == "<hr>"


class TestHtmlToMarkdown:
    def test_round_trip_of_supported_markdown(self):
        md = (
            "# Title\n\nSome *italic* and **bold** with `code` and [a link](https://x.org).\n\n"
            "- one\n- two\n  - nested\n\n1. first\n2. second\n\n"
            "> quoted\n\n```\nx = 1 < 2\n```\n\n---\n"
        )
        assert notes.html_to_markdown(notes.markdown_to_html(md)) == md

    def test_zotero_editor_html(self):
        html = (
            '<div data-schema-version="9"><h1>Paper</h1>'
            "<p>First&nbsp;line &amp; <b>more</b><br>second</p>"
            "<ul><li><p>item</p></li></ul></div>"
        )
        assert notes.html_to_markdown(html) == (
            "# Paper\n\nFirst line & **more**\nsecond\n\n- item\n"
        )

    def test_empty(self):
        assert notes.html_to_markdown("") == "\n"


class TestNoteTitle:
    @pytest.mark.parametrize(
        ("html", "title"),
        [
            ("<h1>Title</h1><p>Body</p>", "Title"),
            ("<p>First &amp; line</p><p>Second</p>", "First & line"),
            ('<div data-schema-version="9"><p>  </p><p>Real</p></div>', "Real"),
            ("plain text<br>more", "plain text"),
            ("", ""),
        ],
    )
    def test_first_line(self, html, title):
        assert notes.note_title(html) == title


class TestAddNote:
    def test_child_note_payload(self, zot):
        key = notes.add_note(zot, "Hello", title="T", parent="PAPER001")
        assert key == "NEWNOTE"
        (payload,) = zot.create_items.call_args[0][0]
        assert payload == {
            "itemType": "note",
            "note": "<h1>T</h1>\n<p>Hello</p>",
            "tags": [],
            "collections": [],
            "relations": {},
            "parentItem": "PAPER001",
        }

    def test_standalone_note_in_collection_with_tags(self, zot):
        notes.add_note(zot, "Idea", collection="COLL0001", tags=["x"])
        (payload,) = zot.create_items.call_args[0][0]
        assert "parentItem" not in payload
        assert payload["collections"] == ["COLL0001"]
        assert payload["tags"] == [{"tag": "x"}]
        assert payload["note"] == "<p>Idea</p>"

    def test_parent_and_collection_conflict(self, zot):
        with pytest.raises(ValueError, match="cannot be filed"):
            notes.add_note(zot, "x", parent="P", collection="C")
        zot.create_items.assert_not_called()

    def test_empty_note_rejected(self, zot):
        with pytest.raises(ValueError, match="empty"):
            notes.add_note(zot, "  \n ")
        zot.create_items.assert_not_called()

    def test_rejection_raises(self, zot):
        zot.create_items.return_value = {
            "success": {},
            "failed": {"0": {"code": 400, "message": "bad parent"}},
        }
        with pytest.raises(RuntimeError, match="bad parent"):
            notes.add_note(zot, "x", parent="NOPE")

    def test_title_is_escaped(self, zot):
        notes.add_note(zot, "x", title="A <b> & B")
        assert zot.create_items.call_args[0][0][0]["note"].startswith(
            "<h1>A &lt;b&gt; &amp; B</h1>"
        )


class TestListAndGet:
    def test_list_notes_skips_other_children(self, zot):
        zot.children.return_value = [
            note_item("N1", "<h1>One</h1>"),
            {"key": "A1", "data": {"itemType": "attachment", "title": "pdf"}},
            note_item("N2", "<p>Two</p>"),
        ]
        assert notes.list_notes(zot, "PAPER001") == [
            {"key": "N1", "title": "One", "dateModified": "2026-01-02T03:04:05Z"},
            {"key": "N2", "title": "Two", "dateModified": "2026-01-02T03:04:05Z"},
        ]
        zot.children.assert_called_once_with("PAPER001")

    def test_get_note(self, zot):
        zot.item.return_value = note_item(
            "N1", "<h1>One</h1><p>x</p>", parentItem="PAPER001"
        )
        assert notes.get_note(zot, "N1") == {
            "key": "N1",
            "version": 7,
            "parent": "PAPER001",
            "title": "One",
            "dateModified": "2026-01-02T03:04:05Z",
            "html": "<h1>One</h1><p>x</p>",
        }

    def test_get_note_rejects_other_item_types(self, zot):
        zot.item.return_value = {
            "key": "B1",
            "version": 1,
            "data": {"itemType": "book"},
        }
        with pytest.raises(ValueError, match="B1 is not a note"):
            notes.get_note(zot, "B1")


class TestAppend:
    def test_patch_carries_the_notes_version(self, zot):
        zot.item.return_value = note_item("N1", "<p>Old</p>", version=12)
        result = notes.append_note(zot, "N1", "## More\n\n- a")
        assert result == {"key": "N1", "title": "Old"}
        zot.update_item.assert_called_once_with(
            {
                "key": "N1",
                "version": 12,
                "note": "<p>Old</p>\n<h2>More</h2>\n<ul><li>a</li></ul>",
            }
        )

    def test_goes_inside_the_zotero_wrapper(self, zot):
        zot.item.return_value = note_item(
            "N1", '<div data-schema-version="9"><p>Old</p></div>'
        )
        notes.append_note(zot, "N1", "New")
        sent = zot.update_item.call_args[0][0]["note"]
        assert sent == '<div data-schema-version="9"><p>Old</p>\n<p>New</p>\n</div>'

    def test_stale_version_is_retried_once_on_fresh_content(self, zot):
        zot.item.side_effect = [
            note_item("N1", "<p>Old</p>", version=1),
            note_item("N1", "<p>Old</p><p>Edited</p>", version=2),
        ]
        zot.update_item.side_effect = [ze.PreConditionFailedError("412"), None]
        notes.append_note(zot, "N1", "New")
        first, second = (c[0][0] for c in zot.update_item.call_args_list)
        assert first["version"] == 1
        assert second["version"] == 2  # noqa: PLR2004
        assert "Edited" in second["note"]
        assert second["note"].endswith("<p>New</p>")

    def test_second_conflict_propagates(self, zot):
        zot.update_item.side_effect = ze.PreConditionFailedError("412")
        with pytest.raises(ze.PreConditionFailedError):
            notes.append_note(zot, "N1", "New")
        assert zot.update_item.call_count == 2  # noqa: PLR2004

    def test_empty_text_is_rejected_before_any_request(self, zot):
        with pytest.raises(ValueError, match="empty"):
            notes.append_note(zot, "N1", " ")
        zot.item.assert_not_called()

    def test_not_a_note(self, zot):
        zot.item.return_value = {"key": "B", "version": 1, "data": {"itemType": "book"}}
        with pytest.raises(ValueError, match="not a note"):
            notes.append_note(zot, "B", "x")
        zot.update_item.assert_not_called()


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli_zot(zot):
    """Patch both client factories of the CLI to return the one mock."""
    with (
        patch("pyzotero.cli.get_write_client", return_value=zot),
        patch("pyzotero.cli.get_zotero_client", return_value=zot),
    ):
        yield zot


class TestCli:
    def test_add_child_note_from_text(self, runner, cli_zot):
        result = runner.invoke(
            cli.main, ["note", "add", "PAPER001", "--text", "Hi", "--title", "T"]
        )
        assert result.exit_code == 0, result.output
        assert "Created note NEWNOTE" in result.output
        payload = cli_zot.create_items.call_args[0][0][0]
        assert payload["parentItem"] == "PAPER001"
        assert payload["note"] == "<h1>T</h1>\n<p>Hi</p>"

    def test_add_standalone_note_from_stdin_as_json(self, runner, cli_zot):
        result = runner.invoke(
            cli.main,
            [
                "note",
                "add",
                "none",
                "--file",
                "-",
                "--collection",
                "C1",
                "--tag",
                "t",
                "--json",
            ],
            input="# Idea\n\ntext",
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "created": "NEWNOTE",
            "parent": None,
            "collection": "C1",
            "tags": ["t"],
        }
        payload = cli_zot.create_items.call_args[0][0][0]
        assert "parentItem" not in payload
        assert payload["collections"] == ["C1"]

    def test_add_from_file(self, runner, cli_zot, tmp_path):
        path = tmp_path / "n.md"
        path.write_text("- a\n- b")
        result = runner.invoke(cli.main, ["note", "add", "P", "--file", str(path)])
        assert result.exit_code == 0, result.output
        assert cli_zot.create_items.call_args[0][0][0]["note"] == (
            "<ul><li>a</li><li>b</li></ul>"
        )

    @pytest.mark.parametrize(
        "args",
        [[], ["--text", "a", "--file", "-"]],
    )
    def test_needs_exactly_one_input(self, runner, cli_zot, args):
        result = runner.invoke(cli.main, ["note", "add", "P", *args], input="b")
        assert result.exit_code == 1
        assert "--text" in result.output
        cli_zot.create_items.assert_not_called()

    def test_parent_and_collection_conflict(self, runner, cli_zot):
        result = runner.invoke(
            cli.main, ["note", "add", "P", "--text", "x", "--collection", "C"]
        )
        assert result.exit_code == 1
        assert "cannot be filed" in result.output

    def test_rejection_exits_1(self, runner, cli_zot):
        cli_zot.create_items.return_value = {
            "success": {},
            "failed": {"0": {"message": "bad parent"}},
        }
        result = runner.invoke(cli.main, ["note", "add", "P", "--text", "x"])
        assert result.exit_code == 1
        assert "bad parent" in result.output

    def test_add_refuses_without_key(self, runner):
        result = runner.invoke(cli.main, ["note", "add", "P", "--text", "x"])
        assert result.exit_code == 1
        assert "pyzotero authorize" in result.output

    def test_list(self, runner, cli_zot):
        cli_zot.children.return_value = [note_item("N1", "<h1>One</h1>")]
        result = runner.invoke(cli.main, ["note", "list", "PAPER001"])
        assert result.exit_code == 0, result.output
        assert "N1  2026-01-02  One" in result.output
        assert "1 notes" in result.output

    def test_list_json(self, runner, cli_zot):
        cli_zot.children.return_value = [note_item("N1", "<h1>One</h1>")]
        result = runner.invoke(cli.main, ["note", "list", "PAPER001", "--json"])
        assert json.loads(result.output)[0]["title"] == "One"

    def test_show_html_and_markdown(self, runner, cli_zot):
        cli_zot.item.return_value = note_item("N1", "<h1>One</h1><p>a <b>b</b></p>")
        raw = runner.invoke(cli.main, ["note", "show", "N1"])
        assert raw.output.strip() == "<h1>One</h1><p>a <b>b</b></p>"
        md = runner.invoke(cli.main, ["note", "show", "N1", "--markdown"])
        assert md.output.strip() == "# One\n\na **b**"

    def test_show_json(self, runner, cli_zot):
        cli_zot.item.return_value = note_item("N1", "<p>x</p>")
        result = runner.invoke(cli.main, ["note", "show", "N1", "--markdown", "--json"])
        data = json.loads(result.output)
        assert data["html"] == "<p>x</p>"
        assert data["markdown"] == "x\n"

    def test_show_rejects_non_note(self, runner, cli_zot):
        cli_zot.item.return_value = {
            "key": "B",
            "version": 1,
            "data": {"itemType": "book"},
        }
        result = runner.invoke(cli.main, ["note", "show", "B"])
        assert result.exit_code == 1
        assert "B is not a note" in result.output

    def test_append_uses_the_notes_version(self, runner, cli_zot):
        cli_zot.item.return_value = note_item("N1", "<p>Old</p>", version=9)
        result = runner.invoke(cli.main, ["note", "append", "N1", "--text", "New"])
        assert result.exit_code == 0, result.output
        assert "Appended to note N1: Old" in result.output
        sent = cli_zot.update_item.call_args[0][0]
        assert sent["version"] == 9  # noqa: PLR2004
        assert sent["note"] == "<p>Old</p>\n<p>New</p>"

    def test_locale_reaches_write_client(self, runner, zot):
        with patch("pyzotero.cli.get_write_client", return_value=zot) as gwc:
            runner.invoke(
                cli.main, ["--locale", "de-DE", "note", "add", "P", "--text", "x"]
            )
        gwc.assert_called_once_with("de-DE")


class _FakeServer:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class TestMcp:
    def test_write_tools_registered_only_by_register_write_tools(self):
        for name in ("add_note", "append_note"):
            assert not hasattr(mcp_server, name)
        server = _FakeServer()
        names = mcp_server.register_write_tools(server)
        assert {"add_note", "append_note"} <= set(names)
        assert {"add_note", "append_note"} <= set(server.tools)

    def test_read_tools_are_module_level_and_not_write_tools(self):
        server = _FakeServer()
        mcp_server.register_write_tools(server, enable_deletes=True)
        assert "list_notes" not in server.tools
        assert "get_note" not in server.tools
        assert callable(mcp_server.list_notes)
        assert callable(mcp_server.get_note)

    def test_list_notes(self, zot):
        zot.children.return_value = [note_item("N1", "<h1>One</h1>")]
        with patch("pyzotero.mcp_server.get_zotero_client", return_value=zot):
            result = json.loads(mcp_server.list_notes("PAPER001"))
        assert result == [
            {"key": "N1", "title": "One", "dateModified": "2026-01-02T03:04:05Z"}
        ]

    def test_get_note_markdown_by_default(self, zot):
        zot.item.return_value = note_item("N1", "<h1>One</h1><p>x</p>")
        with patch("pyzotero.mcp_server.get_zotero_client", return_value=zot):
            result = json.loads(mcp_server.get_note("N1"))
            raw = json.loads(mcp_server.get_note("N1", markdown=False))
        assert result["content"] == "# One\n\nx\n"
        assert result["format"] == "markdown"
        assert "html" not in result
        assert raw["content"] == "<h1>One</h1><p>x</p>"
        assert raw["format"] == "html"

    def test_get_note_error_is_json(self, zot):
        zot.item.return_value = {"key": "B", "version": 1, "data": {"itemType": "book"}}
        with patch("pyzotero.mcp_server.get_zotero_client", return_value=zot):
            assert "B is not a note" in json.loads(mcp_server.get_note("B"))["error"]

    @pytest.fixture
    def write_tools(self, zot):
        server = _FakeServer()
        mcp_server.register_write_tools(server)
        with patch("pyzotero.mcp_server._write_client", return_value=zot):
            yield server.tools

    def test_add_note(self, write_tools, zot):
        result = json.loads(write_tools["add_note"]("PAPER001", "Hi", title="T"))
        assert result == {"created": "NEWNOTE", "parent": "PAPER001"}
        payload = zot.create_items.call_args[0][0][0]
        assert payload["parentItem"] == "PAPER001"
        assert payload["note"] == "<h1>T</h1>\n<p>Hi</p>"

    def test_add_standalone_note(self, write_tools, zot):
        result = json.loads(write_tools["add_note"]("", "Hi"))
        assert result["parent"] is None
        assert "parentItem" not in zot.create_items.call_args[0][0][0]

    def test_add_note_rejection_is_json(self, write_tools, zot):
        zot.create_items.return_value = {"success": {}, "failed": {"0": "no"}}
        assert "rejected" in json.loads(write_tools["add_note"]("P", "x"))["error"]

    def test_append_note(self, write_tools, zot):
        zot.item.return_value = note_item("N1", "<p>Old</p>", version=4)
        result = json.loads(write_tools["append_note"]("N1", "New"))
        assert result == {"updated": "N1", "title": "Old"}
        sent = zot.update_item.call_args[0][0]
        assert sent["version"] == 4  # noqa: PLR2004
        assert sent["note"] == "<p>Old</p>\n<p>New</p>"
