"""Tests for highlight annotations: text matching, payloads, CLI and MCP."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pyzotero import annotations, cli, mcp_server

pytest.importorskip("pdfplumber")

PAGE_HEIGHT = 792
TWO = 2
THREE = 3
BASELINE = 700
FONT_SIZE = 12

PAGE_ONE = [
    (72, 700, "The quick brown fox jumps over"),
    (72, 686, "the lazy dog and finds a jail-"),
    (72, 672, "break prompt in the middle of"),
    (72, 658, "a long sentence ending here now."),
    (72, 600, "Self-"),
    (72, 588, "Reference is not split."),
    (72, 100, "the end of page one"),
    (300, 40, "Footer 1"),
]
PAGE_TWO = [
    (300, 760, "Running Header"),
    (72, 700, "and the story continues here."),
    (72, 686, "A repeated phrase. A repeated phrase."),
]


def make_pdf(pages: list[list[tuple[float, float, str]]]) -> bytes:
    """Build a small PDF: one Helvetica line of text per (x, y, text)."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    font_ref = 3 + 2 * len(pages)
    for number, lines in enumerate(pages):
        page_ref = 3 + 2 * number
        kids.append(f"{page_ref} 0 R")
        ops = "".join(
            f"BT /F1 {FONT_SIZE} Tf {x} {y} Td ({_escape(text)}) Tj ET\n"
            for x, y, text in lines
        ).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 {PAGE_HEIGHT}] "
            f"/Contents {page_ref + 1} 0 R "
            f"/Resources << /Font << /F1 {font_ref} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(ops) + ops + b"endstream")
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return out


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / "paper.pdf"
    path.write_bytes(make_pdf([PAGE_ONE, PAGE_TWO]))
    return path


@pytest.fixture(scope="module")
def pages(pdf_path) -> list[annotations.PageText]:
    return annotations.load_pages(pdf_path)


def one(pages, phrase, **kwargs) -> annotations.Match:
    matches = annotations.find_matches(pages, phrase, **kwargs)
    assert len(matches) == 1, matches
    return matches[0]


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  a \n  b\tc ", "a b c"),
            ("\ufb01nd the o\ufb03ce", "find the office"),
            ("\u201cquoted\u201d and \u2018this\u2019", "\"quoted\" and 'this'"),
            ("a\u2013b \u2014 c\u2212d", "a-b - c-d"),
            ("soft\u00adhyphen", "softhyphen"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert annotations.normalize(raw) == expected

    def test_parse_color(self):
        assert annotations.parse_color("Blue") == "#2ea8e5"
        assert annotations.parse_color("#ABCDEF") == "#abcdef"
        with pytest.raises(ValueError, match="Unknown color"):
            annotations.parse_color("teal")


class TestMatching:
    def test_single_line(self, pages):
        seg = one(pages, "quick brown").segments[0]
        assert seg.text == "quick brown"
        assert len(seg.rects) == 1

    def test_pdfplumber_words_are_not_glued(self, pages):
        assert pages[0].dropped.tokens[0].text == "The"

    def test_rect_y_flip(self, pages):
        """y1 = height - bottom, y2 = height - top; baseline sits between."""
        seg = one(pages, "quick brown").segments[0]
        ((x1, y1, x2, y2),) = seg.rects
        box = seg.lines[0][0]
        assert y1 == pytest.approx(PAGE_HEIGHT - seg.lines[0][-1].bottom, abs=1e-3)
        assert y2 == pytest.approx(PAGE_HEIGHT - box.top, abs=1e-3)
        assert y1 < BASELINE < y2  # the text baseline is at y=700, origin bottom left
        assert x1 == pytest.approx(box.x0, abs=1e-3)
        assert x1 > 72 + 10  # starts at "quick", not at the line start
        assert x2 > x1

    def test_multi_line_mid_line_start_and_end(self, pages):
        seg = one(
            pages, "jumps over the lazy dog and finds a jailbreak prompt in"
        ).segments[0]
        rects = seg.rects
        assert len(rects) == THREE
        first, second, third = rects
        line_one = pages[0].dropped.tokens
        jumps = next(t for t in line_one if t.text == "jumps").boxes[0]
        over = next(t for t in line_one if t.text == "over").boxes[0]
        assert first[0] == pytest.approx(jumps.x0, abs=1e-3)
        assert first[2] == pytest.approx(over.x1, abs=1e-3)
        assert second[0] == pytest.approx(72, abs=1)  # whole second line
        assert third[0] == pytest.approx(72, abs=1)
        assert third[2] < second[2]  # ends mid-line
        assert first[1] > second[1] > third[1]  # lines run downwards

    def test_hyphenation_join(self, pages):
        seg = one(pages, "jailbreak prompt").segments[0]
        assert seg.text == "jailbreak prompt"
        assert len(seg.rects) == TWO  # "jail-" on one line, "break prompt" on the next

    def test_hyphen_kept_variant_finds_the_same_words_once(self, pages):
        seg = one(pages, "jail-break prompt").segments[0]
        assert seg.text == "jail-break prompt"
        assert len(seg.rects) == TWO

    def test_hyphen_and_newline_in_the_phrase(self, pages):
        assert one(pages, "a jail-\nbreak prompt").segments[0].text == (
            "a jailbreak prompt"
        )

    def test_capitalised_continuation_is_not_joined(self, pages):
        assert one(pages, "Self- Reference is").segments[0].text == "Self- Reference is"
        assert not annotations.find_matches(pages, "SelfReference")

    def test_case_sensitive_by_default(self, pages):
        assert not annotations.find_matches(pages, "QUICK BROWN")
        assert one(pages, "QUICK BROWN", ignore_case=True)

    def test_no_match_lists_closest(self, pages):
        assert not annotations.find_matches(pages, "quick red fox jumps over")
        near = annotations.closest(pages, "quick red fox jumps over")
        assert near
        assert near[0].startswith("p.1: ")
        assert "fox jumps" in near[0]

    def test_empty_phrase(self, pages):
        with pytest.raises(ValueError, match="empty"):
            annotations.find_matches(pages, "   ")

    def test_multiple_matches_need_a_pick(self, pages):
        matches = annotations.find_matches(pages, "A repeated phrase.")
        assert len(matches) == TWO
        with pytest.raises(LookupError, match=r"2 matches.*\n.*1\. p\.2"):
            annotations.select(matches, "A repeated phrase.")
        assert annotations.select(matches, "x", occurrence=2) == [matches[1]]
        assert annotations.select(matches, "x", all_matches=True) == matches
        with pytest.raises(LookupError, match="out of range"):
            annotations.select(matches, "x", occurrence=3)

    def test_spans_a_page_break_with_header_and_footer_between(self, pages):
        match = one(pages, "the end of page one and the story continues")
        first, second = match.segments
        assert (first.page.index, second.page.index) == (0, 1)
        assert first.text == "the end of page one"
        assert second.text == "and the story continues"

    def test_page_restriction(self, pdf_path):
        only_two = annotations.load_pages(pdf_path, page=2)
        assert [p.index for p in only_two] == [1]
        assert not annotations.find_matches(only_two, "quick brown")
        with pytest.raises(IndexError, match="out of range"):
            annotations.load_pages(pdf_path, page=3)

    def test_scanned_pdf_has_no_text(self, tmp_path):
        path = tmp_path / "scan.pdf"
        path.write_bytes(make_pdf([[]]))
        with pytest.raises(LookupError, match="no text layer"):
            annotations.load_pages(path)


class TestPayload:
    def test_fields(self, pages):
        seg = one(pages, "quick brown").segments[0]
        payload = annotations.build_payload("ATT00001", seg, "blue", "note")
        position = json.loads(payload.pop("annotationPosition"))
        sort_key = payload.pop("annotationSortIndex")
        assert payload == {
            "itemType": "annotation",
            "parentItem": "ATT00001",
            "annotationType": "highlight",
            "annotationText": "quick brown",
            "annotationComment": "note",
            "annotationColor": "#2ea8e5",
            "annotationPageLabel": "1",
            "tags": [],
            "relations": {},
        }
        assert position == {"pageIndex": 0, "rects": seg.rects}
        assert sort_key

    def test_sort_index_format(self, pages):
        seg = one(pages, "quick brown").segments[0]
        key = annotations.sort_index(seg)
        assert re.fullmatch(r"\d{5}\|\d{6}\|\d{5}", key)
        page, offset, top = key.split("|")
        assert int(page) == 0
        assert int(offset) == pages[0].dropped.text.index("quick brown")
        assert int(top) == int(PAGE_HEIGHT - seg.rects[0][3])

    def test_sort_index_on_second_page(self, pages):
        seg = one(pages, "story continues").segments[0]
        assert annotations.sort_index(seg).startswith("00001|")

    def test_rotated_page_is_refused(self, pages):
        seg = one(pages, "quick brown").segments[0]
        rotated = annotations.Segment(
            annotations.PageText(0, "1", 792, 90, seg.page.dropped, seg.page.kept),
            seg.first,
            seg.last,
            seg.offset,
            seg.stream,
        )
        with pytest.raises(ValueError, match="rotated"):
            rotated.rects


def attachment_item(key="ATT00001", **data):
    data = {
        "key": key,
        "itemType": "attachment",
        "contentType": "application/pdf",
        "linkMode": "imported_file",
        **data,
    }
    return {"key": key, "data": data}


def annotation_child(payload, key="OLD00001"):
    return {"key": key, "data": {**payload, "key": key}}


@pytest.fixture
def zot():
    mock = MagicMock()
    mock.item.return_value = attachment_item()
    mock.children.return_value = []
    mock.everything.side_effect = lambda query: query
    mock.create_items.side_effect = lambda items: {
        "success": {str(n): f"NEW0000{n}" for n in range(len(items))},
        "failed": {},
    }
    return mock


@pytest.fixture(autouse=True)
def _local_pdf(pdf_path):
    """`_files.download` hands back the generated PDF instead of fetching one."""
    with patch(
        "pyzotero.annotations._files.download",
        return_value={"attachment": "ATT00001", "path": str(pdf_path)},
    ) as download:
        yield download


class TestHighlight:
    def test_creates_a_highlight(self, zot):
        result = annotations.highlight(zot, "ATT00001", "quick brown", color="red")
        (entry,) = result["highlights"]
        assert (entry["status"], entry["key"], entry["page"]) == (
            "created",
            "NEW00000",
            1,
        )
        assert entry["color"] == "#ff6666"
        (payloads,) = zot.create_items.call_args[0]
        assert payloads[0]["parentItem"] == "ATT00001"
        assert json.loads(payloads[0]["annotationPosition"])["rects"] == entry["rects"]

    def test_dry_run_writes_nothing(self, zot):
        result = annotations.highlight(zot, "ATT00001", "quick brown", dry_run=True)
        assert result["highlights"][0]["status"] == "planned"
        zot.create_items.assert_not_called()

    def test_second_run_is_unchanged(self, zot, pages):
        seg = one(pages, "quick brown").segments[0]
        payload = annotations.build_payload("ATT00001", seg, "yellow")
        zot.children.return_value = [annotation_child(payload)]
        result = annotations.highlight(zot, "ATT00001", "quick brown")
        assert result["highlights"][0]["status"] == "unchanged"
        assert result["highlights"][0]["key"] == "OLD00001"
        zot.create_items.assert_not_called()

    def test_other_text_on_the_same_page_is_not_a_duplicate(self, zot, pages):
        seg = one(pages, "quick brown").segments[0]
        zot.children.return_value = [
            annotation_child(annotations.build_payload("ATT00001", seg, "yellow"))
        ]
        result = annotations.highlight(zot, "ATT00001", "lazy dog")
        assert result["highlights"][0]["status"] == "created"

    def test_same_text_elsewhere_on_the_page_is_not_a_duplicate(self, zot, pages):
        first = annotations.find_matches(pages, "A repeated phrase.")[0].segments[0]
        zot.children.return_value = [
            annotation_child(annotations.build_payload("ATT00001", first, "yellow"))
        ]
        result = annotations.highlight(
            zot, "ATT00001", "A repeated phrase.", all_matches=True
        )
        assert [h["status"] for h in result["highlights"]] == ["unchanged", "created"]

    def test_page_spanning_phrase_makes_one_annotation_per_page(self, zot):
        result = annotations.highlight(
            zot, "ATT00001", "the end of page one and the story continues", comment="c"
        )
        assert [(h["status"], h["page"]) for h in result["highlights"]] == [
            ("created", 1),
            ("created", 2),
        ]
        payloads = zot.create_items.call_args[0][0]
        assert {p["annotationComment"] for p in payloads} == {"c"}
        assert len(payloads) == TWO

    def test_no_match_error_names_closest(self, zot):
        with pytest.raises(LookupError, match=r"(?s)No match.*Closest passages.*p\.1"):
            annotations.highlight(zot, "ATT00001", "quick red fox jumps over")

    def test_item_key_resolves_to_its_pdf(self, zot):
        zot.item.return_value = {"key": "ITEM0001", "data": {"itemType": "book"}}
        zot.children.return_value = [
            attachment_item("HTML0001", contentType="text/html"),
            attachment_item("ATT00001"),
        ]
        result = annotations.highlight(zot, "ITEM0001", "quick brown", dry_run=True)
        assert result["attachment"] == "ATT00001"

    def test_non_pdf_attachment(self, zot):
        zot.item.return_value = attachment_item(contentType="text/html")
        with pytest.raises(ValueError, match="not a stored PDF"):
            annotations.highlight(zot, "ATT00001", "x")

    def test_rejected_write_raises(self, zot):
        zot.create_items.side_effect = None
        zot.create_items.return_value = {"success": {}, "failed": {"0": "bad"}}
        with pytest.raises(RuntimeError, match="rejected"):
            annotations.highlight(zot, "ATT00001", "quick brown")

    def test_missing_pdfplumber_names_the_extra(self, zot):
        with (
            patch.dict("sys.modules", {"pdfplumber": None}),
            pytest.raises(RuntimeError, match=r"pyzotero\[pdf\]"),
        ):
            annotations.highlight(zot, "ATT00001", "quick brown")


class TestListAnnotations:
    def test_rows_in_reading_order(self, zot, pages):
        late = annotations.build_payload(
            "ATT00001", one(pages, "story continues").segments[0], "green", "later"
        )
        early = annotations.build_payload(
            "ATT00001", one(pages, "quick brown").segments[0], "yellow"
        )
        zot.children.return_value = [
            annotation_child(late, "LATE0001"),
            annotation_child(early, "EARLY001"),
            {"key": "N", "data": {"itemType": "note"}},
        ]
        rows = annotations.list_annotations(zot, "ATT00001")
        assert [r["key"] for r in rows] == ["EARLY001", "LATE0001"]
        assert rows[1] == {
            "key": "LATE0001",
            "attachment": "ATT00001",
            "type": "highlight",
            "color": "#5fb236",
            "page": "2",
            "text": "story continues",
            "comment": "later",
        }


class TestCli:
    @pytest.fixture
    def runner(self, zot):
        with (
            patch("pyzotero.cli.get_write_client", return_value=zot),
            patch("pyzotero.cli.get_zotero_client", return_value=zot),
        ):
            yield CliRunner()

    def test_highlight_json(self, runner, zot):
        result = runner.invoke(
            cli.main,
            [
                "highlight",
                "ATT00001",
                "--text",
                "quick brown",
                "--color",
                "blue",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["highlights"][0]["status"] == "created"
        assert out["highlights"][0]["color"] == "#2ea8e5"

    def test_highlight_text_output(self, runner):
        result = runner.invoke(
            cli.main, ["highlight", "ATT00001", "--text", "quick brown"]
        )
        assert result.exit_code == 0, result.output
        assert "created" in result.output
        assert "NEW00000" in result.output

    def test_dry_run_uses_read_client(self, zot):
        with (
            patch("pyzotero.cli.get_write_client") as write,
            patch("pyzotero.cli.get_zotero_client", return_value=zot),
        ):
            result = CliRunner().invoke(
                cli.main,
                ["highlight", "ATT00001", "--text", "quick brown", "--dry-run"],
            )
        assert result.exit_code == 0, result.output
        write.assert_not_called()
        assert "planned" in result.output

    def test_occurrence_and_all_are_exclusive(self, runner):
        result = runner.invoke(
            cli.main,
            ["highlight", "A", "--text", "x", "--occurrence", "1", "--all"],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_ambiguous_exits_with_the_listing(self, runner):
        result = runner.invoke(
            cli.main, ["highlight", "ATT00001", "--text", "A repeated phrase."]
        )
        assert result.exit_code == 1
        assert "2 matches" in result.output

    def test_annotations_listing(self, runner, zot, pages):
        payload = annotations.build_payload(
            "ATT00001", one(pages, "quick brown").segments[0], "yellow", "hi"
        )
        zot.children.return_value = [annotation_child(payload)]
        result = runner.invoke(cli.main, ["annotations", "ATT00001"])
        assert result.exit_code == 0, result.output
        assert "OLD00001" in result.output
        assert "quick brown" in result.output
        as_json = runner.invoke(cli.main, ["annotations", "ATT00001", "--json"])
        assert json.loads(as_json.output)[0]["comment"] == "hi"


class _FakeServer:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(func):
            self.tools[func.__name__] = func
            return func

        return register


class TestMcp:
    def test_highlight_tool_only_registered_by_register_write_tools(self):
        assert not hasattr(mcp_server, "highlight_text")
        server = _FakeServer()
        names = mcp_server.register_write_tools(server)
        assert "highlight_text" in names
        assert "highlight_text" in server.tools
        assert "list_annotations" not in server.tools

    def test_list_annotations_is_a_read_tool(self, zot, pages):
        payload = annotations.build_payload(
            "ATT00001", one(pages, "quick brown").segments[0], "yellow"
        )
        zot.children.return_value = [annotation_child(payload)]
        with patch("pyzotero.mcp_server.get_zotero_client", return_value=zot):
            rows = json.loads(mcp_server.list_annotations("ATT00001"))
        assert rows[0]["text"] == "quick brown"

    def test_highlight_text(self, zot):
        server = _FakeServer()
        mcp_server.register_write_tools(server)
        with patch("pyzotero.mcp_server._write_client", return_value=zot):
            result = json.loads(
                server.tools["highlight_text"]("ATT00001", "quick brown", color="green")
            )
            second = json.loads(
                server.tools["highlight_text"](
                    "ATT00001", "A repeated phrase.", occurrence=2
                )
            )
        assert result["highlights"][0]["status"] == "created"
        assert second["highlights"][0]["page"] == TWO

    def test_highlight_text_error_is_json(self, zot):
        server = _FakeServer()
        mcp_server.register_write_tools(server)
        with patch("pyzotero.mcp_server._write_client", return_value=zot):
            result = json.loads(
                server.tools["highlight_text"]("ATT00001", "nonexistent")
            )
        assert "No match" in result["error"]
