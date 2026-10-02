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


def metric_font(ascent: int, descent: int) -> list[bytes]:
    """Return a font dict and descriptor with the given ascent and descent (per 1000 em)."""
    widths = " ".join(["500"] * 95)
    return [
        (
            f"<< /Type /Font /Subtype /Type1 /BaseFont /TestSerif /FirstChar 32 "
            f"/LastChar 126 /Widths [{widths}] /FontDescriptor {{descriptor}} 0 R >>"
        ).encode(),
        (
            f"<< /Type /FontDescriptor /FontName /TestSerif /Flags 34 "
            f"/Ascent {ascent} /Descent {descent} /CapHeight {ascent} "
            f"/ItalicAngle 0 /StemV 80 /FontBBox [0 {descent} 1000 {ascent}] >>"
        ).encode(),
    ]


def make_pdf(
    pages: list[list[tuple[float, float, str]]],
    size: float = FONT_SIZE,
    font: list[bytes] | None = None,
    media: str = f"0 0 612 {PAGE_HEIGHT}",
    page_extra: str = "",
) -> bytes:
    """Build a small PDF: one line of text per (x, y, text).

    The font is Helvetica, or the objects of ``font`` (see ``metric_font``).
    ``media`` is the MediaBox, ``page_extra`` more entries of the page dict
    (a CropBox, a Rotate).
    """
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    kids = []
    font_ref = 3 + 2 * len(pages)
    for number, lines in enumerate(pages):
        page_ref = 3 + 2 * number
        kids.append(f"{page_ref} 0 R")
        ops = "".join(
            f"BT /F1 {size} Tf {x} {y} Td ({_escape(text)}) Tj ET\n"
            for x, y, text in lines
        ).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [{media}] {page_extra} "
            f"/Contents {page_ref + 1} 0 R "
            f"/Resources << /Font << /F1 {font_ref} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(ops) + ops + b"endstream")
    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    )
    if font is None:
        font = [b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    font[0] = font[0].replace(b"{descriptor}", str(font_ref + 1).encode())
    objects.extend(font)
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


def load_synthetic(tmp_path, lines, size, ascent, descent):
    path = tmp_path / "metrics.pdf"
    path.write_bytes(make_pdf([lines], size=size, font=metric_font(ascent, descent)))
    return annotations.load_pages(path)


class TestRectHeightMatchesZoteroApp:
    """Rects span descent to ascent of the font, as in poppler and the desktop app.

    The expected numbers come from a real paper: a highlight made in the app,
    and a five-line one checked by eye in it. Here they are rebuilt from the
    same baselines, sizes and font metrics.
    """

    def test_title_line(self, tmp_path):
        # NimbusRomNo9L-Medi, 14.35 pt: ascent 0.69, descent -0.209
        pages = load_synthetic(
            tmp_path, [(328.39, 691.077, "Genetic Algorithm")], 14.35, 690, -209
        )
        seg = one(pages, "Genetic Algorithm").segments[0]
        (rect,) = seg.rects
        assert rect[:2] == pytest.approx([328.39, 688.079], abs=0.05)
        assert rect[3] == pytest.approx(700.976, abs=0.05)
        assert annotations.sort_index(seg).endswith("|00091")

    def test_five_line_sentence(self, tmp_path):
        # NimbusRomNo9L-Regu, 9.9626 pt: ascent 0.678, descent -0.216
        lines = [
            (218.681, 430.670, "In this paper,"),
            (75.007, 418.715, "we introduce a Semantic Mirror Jailbreak (SMJ)"),
            (75.366, 406.759, "approach that bypasses LLMs by generating jail-"),
            (75.366, 394.804, "break prompts that are semantically similar to"),
            (75.366, 382.849, "the original question."),
        ]
        pages = load_synthetic(tmp_path, lines, 9.9626, 678, -216)
        seg = one(
            pages,
            "In this paper, we introduce a Semantic Mirror Jailbreak (SMJ) approach "
            "that bypasses LLMs by generating jailbreak prompts that are "
            "semantically similar to the original question.",
        ).segments[0]
        expected = [
            (218.681, 428.518, 437.425),
            (75.007, 416.563, 425.47),
            (75.366, 404.607, 413.514),
            (75.366, 392.652, 401.559),
            (75.366, 380.697, 389.604),
        ]
        assert len(seg.rects) == len(expected)
        for rect, (x1, y1, y2) in zip(seg.rects, expected, strict=True):
            assert rect[0] == pytest.approx(x1, abs=0.05)
            assert rect[1] == pytest.approx(y1, abs=0.05)
            assert rect[3] == pytest.approx(y2, abs=0.05)


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
            annotations.PageText(
                0, "1", 792, 90, seg.page.dropped, seg.page.kept, seg.page.box
            ),
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


def rect_of(pages, phrase) -> list[float]:
    """Return the rect of the first line of ``phrase``, as the text path makes it."""
    return one(pages, phrase).segments[0].rects[0]


def with_pdf(path: Path):
    return patch(
        "pyzotero.annotations._files.download",
        return_value={"attachment": "ATT00001", "path": str(path)},
    )


def created(zot) -> dict:
    (payloads,) = zot.create_items.call_args[0]
    (payload,) = payloads
    return payload


class TestValidateRects:
    def test_converts_to_rounded_floats(self):
        assert annotations.validate_rects([(1, 2, 3, 4), ["1.23456", 2, 3.5, 4]]) == [
            [1.0, 2.0, 3.0, 4.0],
            [1.235, 2.0, 3.5, 4.0],
        ]

    @pytest.mark.parametrize(
        ("rects", "message"),
        [
            ([], "at least one"),
            ([[1, 2, 3]], "four numbers"),
            ([["a", 2, 3, 4]], "four numbers"),
            ([[1, 2, 3, None]], "four numbers"),
            ([[1, 2, float("nan"), 4]], "not finite"),
            ([[5, 2, 5, 4]], "x0 < x1"),
            ([[6, 2, 5, 4]], "x0 < x1"),
            ([[1, 4, 3, 2]], "y0 < y1"),
        ],
    )
    def test_rejects(self, rects, message):
        with pytest.raises(ValueError, match=message):
            annotations.validate_rects(rects)


class TestHighlightRects:
    def test_fills_text_from_words_and_matches_the_text_highlight(self, zot, pages):
        seg = one(pages, "quick brown").segments[0]
        expected = annotations.build_payload("ATT00001", seg, "blue", "c")
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, seg.rects, color="blue", comment="c"
        )
        (entry,) = result["highlights"]
        assert (entry["status"], entry["key"], entry["text"]) == (
            "created",
            "NEW00000",
            "quick brown",
        )
        assert created(zot) == expected
        # compact JSON, as the text path writes it
        assert " " not in created(zot)["annotationPosition"]

    def test_text_is_in_reading_order_across_rects(self, zot, pages):
        first, last = rect_of(pages, "quick"), rect_of(pages, "lazy dog and")
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [last, first], dry_run=True
        )
        assert result["highlights"][0]["text"] == "quick lazy dog and"

    def test_explicit_text_is_kept(self, zot, pages):
        annotations.highlight_rects(
            zot, "ATT00001", 1, [rect_of(pages, "quick")], text="Quick!"
        )
        assert created(zot)["annotationText"] == "Quick!"

    def test_page_label_and_index(self, zot, pages):
        annotations.highlight_rects(zot, "ATT00001", 2, [rect_of(pages[1:], "story")])
        payload = created(zot)
        assert payload["annotationPageLabel"] == "2"
        assert json.loads(payload["annotationPosition"])["pageIndex"] == 1
        assert payload["annotationSortIndex"].startswith("00001|")

    def test_page_labels_from_the_pdf(self, zot, tmp_path):
        data = make_pdf([PAGE_ONE]).replace(
            b"/Type /Catalog",
            b"/Type /Catalog /PageLabels << /Nums [0 << /S /r /St 4 >>] >>",
        )
        path = tmp_path / "labels.pdf"
        path.write_bytes(data)
        with with_pdf(path):
            annotations.highlight_rects(zot, "ATT00001", 1, [[72, 698, 100, 710]])
        assert created(zot)["annotationPageLabel"] == "iv"

    def test_load_pages_ends_with_page_labels(self, tmp_path):
        """The pdfminer label iterator is infinite: a PDF with /PageLabels hung."""
        path = tmp_path / "labels.pdf"
        path.write_bytes(
            make_pdf([PAGE_ONE, PAGE_TWO]).replace(
                b"/Type /Catalog",
                b"/Type /Catalog /PageLabels << /Nums [0 << /S /r >>] >>",
            )
        )
        assert [p.label for p in annotations.load_pages(path)] == ["i", "ii"]

    def test_sort_index_offset_and_top(self, zot, pages):
        rect = rect_of(pages, "jumps over")
        annotations.highlight_rects(zot, "ATT00001", 1, [rect])
        page, offset, top = created(zot)["annotationSortIndex"].split("|")
        assert int(page) == 0
        assert int(offset) == pages[0].dropped.text.index("jumps over")
        assert int(top) == int(PAGE_HEIGHT - rect[3])

    def test_sort_index_top_is_the_highest_rect(self, zot, pages):
        low, high = rect_of(pages, "lazy dog"), rect_of(pages, "quick")
        annotations.highlight_rects(zot, "ATT00001", 1, [low, high])
        top = created(zot)["annotationSortIndex"].split("|")[2]
        assert int(top) == int(PAGE_HEIGHT - max(low[3], high[3]))

    def test_offset_is_zero_without_a_word_in_the_first_rect(self, zot, pages):
        empty = [400, 300, 450, 320]
        annotations.highlight_rects(
            zot, "ATT00001", 1, [empty, rect_of(pages, "quick")], text="x"
        )
        assert created(zot)["annotationSortIndex"].split("|")[1] == "000000"

    def test_no_word_in_rects_needs_text(self, zot):
        with pytest.raises(LookupError, match="pass the text"):
            annotations.highlight_rects(zot, "ATT00001", 1, [[400, 300, 450, 320]])
        zot.create_items.assert_not_called()

    @pytest.mark.parametrize(
        ("rect", "ok"),
        [
            ([0, 0, 612, 792], True),
            ([-2, -2, 614, 794], True),
            ([-2.5, 0, 100, 100], False),
            ([0, 0, 614.5, 100], False),
            ([0, 0, 100, 795], False),
            ([0, -3, 100, 100], False),
        ],
    )
    def test_page_box_margin(self, zot, rect, ok):
        if ok:
            annotations.highlight_rects(zot, "ATT00001", 1, [rect], text="x")
        else:
            with pytest.raises(ValueError, match="outside page 1"):
                annotations.highlight_rects(zot, "ATT00001", 1, [rect], text="x")

    def test_invalid_rect_fails_before_any_download(self, zot, _local_pdf):
        with pytest.raises(ValueError, match="x0 < x1"):
            annotations.highlight_rects(zot, "ATT00001", 1, [[5, 5, 5, 9]])
        _local_pdf.assert_not_called()

    def test_page_out_of_range(self, zot):
        with pytest.raises(IndexError, match="out of range"):
            annotations.highlight_rects(zot, "ATT00001", 3, [[1, 1, 5, 5]], text="x")

    def test_unknown_color(self, zot):
        with pytest.raises(ValueError, match="Unknown color"):
            annotations.highlight_rects(
                zot, "ATT00001", 1, [[1, 1, 5, 5]], color="teal"
            )

    def test_dry_run_writes_nothing(self, zot, pages):
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [rect_of(pages, "quick")], dry_run=True
        )
        assert result["highlights"][0]["status"] == "planned"
        assert result["dry_run"] is True
        zot.create_items.assert_not_called()

    def test_item_key_resolves_to_its_pdf(self, zot, pages):
        zot.item.return_value = {"key": "ITEM0001", "data": {"itemType": "book"}}
        zot.children.return_value = [attachment_item("ATT00001")]
        result = annotations.highlight_rects(
            zot, "ITEM0001", 1, [rect_of(pages, "quick")], dry_run=True
        )
        assert result["attachment"] == "ATT00001"

    def test_scanned_pdf_works_with_text(self, zot, tmp_path):
        path = tmp_path / "scan.pdf"
        path.write_bytes(make_pdf([[]]))
        with with_pdf(path):
            annotations.highlight_rects(
                zot, "ATT00001", 1, [[72, 600, 300, 620]], text="from OCR"
            )
            with pytest.raises(LookupError, match="pass the text"):
                annotations.highlight_rects(zot, "ATT00001", 1, [[72, 600, 300, 620]])
        assert created(zot)["annotationText"] == "from OCR"

    def test_rotated_page_is_refused(self, zot, tmp_path):
        path = tmp_path / "rot.pdf"
        path.write_bytes(make_pdf([PAGE_ONE], page_extra="/Rotate 90"))
        with with_pdf(path), pytest.raises(ValueError, match="rotated"):
            annotations.highlight_rects(zot, "ATT00001", 1, [[72, 600, 100, 620]])

    def test_rejected_write_raises(self, zot, pages):
        zot.create_items.side_effect = None
        zot.create_items.return_value = {"success": {}, "failed": {"0": "bad"}}
        with pytest.raises(RuntimeError, match="rejected"):
            annotations.highlight_rects(zot, "ATT00001", 1, [rect_of(pages, "quick")])


class TestHighlightRectsIdempotency:
    @pytest.fixture
    def existing(self, zot, pages):
        rect = rect_of(pages, "quick brown")
        annotations.highlight_rects(zot, "ATT00001", 1, [rect], color="green")
        zot.children.return_value = [annotation_child(created(zot))]
        zot.create_items.reset_mock()
        return rect

    def test_same_rect_and_text_is_unchanged_whatever_the_color(self, zot, existing):
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [existing], color="red"
        )
        assert result["highlights"][0]["status"] == "unchanged"
        assert result["highlights"][0]["key"] == "OLD00001"
        zot.create_items.assert_not_called()

    def test_overlapping_rect_is_unchanged(self, zot, existing):
        nudged = [existing[0] + 1, existing[1] + 1, existing[2] - 1, existing[3] - 1]
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [nudged], text="quick brown"
        )
        assert result["highlights"][0]["status"] == "unchanged"

    def test_other_text_is_created(self, zot, existing):
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [existing], text="something else"
        )
        assert result["highlights"][0]["status"] == "created"

    def test_disjoint_rect_is_created(self, zot, existing, pages):
        result = annotations.highlight_rects(
            zot, "ATT00001", 1, [rect_of(pages, "lazy dog")], text="quick brown"
        )
        assert result["highlights"][0]["status"] == "created"

    def test_same_rect_on_another_page_is_created(self, zot, existing):
        result = annotations.highlight_rects(
            zot, "ATT00001", 2, [existing], text="quick brown"
        )
        assert result["highlights"][0]["status"] == "created"

    def test_text_highlight_is_the_same_highlight(self, zot, existing):
        """A phrase highlight and a rect highlight of the same words dedupe."""
        result = annotations.highlight(zot, "ATT00001", "quick brown")
        assert result["highlights"][0]["status"] == "unchanged"


class TestCropBoxAndMediaBox:
    CROP = "/CropBox [0 100 612 700]"

    def pdf(self, tmp_path, **kwargs):
        path = tmp_path / "boxed.pdf"
        path.write_bytes(
            make_pdf(
                [[(72, 650, "Alpha beta gamma"), (72, 400, "Delta epsilon")]], **kwargs
            )
        )
        return path

    def test_load_pages_box(self, tmp_path):
        (plain,) = annotations.load_pages(self.pdf(tmp_path))
        assert plain.box == (0, 0, 612, 792)
        (cropped,) = annotations.load_pages(self.pdf(tmp_path, page_extra=self.CROP))
        assert cropped.box == (0, 100, 612, 700)

    def test_crop_outside_media_is_cut(self, tmp_path):
        path = self.pdf(tmp_path, page_extra="/CropBox [-10 -20 700 900]")
        assert annotations.load_pages(path)[0].box == (0, 0, 612, 792)

    def test_sort_top_is_measured_from_the_crop_top(self, zot, tmp_path):
        path = self.pdf(tmp_path, page_extra=self.CROP)
        (page,) = annotations.load_pages(path)
        rect = [72, 648, 140, 660]
        with with_pdf(path):
            result = annotations.highlight_rects(zot, "ATT00001", 1, [rect])
        assert result["highlights"][0]["text"] == "Alpha beta"
        # absolute coordinates are kept, the top is below the crop top (700)
        assert created(zot)["annotationSortIndex"] == "00000|000000|00040"
        assert page.box[3] - rect[3] == 40  # noqa: PLR2004

    def test_rect_outside_the_crop_box_is_refused(self, zot, tmp_path):
        path = self.pdf(tmp_path, page_extra=self.CROP)
        with with_pdf(path):
            # inside the MediaBox, 5 pt above the CropBox top: more than the margin
            with pytest.raises(ValueError, match="outside page 1"):
                annotations.highlight_rects(
                    zot, "ATT00001", 1, [[72, 690, 140, 705]], text="x"
                )
            # inside the crop box top margin
            annotations.highlight_rects(
                zot, "ATT00001", 1, [[72, 690, 140, 701.5]], text="x"
            )

    def test_text_highlight_and_rect_highlight_sort_alike_on_a_cropped_page(
        self, zot, tmp_path
    ):
        path = self.pdf(tmp_path, page_extra=self.CROP)
        pages = annotations.load_pages(path)
        seg = one(pages, "Alpha beta").segments[0]
        with with_pdf(path):
            annotations.highlight_rects(zot, "ATT00001", 1, seg.rects)
        assert created(zot)["annotationSortIndex"] == annotations.sort_index(seg)

    def test_media_box_with_an_origin(self, zot, tmp_path):
        path = self.pdf(tmp_path, media="0 50 612 842")
        (page,) = annotations.load_pages(path)
        assert page.box == (0, 50, 612, 842)
        with with_pdf(path):
            result = annotations.highlight_rects(
                zot, "ATT00001", 1, [[72, 648, 140, 660]]
            )
        assert result["highlights"][0]["text"] == "Alpha beta"
        # distance from the top of the visible page: 842 - 660
        assert created(zot)["annotationSortIndex"].endswith("|00182")


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

    def test_rect_json(self, runner, zot, pages):
        x0, y0, x1, y1 = (str(v) for v in rect_of(pages, "quick brown"))
        result = runner.invoke(
            cli.main,
            [
                *["highlight", "ATT00001", "--rect", "1", x0, y0, x1, y1],
                *["--color", "blue", "--comment", "hi", "--json"],
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["highlights"][0]["status"] == "created"
        assert out["highlights"][0]["text"] == "quick brown"
        assert out["highlights"][0]["color"] == "#2ea8e5"
        assert created(zot)["annotationComment"] == "hi"

    def test_repeated_rect_makes_one_multi_rect_highlight(self, runner, zot, pages):
        first, second = rect_of(pages, "quick"), rect_of(pages, "lazy dog")
        args = ["highlight", "ATT00001"]
        for r in (first, second):
            args += ["--rect", "1", *(str(v) for v in r)]
        result = runner.invoke(cli.main, args)
        assert result.exit_code == 0, result.output
        position = json.loads(created(zot)["annotationPosition"])
        assert position["rects"] == [first, second]
        assert "quick lazy dog" in result.output

    def test_rect_dry_run_uses_read_client(self, zot, pages):
        with (
            patch("pyzotero.cli.get_write_client") as write,
            patch("pyzotero.cli.get_zotero_client", return_value=zot),
        ):
            result = CliRunner().invoke(
                cli.main,
                [
                    "highlight",
                    "ATT00001",
                    "--dry-run",
                    "--rect",
                    "1",
                    "72",
                    "698",
                    "90",
                    "712",
                ],
            )
        assert result.exit_code == 0, result.output
        write.assert_not_called()
        assert "planned" in result.output

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--text", "x", "--rect", "1", "1", "1", "5", "5"], "--text or --rect"),
            ([], "--text or --rect"),
            (
                ["--rect", "1", "1", "1", "5", "5", "--rect", "2", "1", "1", "5", "5"],
                "same page",
            ),
            (["--rect", "1", "1", "1", "5", "5", "--page", "1"], "go with --text"),
            (["--rect", "1", "1", "1", "5", "5", "--all"], "go with --text"),
            (["--rect", "0", "1", "1", "5", "5"], "Invalid value"),
            (["--rect", "1", "1", "1", "5"], "requires 5 arguments"),
        ],
    )
    def test_rect_usage_errors(self, runner, args, message):
        result = runner.invoke(cli.main, ["highlight", "ATT00001", *args])
        assert result.exit_code != 0, result.output
        assert message in result.output

    def test_bad_rect_exits_with_the_reason(self, runner):
        result = runner.invoke(
            cli.main, ["highlight", "ATT00001", "--rect", "1", "50", "50", "10", "60"]
        )
        assert result.exit_code == 1
        assert "x0 < x1" in result.output

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

    def test_highlight_rects_registered_only_with_writes(self):
        assert not hasattr(mcp_server, "highlight_rects")
        server = _FakeServer()
        names = mcp_server.register_write_tools(server)
        assert "highlight_rects" in names
        assert "highlight_rects" in server.tools

    def test_highlight_rects(self, zot, pages):
        server = _FakeServer()
        mcp_server.register_write_tools(server)
        rect = rect_of(pages, "quick brown")
        with patch("pyzotero.mcp_server._write_client", return_value=zot):
            result = json.loads(
                server.tools["highlight_rects"]("ATT00001", 1, [rect], color="green")
            )
            bad = json.loads(
                server.tools["highlight_rects"]("ATT00001", 1, [[5, 5, 1, 9]])
            )
        assert result["highlights"][0]["status"] == "created"
        assert result["highlights"][0]["text"] == "quick brown"
        assert "x0 < x1" in bad["error"]
