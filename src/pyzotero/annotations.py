"""Highlight annotations: find a phrase in a PDF and create the Zotero highlight.

The work splits in three, so that the first two run without a network:

1. :func:`load_pages` reads a PDF into words with positions (``pdfplumber``).
2. :func:`find_matches` finds a phrase in them. :func:`build_payload` turns a
   match into the item that the Zotero API takes.
3. :func:`highlight` downloads the PDF, runs the two steps and writes.

Coordinates in the payload are PDF user space in points, with the origin at
the bottom left. pdfplumber measures from the top, so ``y = height - top``.

Matching is on normalized text: NFKC (ligatures), straight quotes and
hyphens, collapsed white space. A word that a line break splits after a
hyphen matches both ways: ``jail-`` + ``break`` is ``jailbreak`` and also
``jail-break``. A match covers whole words: a phrase that starts or ends
inside a word highlights that word.

Not supported: rotated pages, and phrases that span more than two pages.
"""

from __future__ import annotations

import bisect
import difflib
import json
import re
import tempfile
import unicodedata
from itertools import pairwise
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyzotero import _files
from pyzotero.webdav import STORED_LINK_MODES

if TYPE_CHECKING:
    from pyzotero._client import Zotero

COLORS = {
    "yellow": "#ffd400",
    "red": "#ff6666",
    "green": "#5fb236",
    "blue": "#2ea8e5",
    "purple": "#a28ae5",
    "magenta": "#e56eee",
    "orange": "#f19837",
    "gray": "#aaaaaa",
}

# A phrase that spans two pages ends within this many words of the end of the
# first page and starts within this many of the start of the next. The slack
# is for running headers, footers and page numbers between the two halves.
PAGE_EDGE_WORDS = 25

# A gap between letters wider than this fraction of the font size is a space.
# Some PDFs (LaTeX with tight spacing) have no space characters at all.
SPACE_RATIO = 0.15

_QUOTES = str.maketrans("\u2018\u2019\u201a\u201b\u2032", "'" * 5)
_DQUOTES = str.maketrans("\u201c\u201d\u201e\u201f\u2033", '"' * 5)
_DASHES = str.maketrans(
    dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-")
)
_LINE_BREAK_HYPHEN = "-\u2010\u2011\u00ad"
_HYPHEN_AT_BREAK = re.compile("[-\u2010\u2011\u00ad][ \\t]*\\n\\s*")
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
_MIN_SIMILARITY = 0.5


def normalize(text: str) -> str:
    """Return ``text`` as the matcher sees it: NFKC, plain quotes and dashes."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = text.translate(_QUOTES).translate(_DQUOTES).translate(_DASHES)
    return " ".join(text.split())


def parse_color(color: str) -> str:
    """Return the hex code for a Zotero color name or a ``#rrggbb`` code."""
    if _HEX_COLOR.fullmatch(color):
        return color.lower()
    if color.lower() in COLORS:
        return COLORS[color.lower()]
    msg = f"Unknown color {color!r}: use #rrggbb or one of {', '.join(COLORS)}"
    raise ValueError(msg)


@dataclass(frozen=True)
class Box:
    """A word's box in pdfplumber coordinates (points, origin top left)."""

    x0: float
    x1: float
    top: float
    bottom: float


@dataclass(frozen=True)
class Token:
    """A word of normalized text, with one box per line it occupies."""

    text: str
    boxes: tuple[Box, ...]


@dataclass
class Stream:
    """The tokens of a page, and the text they make when joined by a space."""

    tokens: list[Token]
    text: str = field(init=False)
    starts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.text = " ".join(t.text for t in self.tokens)
        self.starts = []
        pos = 0
        for token in self.tokens:
            self.starts.append(pos)
            pos += len(token.text) + 1

    def token_at(self, char: int) -> int:
        """Return the index of the token that holds character ``char``."""
        return bisect.bisect_right(self.starts, char) - 1


@dataclass
class PageText:
    """One page: two views of its text (hyphen dropped or kept at line breaks)."""

    index: int
    label: str
    height: float
    rotation: int
    dropped: Stream
    kept: Stream


@dataclass
class Segment:
    """The part of a match on one page."""

    page: PageText
    first: int
    last: int
    offset: int
    stream: Stream

    @property
    def tokens(self) -> list[Token]:
        return self.stream.tokens[self.first : self.last + 1]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def lines(self) -> list[list[Box]]:
        """The boxes of the match, grouped by line of text."""
        lines: list[list[Box]] = []
        for token in self.tokens:
            for box in token.boxes:
                if lines and _same_line(lines[-1], box):
                    lines[-1].append(box)
                else:
                    lines.append([box])
        return lines

    @property
    def rects(self) -> list[list[float]]:
        """One ``[x1, y1, x2, y2]`` per line, in PDF user space."""
        if self.page.rotation:
            msg = f"Page {self.page.index + 1} is rotated, which is not supported"
            raise ValueError(msg)
        height = self.page.height
        return [
            [
                round(min(b.x0 for b in line), 3),
                round(height - max(b.bottom for b in line), 3),
                round(max(b.x1 for b in line), 3),
                round(height - min(b.top for b in line), 3),
            ]
            for line in self.lines
        ]

    @property
    def top(self) -> float:
        """Distance in points from the page top to the first line."""
        return min(b.top for b in self.lines[0])


@dataclass
class Match:
    """A phrase found in a PDF: one segment, or two if it spans a page break."""

    segments: list[Segment]

    @property
    def context(self) -> str:
        seg = self.segments[0]
        text = seg.stream.text
        start = max(0, seg.offset - 30)
        return "..." + text[start : seg.offset + 70] + "..."

    @property
    def page(self) -> int:
        return self.segments[0].page.index + 1


def _same_line(line: list[Box], box: Box) -> bool:
    """Whether ``box`` continues ``line``: level with it, and further right."""
    centre = (box.top + box.bottom) / 2
    level = min(b.top for b in line) <= centre <= max(b.bottom for b in line)
    return level and box.x0 >= line[-1].x0


def _hyphenated(raw: str, box: Box, nxt: Box) -> bool:
    """Whether a line break splits a word after the hyphen that ends ``raw``."""
    if not raw or raw[-1] not in _LINE_BREAK_HYPHEN or not raw[-2:-1].isalpha():
        return False
    centre = (nxt.top + nxt.bottom) / 2
    return not box.top <= centre <= box.bottom or nxt.x0 < box.x0


def _streams(words: list[dict[str, Any]]) -> tuple[Stream, Stream]:
    """Build the dropped-hyphen and kept-hyphen views from pdfplumber words."""
    raw: list[tuple[str, Box]] = [
        (
            str(w["text"]),
            Box(float(w["x0"]), float(w["x1"]), float(w["top"]), float(w["bottom"])),
        )
        for w in words
    ]
    dropped: list[Token] = []
    kept: list[Token] = []
    i = 0
    while i < len(raw):
        text, box = raw[i]
        if i + 1 < len(raw) and _hyphenated(text, box, raw[i + 1][1]):
            next_text, next_box = raw[i + 1]
            boxes = (box, next_box)
            # A lowercase continuation is a split word; "Self-" + "Reference"
            # is not, and only the kept view may join it.
            if next_text[:1].islower():
                dropped.append(Token(normalize(text[:-1] + next_text), boxes))
            else:
                dropped.extend(
                    t for t in (_token(text, box), _token(next_text, next_box)) if t
                )
            kept.append(Token(normalize(text + next_text), boxes))
            i += 2
            continue
        for stream in (dropped, kept):
            if token := _token(text, box):
                stream.append(token)
        i += 1
    return Stream(dropped), Stream(kept)


def _token(text: str, box: Box) -> Token | None:
    normalized = normalize(text)
    return Token(normalized, (box,)) if normalized else None


def _pdfplumber() -> Any:
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError as exc:
        msg = "Highlighting needs pdfplumber: install 'pyzotero[pdf]'"
        raise RuntimeError(msg) from exc
    return pdfplumber


def load_pages(path: Path, page: int = 0) -> list[PageText]:
    """Read the text and word positions of a PDF.

    ``page`` is 1-based: with it, only that page is read. Raises LookupError
    if no page has text (a scanned PDF) and IndexError for a page out of range.
    """
    pdfplumber = _pdfplumber()
    from pdfminer.pdfdocument import PDFNoPageLabels  # noqa: PLC0415

    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        if page and not 1 <= page <= total:
            msg = f"Page {page} is out of range: the PDF has {total} pages"
            raise IndexError(msg)
        try:
            labels = list(pdf.doc.get_page_labels())
        except PDFNoPageLabels:
            # Without page labels Zotero labels a page with its number.
            labels = []
        for index, plumber_page in enumerate(pdf.pages):
            if page and index != page - 1:
                continue
            words = plumber_page.extract_words(
                use_text_flow=True, x_tolerance_ratio=SPACE_RATIO
            )
            dropped, kept = _streams(words)
            label = labels[index] if index < len(labels) else str(index + 1)
            pages.append(
                PageText(
                    index,
                    label,
                    float(plumber_page.height),
                    int(plumber_page.rotation),
                    dropped,
                    kept,
                )
            )
    if not any(p.dropped.tokens for p in pages):
        msg = "The PDF has no text layer (a scanned PDF?): run OCR on it first"
        raise LookupError(msg)
    return pages


def _pattern(phrase: str, hyphen: str, ignore_case: bool) -> re.Pattern[str]:
    """Compile the phrase for one view. A hyphen before a newline is a break."""
    joined = _HYPHEN_AT_BREAK.sub("" if hyphen == "drop" else "-", phrase)
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(re.escape(normalize(joined)), flags)


def _search(
    page: PageText, phrase: str, ignore_case: bool
) -> list[tuple[int, int, int, Stream]]:
    """Return (first token, last token, offset, stream) per match on a page."""
    found: dict[tuple[int, int], tuple[int, int, int, Stream]] = {}
    for hyphen, stream in (("drop", page.dropped), ("keep", page.kept)):
        pattern = _pattern(phrase, hyphen, ignore_case)
        for m in pattern.finditer(stream.text):
            first = stream.token_at(m.start())
            last = stream.token_at(m.end() - 1)
            found.setdefault((first, last), (first, last, m.start(), stream))
    return sorted(found.values(), key=lambda f: f[2])


def _find_spanning(
    pages: list[PageText], phrase: str, ignore_case: bool
) -> list[Match]:
    """Find a phrase that starts at the end of a page and ends on the next."""
    words = normalize(phrase).split(" ")
    matches: list[Match] = []
    for here, there in pairwise(pages):
        if there.index != here.index + 1:
            continue
        for split in range(1, len(words)):
            head = _pattern(" ".join(words[:split]), "drop", ignore_case)
            tail = _pattern(" ".join(words[split:]), "drop", ignore_case)
            n_here = len(here.dropped.tokens)
            heads = [
                m
                for m in head.finditer(here.dropped.text)
                if here.dropped.token_at(m.end() - 1) >= n_here - PAGE_EDGE_WORDS
            ]
            tails = [
                m
                for m in tail.finditer(there.dropped.text)
                if there.dropped.token_at(m.start()) < PAGE_EDGE_WORDS
            ]
            if heads and tails:
                h, t = heads[-1], tails[0]
                matches.append(
                    Match(
                        [
                            _segment(here, here.dropped, h.start(), h.end()),
                            _segment(there, there.dropped, t.start(), t.end()),
                        ]
                    )
                )
    return matches


def _segment(page: PageText, stream: Stream, start: int, end: int) -> Segment:
    return Segment(
        page, stream.token_at(start), stream.token_at(end - 1), start, stream
    )


def find_matches(
    pages: list[PageText], phrase: str, ignore_case: bool = False
) -> list[Match]:
    """Find ``phrase`` on ``pages``, in reading order.

    A phrase that no single page holds is looked for across each pair of
    neighbouring pages.
    """
    if not normalize(phrase):
        msg = "The text to highlight is empty"
        raise ValueError(msg)
    matches = [
        Match([Segment(page, first, last, offset, stream)])
        for page in pages
        for first, last, offset, stream in _search(page, phrase, ignore_case)
    ]
    return matches or _find_spanning(pages, phrase, ignore_case)


def closest(pages: list[PageText], phrase: str, count: int = 3) -> list[str]:
    """Return the passages that look most like ``phrase``, as "p.N: text"."""
    target = normalize(phrase)
    size = len(target.split(" "))
    step = max(1, size // 2)
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    scored: list[tuple[float, str]] = []
    for page in pages:
        tokens = [t.text for t in page.dropped.tokens]
        for i in range(0, max(1, len(tokens) - size + 1), step):
            window = " ".join(tokens[i : i + size])
            matcher.set_seq1(window)
            if (
                matcher.real_quick_ratio() >= _MIN_SIMILARITY
                and matcher.quick_ratio() >= _MIN_SIMILARITY
            ):
                scored.append((matcher.ratio(), f"p.{page.index + 1}: {window}"))
    best = sorted(scored, reverse=True)[:count]
    return [text for score, text in best if score >= _MIN_SIMILARITY]


def sort_index(segment: Segment) -> str:
    """Return Zotero's ``PPPPP|OOOOOO|TTTTT`` sort key for a highlight."""
    return (
        f"{segment.page.index:05d}|{min(segment.offset, 999999):06d}"
        f"|{min(int(segment.top), 99999):05d}"
    )


def build_payload(
    attachment: str, segment: Segment, color: str, comment: str = ""
) -> dict[str, Any]:
    """Return the annotation item that the Zotero API creates for a segment."""
    return {
        "itemType": "annotation",
        "parentItem": attachment,
        "annotationType": "highlight",
        "annotationText": segment.text,
        "annotationComment": comment,
        "annotationColor": parse_color(color),
        "annotationPageLabel": segment.page.label,
        "annotationSortIndex": sort_index(segment),
        "annotationPosition": json.dumps(
            {"pageIndex": segment.page.index, "rects": segment.rects}
        ),
        "tags": [],
        "relations": {},
    }


def select(
    matches: list[Match], phrase: str, occurrence: int = 0, all_matches: bool = False
) -> list[Match]:
    """Pick the matches to highlight. Raises LookupError if the pick is unclear."""
    if not matches:
        msg = f"No match for {phrase!r}"
        raise LookupError(msg)
    if all_matches:
        return matches
    if occurrence:
        if not 1 <= occurrence <= len(matches):
            msg = f"--occurrence {occurrence} is out of range: {len(matches)} matches"
            raise LookupError(msg)
        return [matches[occurrence - 1]]
    if len(matches) > 1:
        listing = "\n".join(
            f"  {n}. p.{m.page}: {m.context}" for n, m in enumerate(matches, 1)
        )
        msg = (
            f"{len(matches)} matches for {phrase!r}; choose one with an occurrence "
            f"number, or highlight all of them:\n{listing}"
        )
        raise LookupError(msg)
    return matches


def pdf_attachments(zot: Zotero, key: str) -> list[dict[str, Any]]:
    """Return the stored PDF attachments of ``key``: itself, or its children."""
    item = zot.item(key)
    data = item["data"]
    if data.get("itemType") == "attachment":
        if (
            data.get("contentType") != "application/pdf"
            or data.get("linkMode") not in STORED_LINK_MODES
        ):
            msg = f"Attachment {key} is not a stored PDF"
            raise ValueError(msg)
        return [item]
    return [
        c
        for c in zot.everything(zot.children(key))
        if c["data"].get("itemType") == "attachment"
        and c["data"].get("contentType") == "application/pdf"
        and c["data"].get("linkMode") in STORED_LINK_MODES
    ]


def _annotations_of(zot: Zotero, attachment: str) -> list[dict[str, Any]]:
    return [
        c["data"]
        for c in zot.everything(zot.children(attachment))
        if c["data"].get("itemType") == "annotation"
    ]


def list_annotations(zot: Zotero, key: str) -> list[dict[str, Any]]:
    """Return the annotations on the PDF(s) of ``key``, in reading order."""
    return [
        {
            "key": data["key"],
            "attachment": att["key"],
            "type": data.get("annotationType"),
            "color": data.get("annotationColor"),
            "page": data.get("annotationPageLabel"),
            "text": data.get("annotationText", ""),
            "comment": data.get("annotationComment", ""),
        }
        for att in pdf_attachments(zot, key)
        for data in sorted(
            _annotations_of(zot, att["key"]),
            key=lambda d: d.get("annotationSortIndex", ""),
        )
    ]


def _overlaps(a: list[list[float]], b: list[list[float]]) -> bool:
    """Whether any rect of ``a`` intersects any rect of ``b``."""
    return any(
        r[0] < s[2] and s[0] < r[2] and r[1] < s[3] and s[1] < r[3]
        for r in a
        for s in b
    )


def find_duplicate(
    payload: dict[str, Any], existing: list[dict[str, Any]]
) -> str | None:
    """Return the key of a highlight that already covers ``payload``, or None.

    Same page, same normalized text, and rects that overlap. The overlap
    keeps two occurrences of one phrase on a page apart.
    """
    position = json.loads(payload["annotationPosition"])
    text = normalize(payload["annotationText"])
    for data in existing:
        if data.get("annotationType") != "highlight":
            continue
        other = json.loads(data.get("annotationPosition") or "{}")
        if (
            other.get("pageIndex") == position["pageIndex"]
            and normalize(data.get("annotationText", "")) == text
            and _overlaps(position["rects"], other.get("rects", []))
        ):
            return data["key"]
    return None


def _entry(status: str, payload: dict[str, Any], key: str | None) -> dict[str, Any]:
    position = json.loads(payload["annotationPosition"])
    return {
        "status": status,
        "key": key,
        "page": position["pageIndex"] + 1,
        "label": payload["annotationPageLabel"],
        "text": payload["annotationText"],
        "color": payload["annotationColor"],
        "sortIndex": payload["annotationSortIndex"],
        "rects": position["rects"],
    }


def highlight(
    zot: Zotero,
    key: str,
    text: str,
    *,
    color: str = "yellow",
    comment: str = "",
    page: int = 0,
    occurrence: int = 0,
    all_matches: bool = False,
    ignore_case: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Highlight ``text`` in the PDF of ``key`` (an attachment, or an item).

    Returns ``{"attachment", "dry_run", "found", "highlights": [...]}`` with
    one entry per page of each highlight: ``status`` is ``created``,
    ``unchanged`` (the same highlight exists) or ``planned`` (dry run).
    Raises LookupError for no match or an unclear pick, RuntimeError if
    Zotero rejects a write.
    """
    parse_color(color)
    attachments = pdf_attachments(zot, key)
    if not attachments:
        msg = f"Item {key} has no stored PDF attachment"
        raise LookupError(msg)
    attachment = attachments[0]["key"]
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = _files.download(zot, attachment, Path(tmp))
        pages = load_pages(Path(downloaded["path"]), page)
    matches = find_matches(pages, text, ignore_case)
    if not matches:
        near = closest(pages, text)
        hint = (
            "\nClosest passages:\n" + "\n".join(f"  {n}" for n in near) if near else ""
        )
        msg = f"No match for {text!r}{hint}"
        raise LookupError(msg)
    chosen = select(matches, text, occurrence, all_matches)
    payloads = [
        build_payload(attachment, segment, color, comment)
        for match in chosen
        for segment in match.segments
    ]
    existing = _annotations_of(zot, attachment)
    entries: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any]]] = []
    for payload in payloads:
        if duplicate := find_duplicate(payload, existing):
            entries.append(_entry("unchanged", payload, duplicate))
        else:
            pending.append((len(entries), payload))
            entries.append(_entry("planned", payload, None))
    if pending and not dry_run:
        response = zot.create_items([p for _, p in pending])
        success = response.get("success") or {}
        if len(success) != len(pending):
            msg = f"Zotero rejected the highlight: {response.get('failed')}"
            raise RuntimeError(msg)
        for n, (slot, _) in enumerate(pending):
            entries[slot]["status"] = "created"
            entries[slot]["key"] = success[str(n)]
    return {
        "attachment": attachment,
        "dry_run": dry_run,
        "found": len(matches),
        "highlights": entries,
    }
