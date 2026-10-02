"""Highlight annotations: find a phrase in a PDF and create the Zotero highlight.

The work splits in three, so that the first two run without a network:

1. :func:`load_pages` reads a PDF into words with positions (``pdfplumber``).
2. :func:`find_matches` finds a phrase in them. :func:`build_payload` turns a
   match into the item that the Zotero API takes.
3. :func:`highlight` downloads the PDF, runs the two steps and writes.

:func:`highlight_rects` highlights given rectangles instead of a phrase, for
callers that know where the text is (for example from OCR). It needs no text
layer if the caller passes the text.

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
import math
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from itertools import islice, pairwise
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
    # Per token, the characters before it that are not white space. This is
    # the text offset in Zotero's sort index (the app's own highlight of a
    # title that follows "Semantic Mirror Jailbreak: " has offset 24, not 27).
    letters: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.text = " ".join(t.text for t in self.tokens)
        self.starts = []
        self.letters = []
        pos = count = 0
        for token in self.tokens:
            self.starts.append(pos)
            self.letters.append(count)
            pos += len(token.text) + 1
            count += len(token.text)

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
    # The visible page in PDF user space: CropBox within MediaBox,
    # as (x0, y0, x1, y1) with the origin at the bottom left.
    box: tuple[float, float, float, float]


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
        """Distance in points from the top of the visible page to the first line."""
        page = self.page
        return min(b.top for b in self.lines[0]) - (page.height - page.box[3])


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


def _read_words(pdf: Any, plumber_page: Any) -> list[dict[str, Any]]:
    """Return the words of a page, with the vertical extent the Zotero app uses.

    pdfplumber boxes a character as one font size tall, from the descent up.
    Poppler (``pdftotext -bbox``) and the desktop reader box it from the
    descent to the ascent that the font descriptor gives, which is shorter
    (0.89 of the size for a typical serif). Only the top changes.
    """
    words = plumber_page.extract_words(
        use_text_flow=True, x_tolerance_ratio=SPACE_RATIO, return_chars=True
    )
    # pdfminer caches the fonts it loaded for this page, keyed by object id.
    extents = {
        f.fontname: (f.get_ascent(), f.get_descent())
        for f in pdf.rsrcmgr._cached_fonts.values()
    }
    for word in words:
        tops = []
        for char in word["chars"]:
            ascent, descent = extents.get(char["fontname"], (0, 0))
            if ascent <= 0 or not char["upright"]:
                # No ascent to use, or text that runs sideways: keep the
                # pdfplumber box, which is taller but never too short.
                tops.append(char["top"])
                continue
            baseline = char["bottom"] + descent * char["size"]
            tops.append(baseline - ascent * char["size"])
        word["top"] = min(tops)
    return words


def _user_box(plumber_page: Any) -> tuple[float, float, float, float]:
    """Return the visible box of a page in PDF user space (origin bottom left).

    pdfplumber keeps the CropBox out of its page size and flips the boxes it
    reports, so ``y = height - y_plumber`` gives user space again. The
    CropBox is cut to the MediaBox, as readers do.
    """
    height = float(plumber_page.height)

    def user(box: tuple[float, float, float, float]) -> tuple[float, ...]:
        return (box[0], height - box[3], box[2], height - box[1])

    media, crop = user(plumber_page.mediabox), user(plumber_page.cropbox)
    x0, y0 = max(media[0], crop[0]), max(media[1], crop[1])
    x1, y1 = min(media[2], crop[2]), min(media[3], crop[3])
    return (x0, y0, x1, y1)


def load_pages(path: Path, page: int = 0, require_text: bool = True) -> list[PageText]:
    """Read the text and word positions of a PDF.

    ``page`` is 1-based: with it, only that page is read. Raises LookupError
    if no page has text (a scanned PDF), unless ``require_text`` is false,
    and IndexError for a page out of range.
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
            # pdfminer's label stream never ends: the last range runs on forever.
            labels = list(islice(pdf.doc.get_page_labels(), total))
        except PDFNoPageLabels:
            # Without page labels Zotero labels a page with its number.
            labels = []
        for index, plumber_page in enumerate(pdf.pages):
            if page and index != page - 1:
                continue
            dropped, kept = _streams(_read_words(pdf, plumber_page))
            label = labels[index] if index < len(labels) else str(index + 1)
            pages.append(
                PageText(
                    index,
                    label,
                    float(plumber_page.height),
                    int(plumber_page.rotation),
                    dropped,
                    kept,
                    _user_box(plumber_page),
                )
            )
    if require_text and not any(p.dropped.tokens for p in pages):
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


def _sort_key(page_index: int, offset: int, top: float) -> str:
    return f"{page_index:05d}|{min(offset, 999999):06d}|{min(int(top), 99999):05d}"


def sort_index(segment: Segment) -> str:
    """Return Zotero's ``PPPPP|OOOOOO|TTTTT`` sort key for a highlight."""
    return _sort_key(
        segment.page.index, segment.stream.letters[segment.first], segment.top
    )


def _payload(
    attachment: str,
    page: PageText,
    text: str,
    rects: list[list[float]],
    sort_key: str,
    color: str,
    comment: str,
) -> dict[str, Any]:
    return {
        "itemType": "annotation",
        "parentItem": attachment,
        "annotationType": "highlight",
        "annotationText": text,
        "annotationComment": comment,
        "annotationColor": parse_color(color),
        "annotationPageLabel": page.label,
        "annotationSortIndex": sort_key,
        "annotationPosition": json.dumps(
            {"pageIndex": page.index, "rects": rects}, separators=(",", ":")
        ),
        "tags": [],
        "relations": {},
    }


def build_payload(
    attachment: str, segment: Segment, color: str, comment: str = ""
) -> dict[str, Any]:
    """Return the annotation item that the Zotero API creates for a segment."""
    return _payload(
        attachment,
        segment.page,
        segment.text,
        segment.rects,
        sort_index(segment),
        color,
        comment,
    )


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


def _first_attachment(zot: Zotero, key: str) -> str:
    attachments = pdf_attachments(zot, key)
    if not attachments:
        msg = f"Item {key} has no stored PDF attachment"
        raise LookupError(msg)
    return attachments[0]["key"]


def _store(
    zot: Zotero,
    attachment: str,
    payloads: list[dict[str, Any]],
    found: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Create the payloads that do not exist yet, and report each one."""
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
        "found": found,
        "highlights": entries,
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
    attachment = _first_attachment(zot, key)
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
    return _store(zot, attachment, payloads, len(matches), dry_run)


# Rectangles may stick out of the page box by this much (points): hand-made
# or OCR-derived boxes are often a little loose at the edge.
RECT_MARGIN = 2.0


def validate_rects(rects: list[list[float]]) -> list[list[float]]:
    """Return ``rects`` as ``[x0, y0, x1, y1]`` floats, rounded like the text path.

    Raises ValueError for an empty list, a rect that is not four finite numbers,
    or one with ``x0 >= x1`` or ``y0 >= y1``.
    """
    if not rects:
        msg = "Give at least one rect"
        raise ValueError(msg)
    checked = []
    for rect in rects:
        try:
            x0, y0, x1, y1 = (float(v) for v in rect)
        except (TypeError, ValueError) as exc:
            msg = f"Rect {rect!r} is not four numbers [x0, y0, x1, y1]"
            raise ValueError(msg) from exc
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            msg = f"Rect {rect!r} has a value that is not finite"
            raise ValueError(msg)
        if not (x0 < x1 and y0 < y1):
            msg = f"Rect {rect!r} needs x0 < x1 and y0 < y1 (origin bottom left)"
            raise ValueError(msg)
        checked.append([round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)])
    return checked


def _check_in_page(page: PageText, rects: list[list[float]]) -> None:
    bx0, by0, bx1, by1 = page.box
    for rect in rects:
        if (
            rect[0] < bx0 - RECT_MARGIN
            or rect[1] < by0 - RECT_MARGIN
            or rect[2] > bx1 + RECT_MARGIN
            or rect[3] > by1 + RECT_MARGIN
        ):
            msg = (
                f"Rect {rect} is outside page {page.index + 1}, whose box is "
                f"[{bx0:g}, {by0:g}, {bx1:g}, {by1:g}] (+-{RECT_MARGIN:g} pt)"
            )
            raise ValueError(msg)


def _words_in(page: PageText, rect: list[float]) -> list[int]:
    """Return the indexes of the tokens that have a box centred inside ``rect``."""
    found = []
    for n, token in enumerate(page.dropped.tokens):
        for box in token.boxes:
            cx = (box.x0 + box.x1) / 2
            cy = page.height - (box.top + box.bottom) / 2
            if rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
                found.append(n)
                break
    return found


def highlight_rects(
    zot: Zotero,
    key: str,
    page: int,
    rects: list[list[float]],
    text: str = "",
    color: str = "yellow",
    comment: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Highlight rectangles on one page of the PDF of ``key``.

    ``page`` is 1-based. ``rects`` are ``[x0, y0, x1, y1]`` in PDF user space
    (points, origin bottom left), the convention of Zotero's
    ``annotationPosition``. Without ``text``, it is the words of the PDF
    whose centre lies in a rect, in reading order; with ``text`` the PDF
    needs no text layer. The sort index takes its text offset from the first
    word in the first rect (0 if there is none).

    The result has the shape of :func:`highlight`'s, with one entry. A
    highlight on the same page with overlapping rects and the same text is
    ``unchanged``, whatever its color. Raises ValueError for bad rects or a
    rotated page, IndexError for a page out of range, LookupError if no
    attachment is found or no word is in the rects and ``text`` is empty.
    """
    parse_color(color)
    checked = validate_rects(rects)
    attachment = _first_attachment(zot, key)
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = _files.download(zot, attachment, Path(tmp))
        (target,) = load_pages(Path(downloaded["path"]), page, require_text=False)
    if target.rotation:
        msg = f"Page {page} is rotated, which is not supported"
        raise ValueError(msg)
    _check_in_page(target, checked)
    inside = [_words_in(target, rect) for rect in checked]
    if not normalize(text):
        words = sorted({n for found in inside for n in found})
        text = " ".join(target.dropped.tokens[n].text for n in words)
        if not text:
            msg = (
                f"No words of page {page} lie in the rects: "
                "pass the text, or fix the rects"
            )
            raise LookupError(msg)
    offset = target.dropped.letters[inside[0][0]] if inside[0] else 0
    sort_key = _sort_key(
        target.index, offset, target.box[3] - max(rect[3] for rect in checked)
    )
    payload = _payload(attachment, target, text, checked, sort_key, color, comment)
    return _store(zot, attachment, [payload], 1, dry_run)
