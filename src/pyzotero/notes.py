"""Zotero notes: create, list, read and append, for the CLI and the MCP server.

A Zotero note is an item of type ``note`` whose ``note`` field holds HTML.
Zotero shows the first line of that HTML as the note's title. The notes here
are written in Markdown and converted with the standard library only.

The Markdown converter handles headings, paragraphs, bold, italic, inline
code, fenced code blocks, bullet and numbered lists (nested by indentation),
block quotes, horizontal rules and links. It does not handle tables, images,
footnotes, raw HTML (which is escaped), bare-URL autolinks or LaTeX math.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from . import errors as ze

if TYPE_CHECKING:
    from ._client import Zotero

# Link schemes that Zotero would run or embed. A link with one of these keeps
# its text and loses its href.
_UNSAFE_SCHEMES = ("javascript:", "data:", "vbscript:")

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^(`{3,}|~{3,})[ \t]*([\w+.-]*)[ \t]*$")
_BULLET = re.compile(r"^(\s*)[-*+][ \t]+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d+[.)][ \t]+(.*)$")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_QUOTE = re.compile(r"^ {0,3}>[ \t]?(.*)$")

_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
_ESCAPED = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|<])")
_LINK = re.compile(r"\[([^\]]+)\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+\"[^\"]*\")?\)")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?![\s_])(.+?)(?<![\s_])_(?![\w_])")


def _inline(text: str) -> str:
    """Convert the inline Markdown in ``text`` to HTML, escaping the rest."""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    text = _CODE_SPAN.sub(
        lambda m: keep(f"<code>{html.escape(m.group(2).strip(), quote=False)}</code>"),
        text,
    )
    text = _ESCAPED.sub(lambda m: keep(html.escape(m.group(1), quote=False)), text)
    text = html.escape(text, quote=False)

    def link(m: re.Match[str]) -> str:
        label, href = m.group(1), html.unescape(m.group(2))
        if href.lower().startswith(_UNSAFE_SCHEMES):
            return label
        return keep(f'<a href="{html.escape(href)}">') + label + keep("</a>")

    text = _LINK.sub(link, text)
    text = _BOLD.sub(r"<strong>\2</strong>", text)
    text = _ITALIC_STAR.sub(r"<em>\1</em>", text)
    text = _ITALIC_UNDER.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def _list_item(line: str) -> tuple[int, str, str] | None:
    """Return (indent, tag, text) if ``line`` starts a list item."""
    for pattern, tag in ((_BULLET, "ul"), (_NUMBERED, "ol")):
        if m := pattern.match(line):
            return len(m.group(1).expandtabs(4)), tag, m.group(2)
    return None


def _collect_list(lines: list[str], start: int) -> tuple[list[list[Any]], int]:
    """Read the list items that begin at ``lines[start]``.

    Return ([indent, tag, text] per item, index of the first line after the
    list). An indented line that is not itself an item continues the text of
    the item before it. Blank lines between items do not end the list.
    """
    items: list[list[Any]] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if item := _list_item(line):
            items.append(list(item))
        elif not line.strip():
            ahead = i
            while ahead < len(lines) and not lines[ahead].strip():
                ahead += 1
            if ahead >= len(lines) or not _list_item(lines[ahead]):
                break
            i = ahead
            continue
        elif line[:1] in " \t" and items and not _FENCE.match(line.strip()):
            items[-1][2] += " " + line.strip()
        else:
            break
        i += 1
    return items, i


def _nest(items: list[list[Any]], pos: int) -> tuple[str, int]:
    """Render the list that begins at ``items[pos]``. Return (HTML, next pos).

    An item indented deeper than the ones before it opens a sublist, inside
    the item above. An item of another list type at the same indent ends
    this list.
    """
    base, tag = items[pos][0], items[pos][1]
    out = [f"<{tag}>"]
    while pos < len(items):
        indent, item_tag, text = items[pos]
        if indent < base:
            break
        if indent > base:
            sub, pos = _nest(items, pos)
            out[-1] = out[-1].removesuffix("</li>") + sub + "</li>"
        elif item_tag != tag:
            break
        else:
            out.append(f"<li>{_inline(text)}</li>")
            pos += 1
    out.append(f"</{tag}>")
    return "".join(out), pos


def _code_block(lines: list[str], start: int, marker: str) -> tuple[str, int]:
    """Render the fenced code block whose opening fence is ``lines[start]``.

    An unclosed fence runs to the end of the input.
    """
    i = start + 1
    code: list[str] = []
    while i < len(lines) and not lines[i].strip().startswith(marker):
        code.append(lines[i])
        i += 1
    escaped = html.escape("\n".join(code), quote=False)
    return f"<pre><code>{escaped}</code></pre>", i + 1


def _quote_block(lines: list[str], start: int) -> tuple[str, int]:
    """Render the run of ``>`` lines that begins at ``lines[start]``."""
    i = start
    quoted: list[str] = []
    while i < len(lines) and _QUOTE.match(lines[i]):
        quoted.append(_QUOTE.sub(r"\1", lines[i]))
        i += 1
    return f"<blockquote>{markdown_to_html(chr(10).join(quoted))}</blockquote>", i


def markdown_to_html(markdown: str) -> str:
    """Convert Markdown to the HTML that a Zotero note holds.

    Block elements are separated by newlines. Raw HTML in the input is
    escaped, not passed through.
    """
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append("<p>" + _inline(" ".join(s.strip() for s in para)) + "</p>")
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if fence := _FENCE.match(line.strip()):
            flush()
            rendered, i = _code_block(lines, i, fence.group(1))
            blocks.append(rendered)
        elif not line.strip():
            flush()
            i += 1
        elif m := _HEADING.match(line):
            flush()
            level = len(m.group(1))
            blocks.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif _RULE.match(line):
            flush()
            blocks.append("<hr>")
            i += 1
        elif _QUOTE.match(line):
            flush()
            rendered, i = _quote_block(lines, i)
            blocks.append(rendered)
        elif _list_item(line):
            flush()
            items, i = _collect_list(lines, i)
            pos = 0
            while pos < len(items):
                rendered, pos = _nest(items, pos)
                blocks.append(rendered)
        else:
            para.append(line)
            i += 1
    flush()
    return "\n".join(blocks)


class _MarkdownWriter(HTMLParser):
    """Turn note HTML into readable Markdown. Lossless output is not a goal."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._newlines = 0  # newlines at the end of ``out``
        self._lists: list[list[Any]] = []  # [tag, item counter] of the open lists
        self._hrefs: list[str | None] = []
        self._quotes: list[int] = []  # index in ``out`` where each quote began
        self._in_pre = False

    def _emit(self, text: str) -> None:
        if not text:
            return
        self.out.append(text)
        trailing = len(text) - len(text.rstrip("\n"))
        self._newlines = (
            self._newlines + trailing if trailing == len(text) else trailing
        )

    def _break(self, count: int) -> None:
        if self.out and self._newlines < count:
            self._emit("\n" * (count - self._newlines))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: PLR0912
        if re.fullmatch(r"h[1-6]", tag):
            self._break(2)
            self._emit("#" * int(tag[1]) + " ")
        elif tag == "p":
            self._break(1 if self._lists else 2)
        elif tag == "br":
            self._emit("\n")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code" and not self._in_pre:
            self._emit("`")
        elif tag == "pre":
            self._break(2)
            self._emit("```\n")
            self._in_pre = True
        elif tag in ("ul", "ol"):
            if not self._lists:
                self._break(2)
            self._lists.append([tag, 0])
        elif tag == "li" and self._lists:
            self._break(1)
            top = self._lists[-1]
            top[1] += 1
            marker = "-" if top[0] == "ul" else f"{top[1]}."
            self._emit("  " * (len(self._lists) - 1) + marker + " ")
            # Count the marker as a line start, so that a <p> inside the item
            # (as Zotero's editor writes) continues the marker's line.
            self._newlines = 1
        elif tag == "a":
            self._hrefs.append(dict(attrs).get("href"))
            self._emit("[")
        elif tag == "blockquote":
            self._break(2)
            self._quotes.append(len(self.out))
        elif tag == "hr":
            self._break(2)
            self._emit("---\n\n")

    def handle_endtag(self, tag: str) -> None:
        if re.fullmatch(r"h[1-6]", tag) or tag == "p":
            self._break(1 if self._lists else 2)
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code" and not self._in_pre:
            self._emit("`")
        elif tag == "pre":
            self._break(1)
            self._emit("```\n\n")
            self._in_pre = False
        elif tag in ("ul", "ol") and self._lists:
            self._lists.pop()
            self._break(1 if self._lists else 2)
        elif tag == "a" and self._hrefs:
            href = self._hrefs.pop()
            self._emit(f"]({href})" if href else "]")
        elif tag == "blockquote" and self._quotes:
            start = self._quotes.pop()
            body = "".join(self.out[start:]).strip("\n")
            del self.out[start:]
            self._emit("\n".join(f"> {s}".rstrip() for s in body.split("\n")) + "\n\n")

    def handle_data(self, data: str) -> None:
        if not self._in_pre:
            data = re.sub(r"\s+", " ", data)
            if not self.out or self._newlines:
                data = data.lstrip()
        self._emit(data)


def html_to_markdown(note_html: str) -> str:
    """Convert note HTML to Markdown, for reading. Not a lossless round trip."""
    writer = _MarkdownWriter()
    writer.feed(note_html)
    writer.close()
    return re.sub(r"\n{3,}", "\n\n", "".join(writer.out)).strip() + "\n"


class _TitleParser(HTMLParser):
    """Collect the text of a note up to the end of its first non-empty line."""

    _BREAKS = frozenset(
        {"p", "br", "div", "li", "ul", "ol", "pre", "blockquote", "hr"}
        | {f"h{n}" for n in range(1, 7)}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = [""]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BREAKS:
            self.lines.append("")

    def handle_endtag(self, tag: str) -> None:
        self.handle_starttag(tag, [])

    def handle_data(self, data: str) -> None:
        self.lines[-1] += data


def note_title(note_html: str) -> str:
    """Return the first non-empty line of a note, without tags, as Zotero does."""
    parser = _TitleParser()
    parser.feed(note_html)
    parser.close()
    for line in parser.lines:
        if line.strip():
            return " ".join(line.split())
    return ""


def _check_markdown(markdown: str) -> None:
    if not markdown.strip():
        msg = "The note is empty"
        raise ValueError(msg)


def _body(markdown: str, title: str = "") -> str:
    """Return the note HTML for ``markdown``, with ``title`` as an <h1> if given."""
    _check_markdown(markdown)
    body = markdown_to_html(markdown)
    if title.strip():
        return f"<h1>{html.escape(title.strip(), quote=False)}</h1>\n{body}"
    return body


def build_note(
    markdown: str,
    *,
    title: str = "",
    parent: str | None = None,
    collection: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return the item data of a new note.

    A child note (``parent`` given) cannot also name a ``collection``: Zotero
    files a child note with its parent.
    """
    if parent and collection:
        msg = "A note with a parent item cannot be filed in a collection"
        raise ValueError(msg)
    item: dict[str, Any] = {
        "itemType": "note",
        "note": _body(markdown, title),
        "tags": [{"tag": tag} for tag in tags],
        "collections": [collection] if collection else [],
        "relations": {},
    }
    if parent:
        item["parentItem"] = parent
    return item


def add_note(
    zot: Zotero,
    markdown: str,
    *,
    title: str = "",
    parent: str | None = None,
    collection: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
) -> str:
    """Create a note and return its key. Raises RuntimeError if Zotero rejects it."""
    item = build_note(
        markdown, title=title, parent=parent, collection=collection, tags=tags
    )
    resp = zot.create_items([item])
    success = resp.get("success") or {}
    if not success:
        msg = f"Zotero rejected the note: {resp.get('failed')}"
        raise RuntimeError(msg)
    return success["0"]


def _is_note(item: dict[str, Any]) -> bool:
    return item.get("data", {}).get("itemType") == "note"


def list_notes(zot: Zotero, parent: str) -> list[dict[str, Any]]:
    """Return the child notes of item ``parent``: key, title, dateModified."""
    return [
        {
            "key": child["key"],
            "title": note_title(child["data"].get("note", "")),
            "dateModified": child["data"].get("dateModified"),
        }
        for child in zot.everything(zot.children(parent))
        if _is_note(child)
    ]


def get_note(zot: Zotero, key: str) -> dict[str, Any]:
    """Return a note as key, version, parent, title, dateModified and HTML.

    Raises ValueError if the item is not a note.
    """
    item = zot.item(key)
    if not _is_note(item):
        msg = f"{key} is not a note"
        raise ValueError(msg)
    data = item["data"]
    return {
        "key": item["key"],
        "version": item["version"],
        "parent": data.get("parentItem"),
        "title": note_title(data.get("note", "")),
        "dateModified": data.get("dateModified"),
        "html": data.get("note", ""),
    }


def _appended(existing: str, addition: str) -> str:
    """Put ``addition`` at the end of a note's HTML.

    Zotero's editor wraps a note in one ``<div data-schema-version=...>``.
    The addition goes inside that wrapper when the note has one.
    """
    stripped = existing.rstrip()
    if stripped.startswith("<div data-schema-version") and stripped.endswith("</div>"):
        return stripped.removesuffix("</div>") + "\n" + addition + "\n</div>"
    return stripped + "\n" + addition if stripped else addition


def append_note(zot: Zotero, key: str, markdown: str) -> dict[str, Any]:
    """Append Markdown to note ``key``. Return its key and the title.

    The write sends the note's version, so Zotero refuses it if the note
    changed meanwhile. The note is then read again and the text appended to
    the new content, once. Raises ValueError if the item is not a note.
    """
    addition = _body(markdown)
    try:
        note = get_note(zot, key)
        zot.update_item(
            {
                "key": key,
                "version": note["version"],
                "note": _appended(note["html"], addition),
            }
        )
    except ze.PreConditionFailedError:
        note = get_note(zot, key)
        zot.update_item(
            {
                "key": key,
                "version": note["version"],
                "note": _appended(note["html"], addition),
            }
        )
    return {"key": key, "title": note["title"]}
