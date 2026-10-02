# CLAUDE.md

Fork of [urschrei/pyzotero](https://github.com/urschrei/pyzotero) at `fredericosantos/pyzotero`, used as the CLI and MCP server for the owner's Zotero library on a headless server (remote mode, files on a self-hosted WebDAV).

## Git

- `origin` is the fork, `upstream` is urschrei. Merge into the fork's `main` only; never open a PR or push to upstream without an explicit request.
- Keep fork-only code in its own modules (`_config.py`, `_files.py`, `webdav.py`, `openaccess.py`, `duplicates.py`, `notes.py`, `annotations.py`) and keep edits to upstream files small, so rebasing onto upstream stays cheap.
- Worktrees: `.claude/worktrees/<name>` (gitignored).

## Checks

```bash
uv run ruff format && uv run ruff check src tests
uv run ty check src/pyzotero      # one pre-existing warning in _client.py is upstream's
uv run pytest -q -p no:cacheprovider --no-cov
```

`tests/conftest.py` points `XDG_CONFIG_HOME` at a temp dir and clears `PYZOTERO_*`. Keep it: without it, tests read the real `~/.config/pyzotero/config.json` and once uploaded mock files to the real WebDAV server.

## Verified zotero.org behaviour (live, 2026-10)

- Attachments on WebDAV: upload `KEY.zip` + `KEY.prop`, then PATCH the item's `md5` and `mtime`; desktops download a file only when the item's `mtime` changes. Zotero's own protocol is in its `xpcom/storage/webdav.js`.
- `dateAdded` is writable via PATCH.
- `dc:relation` (related items) is kept symmetric by the server: removing a link from one side has no effect while the other side still has it; removing it from either side removes both.
- Annotation items (`itemType: "annotation"`, child of a PDF attachment) can be created via the API and show in the desktop reader. `annotationPosition` is a JSON string `{"pageIndex", "rects": [[x1, y1, x2, y2]]}` in PDF user space (points, origin bottom-left). From `pdftotext -bbox` (origin top-left): `y1 = pageHeight - yMax`, `y2 = pageHeight - yMin`. A highlight made in the desktop app had exactly these coordinates. `annotationSortIndex` is `PPPPP|OOOOOO|TTTTT` (page index, text offset, top in points).
- Annotation rect height is the font's descent to ascent (FontDescriptor `Ascent`/`Descent` times size), which is what `pdftotext -bbox` and the desktop app use, not the font size: the app's own highlight of a 14.35 pt title is 12.897 pt tall (0.899 of the size), and 8.907 pt for 9.96 pt text. pdfplumber boxes a character one font size tall, so its tops are 1 to 1.5 pt too high; `annotations._read_words` recomputes the top from the pdfminer font and then matches poppler on every word of a real paper (max difference 1e-6 pt). The sort-index top is the floor of that corrected top (91 for the title). pdfplumber's default `x_tolerance=3` glues words of PDFs that have no space characters ("LargeLanguageModels(LLMs)"), so words are extracted with `x_tolerance_ratio=0.15`.
- pdfplumber ignores the CropBox for `page.height`, `top` and `bottom`: `y_user = page.height - top` holds in absolute PDF user space even for a MediaBox with a non-zero origin (checked with `[0 50 612 842]`), while `page.cropbox` and `page.mediabox` come back flipped (`y_user = height - y`). `annotations._user_box` rebuilds the visible box (CropBox cut to MediaBox) from them. The sort-index top of a rect highlight is measured from that box's top, which is an assumption about Zotero's own value (it is only an ordering key; not checked against the desktop app).
- The text offset in `annotationSortIndex` counts the characters before the highlight that are not white space: the app's highlight IZB6P65K ("Genetic Algorithm", preceded by "Semantic Mirror Jailbreak: ") has offset 24 = len("SemanticMirrorJailbreak:"), not 27. One data point; `Stream.letters` implements it, and it reproduces that sort index exactly.
- pdfminer's `doc.get_page_labels()` never ends on a PDF with `/PageLabels` (the last range is `itertools.count`): `list()` of it ate 42 GB. `load_pages` takes `islice(..., page_count)`.
- `update_item` raises on a rejected write (via `@backoff_check`); batch `update_items` does not check per-object failures in a 200 response.

## Library conventions

Filing rules for the library itself live in the owner's global `~/.claude/CLAUDE.md` (Workflows → Research). Show a dry run before bulk library changes; they sync to every device.
