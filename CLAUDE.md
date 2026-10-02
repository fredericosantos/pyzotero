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
- `update_item` raises on a rejected write (via `@backoff_check`); batch `update_items` does not check per-object failures in a 200 response.

## Library conventions

Filing rules for the library itself live in the owner's global `~/.claude/CLAUDE.md` (Workflows → Research). Show a dry run before bulk library changes; they sync to every device.
