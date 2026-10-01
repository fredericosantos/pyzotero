"""Attachment file operations shared by the CLI and MCP server.

Each function sends the file to the configured storage: WebDAV when
``storage`` is ``webdav`` (see :mod:`pyzotero._config`), else Zotero File
Storage through the API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pyzotero import zotero
from pyzotero._config import load_settings
from pyzotero._helpers import get_webdav_storage
from pyzotero.openaccess import download_pdf, find_pdf
from pyzotero.webdav import STORED_LINK_MODES, WebDAVStorage, attach_file, file_props


def _upload_storage() -> WebDAVStorage | None:
    """Return the WebDAV storage that uploads go to, or None for the API path.

    Uploads go to WebDAV only in remote mode for a user library. In local
    mode the desktop app syncs files to WebDAV itself, after the local API
    has stored them. Zotero syncs group files only to Zotero File Storage.
    """
    settings = load_settings()
    if settings.mode != "remote" or settings.library_type != "user":
        return None
    return get_webdav_storage()


def find_existing(zot: zotero.Zotero, parent_key: str, path: Path) -> str | None:
    """Return the key of a child of ``parent_key`` that holds this exact file."""
    checksum = file_props(path).md5
    for child in zot.children(parent_key):
        data = child.get("data", {})
        if data.get("filename") == path.name and data.get("md5") == checksum:
            return child["key"]
    return None


def attach(
    zot: zotero.Zotero, parent_key: str, path: Path, title: str | None = None
) -> dict[str, Any]:
    """Attach the file at ``path`` to item ``parent_key``.

    Returns ``{"attached": key, ...}`` for a new attachment,
    ``{"unchanged": key, ...}`` if the identical file is already attached
    (so a retried call is safe), or ``{"error": ..., "detail": ...}`` if
    Zotero File Storage rejected the upload.
    """
    if not path.is_file():
        msg = f"No file at {path}"
        raise FileNotFoundError(msg)
    if existing := find_existing(zot, parent_key, path):
        return {"unchanged": existing, "parent": parent_key, "filename": path.name}
    storage = _upload_storage()
    if storage is not None:
        key = attach_file(zot, storage, path, parent_key, title)
        return {
            "attached": key,
            "parent": parent_key,
            "filename": path.name,
            "storage": "webdav",
        }
    # item_template() is not available on the local API (no /items/new), so
    # build the template directly. The upload code finds the contentType.
    template = {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "title": title or path.name,
        "filename": str(path),
        "note": "",
        "tags": [],
        "relations": {},
    }
    result = zot.upload_attachments([template], parent_key)
    if result["success"]:
        return {
            "attached": result["success"][0]["key"],
            "parent": parent_key,
            "filename": path.name,
            "storage": "zotero",
        }
    if result["unchanged"]:
        return {"unchanged": parent_key, "filename": path.name}
    detail = result["failure"][0] if result["failure"] else None
    return {"error": "Attachment was rejected", "detail": detail}


def resolve_attachment(zot: zotero.Zotero, key: str) -> dict[str, Any]:
    """Return the stored attachment for ``key``.

    ``key`` may name an attachment, or a regular item: then its first PDF
    attachment is used, or else its first stored attachment of any type.
    """
    item = zot.item(key)
    data = item["data"]
    if data.get("itemType") == "attachment":
        if data.get("linkMode") not in STORED_LINK_MODES:
            msg = f"Attachment {key} is a link ({data.get('linkMode')}), not a stored file"
            raise ValueError(msg)
        return item
    stored = [
        c
        for c in zot.children(key)
        if c["data"].get("itemType") == "attachment"
        and c["data"].get("linkMode") in STORED_LINK_MODES
    ]
    if not stored:
        msg = f"Item {key} has no stored attachments"
        raise LookupError(msg)
    pdfs = [c for c in stored if c["data"].get("contentType") == "application/pdf"]
    return (pdfs or stored)[0]


def download(zot: zotero.Zotero, key: str, dest_dir: Path) -> dict[str, Any]:
    """Download the file of attachment (or item) ``key`` into ``dest_dir``.

    Returns ``{"attachment": key, "path": main file, "files": [...]}``.
    """
    att = resolve_attachment(zot, key)
    att_key, filename = att["key"], att["data"].get("filename")
    storage = get_webdav_storage()
    if storage is not None:
        files = storage.download(att_key, dest_dir)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        zot.dump(att_key, filename, dest_dir)
        files = [(dest_dir / filename).resolve()] if filename else []
    main = next((f for f in files if f.name == filename), files[0] if files else None)
    return {
        "attachment": att_key,
        "path": str(main) if main else None,
        "files": [str(f) for f in files],
    }


def has_pdf(zot: zotero.Zotero, parent_key: str) -> str | None:
    """Return the key of a stored PDF attachment of ``parent_key``, or None."""
    for child in zot.children(parent_key):
        data = child.get("data", {})
        if (
            data.get("contentType") == "application/pdf"
            and data.get("linkMode") in STORED_LINK_MODES
        ):
            return child["key"]
    return None


def fetch_pdf(zot: zotero.Zotero, key: str, force: bool = False) -> dict[str, Any]:
    """Find an open-access PDF for item ``key`` and attach it.

    Unless ``force`` is set, an item that already has a stored PDF is left
    alone, and the result reports it as ``unchanged``.
    """
    existing = None if force else has_pdf(zot, key)
    if existing:
        return {
            "unchanged": existing,
            "parent": key,
            "detail": "the item already has a PDF",
        }
    data = zot.item(key)["data"]
    source = find_pdf(data, email=load_settings().unpaywall_email)
    with tempfile.TemporaryDirectory() as tmp:
        stem = source.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".pdf") or key
        path = download_pdf(source, Path(tmp) / f"{stem}.pdf")
        result = attach(zot, key, path, title=f"Full Text PDF ({source.source})")
    return {**result, "source": source.source, "url": source.url}
