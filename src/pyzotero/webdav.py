"""Attachment file storage on a WebDAV server, in the layout Zotero uses.

Zotero desktop syncs each stored attachment as two files in a ``zotero/``
directory under the WebDAV URL that the user configures:

- ``<KEY>.zip``: a ZIP archive with the attachment's files at its root.
- ``<KEY>.prop``: ``<properties version="1"><mtime>MS</mtime><hash>MD5</hash>
  </properties>``, with the modification time (in milliseconds) and the MD5
  of the attachment's main file, not of the ZIP.

The item data on zotero.org carries the same ``mtime`` and ``md5``. A desktop
client downloads a file only when the ``mtime`` in the item data differs from
its local copy, so an upload must update both the server files and the item.
:func:`attach_file` does both.

Reference: ``chrome/content/zotero/xpcom/storage/webdav.js`` and
``storageLocal.js`` (``processDownload``) in the Zotero client.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import mimetypes
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

import httpx2

from ._utils import DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from ._client import Zotero

# Upload and download of a large PDF can take far longer than an API call.
TRANSFER_TIMEOUT = httpx2.Timeout(DEFAULT_TIMEOUT, read=600, write=600)
# Attachment link modes whose file lives in storage (the others are links).
STORED_LINK_MODES = frozenset({"imported_file", "imported_url"})
_B64_SUFFIX = "%ZB64"
_PROPFIND_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<D:propfind xmlns:D="DAV:"><D:prop><D:getlastmodified/></D:prop></D:propfind>'
)
_HREF_RE = re.compile(r"<(?:\w+:)?href>([^<]+)</(?:\w+:)?href>", re.IGNORECASE)


class WebDAVError(RuntimeError):
    """A WebDAV request failed or the server holds unexpected data."""


@dataclasses.dataclass(frozen=True)
class StorageProps:
    """The contents of a ``.prop`` file."""

    mtime: int
    """Modification time of the attachment file, in milliseconds."""
    md5: str | None


def file_props(path: Path) -> StorageProps:
    """Return the mtime (ms) and MD5 of a local file, as Zotero computes them."""
    digest = hashlib.md5()  # noqa: S324 -- Zotero's file identity, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return StorageProps(mtime=int(path.stat().st_mtime * 1000), md5=digest.hexdigest())


def parse_props(text: str) -> StorageProps | None:
    """Parse a ``.prop`` file. Return None if it holds no valid mtime.

    Old Zotero versions wrote a bare Unix timestamp in seconds; Zotero still
    accepts that, so this function does too.
    """
    mtime_m = re.search(r"<mtime>\s*(\d+)\s*</mtime>", text)
    hash_m = re.search(r"<hash>\s*([0-9a-fA-F]{32})\s*</hash>", text)
    if mtime_m and len(mtime_m.group(1)) <= 13:  # noqa: PLR2004 -- Zotero's limit
        return StorageProps(int(mtime_m.group(1)), hash_m.group(1) if hash_m else None)
    legacy = text.strip()
    if not mtime_m and re.fullmatch(r"\d{1,10}", legacy):
        return StorageProps(int(legacy) * 1000, None)
    return None


def format_props(props: StorageProps) -> str:
    """Return the ``.prop`` file text for ``props``."""
    return (
        '<properties version="1">'
        f"<mtime>{props.mtime}</mtime>"
        f"<hash>{escape(props.md5 or '')}</hash>"
        "</properties>"
    )


def _entry_name(name: str) -> str:
    """Decode a ZIP entry name the way Zotero does."""
    if name.endswith(_B64_SUFFIX):
        return base64.b64decode(name[: -len(_B64_SUFFIX)]).decode("utf-8")
    return name


class WebDAVStorage:
    """Read and write Zotero attachment files on a WebDAV server.

    Args:
        url: The URL as entered in Zotero's sync settings, without the
            ``zotero/`` part. Zotero adds ``zotero/`` itself, and so does this
            class.
        username: The WebDAV user name.
        password: The WebDAV password.
        client: Optional HTTP client, for tests or connection reuse.

    """

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        client: httpx2.Client | None = None,
    ) -> None:
        self.base_url = url.rstrip("/") + "/zotero/"
        # The auth goes with each request, not on the client, so that a
        # client that is passed in is not changed.
        self.auth = httpx2.BasicAuth(username, password)
        self.client = client or httpx2.Client(
            timeout=TRANSFER_TIMEOUT, follow_redirects=True
        )

    def _url(self, name: str) -> str:
        return self.base_url + name

    def _request(
        self, method: str, name: str, ok: tuple[int, ...], **kwargs: Any
    ) -> httpx2.Response:
        resp = self.client.request(method, self._url(name), auth=self.auth, **kwargs)
        if resp.status_code not in ok:
            msg = f"WebDAV {method} {self._url(name)} returned {resp.status_code}"
            if resp.status_code == 401:  # noqa: PLR2004
                msg += " (check the WebDAV user name and password)"
            elif resp.status_code == 507:  # noqa: PLR2004
                msg += " (the server has no space left)"
            raise WebDAVError(msg)
        return resp

    def check(self) -> None:
        """Verify that the server is reachable and writable, as Zotero does.

        Raises WebDAVError with the reason if not.
        """
        self._request(
            "PROPFIND", "", (207,), headers={"Depth": "0"}, content=_PROPFIND_BODY
        )
        test = "zotero-test-file.prop"
        self._request("PUT", test, (200, 201, 204), content=b" ")
        self._request("GET", test, (200,))
        self._request("DELETE", test, (200, 204))

    def get_props(self, key: str) -> StorageProps | None:
        """Return the ``.prop`` contents for ``key``, or None if there are none."""
        # mod_speling can answer 300 for a missing file with a similar name.
        resp = self._request("GET", f"{key}.prop", (200, 300, 404))
        if resp.status_code != 200:  # noqa: PLR2004
            return None
        return parse_props(resp.text)

    def upload(self, key: str, path: Path) -> StorageProps:
        """Store ``path`` as the file of attachment ``key``. Return its props.

        The ``.prop`` file is deleted first and written last, in the order
        Zotero uses: a reader never sees new props with an old ZIP.
        """
        props = file_props(path)
        self._request("DELETE", f"{key}.prop", (200, 204, 404))
        # A spooled file keeps small ZIPs in memory, and large ones on disk.
        with tempfile.SpooledTemporaryFile(max_size=32 << 20) as buf:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(path, arcname=path.name)
            buf.seek(0)
            self._request(
                "PUT",
                f"{key}.zip",
                (200, 201, 204),
                content=buf,
                headers={"Content-Type": "application/zip"},
            )
        self._request(
            "PUT",
            f"{key}.prop",
            (200, 201, 204),
            content=format_props(props).encode(),
            headers={"Content-Type": "text/xml"},
        )
        return props

    def download(self, key: str, dest_dir: Path) -> list[Path]:
        """Extract the files of attachment ``key`` into ``dest_dir``.

        Returns the extracted paths. Entries that would land outside
        ``dest_dir`` are refused.

        Raises:
            FileNotFoundError: The server has no ZIP for ``key``.
            WebDAVError: The request failed or the ZIP is invalid.

        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        root = dest_dir.resolve()
        written: list[Path] = []
        with tempfile.TemporaryFile() as buf:
            with self.client.stream(
                "GET", self._url(f"{key}.zip"), auth=self.auth
            ) as resp:
                if resp.status_code == 404:  # noqa: PLR2004
                    msg = f"No file on the WebDAV server for attachment {key}"
                    raise FileNotFoundError(msg)
                if resp.status_code != 200:  # noqa: PLR2004
                    msg = f"WebDAV GET {key}.zip returned {resp.status_code}"
                    raise WebDAVError(msg)
                for chunk in resp.iter_bytes():
                    buf.write(chunk)
            buf.seek(0)
            try:
                zf = zipfile.ZipFile(buf)
            except zipfile.BadZipFile as exc:
                msg = f"{key}.zip on the WebDAV server is not a valid ZIP file"
                raise WebDAVError(msg) from exc
            with zf:
                for info in zf.infolist():
                    name = _entry_name(info.filename)
                    if info.is_dir() or name.startswith(".zotero"):
                        continue
                    target = (root / PurePosixPath(name)).resolve()
                    if not target.is_relative_to(root):
                        msg = f"{key}.zip has an entry outside its folder: {name!r}"
                        raise WebDAVError(msg)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, target.open("wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
                    written.append(target)
        return written

    def delete(self, key: str) -> None:
        """Delete the files of attachment ``key``. Missing files are ignored."""
        # Zotero deletes the .prop first, so that no reader trusts a lone ZIP.
        self._request("DELETE", f"{key}.prop", (200, 204, 404))
        self._request("DELETE", f"{key}.zip", (200, 204, 404))

    def list_files(self) -> dict[str, set[str]]:
        """Return ``{key: {"zip", "prop"}}`` for the files in the directory."""
        resp = self._request(
            "PROPFIND", "", (207,), headers={"Depth": "1"}, content=_PROPFIND_BODY
        )
        found: dict[str, set[str]] = {}
        for href in _HREF_RE.findall(resp.text):
            name = PurePosixPath(httpx2.URL(href).path).name
            stem, _, ext = name.rpartition(".")
            if ext in {"zip", "prop"} and re.fullmatch(r"[A-Z0-9]{8}", stem):
                found.setdefault(stem, set()).add(ext)
        return found


def attach_file(
    zot: Zotero,
    storage: WebDAVStorage,
    path: Path,
    parent_key: str | None = None,
    title: str | None = None,
) -> str:
    """Create an attachment item for ``path`` and store its file on WebDAV.

    Returns the new attachment's key. If the upload fails after the item was
    created, the item is deleted again, so no attachment without a file is
    left in the library.
    """
    if not path.is_file():
        msg = f"No file at {path}"
        raise FileNotFoundError(msg)
    template = zot.item_template("attachment", "imported_file")
    template.update(
        title=title or path.name,
        filename=path.name,
        contentType=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )
    resp = zot.create_items([template], parentid=parent_key)
    if not resp.get("success"):
        msg = f"Zotero rejected the attachment item: {resp.get('failed')}"
        raise RuntimeError(msg)
    key = resp["success"]["0"]
    try:
        props = storage.upload(key, path)
        item = zot.item(key)
        item["data"]["md5"] = props.md5
        item["data"]["mtime"] = props.mtime
        zot.update_item(item)
    except Exception as exc:
        try:
            zot.delete_item(zot.item(key))
            storage.delete(key)
        except Exception as cleanup_exc:
            msg = f"Upload of {key} failed ({exc}), and so did the removal of its item"
            raise RuntimeError(msg) from cleanup_exc
        raise
    return key


@dataclasses.dataclass
class StorageReport:
    """The result of :func:`check_storage`. Each field lists attachment keys."""

    ok: list[str] = dataclasses.field(default_factory=list)
    missing: list[str] = dataclasses.field(default_factory=list)
    """In the library with an mtime, but no ZIP on the server."""
    not_uploaded: list[str] = dataclasses.field(default_factory=list)
    """In the library with no mtime: no client has uploaded a file yet."""
    mismatched: list[str] = dataclasses.field(default_factory=list)
    """The ``.prop`` hash differs from the item's md5."""
    orphaned: list[str] = dataclasses.field(default_factory=list)
    """On the server, but no stored attachment in the library has this key."""

    def as_dict(self) -> dict[str, list[str]]:
        """Return the report as a dict of sorted key lists."""
        return {k: sorted(v) for k, v in dataclasses.asdict(self).items()}


def check_storage(
    zot: Zotero, storage: WebDAVStorage, verify_hashes: bool = False
) -> StorageReport:
    """Compare the library's stored attachments with the files on WebDAV.

    With ``verify_hashes``, each ``.prop`` file is read and its hash compared
    with the item's md5. This takes one request per attachment.

    Items in the trash count as part of the library: Zotero keeps their files.
    """
    attachments = zot.everything(zot.items(itemType="attachment", includeTrashed=1))
    on_server = storage.list_files()
    report = StorageReport()
    seen: set[str] = set()
    for att in attachments:
        data = att["data"]
        if data.get("linkMode") not in STORED_LINK_MODES:
            continue
        key = att["key"]
        seen.add(key)
        if "zip" not in on_server.get(key, set()):
            (report.missing if data.get("mtime") else report.not_uploaded).append(key)
            continue
        if verify_hashes and data.get("md5"):
            props = storage.get_props(key)
            if props is None or props.md5 != data["md5"]:
                report.mismatched.append(key)
                continue
        report.ok.append(key)
    report.orphaned = [k for k in on_server if k not in seen]
    return report
