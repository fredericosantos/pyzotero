"""Tests for WebDAV attachment storage, settings and open-access PDFs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
import zipfile
from unittest.mock import MagicMock

import httpx2
import pytest

from pyzotero import _config, _files, _helpers, openaccess
from pyzotero.webdav import (
    StorageProps,
    WebDAVError,
    WebDAVStorage,
    attach_file,
    check_storage,
    format_props,
    parse_props,
)

BASE = "https://dav.example.org"
MTIME_MS = 1_700_000_000_123
PRIVATE = 0o600
PASSWORD = "p"  # noqa: S105 -- test fixture


class FakeDAV:
    """An in-memory WebDAV server for one directory. Records each request."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.log: list[tuple[str, str]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:  # noqa: PLR0911
        path = request.url.path
        self.log.append((request.method, path))
        if (
            request.headers.get("authorization")
            != httpx2.BasicAuth("u", "p")._auth_header
        ):
            return httpx2.Response(401)
        if request.method == "PUT":
            self.files[path] = request.read()
            return httpx2.Response(201)
        if request.method == "GET":
            if path not in self.files:
                return httpx2.Response(404)
            return httpx2.Response(200, content=self.files[path])
        if request.method == "DELETE":
            return httpx2.Response(
                204 if self.files.pop(path, None) is not None else 404
            )
        if request.method == "PROPFIND":
            hrefs = "".join(
                f"<D:response><D:href>{p}</D:href></D:response>" for p in self.files
            )
            body = f'<D:multistatus xmlns:D="DAV:"><D:response><D:href>/zotero/</D:href></D:response>{hrefs}</D:multistatus>'
            return httpx2.Response(207, text=body)
        return httpx2.Response(405)


@pytest.fixture
def dav():
    return FakeDAV()


@pytest.fixture
def storage(dav):
    return WebDAVStorage(
        BASE, "u", "p", client=httpx2.Client(transport=httpx2.MockTransport(dav))
    )


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\nhello\n")
    os.utime(path, (1_700_000_000, 1_700_000_000.123))
    return path


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestProps:
    def test_round_trip(self):
        props = StorageProps(1772548674716, "69f51b0d421ee7f44899c04878d48431")
        assert parse_props(format_props(props)) == props

    def test_real_zotero_file(self):
        text = '<properties version="1"><mtime>1772548674716</mtime><hash>69f51b0d421ee7f44899c04878d48431</hash></properties>'
        assert parse_props(text) == StorageProps(
            1772548674716, "69f51b0d421ee7f44899c04878d48431"
        )

    def test_legacy_seconds(self):
        assert parse_props("1300000000\n") == StorageProps(1300000000000, None)

    @pytest.mark.parametrize("text", ["", "garbage", "<mtime>12345678901234</mtime>"])
    def test_invalid(self, text):
        assert parse_props(text) is None


class TestStorage:
    def test_upload_layout_and_order(self, storage, dav, pdf):
        props = storage.upload("ABCD1234", pdf)
        assert props.md5 == hashlib.md5(pdf.read_bytes()).hexdigest()  # noqa: S324
        assert props.mtime == MTIME_MS
        with zipfile.ZipFile(io.BytesIO(dav.files["/zotero/ABCD1234.zip"])) as zf:
            assert zf.namelist() == ["paper.pdf"]
            assert zf.read("paper.pdf") == pdf.read_bytes()
        assert parse_props(dav.files["/zotero/ABCD1234.prop"].decode()) == props
        # The .prop is deleted first and written last, as Zotero does.
        assert dav.log == [
            ("DELETE", "/zotero/ABCD1234.prop"),
            ("PUT", "/zotero/ABCD1234.zip"),
            ("PUT", "/zotero/ABCD1234.prop"),
        ]

    def test_download_round_trip(self, storage, pdf, tmp_path):
        storage.upload("ABCD1234", pdf)
        files = storage.download("ABCD1234", tmp_path / "out")
        assert [f.name for f in files] == ["paper.pdf"]
        assert files[0].read_bytes() == pdf.read_bytes()

    def test_download_decodes_b64_names_and_skips_zotero_files(
        self, storage, dav, tmp_path
    ):
        encoded = base64.b64encode("résumé.pdf".encode()).decode() + "%ZB64"
        dav.files["/zotero/ABCD1234.zip"] = _zip(
            {encoded: b"%PDF", ".zotero-ft-cache": b"x"}
        )
        files = storage.download("ABCD1234", tmp_path)
        assert [f.name for f in files] == ["résumé.pdf"]

    def test_download_refuses_path_traversal(self, storage, dav, tmp_path):
        dav.files["/zotero/ABCD1234.zip"] = _zip({"../evil.txt": b"x"})
        with pytest.raises(WebDAVError, match="outside its folder"):
            storage.download("ABCD1234", tmp_path / "out")
        assert not (tmp_path / "evil.txt").exists()

    def test_download_missing(self, storage, tmp_path):
        with pytest.raises(FileNotFoundError):
            storage.download("ABCD1234", tmp_path)

    def test_list_files(self, storage, dav):
        dav.files.update(
            {
                "/zotero/ABCD1234.zip": b"",
                "/zotero/ABCD1234.prop": b"",
                "/zotero/WXYZ9876.prop": b"",
                "/zotero/lastsync.txt": b"",
            }
        )
        assert storage.list_files() == {
            "ABCD1234": {"zip", "prop"},
            "WXYZ9876": {"prop"},
        }

    def test_bad_credentials(self, dav):
        bad = WebDAVStorage(
            BASE,
            "u",
            "wrong",
            client=httpx2.Client(transport=httpx2.MockTransport(dav)),
        )
        with pytest.raises(WebDAVError, match="user name and password"):
            bad.check()

    def test_check(self, storage, dav):
        storage.check()
        assert "/zotero/zotero-test-file.prop" not in dav.files


def _zot(children=()):
    zot = MagicMock()
    zot.item_template.return_value = {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "title": "",
        "filename": "",
        "contentType": "",
    }
    zot.create_items.return_value = {"success": {"0": "NEWKEY01"}, "failed": {}}
    zot.item.return_value = {
        "key": "NEWKEY01",
        "data": {"key": "NEWKEY01", "version": 5},
    }
    zot.children.return_value = list(children)
    return zot


class TestAttachFile:
    def test_creates_item_uploads_and_sets_props(self, storage, dav, pdf):
        zot = _zot()
        key = attach_file(zot, storage, pdf, "PARENT01")
        assert key == "NEWKEY01"
        sent = zot.create_items.call_args
        assert sent.kwargs["parentid"] == "PARENT01"
        assert sent.args[0][0]["filename"] == "paper.pdf"
        assert sent.args[0][0]["contentType"] == "application/pdf"
        updated = zot.update_item.call_args.args[0]["data"]
        assert updated["md5"] == hashlib.md5(pdf.read_bytes()).hexdigest()  # noqa: S324
        assert updated["mtime"] == MTIME_MS
        assert "/zotero/NEWKEY01.zip" in dav.files

    def test_failed_upload_removes_item(self, pdf):
        zot = _zot()
        broken = MagicMock()
        broken.upload.side_effect = WebDAVError("boom")
        with pytest.raises(WebDAVError):
            attach_file(zot, broken, pdf, "PARENT01")
        zot.delete_item.assert_called_once()
        broken.delete.assert_called_once_with("NEWKEY01")


class TestCheckStorage:
    def test_categories(self, storage, dav):
        dav.files.update(
            {
                "/zotero/OKOKOK01.zip": b"",
                "/zotero/OKOKOK01.prop": format_props(
                    StorageProps(1, "a" * 32)
                ).encode(),
                "/zotero/BADHASH1.zip": b"",
                "/zotero/BADHASH1.prop": format_props(
                    StorageProps(1, "b" * 32)
                ).encode(),
                "/zotero/ORPHAN01.zip": b"",
            }
        )

        def att(key, **data):
            return {"key": key, "data": {"linkMode": "imported_file", **data}}

        zot = MagicMock()
        zot.everything.return_value = [
            att("OKOKOK01", mtime=1, md5="a" * 32),
            att("BADHASH1", mtime=1, md5="c" * 32),
            att("MISSING1", mtime=1, md5="d" * 32),
            att("NEVERUP1"),
            {"key": "LINKED01", "data": {"linkMode": "linked_url"}},
        ]
        report = check_storage(zot, storage, verify_hashes=True).as_dict()
        assert report == {
            "ok": ["OKOKOK01"],
            "missing": ["MISSING1"],
            "not_uploaded": ["NEVERUP1"],
            "mismatched": ["BADHASH1"],
            "orphaned": ["ORPHAN01"],
        }


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in _config.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestSettings:
    def test_defaults_keep_upstream_behaviour(self, clean_env):
        settings = _config.load_settings()
        assert (settings.mode, settings.storage) == ("local", "zotero")
        assert _helpers.get_zotero_client().local is True
        assert _helpers.get_webdav_storage() is None

    def test_env_overrides_file(self, clean_env):
        _config.save_settings(
            _config.Settings(mode="remote", api_key="filekey", library_id="1")
        )
        clean_env.setenv("PYZOTERO_API_KEY", "envkey")
        settings = _config.load_settings()
        assert (settings.mode, settings.api_key) == ("remote", "envkey")

    def test_file_is_private(self, clean_env):
        path = _config.save_settings(_config.Settings(api_key="secret"))
        assert stat.S_IMODE(path.stat().st_mode) == PRIVATE

    def test_invalid_mode(self, clean_env):
        clean_env.setenv("PYZOTERO_MODE", "cloud")
        with pytest.raises(_config.ConfigError, match="mode must be one of"):
            _config.load_settings()

    def test_unknown_key_in_file(self, clean_env):
        path = _config.config_path()
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"api_kee": "x"}))
        with pytest.raises(_config.ConfigError, match="api_kee"):
            _config.load_settings()

    def test_remote_client(self, clean_env):
        clean_env.setenv("PYZOTERO_MODE", "remote")
        with pytest.raises(_config.ConfigError, match="api_key"):
            _helpers.get_zotero_client()
        clean_env.setenv("PYZOTERO_API_KEY", "k")
        clean_env.setenv("PYZOTERO_LIBRARY_ID", "42")
        zot = _helpers.get_write_client()
        assert (zot.local, zot.endpoint, zot.library_id, zot.api_key) == (
            False,
            "https://api.zotero.org",
            "42",
            "k",
        )

    def test_webdav_needs_credentials(self, clean_env):
        clean_env.setenv("PYZOTERO_STORAGE", "webdav")
        clean_env.setenv("PYZOTERO_WEBDAV_URL", BASE)
        with pytest.raises(_config.ConfigError, match="webdav_username"):
            _helpers.get_webdav_storage()


class TestSetupCommand:
    def test_non_interactive(self, clean_env):
        pytest.importorskip("click")
        from click.testing import CliRunner  # noqa: PLC0415

        from pyzotero import cli  # noqa: PLC0415

        args = [
            "setup", "--no-input", "--no-check", "--mode", "remote", "--api-key", "k",
            "--library-id", "42", "--storage", "webdav", "--webdav-url", BASE,
            "--webdav-username", "u", "--webdav-password", "p",
        ]  # fmt: skip
        result = CliRunner().invoke(cli.main, args)
        assert result.exit_code == 0, result.output
        saved = json.loads(_config.config_path().read_text())
        assert saved["storage"] == "webdav"
        assert saved["webdav_password"] == PASSWORD
        shown = CliRunner().invoke(cli.main, ["setup", "--show"])
        assert '"api_key": "****k"' in shown.output
        assert '"p"' not in shown.output


class TestOpenAccess:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"archiveID": "arXiv:2402.14872"}, "2402.14872"),
            ({"url": "https://arxiv.org/abs/2402.14872v2"}, "2402.14872v2"),
            ({"url": "http://arxiv.org/pdf/2402.14872.pdf"}, "2402.14872"),
            ({"DOI": "10.48550/arXiv.2402.14872"}, "2402.14872"),
            ({"extra": "arXiv: hep-th/9901001"}, "hep-th/9901001"),
            ({"DOI": "10.1000/xyz"}, None),
        ],
    )
    def test_find_arxiv_id(self, data, expected):
        assert openaccess.find_arxiv_id(data) == expected

    def test_unpaywall(self):
        def handler(request):
            assert request.url.params["email"] == "me@example.org"
            assert request.url.path == "/v2/10.1000/xyz"
            return httpx2.Response(
                200,
                json={
                    "best_oa_location": None,
                    "oa_locations": [{"url_for_pdf": "https://x.org/a.pdf"}],
                },
            )

        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        source = openaccess.find_pdf(
            {"DOI": "https://doi.org/10.1000/XYZ"}, "me@example.org", client
        )
        assert source == openaccess.PDFSource("https://x.org/a.pdf", "unpaywall")

    def test_doi_without_email(self):
        with pytest.raises(openaccess.NoOpenAccessPDFError, match="email"):
            openaccess.find_pdf({"DOI": "10.1000/xyz"})

    def test_download_rejects_html(self, tmp_path):
        client = httpx2.Client(
            transport=httpx2.MockTransport(
                lambda r: httpx2.Response(200, text="<html>paywall</html>")
            )
        )
        with pytest.raises(
            openaccess.NoOpenAccessPDFError, match="did not return a PDF"
        ):
            openaccess.download_pdf(
                openaccess.PDFSource("https://x.org/a.pdf", "unpaywall"),
                tmp_path / "a.pdf",
                client,
            )
        assert not (tmp_path / "a.pdf").exists()


class TestFiles:
    def test_attach_routes_to_webdav(self, clean_env, monkeypatch, storage, dav, pdf):
        monkeypatch.setattr(_files, "get_webdav_storage", lambda: storage)
        clean_env.setenv("PYZOTERO_MODE", "remote")
        result = _files.attach(_zot(), "PARENT01", pdf)
        assert result["attached"] == "NEWKEY01"
        assert result["storage"] == "webdav"
        assert "/zotero/NEWKEY01.zip" in dav.files

    @pytest.mark.parametrize(
        "env", [{}, {"PYZOTERO_MODE": "remote", "PYZOTERO_LIBRARY_TYPE": "group"}]
    )
    def test_attach_uses_api_outside_remote_user_library(
        self, clean_env, monkeypatch, storage, dav, pdf, env
    ):
        monkeypatch.setattr(_files, "get_webdav_storage", lambda: storage)
        for var, value in env.items():
            clean_env.setenv(var, value)
        zot = _zot()
        zot.upload_attachments.return_value = {
            "success": [{"key": "APIKEY01"}],
            "failure": [],
            "unchanged": [],
        }
        assert _files.attach(zot, "PARENT01", pdf)["storage"] == "zotero"
        assert not dav.files

    def test_attach_is_idempotent(self, clean_env, pdf):
        md5 = hashlib.md5(pdf.read_bytes()).hexdigest()  # noqa: S324
        zot = _zot(
            children=[
                {"key": "OLDKEY01", "data": {"filename": "paper.pdf", "md5": md5}}
            ]
        )
        assert _files.attach(zot, "PARENT01", pdf)["unchanged"] == "OLDKEY01"
        zot.create_items.assert_not_called()

    def test_resolve_prefers_pdf(self):
        zot = MagicMock()
        zot.item.return_value = {
            "key": "PARENT01",
            "data": {"itemType": "journalArticle"},
        }
        zot.children.return_value = [
            {"key": "NOTE0001", "data": {"itemType": "note"}},
            {
                "key": "HTML0001",
                "data": {
                    "itemType": "attachment",
                    "linkMode": "imported_url",
                    "contentType": "text/html",
                },
            },
            {
                "key": "PDF00001",
                "data": {
                    "itemType": "attachment",
                    "linkMode": "imported_file",
                    "contentType": "application/pdf",
                },
            },
        ]
        assert _files.resolve_attachment(zot, "PARENT01")["key"] == "PDF00001"

    def test_fetch_pdf_skips_item_with_pdf(self, clean_env):
        zot = _zot(
            children=[
                {
                    "key": "PDF00001",
                    "data": {
                        "contentType": "application/pdf",
                        "linkMode": "imported_file",
                    },
                }
            ]
        )
        assert _files.fetch_pdf(zot, "PARENT01")["unchanged"] == "PDF00001"
