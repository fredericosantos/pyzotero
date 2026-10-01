"""Find and download open-access PDFs for Zotero items.

Two sources are used, in this order:

1. arXiv, for items with an arXiv ID (in ``archiveID``, ``url``, ``extra`` or
   an arXiv DOI). No account is needed.
2. Unpaywall, for items with a DOI. Unpaywall requires an email address with
   each request (``unpaywall_email`` in the settings).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx2

from ._helpers import normalise_doi
from .webdav import TRANSFER_TIMEOUT

UNPAYWALL_URL = "https://api.unpaywall.org/v2/"
ARXIV_PDF_URL = "https://arxiv.org/pdf/"

# New-style IDs (2402.14872, optionally with a version) and old-style IDs
# (hep-th/9901001, math.GT/0309136).
_ARXIV_ID = r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
_ARXIV_PATTERNS = (
    re.compile(r"arxiv\.org/(?:abs|pdf)/" + _ARXIV_ID, re.IGNORECASE),
    re.compile(r"arxiv:\s*" + _ARXIV_ID, re.IGNORECASE),
    re.compile(r"10\.48550/arxiv\." + _ARXIV_ID, re.IGNORECASE),
)


class NoOpenAccessPDFError(LookupError):
    """No open-access PDF was found for the item."""


@dataclasses.dataclass(frozen=True)
class PDFSource:
    """Where a PDF can be downloaded, and which service reported it."""

    url: str
    source: str


def find_arxiv_id(data: dict[str, Any]) -> str | None:
    """Return the arXiv ID in an item's data, or None."""
    for field in ("archiveID", "url", "DOI", "extra"):
        value = data.get(field) or ""
        for pattern in _ARXIV_PATTERNS:
            if match := pattern.search(value):
                return match.group(1).removesuffix(".pdf")
    return None


def find_pdf(
    data: dict[str, Any],
    email: str | None = None,
    client: httpx2.Client | None = None,
) -> PDFSource:
    """Return an open-access PDF location for an item's data.

    Raises:
        NoOpenAccessPDFError: Neither source has a PDF, or the item has no
            identifier that a source can use.

    """
    if arxiv_id := find_arxiv_id(data):
        return PDFSource(ARXIV_PDF_URL + arxiv_id, "arxiv")
    doi = normalise_doi(data.get("DOI") or "")
    if not doi:
        msg = "The item has neither an arXiv ID nor a DOI"
        raise NoOpenAccessPDFError(msg)
    if not email:
        msg = (
            "Unpaywall needs an email address for DOI lookups: set "
            "unpaywall_email with 'pyzotero setup' or PYZOTERO_UNPAYWALL_EMAIL"
        )
        raise NoOpenAccessPDFError(msg)
    http = client or httpx2.Client(timeout=30, follow_redirects=True)
    resp = http.get(UNPAYWALL_URL + quote(doi, safe="/"), params={"email": email})
    if resp.status_code == 404:  # noqa: PLR2004
        msg = f"Unpaywall does not know DOI {doi}"
        raise NoOpenAccessPDFError(msg)
    resp.raise_for_status()
    record = resp.json()
    locations = [record.get("best_oa_location"), *(record.get("oa_locations") or [])]
    for loc in locations:
        if loc and loc.get("url_for_pdf"):
            return PDFSource(loc["url_for_pdf"], "unpaywall")
    msg = f"Unpaywall has no open-access PDF for DOI {doi}"
    raise NoOpenAccessPDFError(msg)


def download_pdf(
    source: PDFSource, dest: Path, client: httpx2.Client | None = None
) -> Path:
    """Download the PDF at ``source`` to ``dest``. Return ``dest``.

    Raises:
        NoOpenAccessPDFError: The URL does not return a PDF. Publisher sites
            often answer a PDF link with an HTML page.

    """
    http = client or httpx2.Client(timeout=TRANSFER_TIMEOUT, follow_redirects=True)
    # Some publishers refuse requests without a browser-like Accept header.
    headers = {"Accept": "application/pdf,*/*;q=0.8"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with http.stream("GET", source.url, headers=headers) as resp:
        resp.raise_for_status()
        chunks = resp.iter_bytes()
        first = next(chunks, b"")
        if not first.startswith(b"%PDF"):
            msg = f"{source.url} did not return a PDF"
            raise NoOpenAccessPDFError(msg)
        with dest.open("wb") as out:
            out.write(first)
            for chunk in chunks:
                out.write(chunk)
    return dest
