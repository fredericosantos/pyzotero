"""MCP server exposing a Zotero library (local or zotero.org) and Semantic Scholar."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer

from pyzotero import _files
from pyzotero._helpers import (
    LOCAL_KEY_ENV,
    LOCAL_SERVER_ID_ENV,
    annotate_with_library,
    build_doi_index,
    format_creators,
    format_s2_paper,
    get_webdav_storage,
    get_write_client,
    get_zotero_client,
)
from pyzotero.semantic_scholar import (
    PaperNotFoundError,
    RateLimitError,
    SemanticScholarError,
    filter_by_citations,
    get_citations as s2_get_citations,
    get_recommendations,
    get_references as s2_get_references,
    search_papers,
)
from pyzotero.duplicates import apply_merge, find_duplicates, match_basis, plan_merge
from pyzotero.webdav import check_storage
from pyzotero.zotero import chunks

mcp = MCPServer("zotero")

F = TypeVar("F", bound=Callable[..., str])


def _json(obj: Any) -> str:
    """Serialise an object to a JSON string."""
    return json.dumps(obj, indent=2)


def _error(msg: str) -> str:
    """Return a JSON-encoded error message."""
    return _json({"error": msg})


def _run_s2_tool_lookup(
    doi: str,
    limit: int,
    min_citations: int,
    check_library: bool,
    lookup: Callable[..., dict[str, Any]],
) -> str:
    """Drive the shared Semantic-Scholar-by-DOI lookup flow and return JSON.

    Fetches papers via ``lookup(doi, id_type="doi", limit=limit)``, applies
    the ``min_citations`` filter, optionally annotates each paper with its
    presence in the local Zotero library, and serialises the payload.
    """
    result = lookup(doi, id_type="doi", limit=limit)
    papers = result.get("papers", [])

    if min_citations > 0:
        papers = filter_by_citations(papers, min_citations)

    if not papers:
        return _json({"count": 0, "papers": []})

    if check_library:
        zot = get_zotero_client()
        doi_map = build_doi_index(zot)
        output_papers = annotate_with_library(papers, doi_map)
    else:
        output_papers = [format_s2_paper(p) for p in papers]

    return _json({"count": len(output_papers), "papers": output_papers})


def mcp_error_handler(func: F) -> F:
    """Translate exceptions raised inside an MCP tool to a JSON error payload.

    Semantic Scholar errors get short, specific messages; everything else
    falls back to the exception's str() representation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except PaperNotFoundError:
            return _error("Paper not found in Semantic Scholar.")
        except RateLimitError:
            return _error("Rate limit exceeded. Please wait and try again.")
        except SemanticScholarError as e:
            return _error(str(e))
        except Exception as e:
            return _error(str(e))

    return wrapper  # type: ignore[return-value]


@mcp.tool()
@mcp_error_handler
def search(  # noqa: PLR0912
    query: str = "",
    fulltext: bool = False,
    itemtype: str = "",
    collection: str = "",
    tag: str = "",
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Search the local Zotero library.

    Args:
        query: Search query string.
        fulltext: If true, search full-text content including PDFs.
        itemtype: Filter by item type. Use || to combine types, e.g. "book || journalArticle".
        collection: Filter by collection key.
        tag: Filter by tag. Use comma-separated values for AND search, e.g. "climate,adaptation".
        limit: Maximum results to return (default 50).
        offset: Number of results to skip for pagination.

    Returns:
        JSON with count and items list.

    """
    zot = get_zotero_client()

    params: dict[str, Any] = {"limit": limit}

    if offset > 0:
        params["start"] = offset
    if query:
        params["q"] = query
    if fulltext:
        params["qmode"] = "everything"
    if itemtype:
        params["itemType"] = itemtype
    if tag:
        # Split comma-separated tags for AND search
        tags = [t.strip() for t in tag.split(",")]
        params["tag"] = tags if len(tags) > 1 else tags[0]

    if fulltext:
        if collection:
            results = zot.collection_items(collection, **params)
        else:
            results = zot.items(**params)

        # Retrieve parent items for attachment matches
        top_level_items = []
        attachment_items = []

        for item in results:
            data = item.get("data", {})
            if "parentItem" in data:
                attachment_items.append(item)
            else:
                top_level_items.append(item)

        parent_items = []
        if attachment_items:
            parent_ids = list({item["data"]["parentItem"] for item in attachment_items})
            for chunk in chunks(parent_ids, 50):
                parent_items.extend(zot.get_subset(chunk))

        all_items = top_level_items + parent_items
        items_dict = {item["data"]["key"]: item for item in all_items}
        results = list(items_dict.values())
    elif collection:
        results = zot.collection_items_top(collection, **params)
    else:
        results = zot.top(**params)

    output_items = []
    for item in results:
        data = item.get("data", {})
        creator_names = format_creators(data.get("creators", []))

        output_items.append(
            {
                "key": data.get("key", ""),
                "itemType": data.get("itemType", "Unknown"),
                "title": data.get("title", "No title"),
                "creators": creator_names,
                "date": data.get("date", ""),
                "publication": data.get("publicationTitle", ""),
                "volume": data.get("volume", ""),
                "issue": data.get("issue", ""),
                "doi": data.get("DOI", ""),
                "url": data.get("url", ""),
            }
        )

    return _json({"count": len(output_items), "items": output_items})


@mcp.tool()
@mcp_error_handler
def get_item(key: str) -> str:
    """Get a single Zotero item by its key.

    Args:
        key: The Zotero item key.

    Returns:
        JSON with the full item data.

    """
    zot = get_zotero_client()
    result = zot.item(key)
    if not result:
        return _error(f"Item not found: {key}")
    return _json(result)


@mcp.tool()
@mcp_error_handler
def get_children(key: str) -> str:
    """Get child items (attachments, notes) of a Zotero item.

    Args:
        key: The Zotero item key.

    Returns:
        JSON array of child items.

    """
    zot = get_zotero_client()
    results = zot.children(key)
    return _json(results)


@mcp.tool()
@mcp_error_handler
def list_collections(limit: int = 0) -> str:
    """List all collections in the local Zotero library.

    Args:
        limit: Maximum number of collections to return (0 for all).

    Returns:
        JSON array of collections with id, name, items count, and parent info.

    """
    zot = get_zotero_client()

    params: dict[str, Any] = {}
    if limit > 0:
        params["limit"] = limit

    collections = zot.collections(**params)

    # Build parent name lookup
    collection_map = {}
    for coll in collections:
        data = coll.get("data", {})
        ckey = data.get("key", "")
        cname = data.get("name", "")
        if ckey:
            collection_map[ckey] = cname or None

    output = []
    for coll in collections:
        data = coll.get("data", {})
        meta = coll.get("meta", {})
        parent_key = data.get("parentCollection", "")

        obj: dict[str, Any] = {
            "id": data.get("key", ""),
            "name": data.get("name", "") or None,
            "items": meta.get("numItems", 0),
        }

        if parent_key:
            obj["parent"] = {
                "id": parent_key,
                "name": collection_map.get(parent_key),
            }
        else:
            obj["parent"] = None

        output.append(obj)

    return _json(output)


@mcp.tool()
@mcp_error_handler
def list_tags(collection: str = "") -> str:
    """List all tags in the Zotero library, optionally filtered by collection.

    Args:
        collection: Optional collection key to filter tags.

    Returns:
        JSON array of tag strings.

    """
    zot = get_zotero_client()
    if collection:
        results = zot.collection_tags(collection)
    else:
        results = zot.tags()
    return _json(results)


@mcp.tool()
@mcp_error_handler
def download_attachment(key: str, dest_dir: str) -> str:
    """Download an attachment's file to a directory, for example to read a PDF.

    Args:
        key: The key of an attachment, or of a regular item: then its first
            PDF attachment is used.
        dest_dir: An absolute path to the directory for the file. It is
            created if needed.

    Returns:
        JSON with the attachment key, ``path`` (the main file) and ``files``.

    """
    dest = Path(dest_dir)
    if not dest.is_absolute():
        return _error(f"dest_dir must be an absolute path, got {dest_dir!r}")
    return _json(_files.download(get_zotero_client(), key, dest))


@mcp.tool()
@mcp_error_handler
def storage_check(verify_hashes: bool = False) -> str:
    """Compare the library's attachments with the files on the WebDAV server.

    Only for storage = webdav. Nothing is changed.

    Args:
        verify_hashes: Also compare each file's hash with the item's md5.
            This takes one request per attachment.

    Returns:
        JSON with lists of attachment keys: ok, missing (no file on the
        server), not_uploaded, mismatched and orphaned (a file that no
        attachment owns).

    """
    storage = get_webdav_storage()
    if storage is None:
        return _error("storage_check needs storage = webdav (see 'pyzotero setup')")
    report = check_storage(get_zotero_client(), storage, verify_hashes=verify_hashes)
    return _json(report.as_dict())


@mcp.tool()
@mcp_error_handler
def find_duplicate_items() -> str:
    """List groups of duplicate items, using Zotero desktop's rules.

    Items match on DOI, on ISBN (books), or on a normalized title with at
    least one common author and years at most one apart. Each group lists
    the oldest item first.

    Returns:
        JSON list of groups: ``basis`` (doi, title, isbn or mixed; check
        isbn groups by hand, since a proceedings volume and one of its
        chapters share an ISBN) and ``items`` (key, itemType, title, date,
        DOI, dateAdded, children).

    """
    zot = get_zotero_client()
    groups = find_duplicates(zot.everything(zot.items(limit=100)))
    return _json(
        [
            {
                "basis": match_basis(g),
                "items": [
                    {
                        "key": i["key"],
                        **{
                            f: i["data"].get(f)
                            for f in ("itemType", "title", "date", "DOI", "dateAdded")
                        },
                        "children": i.get("meta", {}).get("numChildren", 0),
                    }
                    for i in g
                ],
            }
            for g in groups
        ]
    )


@mcp.tool()
@mcp_error_handler
def get_fulltext(key: str) -> str:
    """Get full-text content of a Zotero attachment.

    Args:
        key: The key of an attachment item (not a top-level item).

    Returns:
        JSON with content, indexedPages, and totalPages.

    """
    zot = get_zotero_client()
    result = zot.fulltext_item(key)
    if not result:
        return _error("No full-text content available")
    return _json(result)


@mcp.tool()
@mcp_error_handler
def find_related(
    doi: str,
    limit: int = 20,
    min_citations: int = 0,
    check_library: bool = True,
) -> str:
    """Find papers related to a given paper using Semantic Scholar.

    Uses SPECTER2 embeddings for semantic similarity.

    Args:
        doi: DOI of the paper.
        limit: Maximum results (default 20, max 500).
        min_citations: Minimum citation count filter.
        check_library: If true, annotate results with local Zotero presence.

    Returns:
        JSON with count and papers list.

    """
    return _run_s2_tool_lookup(
        doi, limit, min_citations, check_library, get_recommendations
    )


@mcp.tool()
@mcp_error_handler
def get_citations(
    doi: str,
    limit: int = 100,
    min_citations: int = 0,
    check_library: bool = True,
) -> str:
    """Find papers that cite a given paper using Semantic Scholar.

    Args:
        doi: DOI of the paper.
        limit: Maximum results (default 100, max 1000).
        min_citations: Minimum citation count filter.
        check_library: If true, annotate results with local Zotero presence.

    Returns:
        JSON with count and papers list.

    """
    return _run_s2_tool_lookup(
        doi, limit, min_citations, check_library, s2_get_citations
    )


@mcp.tool()
@mcp_error_handler
def get_references(
    doi: str,
    limit: int = 100,
    min_citations: int = 0,
    check_library: bool = True,
) -> str:
    """Find papers referenced by a given paper using Semantic Scholar.

    Args:
        doi: DOI of the paper.
        limit: Maximum results (default 100, max 1000).
        min_citations: Minimum citation count filter.
        check_library: If true, annotate results with local Zotero presence.

    Returns:
        JSON with count and papers list.

    """
    return _run_s2_tool_lookup(
        doi, limit, min_citations, check_library, s2_get_references
    )


@mcp.tool()
@mcp_error_handler
def search_semantic_scholar(
    query: str,
    limit: int = 20,
    year: str = "",
    open_access: bool = False,
    sort: str = "",
    min_citations: int = 0,
    check_library: bool = True,
) -> str:
    """Search for papers on Semantic Scholar.

    Args:
        query: Search query string.
        limit: Maximum results (default 20, max 100).
        year: Year filter, e.g. "2020", "2018-2022", or "2020-".
        open_access: Only return open access papers.
        sort: Sort by "citations" or "year" (descending).
        min_citations: Minimum citation count filter.
        check_library: If true, annotate results with local Zotero presence.

    Returns:
        JSON with count, total, and papers list.

    """
    result = search_papers(
        query,
        limit=limit,
        year=year or None,
        open_access_only=open_access,
        sort=sort or None,
        min_citations=min_citations,
    )
    papers = result.get("papers", [])
    total = result.get("total", len(papers))

    if not papers:
        return _json({"count": 0, "total": total, "papers": []})

    if check_library:
        zot = get_zotero_client()
        doi_map = build_doi_index(zot)
        output_papers = annotate_with_library(papers, doi_map)
    else:
        output_papers = [format_s2_paper(p) for p in papers]

    return _json({"count": len(output_papers), "total": total, "papers": output_papers})


WRITE_KEY_ENV = LOCAL_KEY_ENV
SERVER_ID_ENV = LOCAL_SERVER_ID_ENV


def _write_client() -> Any:
    """Return a Zotero client that has write access to the local API.

    The key comes from the environment, or else from the file that
    ``pyzotero authorize`` writes. See :func:`get_write_client`.
    """
    return get_write_client()


AddTool = Callable[[Callable[..., str]], None]


def _register_item_tools(add: AddTool) -> None:
    """Register tools that create and modify items."""

    def list_item_fields(item_type: str) -> str:
        """List the fields and creator types valid for a Zotero item type.

        Call this before create_item to find the fields that an item type accepts.

        Args:
            item_type: A Zotero item type, e.g. "journalArticle" or "book".

        Returns:
            JSON with the field names and creator types for that item type.

        """
        zot = get_zotero_client()
        return _json(
            {
                "itemType": item_type,
                "fields": [
                    f["field"]  # ty: ignore[invalid-argument-type]
                    for f in zot.item_type_fields(item_type)
                ],
                "creatorTypes": [
                    c["creatorType"]  # ty: ignore[invalid-argument-type]
                    for c in zot.item_creator_types(item_type)
                ],
            }
        )

    def create_item(
        item_type: str,
        fields: dict[str, Any] | None = None,
        creators: list[dict[str, str]] | None = None,
        tags: list[str] | None = None,
        collections: list[str] | None = None,
    ) -> str:
        """Create a new item in the local Zotero library.

        Args:
            item_type: A Zotero item type, e.g. "journalArticle" or "book".
            fields: Field values, e.g. {"title": "...", "date": "2024"}. Use
                list_item_fields to find the fields that the item type accepts.
            creators: Creator dicts, each with creatorType and either
                (firstName, lastName) or name.
            tags: Tag names to attach.
            collections: Collection keys to file the item under.

        Returns:
            JSON with the created item's key, or the server's rejection reason.

        """
        zot = _write_client()
        item: dict[str, Any] = {"itemType": item_type, **(fields or {})}
        if creators:
            item["creators"] = creators
        if tags:
            item["tags"] = [{"tag": tag} for tag in tags]
        if collections:
            item["collections"] = collections
        resp = zot.create_items([item])
        if resp.get("success"):
            return _json({"created": resp["success"]["0"], "itemType": item_type})
        return _json({"error": "Item was rejected", "detail": resp.get("failed")})

    def update_item(key: str, fields: dict[str, Any]) -> str:
        """Update fields on an existing item. Other fields stay unchanged.

        Args:
            key: The item key.
            fields: Field values to set, e.g. {"title": "New title"}.

        Returns:
            JSON confirming the key and the fields written.

        """
        zot = _write_client()
        item = zot.item(key)
        item["data"].update(fields)
        zot.update_item(item)
        return _json({"updated": key, "fields": sorted(fields)})

    def add_tags(key: str, tags: list[str]) -> str:
        """Add one or more tags to an existing item.

        Args:
            key: The item key.
            tags: Tag names to add.

        Returns:
            JSON confirming the key and the item's full tag list.

        """
        zot = _write_client()
        zot.add_tags(zot.item(key), *tags)
        return _json(
            {"updated": key, "tags": [t["tag"] for t in zot.item(key)["data"]["tags"]]}
        )

    for tool in (list_item_fields, create_item, update_item, add_tags):
        add(tool)


def _register_collection_tools(add: AddTool) -> None:
    """Register tools that create collections and change item membership."""

    def create_collection(name: str, parent: str = "") -> str:
        """Create a new collection.

        Args:
            name: The collection name.
            parent: Optional parent collection key, to nest the new collection.

        Returns:
            JSON with the created collection's key.

        """
        zot = _write_client()
        payload: dict[str, Any] = {"name": name}
        if parent:
            payload["parentCollection"] = parent
        resp = zot.create_collections([payload])
        if resp.get("success"):
            return _json({"created": resp["success"]["0"], "name": name})
        return _json({"error": "Collection was rejected", "detail": resp.get("failed")})

    def add_to_collection(item_key: str, collection_key: str) -> str:
        """File an existing item under a collection.

        Args:
            item_key: The item key.
            collection_key: The collection key.

        Returns:
            JSON confirming the item and collection.

        """
        zot = _write_client()
        zot.addto_collection(collection_key, zot.item(item_key))
        return _json({"item": item_key, "addedTo": collection_key})

    def remove_from_collection(item_key: str, collection_key: str) -> str:
        """Remove an item from a collection. The item itself is unchanged.

        This is the inverse of add_to_collection. The item stays in the
        library and in its other collections.

        Args:
            item_key: The item key.
            collection_key: The collection key.

        Returns:
            JSON confirming the item and collection.

        """
        zot = _write_client()
        zot.deletefrom_collection(collection_key, zot.item(item_key))
        return _json({"item": item_key, "removedFrom": collection_key})

    def move_to_collection(
        item_key: str, from_collection_key: str, to_collection_key: str
    ) -> str:
        """Move an item from one collection to another in one step.

        Prefer this over get_item followed by update_item with a rewritten
        collections list: membership of other collections is kept. If the
        item is not in the source collection, it is added to the destination
        all the same.

        Args:
            item_key: The item key.
            from_collection_key: The key of the collection to remove the item from.
            to_collection_key: The key of the collection to add the item to.

        Returns:
            JSON confirming the item, the source and the destination.

        """
        zot = _write_client()
        zot.moveto_collection(
            from_collection_key, to_collection_key, zot.item(item_key)
        )
        return _json(
            {
                "item": item_key,
                "movedFrom": from_collection_key,
                "movedTo": to_collection_key,
            }
        )

    for tool in (
        create_collection,
        add_to_collection,
        remove_from_collection,
        move_to_collection,
    ):
        add(tool)


def _register_attachment_tools(add: AddTool) -> None:
    """Register the file attachment tools."""

    def add_attachment(item_key: str, file_path: str, title: str = "") -> str:
        """Attach a file on disk to an existing Zotero item.

        The file goes to the configured storage: the WebDAV server, or Zotero
        File Storage (see ``pyzotero setup``). The attachment stays available
        if the source file moves or is deleted. If the library syncs, the
        attachment also syncs.

        Args:
            item_key: The key of the item that gets the attachment.
            file_path: An absolute path to the file. This server runs as a
                subprocess of the MCP client, so its working directory is not
                yours. A relative path would point to an unknown location.
                Thus the server rejects relative paths.
            title: Optional title for the attachment. The default is the
                filename.

        Returns:
            JSON with the key of the new attachment, or the reason for a
            rejection. If the file is already attached to the item, the
            result reports it as unchanged and no copy occurs. Thus a retried
            call is safe. The server does not replace the contents of an
            attachment: a file with different contents becomes a second
            attachment.

        """
        path = Path(file_path)
        if not path.is_absolute():
            msg = f"file_path must be an absolute path, got {file_path!r}"
            raise ValueError(msg)
        return _json(_files.attach(_write_client(), item_key, path, title or None))

    def fetch_pdf(item_key: str, force: bool = False) -> str:
        """Find an open-access PDF for an item and attach it.

        arXiv is tried first, from the item's arXiv ID; then Unpaywall, from
        its DOI.

        Args:
            item_key: The key of a regular (not attachment) item.
            force: Attach even if the item already has a stored PDF.

        Returns:
            JSON with the new attachment's key, the source and the URL; or
            ``unchanged`` with the existing PDF's key; or the reason no PDF
            was found.

        """
        return _json(_files.fetch_pdf(_write_client(), item_key, force=force))

    add(add_attachment)
    add(fetch_pdf)


def _register_merge_tools(add: AddTool) -> None:
    """Register the duplicate merge tool."""

    def merge_items(
        keys: list[str], master: str = "", fill_empty: bool = False, apply: bool = False
    ) -> str:
        """Merge duplicate items into one, as Zotero desktop does.

        The master keeps its fields and gets the others' notes, attachments,
        collections, tags and relations, and the earliest date added. A PDF
        that is byte-identical to one of the master's is trashed. The other
        items go to the trash, from where Zotero can restore them.

        Args:
            keys: Two or more item keys, for example a group from
                find_duplicate_items.
            master: The key to keep. Default: the oldest by dateAdded.
            fill_empty: Copy fields that are empty on the master from the
                other items.
            apply: Make the changes. With False (the default), only the
                plan is returned: show it to the user before applying.

        Returns:
            JSON with ``applied`` and the plan: master, trashed items,
            changed master fields, moved children, trashed duplicate PDFs
            and items whose related-item links were updated.

        """
        if len(set(keys)) < 2:  # noqa: PLR2004
            return _error("Give at least two different item keys")
        zot = _write_client() if apply else get_zotero_client()
        if not master:
            master = min(
                (zot.item(k) for k in keys),
                key=lambda i: i["data"].get("dateAdded", ""),
            )["key"]
        elif master not in keys:
            return _error(f"master {master} is not one of the given keys")
        plan = plan_merge(
            zot, master, [k for k in keys if k != master], fill_empty=fill_empty
        )
        if apply:
            apply_merge(zot, plan)
        return _json({"applied": apply, **plan.summary()})

    add(merge_items)


def _register_delete_tools(add: AddTool) -> None:
    """Register the delete tools. This runs only for --enable-deletes."""

    def delete_item(key: str) -> str:
        """Permanently delete an item from the local Zotero library.

        You cannot undo this operation. The local API erases the item. It
        does not move the item to the trash. If the library syncs, the
        deletion also syncs. Get the user's confirmation before you call
        this tool.

        Args:
            key: The item key.

        Returns:
            JSON confirming the deleted key.

        """
        zot = _write_client()
        zot.delete_item(zot.item(key))
        return _json({"deleted": key})

    add(delete_item)


def register_write_tools(
    server: MCPServer, *, enable_deletes: bool = False
) -> list[str]:
    """Register the write tools on ``server``. Return the registered names.

    Registration is the gate for writes, not a check in each tool. A tool
    that is not registered does not appear in the model's tool list. Thus
    content in the library cannot cause a call to it. Nothing here runs
    unless ``main()`` receives --enable-writes or --enable-deletes.
    """
    registered: list[str] = []

    def add(func: Callable[..., str]) -> None:
        server.tool()(mcp_error_handler(func))
        registered.append(func.__name__)  # ty: ignore[unresolved-attribute]

    _register_item_tools(add)
    _register_collection_tools(add)
    _register_attachment_tools(add)
    _register_merge_tools(add)
    if enable_deletes:
        _register_delete_tools(add)
    return registered


def main() -> None:
    """Run the MCP server over stdio transport."""
    parser = argparse.ArgumentParser(
        prog="pyzotero-mcp",
        description="MCP server exposing a Zotero library. Read-only by default.",
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="register tools that create and modify library items",
    )
    parser.add_argument(
        "--enable-deletes",
        action="store_true",
        help=(
            "additionally register delete_item. Deletions via the local API are "
            "permanent, not moves to the trash. Implies --enable-writes"
        ),
    )
    args = parser.parse_args()
    if args.enable_writes or args.enable_deletes:
        names = register_write_tools(mcp, enable_deletes=args.enable_deletes)
        print(f"pyzotero-mcp: write tools enabled: {', '.join(names)}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
