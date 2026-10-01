"""Command-line interface for pyzotero."""

from __future__ import annotations

import functools
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, TypeVar

import click
import httpx2

from pyzotero import __version__, _files
from pyzotero._config import (
    ENV_VARS,
    MODES,
    STORAGES,
    Settings,
    config_path,
    load_settings,
    save_settings,
)
from pyzotero._helpers import (
    annotate_with_library,
    build_doi_index,
    build_doi_index_full,
    describe_item_type,
    format_creators,
    format_s2_paper,
    get_webdav_storage,
    get_write_client,
    get_zotero_client,
    normalise_doi,
    save_local_key,
    validate_items,
)
from pyzotero.semantic_scholar import (
    PaperNotFoundError,
    RateLimitError,
    SemanticScholarError,
    filter_by_citations,
    get_citations,
    get_recommendations,
    get_references,
    search_papers,
)
from pyzotero.duplicates import apply_merge, find_duplicates, match_basis, plan_merge
from pyzotero.webdav import WebDAVStorage, check_storage
from pyzotero.zotero import chunks

F = TypeVar("F", bound=Callable[..., Any])


def cli_error_handler(func: F) -> F:
    """Map exceptions raised in a CLI command to stderr messages + exit 1.

    Semantic Scholar errors get short, specific messages; everything else
    falls back to ``"Error: <str(exc)>"``.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except PaperNotFoundError:
            click.echo("Error: Paper not found in Semantic Scholar.", err=True)
            sys.exit(1)
        except RateLimitError:
            click.echo(
                "Error: Rate limit exceeded. Please wait and try again.", err=True
            )
            sys.exit(1)
        except SemanticScholarError as e:
            click.echo(f"Error: {e!s}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {e!s}", err=True)
            sys.exit(1)

    return wrapper  # type: ignore[return-value]


def _zot_from_ctx(ctx: Any) -> Any:
    """Build a local-mode Zotero client using the locale from the CLI context."""
    return get_zotero_client(ctx.obj.get("locale", "en-US"))


def _write_zot_from_ctx(ctx: Any) -> Any:
    """Build a local-mode client that can write, from the stored local API key.

    Raises RuntimeError, which the error handler reports, if no key is stored.
    """
    return get_write_client(ctx.obj.get("locale", "en-US"))


def _change_membership(
    zot: Any, keys: tuple[str, ...], change: Callable[[dict[str, Any]], Any]
) -> list[str]:
    """Fetch each item in ``keys`` and apply ``change`` to it. Return the keys.

    The items change one at a time. If one fails, the error names it and
    lists the items that had already changed, so that the caller can resume.
    """
    done: list[str] = []
    for key in keys:
        try:
            change(zot.item(key))
        except Exception as exc:
            changed = ", ".join(done) if done else "none"
            msg = (
                f"{key}: {exc} "
                f"(changed {len(done)} of {len(keys)} items before this: {changed})"
            )
            raise RuntimeError(msg) from exc
        done.append(key)
    return done


def _read_items(source: IO[str]) -> list[Any]:
    """Read one item, or a list of items, from an open JSON source.

    A single object becomes a list of one item, so that the caller always
    gets a list. Raises RuntimeError, which the error handler reports, if
    the content is not JSON, or holds neither an object nor a list.
    """
    name = getattr(source, "name", "input")
    try:
        parsed = json.loads(source.read())
    except ValueError as exc:
        msg = f"{name} does not hold valid JSON: {exc}"
        raise RuntimeError(msg) from exc
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        msg = f"{name} must hold a JSON object or a list of objects"
        raise TypeError(msg)
    return parsed


def _attach(
    item: dict[str, Any], tags: tuple[str, ...], collection: str | None
) -> None:
    """Add ``tags`` and ``collection`` to an item, in place.

    Tags and collections that the item already has are kept. A tag that the
    item gives as a plain string becomes a Zotero tag object.
    """
    attached = [t if isinstance(t, dict) else {"tag": t} for t in item.get("tags", [])]
    present = {t.get("tag") for t in attached}
    attached.extend({"tag": tag} for tag in tags if tag not in present)
    if attached:
        item["tags"] = attached
    if collection:
        collections = list(item.get("collections", []))
        if collection not in collections:
            collections.append(collection)
        item["collections"] = collections


def _run_s2_lookup(
    ctx: Any,
    doi: str,
    limit: int,
    min_citations: int,
    check_library: bool,
    lookup: Callable[..., dict[str, Any]],
    label: str,
) -> None:
    """Drive the shared Semantic-Scholar-by-DOI lookup flow.

    Fetches papers via ``lookup(doi, id_type="doi", limit=limit)``, applies
    the ``min_citations`` filter, optionally annotates each paper with its
    presence in the local Zotero library, and prints the JSON payload.
    ``label`` appears in the stderr progress message.
    """
    click.echo(f"Fetching {label} for DOI: {doi}...", err=True)
    result = lookup(doi, id_type="doi", limit=limit)
    papers = result.get("papers", [])

    if min_citations > 0:
        papers = filter_by_citations(papers, min_citations)

    if not papers:
        click.echo(json.dumps({"count": 0, "papers": []}))
        return

    if check_library:
        click.echo("Checking local Zotero library...", err=True)
        zot = _zot_from_ctx(ctx)
        doi_map = build_doi_index(zot)
        output_papers = annotate_with_library(papers, doi_map)
    else:
        output_papers = [format_s2_paper(p) for p in papers]

    click.echo(
        json.dumps({"count": len(output_papers), "papers": output_papers}, indent=2)
    )


@click.group()
@click.version_option(version=__version__, prog_name="pyzotero")
@click.option(
    "--locale",
    default="en-US",
    help="Locale for localized strings (default: en-US)",
)
@click.pass_context
def main(ctx: Any, locale: str) -> None:
    """Search and manage a Zotero library: local, or on zotero.org.

    Run 'pyzotero setup' to choose the mode and file storage.
    """
    ctx.ensure_object(dict)
    ctx.obj["locale"] = locale


@main.command()
@click.option(
    "-q",
    "--query",
    help="Search query string",
    default="",
)
@click.option(
    "--fulltext",
    is_flag=True,
    help="Search full-text content including PDFs. Retrieves parent items when attachments match.",
)
@click.option(
    "--itemtype",
    multiple=True,
    help="Filter by item type (can be specified multiple times for OR search)",
)
@click.option(
    "--collection",
    help="Filter by collection key (returns only items in this collection)",
)
@click.option(
    "--tag",
    multiple=True,
    help="Filter by tag (can be specified multiple times for AND search)",
)
@click.option(
    "--limit",
    type=int,
    default=1000000,
    help="Maximum number of results to return (default: 1000000)",
)
@click.option(
    "--offset",
    type=int,
    default=0,
    help="Number of results to skip for pagination (default: 0)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def search(  # noqa: PLR0912, PLR0915
    ctx: Any,
    query: str,
    fulltext: bool,
    itemtype: tuple[str, ...],
    collection: str | None,
    tag: tuple[str, ...],
    limit: int,
    offset: int,
    output_json: bool,
) -> None:
    """Search local Zotero library.

    By default, searches top-level items in titles and metadata.

    When --fulltext is enabled, searches all items including attachment content
    (PDFs, documents, etc.). If a match is found in an attachment, the parent
    bibliographic item is retrieved and included in results.

    Examples:
        pyzotero search -q "machine learning"

        pyzotero search -q "climate change" --fulltext

        pyzotero search -q "methodology" --itemtype book --itemtype journalArticle

        pyzotero search --collection ABC123 -q "test"

        pyzotero search -q "climate" --json

        pyzotero search -q "topic" --limit 20 --offset 20 --json

        pyzotero search -q "topic" --tag "climate" --tag "adaptation" --json

    """
    zot = _zot_from_ctx(ctx)

    # Build query parameters
    params = {"limit": limit}

    if offset > 0:
        params["start"] = offset

    if query:
        params["q"] = query

    if fulltext:
        params["qmode"] = "everything"

    if itemtype:
        # Join multiple item types with || for OR search
        params["itemType"] = " || ".join(itemtype)

    if tag:
        # Multiple tags are passed as a list for AND search
        params["tag"] = list(tag)

    # Execute search
    # When fulltext is enabled, use items() or collection_items() to get both
    # top-level items and attachments. Otherwise use top() or collection_items_top()
    # to only get top-level items.
    if fulltext:
        if collection:
            results = zot.collection_items(collection, **params)
        else:
            results = zot.items(**params)

        # When using fulltext, we need to retrieve parent items for any attachments
        # that matched, since most full-text content comes from PDFs and other attachments
        top_level_items = []
        attachment_items = []

        for item in results:
            data = item.get("data", {})
            if "parentItem" in data:
                attachment_items.append(item)
            else:
                top_level_items.append(item)

        # Retrieve parent items for attachments in batches of 50
        parent_items = []
        if attachment_items:
            parent_ids = list({item["data"]["parentItem"] for item in attachment_items})
            for chunk in chunks(parent_ids, 50):
                parent_items.extend(zot.get_subset(chunk))

        # Combine top-level items and parent items, removing duplicates by key
        all_items = top_level_items + parent_items
        items_dict = {item["data"]["key"]: item for item in all_items}
        results = list(items_dict.values())
    # Non-fulltext search: use top() or collection_items_top() as before
    elif collection:
        results = zot.collection_items_top(collection, **params)
    else:
        results = zot.top(**params)

    # Handle empty results
    if not results:
        if output_json:
            click.echo(json.dumps([]))
        else:
            click.echo("No results found.")
        return

    # Build output data structure
    output_items = []
    for item in results:
        data = item.get("data", {})

        title = data.get("title", "No title")
        item_type = data.get("itemType", "Unknown")
        date = data.get("date", "No date")
        item_key = data.get("key", "")
        publication = data.get("publicationTitle", "")
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        doi = data.get("DOI", "")
        url = data.get("url", "")

        # Format creators (authors, editors, etc.)
        creator_names = format_creators(data.get("creators", []))

        # Check for PDF attachments
        pdf_attachments = []
        num_children = item.get("meta", {}).get("numChildren", 0)
        if num_children > 0:
            children = zot.children(item_key)
            for child in children:
                child_data = child.get("data", {})
                if child_data.get("contentType") == "application/pdf":
                    # Extract file URL from links.enclosure.href
                    file_url = (
                        child.get("links", {}).get("enclosure", {}).get("href", "")
                    )
                    if file_url:
                        pdf_attachments.append(file_url)

        # Build item object for JSON output
        item_obj = {
            "key": item_key,
            "itemType": item_type,
            "title": title,
            "creators": creator_names,
            "date": date,
            "publication": publication,
            "volume": volume,
            "issue": issue,
            "doi": doi,
            "url": url,
            "pdfAttachments": pdf_attachments,
        }
        output_items.append(item_obj)

    # Output results
    if output_json:
        click.echo(
            json.dumps({"count": len(output_items), "items": output_items}, indent=2)
        )
    else:
        click.echo(f"\nFound {len(results)} items:\n")
        for idx, item_obj in enumerate(output_items, 1):
            authors_str = (
                ", ".join(item_obj["creators"])
                if item_obj["creators"]
                else "No authors"
            )

            click.echo(f"{idx}. [{item_obj['itemType']}] {item_obj['title']}")
            click.echo(f"   Authors: {authors_str}")
            click.echo(f"   Date: {item_obj['date']}")
            click.echo(f"   Publication: {item_obj['publication']}")
            click.echo(f"   Volume: {item_obj['volume']}")
            click.echo(f"   Issue: {item_obj['issue']}")
            click.echo(f"   DOI: {item_obj['doi']}")
            click.echo(f"   URL: {item_obj['url']}")
            click.echo(f"   Key: {item_obj['key']}")

            if item_obj["pdfAttachments"]:
                click.echo("   PDF Attachments:")
                for pdf_url in item_obj["pdfAttachments"]:
                    click.echo(f"      {pdf_url}")

            click.echo()


@main.command()
@click.option(
    "--limit",
    type=int,
    help="Maximum number of collections to return (default: all)",
)
@click.pass_context
@cli_error_handler
def listcollections(ctx: Any, limit: int | None) -> None:
    """List all collections in the local Zotero library.

    Examples:
        pyzotero listcollections

        pyzotero listcollections --limit 10

    """
    zot = _zot_from_ctx(ctx)

    # Build query parameters
    params = {}
    if limit:
        params["limit"] = limit

    # Get all collections
    collections = zot.collections(**params)

    if not collections:
        click.echo(json.dumps([]))
        return

    # Build a mapping of collection keys to names for parent lookup
    collection_map = {}
    for collection in collections:
        data = collection.get("data", {})
        key = data.get("key", "")
        name = data.get("name", "")
        if key:
            collection_map[key] = name or None

    # Build JSON output
    output = []
    for collection in collections:
        data = collection.get("data", {})
        meta = collection.get("meta", {})

        name = data.get("name", "")
        key = data.get("key", "")
        num_items = meta.get("numItems", 0)
        parent_collection = data.get("parentCollection", "")

        collection_obj = {
            "id": key,
            "name": name or None,
            "items": num_items,
        }

        # Add parent information if it exists
        if parent_collection:
            parent_name = collection_map.get(parent_collection)
            collection_obj["parent"] = {
                "id": parent_collection,
                "name": parent_name,
            }
        else:
            collection_obj["parent"] = None

        output.append(collection_obj)

    # Output as JSON
    click.echo(json.dumps(output, indent=2))


@main.command()
@click.pass_context
@cli_error_handler
def itemtypes(ctx: Any) -> None:
    """List all valid item types.

    Examples:
        pyzotero itemtypes

    """
    zot = _zot_from_ctx(ctx)

    # Get all item types
    item_types = zot.item_types()

    if not item_types:
        click.echo(json.dumps([]))
        return

    # Output as JSON array
    click.echo(json.dumps(item_types, indent=2))


@main.command()
@click.argument("itemtype")
@click.pass_context
@cli_error_handler
def listitemfields(ctx: Any, itemtype: str) -> None:
    """List the fields and creator types that one item type accepts.

    Run this before 'pyzotero createitem' to find the fields of an item
    type. 'pyzotero itemtypes' lists the types themselves.

    Examples:
        pyzotero listitemfields journalArticle

        pyzotero listitemfields book

    """
    zot = _zot_from_ctx(ctx)

    click.echo(json.dumps(describe_item_type(zot, itemtype), indent=2))


@main.command()
@click.option(
    "--app-name",
    default="Pyzotero",
    show_default=True,
    help="The name that Zotero shows in the authorisation dialog.",
)
@click.option(
    "--no-store",
    is_flag=True,
    help="Print the key only. Do not write it to the key file.",
)
@click.pass_context
@cli_error_handler
def authorize(ctx: Any, app_name: str, no_store: bool) -> None:
    """Get a local API key, which permits writes to your Zotero library.

    Zotero shows a dialog with the options "Allow" (one-time access),
    "Always Allow" (permanent access) and "Deny". Select "Always Allow" to
    get a key that you can keep: the first write uses a one-time key.

    A permanent key is stored in a file that only you can read
    ($XDG_CONFIG_HOME/pyzotero/local-api-key.json, or the same path under
    ~/.config). The CLI's write commands and the MCP server, when
    started with --enable-writes, read the key from that file. Setting
    PYZOTERO_LOCAL_API_KEY in the environment takes precedence over it.

    Local API keys have no relation to zotero.org API keys.

    Examples:
        pyzotero authorize
        pyzotero authorize --app-name "My MCP server"
        pyzotero authorize --no-store

    """
    zot = _zot_from_ctx(ctx)
    click.echo("Requesting authorisation. Confirm the dialog in Zotero...", err=True)
    result = zot.authorize_local(app_name)
    click.echo(f"Key:       {result['key']}")
    click.echo(f"Server ID: {zot.server_id}")
    if not result["remember"]:
        click.echo(
            "\nThis key is single-use: the first successful write consumes it.\n"
            "Re-run and choose 'Always Allow' if you need a persistent key.",
            err=True,
        )
        return
    if no_store:
        click.echo(
            "\nThis key persists. To give the MCP server write access, set it in\n"
            "the server's environment and pass --enable-writes:\n\n"
            '    "env": {"PYZOTERO_LOCAL_API_KEY": "' + result["key"] + '"},\n'
            '    "args": ["--enable-writes"]',
            err=True,
        )
        return
    path = save_local_key(result["key"], zot.server_id)
    click.echo(
        f"\nThis key persists. Stored it in {path}.\n"
        "The CLI's write commands use it. So does the MCP server when\n"
        "started with --enable-writes; no environment variable is needed.",
        err=True,
    )


@main.command()
@click.argument("source", type=click.File("r"))
@click.option(
    "--collection",
    help="Key of a collection. Every created item is filed under it.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Tag for every created item (can be specified multiple times)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def createitem(
    ctx: Any,
    source: IO[str],
    collection: str | None,
    tags: tuple[str, ...],
    output_json: bool,
) -> None:
    """Create one or more items from a JSON file, or from '-' for stdin.

    SOURCE holds one item object, or a list of them, in Zotero's item-data
    format: an "itemType" and the fields that the type accepts, for example
    {"itemType": "book", "title": "Frankenstein", "creators": [...]}. Run
    'pyzotero itemtypes' for the item types, and 'pyzotero listitemfields
    TYPE' for the fields of one type.

    The item type and the field names of every item are checked before
    anything is sent. One invalid item stops the whole batch, and the
    message names its position in the list. Zotero accepts up to 50 items
    in one call.

    Needs a stored local API key: run 'pyzotero authorize' first.

    Examples:
        pyzotero createitem item.json

        cat items.json | pyzotero createitem -

        pyzotero createitem items.json --collection FD9AUNP2 --tag "to read"

        pyzotero createitem items.json --json

    """
    zot = _write_zot_from_ctx(ctx)
    items = _read_items(source)
    if not items:
        msg = "No items to create"
        raise RuntimeError(msg)
    validate_items(zot, items)
    for item in items:
        _attach(item, tags, collection)
    resp = zot.create_items(items)
    success = resp.get("success") or {}
    created = [success[pos] for pos in sorted(success, key=int)]
    failed = resp.get("failed") or {}
    if failed:
        made = ", ".join(created) if created else "none"
        msg = (
            f"Zotero rejected {len(failed)} of {len(items)} items: {failed} "
            f"(created {len(created)}: {made})"
        )
        raise RuntimeError(msg)
    if output_json:
        click.echo(
            json.dumps(
                {
                    "created": created,
                    "collection": collection or None,
                    "tags": list(tags),
                },
                indent=2,
            )
        )
    else:
        click.echo(f"Created {len(created)} items: {', '.join(created)}")


@main.command()
@click.argument("name")
@click.option(
    "--parent",
    help="Key of the parent collection. The new collection nests under it.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def createcollection(
    ctx: Any, name: str, parent: str | None, output_json: bool
) -> None:
    """Create a collection, at the top level or under --parent.

    Needs a stored local API key: run 'pyzotero authorize' first.

    Examples:
        pyzotero createcollection "Frankenstein Cities"

        pyzotero createcollection "Frankenstein Cities" --parent FD9AUNP2 --json

    """
    zot = _write_zot_from_ctx(ctx)
    payload: dict[str, Any] = {"name": name}
    if parent:
        payload["parentCollection"] = parent
    resp = zot.create_collections([payload])
    if not resp.get("success"):
        msg = f"Collection was rejected: {resp.get('failed')}"
        raise RuntimeError(msg)
    key = resp["success"]["0"]
    if output_json:
        click.echo(
            json.dumps(
                {"created": key, "name": name, "parent": parent or None}, indent=2
            )
        )
    else:
        where = f" under {parent}" if parent else ""
        click.echo(f"Created collection {name!r} with key {key}{where}")


@main.command()
@click.argument("collection_key")
@click.argument("item_keys", nargs=-1, required=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def addtocollection(
    ctx: Any, collection_key: str, item_keys: tuple[str, ...], output_json: bool
) -> None:
    """Add one or more items to a collection.

    Needs a stored local API key: run 'pyzotero authorize' first.

    Examples:
        pyzotero addtocollection FD9AUNP2 ABC123 DEF456

        pyzotero addtocollection FD9AUNP2 ABC123 --json

    """
    zot = _write_zot_from_ctx(ctx)
    done = _change_membership(
        zot, item_keys, lambda item: zot.addto_collection(collection_key, item)
    )
    if output_json:
        click.echo(json.dumps({"collection": collection_key, "added": done}, indent=2))
    else:
        click.echo(f"Added {len(done)} items to {collection_key}: {', '.join(done)}")


@main.command()
@click.argument("collection_key")
@click.argument("item_keys", nargs=-1, required=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def removefromcollection(
    ctx: Any, collection_key: str, item_keys: tuple[str, ...], output_json: bool
) -> None:
    """Remove one or more items from a collection. The items are unchanged.

    Needs a stored local API key: run 'pyzotero authorize' first.

    Examples:
        pyzotero removefromcollection FD9AUNP2 ABC123 DEF456

        pyzotero removefromcollection FD9AUNP2 ABC123 --json

    """
    zot = _write_zot_from_ctx(ctx)
    done = _change_membership(
        zot, item_keys, lambda item: zot.deletefrom_collection(collection_key, item)
    )
    if output_json:
        click.echo(
            json.dumps({"collection": collection_key, "removed": done}, indent=2)
        )
    else:
        click.echo(
            f"Removed {len(done)} items from {collection_key}: {', '.join(done)}"
        )


@main.command()
@click.argument("item_keys", nargs=-1, required=True)
@click.option(
    "--from",
    "from_collection",
    required=True,
    help="Key of the collection to remove the items from",
)
@click.option(
    "--to",
    "to_collection",
    required=True,
    help="Key of the collection to add the items to",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def movetocollection(
    ctx: Any,
    item_keys: tuple[str, ...],
    from_collection: str,
    to_collection: str,
    output_json: bool,
) -> None:
    """Move one or more items from one collection to another.

    Each item leaves the --from collection and joins the --to collection in
    one request. Membership of other collections is unchanged. An item that
    is not in the --from collection still joins the --to collection.

    Needs a stored local API key: run 'pyzotero authorize' first.

    Examples:
        pyzotero movetocollection --from FD9AUNP2 --to X7Y8Z9W0 ABC123 DEF456

        pyzotero movetocollection --from FD9AUNP2 --to X7Y8Z9W0 ABC123 --json

    """
    zot = _write_zot_from_ctx(ctx)
    done = _change_membership(
        zot,
        item_keys,
        lambda item: zot.moveto_collection(from_collection, to_collection, item),
    )
    if output_json:
        click.echo(
            json.dumps(
                {"from": from_collection, "to": to_collection, "moved": done},
                indent=2,
            )
        )
    else:
        click.echo(
            f"Moved {len(done)} items from {from_collection} to {to_collection}: "
            f"{', '.join(done)}"
        )


@main.command()
@click.pass_context
@cli_error_handler
def test(ctx: Any) -> None:
    """Test connection to local Zotero instance.

    This command checks whether Zotero is running and accepting local connections.

    Examples:
        pyzotero test

    """
    zot = _zot_from_ctx(ctx)

    try:
        # Call settings() to test the connection
        # This should return {} if Zotero is running and listening
        result = zot.settings()
    except httpx2.ConnectError:
        click.echo(
            "✗ Connection failed: Could not connect to Zotero.\n\n"
            "Possible causes:\n"
            "  • Zotero might not be running\n"
            "  • Local connections might not be enabled\n\n"
            "To enable local connections:\n"
            "  Zotero > Settings > Advanced > Allow other applications on this computer to communicate with Zotero",
            err=True,
        )
        sys.exit(1)

    # If we get here, the connection succeeded
    click.echo("✓ Connection successful: Zotero is running and listening locally.")
    if result == {}:
        click.echo("  Received expected empty settings response.")
    else:
        click.echo(f"  Received response: {json.dumps(result)}")


@main.command()
@click.argument("key")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def item(ctx: Any, key: str, output_json: bool) -> None:
    """Get a single item by its key.

    Returns full item data for the specified Zotero item key.

    Examples:
        pyzotero item ABC123

        pyzotero item ABC123 --json

    """
    zot = _zot_from_ctx(ctx)

    # Fetch the item
    result = zot.item(key)

    if not result:
        if output_json:
            click.echo(json.dumps(None))
        else:
            click.echo(f"Item not found: {key}")
        return

    data = result.get("data", {})

    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        title = data.get("title", "No title")
        item_type = data.get("itemType", "Unknown")
        date = data.get("date", "No date")
        item_key = data.get("key", "")
        doi = data.get("DOI", "")
        url = data.get("url", "")

        # Format creators
        creator_names = format_creators(data.get("creators", []))
        authors_str = ", ".join(creator_names) if creator_names else "No authors"

        click.echo(f"[{item_type}] {title}")
        click.echo(f"Authors: {authors_str}")
        click.echo(f"Date: {date}")
        click.echo(f"DOI: {doi}")
        click.echo(f"URL: {url}")
        click.echo(f"Key: {item_key}")


@main.command()
@click.argument("key")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def children(ctx: Any, key: str, output_json: bool) -> None:
    """Get child items (attachments, notes) of a specific item.

    Returns all child items for the specified Zotero item key.
    Useful for finding PDF attachments without the N+1 overhead during search.

    Examples:
        pyzotero children ABC123

        pyzotero children ABC123 --json

    """
    zot = _zot_from_ctx(ctx)

    # Fetch children
    results = zot.children(key)

    if not results:
        if output_json:
            click.echo(json.dumps([]))
        else:
            click.echo(f"No children found for item: {key}")
        return

    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo(f"\nFound {len(results)} child items:\n")
        for idx, child in enumerate(results, 1):
            data = child.get("data", {})
            item_type = data.get("itemType", "Unknown")
            child_key = data.get("key", "")
            title = data.get("title", data.get("note", "No title")[:50] + "...")
            content_type = data.get("contentType", "")

            click.echo(f"{idx}. [{item_type}] {title}")
            click.echo(f"   Key: {child_key}")
            if content_type:
                click.echo(f"   Content-Type: {content_type}")

            # Show file URL for attachments
            file_url = child.get("links", {}).get("enclosure", {}).get("href", "")
            if file_url:
                click.echo(f"   File: {file_url}")
            click.echo()


@main.command()
@click.option(
    "--collection",
    help="Filter tags to a specific collection key",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def tags(ctx: Any, collection: str | None, output_json: bool) -> None:
    """List all tags in the library.

    Returns all tags used in the library, or only tags from a specific collection.

    Examples:
        pyzotero tags

        pyzotero tags --collection ABC123

        pyzotero tags --json

    """
    zot = _zot_from_ctx(ctx)

    # Fetch tags
    if collection:
        results = zot.collection_tags(collection)
    else:
        results = zot.tags()

    if not results:
        if output_json:
            click.echo(json.dumps([]))
        else:
            click.echo("No tags found.")
        return

    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo(f"\nFound {len(results)} tags:\n")
        for tag in sorted(results):
            click.echo(f"  {tag}")


@main.command()
@click.argument("keys", nargs=-1, required=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def subset(ctx: Any, keys: tuple[str, ...], output_json: bool) -> None:
    """Get multiple items by their keys in a single call.

    Efficiently retrieve up to 50 items by key in a single API call.
    Far more efficient than multiple individual item lookups.

    Examples:
        pyzotero subset ABC123 DEF456 GHI789

        pyzotero subset ABC123 DEF456 --json

    """
    zot = _zot_from_ctx(ctx)

    if len(keys) > 50:  # noqa: PLR2004 - Zotero API limit
        click.echo("Error: Maximum 50 items per call.", err=True)
        sys.exit(1)

    # Fetch items
    results = zot.get_subset(list(keys))

    if not results:
        if output_json:
            click.echo(json.dumps([]))
        else:
            click.echo("No items found.")
        return

    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo(f"\nFound {len(results)} items:\n")
        for idx, item in enumerate(results, 1):
            data = item.get("data", {})
            title = data.get("title", "No title")
            item_type = data.get("itemType", "Unknown")
            item_key = data.get("key", "")

            click.echo(f"{idx}. [{item_type}] {title}")
            click.echo(f"   Key: {item_key}")
            click.echo()


@main.command()
@click.argument("dois", nargs=-1)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
@cli_error_handler
def alldoi(ctx: Any, dois: tuple[str, ...], output_json: bool) -> None:  # noqa: PLR0912
    """Look up DOIs in the local Zotero library and return their Zotero IDs.

    Accepts one or more DOIs as arguments and checks if they exist in the library.
    DOI matching is case-insensitive and handles common prefixes (https://doi.org/, doi:).

    If no DOIs are provided, shows "No items found" (text) or {} (JSON).

    Examples:
        pyzotero alldoi 10.1234/example

        pyzotero alldoi 10.1234/abc https://doi.org/10.5678/def doi:10.9012/ghi

        pyzotero alldoi 10.1234/example --json

    """
    zot = _zot_from_ctx(ctx)

    click.echo("Building DOI index from library...", err=True)
    doi_map = build_doi_index_full(zot)
    click.echo(f"Indexed {len(doi_map)} items with DOIs", err=True)

    # If no DOIs provided, return empty result
    if not dois:
        if output_json:
            click.echo(json.dumps({}))
        else:
            click.echo("No items found")
        return

    # Look up each input DOI
    found = []
    not_found = []

    for input_doi in dois:
        entry = doi_map.get(normalise_doi(input_doi))
        if entry is not None:
            found.append({"doi": entry["original"], "key": entry["key"]})
        else:
            not_found.append(input_doi)

    # Output results
    if output_json:
        result = {"found": found, "not_found": not_found}
        click.echo(json.dumps(result, indent=2))
    else:
        if found:
            click.echo(f"\nFound {len(found)} items:\n")
            for item in found:
                click.echo(f"  {item['doi']} → {item['key']}")
        else:
            click.echo("No items found")

        if not_found:
            click.echo(f"\nNot found ({len(not_found)}):")
            for doi in not_found:
                click.echo(f"  {doi}")


@main.command()
@click.pass_context
@cli_error_handler
def doiindex(ctx: Any) -> None:
    """Output the complete DOI-to-key mapping for the library.

    Returns a JSON mapping of normalised DOIs to item keys and original DOIs.
    This allows the skill to cache the index and avoid repeated full-library scans.

    Output format:
        {
          "10.1234/abc": {"key": "ABC123", "original": "https://doi.org/10.1234/ABC"},
          ...
        }

    Examples:
        pyzotero doiindex

        pyzotero doiindex > doi_cache.json

    """
    zot = _zot_from_ctx(ctx)

    click.echo("Building DOI index from library...", err=True)
    doi_map = build_doi_index_full(zot)
    click.echo(f"Indexed {len(doi_map)} items with DOIs", err=True)

    click.echo(json.dumps(doi_map, indent=2))


@main.command()
@click.argument("key")
@click.pass_context
@cli_error_handler
def fulltext(ctx: Any, key: str) -> None:
    """Get full-text content of an attachment.

    Returns the full-text content extracted from a PDF or other attachment.
    The key should be the key of an attachment item (not a top-level item).

    Output format:
        {
          "content": "Full-text extracted from PDF...",
          "indexedPages": 50,
          "totalPages": 50
        }

    Examples:
        pyzotero fulltext ABC123

    """
    zot = _zot_from_ctx(ctx)

    result = zot.fulltext_item(key)

    if not result:
        click.echo(json.dumps({"error": "No full-text content available"}))
        return

    click.echo(json.dumps(result, indent=2))


@main.command()
@click.option(
    "--doi",
    required=True,
    help="DOI of the paper to find related papers for",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of results to return (default: 20, max: 500)",
)
@click.option(
    "--min-citations",
    type=int,
    default=0,
    help="Minimum citation count filter (default: 0)",
)
@click.option(
    "--check-library/--no-check-library",
    default=True,
    help="Check if papers exist in local Zotero (default: True)",
)
@click.pass_context
@cli_error_handler
def related(
    ctx: Any, doi: str, limit: int, min_citations: int, check_library: bool
) -> None:
    """Find papers related to a given paper using Semantic Scholar.

    Uses SPECTER2 embeddings to find semantically similar papers.

    Examples:
        pyzotero related --doi "10.1038/nature12373"

        pyzotero related --doi "10.1038/nature12373" --limit 50

        pyzotero related --doi "10.1038/nature12373" --min-citations 100

    """
    _run_s2_lookup(
        ctx,
        doi,
        limit,
        min_citations,
        check_library,
        get_recommendations,
        "related papers",
    )


@main.command()
@click.option(
    "--doi",
    required=True,
    help="DOI of the paper to find citations for",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results to return (default: 100, max: 1000)",
)
@click.option(
    "--min-citations",
    type=int,
    default=0,
    help="Minimum citation count filter (default: 0)",
)
@click.option(
    "--check-library/--no-check-library",
    default=True,
    help="Check if papers exist in local Zotero (default: True)",
)
@click.pass_context
@cli_error_handler
def citations(
    ctx: Any, doi: str, limit: int, min_citations: int, check_library: bool
) -> None:
    """Find papers that cite a given paper using Semantic Scholar.

    Examples:
        pyzotero citations --doi "10.1038/nature12373"

        pyzotero citations --doi "10.1038/nature12373" --limit 50

        pyzotero citations --doi "10.1038/nature12373" --min-citations 50

    """
    _run_s2_lookup(
        ctx, doi, limit, min_citations, check_library, get_citations, "citations"
    )


@main.command()
@click.option(
    "--doi",
    required=True,
    help="DOI of the paper to find references for",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results to return (default: 100, max: 1000)",
)
@click.option(
    "--min-citations",
    type=int,
    default=0,
    help="Minimum citation count filter (default: 0)",
)
@click.option(
    "--check-library/--no-check-library",
    default=True,
    help="Check if papers exist in local Zotero (default: True)",
)
@click.pass_context
@cli_error_handler
def references(
    ctx: Any, doi: str, limit: int, min_citations: int, check_library: bool
) -> None:
    """Find papers referenced by a given paper using Semantic Scholar.

    Examples:
        pyzotero references --doi "10.1038/nature12373"

        pyzotero references --doi "10.1038/nature12373" --limit 50

        pyzotero references --doi "10.1038/nature12373" --min-citations 100

    """
    _run_s2_lookup(
        ctx, doi, limit, min_citations, check_library, get_references, "references"
    )


@main.command()
@click.option(
    "-q",
    "--query",
    required=True,
    help="Search query string",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of results to return (default: 20, max: 100)",
)
@click.option(
    "--year",
    help="Year filter (e.g., '2020', '2018-2022', '2020-')",
)
@click.option(
    "--open-access/--no-open-access",
    default=False,
    help="Only return open access papers (default: False)",
)
@click.option(
    "--sort",
    type=click.Choice(["citations", "year"], case_sensitive=False),
    help="Sort results by citation count or year (descending)",
)
@click.option(
    "--min-citations",
    type=int,
    default=0,
    help="Minimum citation count filter (default: 0)",
)
@click.option(
    "--check-library/--no-check-library",
    default=True,
    help="Check if papers exist in local Zotero (default: True)",
)
@click.pass_context
@cli_error_handler
def s2search(
    ctx: Any,
    query: str,
    limit: int,
    year: str | None,
    open_access: bool,
    sort: str | None,
    min_citations: int,
    check_library: bool,
) -> None:
    """Search for papers on Semantic Scholar.

    Search across Semantic Scholar's index of over 200M papers.

    Examples:
        pyzotero s2search -q "climate adaptation"

        pyzotero s2search -q "machine learning" --year 2020-2024

        pyzotero s2search -q "neural networks" --open-access --limit 50

        pyzotero s2search -q "deep learning" --sort citations --min-citations 100

    """
    # Search Semantic Scholar
    click.echo(f'Searching Semantic Scholar for: "{query}"...', err=True)
    result = search_papers(
        query,
        limit=limit,
        year=year,
        open_access_only=open_access,
        sort=sort,
        min_citations=min_citations,
    )
    papers = result.get("papers", [])
    total = result.get("total", len(papers))

    if not papers:
        click.echo(json.dumps({"count": 0, "total": total, "papers": []}))
        return

    # Optionally annotate with library status
    if check_library:
        click.echo("Checking local Zotero library...", err=True)
        zot = _zot_from_ctx(ctx)
        doi_map = build_doi_index(zot)
        output_papers = annotate_with_library(papers, doi_map)
    else:
        output_papers = [format_s2_paper(p) for p in papers]

    click.echo(
        json.dumps(
            {"count": len(output_papers), "total": total, "papers": output_papers},
            indent=2,
        )
    )


SECRET_SETTINGS = ("api_key", "webdav_password")
API_KEY_INFO_URL = "https://api.zotero.org/keys/current"


def _masked(settings: Settings) -> dict[str, Any]:
    shown = settings.to_file_dict()
    for name in SECRET_SETTINGS:
        if shown.get(name):
            shown[name] = "****" + shown[name][-4:]
    return shown


def _ask(
    value: str | None, text: str, default: str | None, secret: bool = False
) -> str | None:
    """Return ``value`` if given, else prompt with ``default``. Blank keeps None."""
    if value is not None:
        return value or None
    if secret and default:
        answer = click.prompt(
            f"{text} (Enter keeps the stored one)",
            default="",
            hide_input=True,
            show_default=False,
        )
        return answer or default
    answer = click.prompt(
        text, default=default or "", hide_input=secret, show_default=bool(default)
    )
    return answer or None


def _check_api_key(api_key: str) -> dict[str, Any]:
    """Return the key's info from zotero.org. Raise RuntimeError if invalid."""
    resp = httpx2.get(API_KEY_INFO_URL, headers={"Zotero-API-Key": api_key}, timeout=30)
    if resp.status_code == 403:  # noqa: PLR2004
        msg = "zotero.org rejected the API key"
        raise RuntimeError(msg)
    resp.raise_for_status()
    return resp.json()


@main.command()
@click.option(
    "--mode",
    type=click.Choice(MODES),
    help="local (desktop app) or remote (zotero.org API)",
)
@click.option("--api-key", help="zotero.org API key (remote mode)")
@click.option("--library-id", help="Library ID; default: the API key's user ID")
@click.option(
    "--library-type", type=click.Choice(["user", "group"]), help="Default: user"
)
@click.option(
    "--storage",
    type=click.Choice(STORAGES),
    help="zotero (Zotero File Storage) or webdav",
)
@click.option("--webdav-url", help="WebDAV URL as entered in Zotero, without /zotero")
@click.option("--webdav-username")
@click.option("--webdav-password")
@click.option(
    "--unpaywall-email", help="Email sent to Unpaywall for DOI lookups (optional)"
)
@click.option(
    "--no-input", is_flag=True, help="Never prompt; use the options and stored values"
)
@click.option(
    "--no-check",
    is_flag=True,
    help="Save without testing the API key and WebDAV server",
)
@click.option(
    "--show", is_flag=True, help="Print the current settings (secrets masked) and exit"
)
@cli_error_handler
def setup(
    mode: str | None,
    api_key: str | None,
    library_id: str | None,
    library_type: str | None,
    storage: str | None,
    webdav_url: str | None,
    webdav_username: str | None,
    webdav_password: str | None,
    unpaywall_email: str | None,
    no_input: bool,
    no_check: bool,
    show: bool,
) -> None:
    """Choose the library mode and file storage, and save the credentials.

    Options that are not given are prompted for, with the stored value as the
    default. With --no-input nothing is prompted, so scripts and agents can
    run it. The API key and the WebDAV server are tested before anything is
    saved. Environment variables override the saved file:

    \b
    PYZOTERO_MODE, PYZOTERO_API_KEY, PYZOTERO_LIBRARY_ID,
    PYZOTERO_LIBRARY_TYPE, PYZOTERO_STORAGE, PYZOTERO_WEBDAV_URL,
    PYZOTERO_WEBDAV_USERNAME, PYZOTERO_WEBDAV_PASSWORD,
    PYZOTERO_UNPAYWALL_EMAIL

    Examples:
        pyzotero setup

        pyzotero setup --show

        pyzotero setup --no-input --mode remote --api-key KEY \\
            --storage webdav --webdav-url https://dav.example.org \\
            --webdav-username me --webdav-password secret

    """
    current = load_settings()
    if show:
        click.echo(
            json.dumps(
                {"path": str(config_path()), "settings": _masked(current)}, indent=2
            )
        )
        return
    if no_input:

        def ask(
            value: str | None, _text: str, default: str | None, secret: bool = False
        ) -> str | None:
            return default if value is None else (value or None)
    else:
        ask = _ask
    mode = ask(mode, "Mode (local, remote)", current.mode) or "local"
    values: dict[str, Any] = {"mode": mode}
    if mode == "remote":
        values["api_key"] = ask(
            api_key,
            "zotero.org API key (zotero.org/settings/keys)",
            current.api_key,
            secret=True,
        )
        if not values["api_key"]:
            msg = "Remote mode needs an API key"
            raise RuntimeError(msg)
        info = {} if no_check else _check_api_key(values["api_key"])
        if info:
            click.echo(
                f"API key OK: user {info.get('username')} ({info.get('userID')})",
                err=True,
            )
            access = info.get("access", {}).get("user", {})
            if not access.get("write"):
                click.echo("Warning: this key cannot write to your library.", err=True)
        values["library_type"] = (
            ask(library_type, "Library type (user, group)", current.library_type)
            or "user"
        )
        default_id = current.library_id or (
            str(info["userID"])
            if values["library_type"] == "user" and "userID" in info
            else None
        )
        values["library_id"] = ask(library_id, "Library ID", default_id)
    values["storage"] = (
        ask(storage, "File storage (zotero, webdav)", current.storage) or "zotero"
    )
    if values["storage"] == "webdav":
        values["webdav_url"] = ask(
            webdav_url, "WebDAV URL (without /zotero)", current.webdav_url
        )
        values["webdav_username"] = ask(
            webdav_username, "WebDAV user name", current.webdav_username
        )
        values["webdav_password"] = ask(
            webdav_password, "WebDAV password", current.webdav_password, secret=True
        )
    values["unpaywall_email"] = ask(
        unpaywall_email, "Email for Unpaywall (blank to skip)", current.unpaywall_email
    )
    new = Settings(**{k: v for k, v in values.items() if v is not None})
    if new.mode == "remote":
        new.require_remote()
    if new.storage == "webdav":
        credentials = new.webdav_credentials()
        if not no_check:
            WebDAVStorage(*credentials).check()
            click.echo("WebDAV server OK: readable and writable", err=True)
    path = save_settings(new)
    overridden = [var for var in ENV_VARS.values() if os.environ.get(var)]
    click.echo(f"Saved settings to {path}")
    if overridden:
        click.echo(
            f"Note: these environment variables override the file: {', '.join(overridden)}",
            err=True,
        )


@main.command()
@click.argument("parent")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--title", help="Attachment title; default: the file name")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def attach(
    ctx: Any, parent: str, file: Path, title: str | None, output_json: bool
) -> None:
    """Attach FILE to the item with key PARENT.

    The file goes to the configured storage (see 'pyzotero setup'). If the
    identical file is already attached, nothing is uploaded.

    Examples:
        pyzotero attach ABC12345 paper.pdf

        pyzotero attach ABC12345 paper.pdf --title "Preprint" --json

    """
    result = _files.attach(_write_zot_from_ctx(ctx), parent, file.resolve(), title)
    if output_json:
        click.echo(json.dumps(result, indent=2))
    elif "error" in result:
        click.echo(f"Error: {result['error']}: {result['detail']}", err=True)
    elif "unchanged" in result:
        click.echo(f"Already attached: {result['unchanged']}")
    else:
        click.echo(f"Attached {result['filename']} as {result['attached']}")
    if "error" in result:
        sys.exit(1)


@main.command()
@click.argument("key")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(),
    help="Directory for the file (default: current directory)",
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def download(ctx: Any, key: str, out_dir: Path, output_json: bool) -> None:
    """Download the file of attachment KEY, or the PDF of item KEY.

    For a regular item, its first PDF attachment is used, or else its first
    stored attachment.

    Examples:
        pyzotero download ABC12345 --out papers/

        pyzotero download ABC12345 --json

    """
    result = _files.download(_zot_from_ctx(ctx), key, out_dir)
    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(result["path"])


@main.command()
@click.argument("keys", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Attach even if the item already has a PDF")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def fetchpdf(ctx: Any, keys: tuple[str, ...], force: bool, output_json: bool) -> None:
    """Find an open-access PDF for each item KEY and attach it.

    arXiv is tried first (from the item's arXiv ID), then Unpaywall (from its
    DOI; needs unpaywall_email, see 'pyzotero setup'). Items that already
    have a PDF are skipped unless --force is given. A failure for one item is
    reported and the others continue.

    Examples:
        pyzotero fetchpdf ABC12345

        pyzotero fetchpdf ABC12345 DEF67890 --json

    """
    zot = _write_zot_from_ctx(ctx)
    results = []
    for key in keys:
        try:
            results.append({"key": key, **_files.fetch_pdf(zot, key, force=force)})
        except Exception as exc:  # noqa: PERF203 -- report per item, continue
            results.append({"key": key, "error": str(exc)})
    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                click.echo(f"{r['key']}: error: {r['error']}")
            elif "unchanged" in r:
                click.echo(f"{r['key']}: already has a PDF ({r['unchanged']})")
            else:
                click.echo(f"{r['key']}: attached {r['attached']} from {r['source']}")
    if any("error" in r for r in results):
        sys.exit(1)


@main.command()
@click.option(
    "--hashes",
    is_flag=True,
    help="Also compare each .prop hash with the item's md5 (one request per file)",
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def storagecheck(ctx: Any, hashes: bool, output_json: bool) -> None:
    """Compare the library's attachments with the files on the WebDAV server.

    Reports attachments whose file is missing on the server, attachments no
    client has uploaded yet, files on the server that no attachment owns
    (orphaned), and, with --hashes, files whose hash differs from the item.
    Nothing is changed. Exits 1 if anything is missing or mismatched.

    Examples:
        pyzotero storagecheck

        pyzotero storagecheck --hashes --json

    """
    storage = get_webdav_storage()
    if storage is None:
        msg = "storagecheck needs storage = webdav (see 'pyzotero setup')"
        raise RuntimeError(msg)
    report = check_storage(_zot_from_ctx(ctx), storage, verify_hashes=hashes).as_dict()
    if output_json:
        click.echo(json.dumps(report, indent=2))
    else:
        for name, keys in report.items():
            line = f"{name}: {len(keys)}"
            if keys and name != "ok":
                line += "  " + " ".join(keys[:20]) + (" ..." if len(keys) > 20 else "")  # noqa: PLR2004
            click.echo(line)
    if report["missing"] or report["mismatched"]:
        sys.exit(1)


def _brief(item: dict[str, Any]) -> dict[str, Any]:
    data = item["data"]
    out = {
        "key": item["key"],
        "itemType": data.get("itemType"),
        "title": data.get("title"),
        "date": data.get("date"),
        "DOI": data.get("DOI") or None,
        "dateAdded": data.get("dateAdded"),
        "children": item.get("meta", {}).get("numChildren", 0),
    }
    return out


@main.command()
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def duplicates(ctx: Any, output_json: bool) -> None:
    """List groups of duplicate items, using Zotero desktop's rules.

    Items match on DOI, on ISBN (books), or on a normalized title, unless
    their DOIs differ, their years are more than one apart, or no author
    matches (last name and first initial). Each group lists the oldest item
    first: 'pyzotero merge' keeps it by default. The "basis" says what the
    items share; check "isbn" groups by hand, since a proceedings volume and
    one of its chapters share an ISBN.

    Examples:
        pyzotero duplicates

        pyzotero duplicates --json

    """
    zot = _zot_from_ctx(ctx)
    groups = find_duplicates(zot.everything(zot.items(limit=100)))
    result: list[dict[str, Any]] = [
        {"basis": match_basis(g), "items": [_brief(i) for i in g]} for g in groups
    ]
    if output_json:
        click.echo(json.dumps(result, indent=2))
        return
    for group in result:
        click.echo(f"[{group['basis']}]")
        for i in group["items"]:
            click.echo(
                f"  {i['key']}  {i['itemType']:<16} added {(i['dateAdded'] or '')[:10]}"
                f"  {i['children']} children  {(i['title'] or '')[:70]}"
            )
    click.echo(f"{len(result)} groups")


@main.command()
@click.argument("keys", nargs=-1, required=True)
@click.option(
    "--master", help="Key of the item to keep (default: the oldest by dateAdded)"
)
@click.option(
    "--fill-empty",
    is_flag=True,
    help="Copy fields that are empty on the master from the other items",
)
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    help="Make the changes. Without it, only the plan is shown",
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
@cli_error_handler
def merge(
    ctx: Any,
    keys: tuple[str, ...],
    master: str | None,
    fill_empty: bool,
    do_apply: bool,
    output_json: bool,
) -> None:
    """Merge duplicate items KEYS into one, as Zotero desktop does.

    The master keeps its fields and gets the others' notes, attachments,
    collections, tags and relations, and the earliest date added. A PDF
    that is byte-identical to one of the master's is trashed, and its
    annotations move to the master's copy. The other items go to the trash,
    from where Zotero can restore them.

    Without --apply, nothing changes: the plan is printed.

    Examples:
        pyzotero merge LDBZ4RJ7 N7WHM5XS

        pyzotero merge LDBZ4RJ7 N7WHM5XS --master N7WHM5XS --apply

    """
    if len(set(keys)) < 2:  # noqa: PLR2004
        msg = "Give at least two different item keys"
        raise RuntimeError(msg)
    zot = _write_zot_from_ctx(ctx) if do_apply else _zot_from_ctx(ctx)
    if master is None:
        items = [zot.item(k) for k in keys]
        master = min(items, key=lambda i: i["data"].get("dateAdded", ""))["key"]
    elif master not in keys:
        msg = f"--master {master} is not one of the given keys"
        raise RuntimeError(msg)
    plan = plan_merge(
        zot, master, [k for k in keys if k != master], fill_empty=fill_empty
    )
    if do_apply:
        apply_merge(zot, plan)
    summary: dict[str, Any] = {"applied": do_apply, **plan.summary()}
    if output_json:
        click.echo(json.dumps(summary, indent=2))
        return
    m = summary["master"]
    click.echo(f"Keep:   {m['key']}  {m['title']}")
    for t in summary["trashed"]:
        click.echo(f"Trash:  {t['key']}  {t['title']}")
    click.echo(
        f"Master fields changed: {', '.join(summary['master_updates']) or 'none'}"
    )
    click.echo(
        f"Children moved: {len(summary['moved_children'])}; duplicate PDFs trashed: {len(summary['trashed_attachments'])}"
    )
    if summary["repointed_relations"]:
        click.echo(
            f"Related-item links updated on: {', '.join(summary['repointed_relations'])}"
        )
    click.echo("Merged." if do_apply else "Nothing changed. Add --apply to merge.")


if __name__ == "__main__":
    main()
