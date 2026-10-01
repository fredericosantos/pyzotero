"""Find and merge duplicate items, as Zotero desktop's Duplicate Items pane does.

Detection follows ``chrome/content/zotero/xpcom/duplicates.js``, and merging
follows ``chrome/content/zotero/mergeItems.mjs`` in the Zotero client. Both
work on item data from the Web API (or the local API), so no desktop app or
database file is needed.

A merge keeps one item (the master) and moves the others to the trash, from
where they can be restored. The master gets the others' child notes and
attachments, collections, tags and relations, and the earliest dateAdded.
It also gets a ``dc:replaces`` relation to each merged item, so that other
clients and group libraries can follow the merge.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from . import errors as ze

if TYPE_CHECKING:
    from ._client import Zotero

REPLACES = "dc:replaces"
# Related items. zotero.org keeps this relation symmetric: a link stays while
# either side has it, and goes from both sides when one side removes it.
RELATED = "dc:relation"
# Item types that are never duplicates of anything (Zotero skips them too).
NON_REGULAR_TYPES = frozenset({"attachment", "note", "annotation"})
WEB_LINK_MODES = frozenset({"imported_url", "linked_url"})
# Fields that a merge never copies from another item.
SYSTEM_FIELDS = frozenset(
    {
        "key",
        "version",
        "itemType",
        "dateAdded",
        "dateModified",
        "collections",
        "tags",
        "relations",
        "creators",
        "deleted",
        "parentItem",
    }
)


def normalize(text: str) -> str:
    """Normalize a title or name the way Zotero's duplicate check does."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    # ASCII punctuation becomes a space: the ranges in Zotero's regex.
    text = re.sub(r"[ !-/:-@\[-`{-~]+", " ", text)
    return text.strip().lower()


def _isbn13(raw: str) -> str | None:
    """Return the first ISBN in ``raw`` as ISBN-13, or None."""
    for candidate in re.split(r"[\s,;]+", raw):
        digits = re.sub(r"[^0-9Xx]", "", candidate).upper()
        if len(digits) == 13 and digits.isdigit():  # noqa: PLR2004
            return digits
        if len(digits) == 10:  # noqa: PLR2004
            core = "978" + digits[:9]
            check = (
                10 - sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(core)) % 10
            ) % 10
            return core + str(check)
    return None


def _year(data: dict[str, Any]) -> int | None:
    match = re.search(r"\b(\d{4})\b", data.get("date") or "")
    return int(match.group(1)) if match else None


def _creator_keys(data: dict[str, Any]) -> set[tuple[str, str]]:
    keys = set()
    for creator in data.get("creators") or []:
        if "name" in creator:  # single-field name: no first initial
            keys.add((normalize(creator["name"]), ""))
        else:
            first = normalize(creator.get("firstName", ""))
            keys.add((normalize(creator.get("lastName", "")), first[:1]))
    return keys


def _doi(data: dict[str, Any]) -> str | None:
    doi = (data.get("DOI") or "").strip().lower()
    if not doi:
        # Some item types have no DOI field; Zotero users put it in Extra.
        match = re.search(
            r"^doi:\s*(10\.\S+)", data.get("extra") or "", re.IGNORECASE | re.MULTILINE
        )
        doi = match.group(1).lower() if match else ""
    return doi.removeprefix("https://doi.org/") or None


def _title_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Apply Zotero's checks to two items whose titles normalize equal."""
    doi_a, doi_b = _doi(a), _doi(b)
    if doi_a and doi_b and doi_a != doi_b:
        return False
    isbn_a, isbn_b = _isbn13(a.get("ISBN") or ""), _isbn13(b.get("ISBN") or "")
    if isbn_a and isbn_b and isbn_a != isbn_b:
        return False
    year_a, year_b = _year(a), _year(b)
    if year_a and year_b and abs(year_a - year_b) > 1:
        return False
    creators_a, creators_b = _creator_keys(a), _creator_keys(b)
    if not creators_a and not creators_b:
        return True
    return bool(creators_a & creators_b)


def find_duplicates(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return groups of duplicate items, each sorted oldest first.

    ``items`` are items from the API (with a ``data`` key). Attachments,
    notes, annotations and trashed items are skipped.
    """
    regular = [
        i
        for i in items
        if i["data"].get("itemType") not in NON_REGULAR_TYPES
        and not i["data"].get("deleted")
    ]
    parent = list(range(len(regular)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, item in enumerate(regular):
        data = item["data"]
        if doi := _doi(data):
            by_key["doi", doi].append(idx)
        isbn = (
            _isbn13(data.get("ISBN") or "") if data.get("itemType") == "book" else None
        )
        if isbn:
            by_key["isbn", isbn].append(idx)
        if title := normalize(data.get("title") or ""):
            by_key["title", title].append(idx)
    for (kind, _), idxs in by_key.items():
        for pos, i in enumerate(idxs):
            for j in idxs[pos + 1 :]:
                if kind != "title" or _title_match(
                    regular[i]["data"], regular[j]["data"]
                ):
                    union(i, j)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(regular):
        groups[find(idx)].append(item)
    return sorted(
        (
            sorted(g, key=lambda i: i["data"].get("dateAdded", ""))
            for g in groups.values()
            if len(g) > 1
        ),
        key=lambda g: normalize(g[0]["data"].get("title") or ""),
    )


def match_basis(group: list[dict[str, Any]]) -> str:
    """Return what the items in a group share: ``doi``, ``title`` or ``isbn``.

    An ``isbn``-only match is the least certain: a proceedings volume and a
    chapter filed as a book share the volume's ISBN.
    """
    datas = [i["data"] for i in group]
    if len({_doi(d) for d in datas}) == 1 and _doi(datas[0]):
        return "doi"
    if len({normalize(d.get("title") or "") for d in datas}) == 1:
        return "title"
    return "isbn" if all(_isbn13(d.get("ISBN") or "") for d in datas) else "mixed"


@dataclasses.dataclass
class MergePlan:
    """The changes a merge makes. :func:`plan_merge` builds one."""

    master: dict[str, Any]
    others: list[dict[str, Any]]
    master_updates: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Field -> new value on the master."""
    moved_children: list[str] = dataclasses.field(default_factory=list)
    """Keys of notes and attachments that get the master as parent."""
    trashed_attachments: list[str] = dataclasses.field(default_factory=list)
    """Keys of attachments that duplicate one of the master's. Their child
    annotations and notes move to the matching master attachment."""
    repointed: list[str] = dataclasses.field(default_factory=list)
    """Keys of other items whose relations pointed at a merged item."""
    # Internal: the full write payloads, in order.
    writes: list[dict[str, Any]] = dataclasses.field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the plan."""

        def brief(item: dict[str, Any]) -> dict[str, Any]:
            data = item["data"]
            return {
                "key": item["key"],
                "itemType": data.get("itemType"),
                "title": data.get("title"),
                "date": data.get("date"),
                "dateAdded": data.get("dateAdded"),
            }

        return {
            "master": brief(self.master),
            "trashed": [brief(o) for o in self.others],
            "master_updates": sorted(self.master_updates),
            "moved_children": self.moved_children,
            "trashed_attachments": self.trashed_attachments,
            "repointed_relations": self.repointed,
        }


def _item_uri(zot: Zotero, key: str) -> str:
    return f"http://zotero.org/{zot.library_type}/{zot.library_id}/items/{key}"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _add_relation(relations: dict[str, Any], predicate: str, uri: str) -> None:
    values = _as_list(relations.get(predicate))
    if uri not in values:
        values.append(uri)
    relations[predicate] = values[0] if len(values) == 1 else values


def _remove_relation(relations: dict[str, Any], predicate: str, uri: str) -> None:
    values = [v for v in _as_list(relations.get(predicate)) if v != uri]
    if not values:
        relations.pop(predicate, None)
    else:
        relations[predicate] = values[0] if len(values) == 1 else values


def _merge_tags(
    master: list[dict[str, Any]], other: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union of tags. A manual tag (type 0) wins over an automatic one."""
    merged = {t["tag"]: dict(t) for t in master}
    for tag in other:
        name = tag["tag"]
        if name not in merged or tag.get("type", 0) == 0:
            merged[name] = {
                "tag": name,
                **({"type": tag["type"]} if tag.get("type") else {}),
            }
    return list(merged.values())


def plan_merge(  # noqa: PLR0912, PLR0915
    zot: Zotero,
    master_key: str,
    other_keys: list[str],
    fill_empty: bool = False,
    library: list[dict[str, Any]] | None = None,
) -> MergePlan:
    """Work out what merging ``other_keys`` into ``master_key`` changes.

    Nothing is written. Pass the plan to :func:`apply_merge`.

    Args:
        zot: The client.
        master_key: The item to keep.
        other_keys: The items to merge into it and move to the trash.
        fill_empty: Copy a field from another item when the master's is
            empty (the first other item that has it wins). Zotero desktop
            asks per field instead; by default only the master's fields stay.
        library: All items of the library, as from
            ``zot.everything(zot.items())``, to find relations that point at
            the merged items. Pass it to reuse one download across several
            merges; by default it is downloaded. Versions in it can be stale:
            :func:`apply_merge` retries those writes with the current one.

    Simplifications compared with Zotero desktop:

    - Duplicate PDFs are matched by MD5 only. Zotero also compares the text
      of PDFs whose bytes differ.
    - Item keys inside note HTML (citations, links to attachments) are not
      rewritten. Such links in notes keep pointing at the trashed items.

    """
    if master_key in other_keys or not other_keys:
        msg = "Give one master and at least one other item, all different"
        raise ValueError(msg)
    master = zot.item(master_key)
    others = [zot.item(k) for k in other_keys]
    mdata = master["data"]
    for item in [master, *others]:
        if item["data"].get("itemType") in NON_REGULAR_TYPES:
            msg = f"{item['key']} is a {item['data']['itemType']}; only regular items can be merged"
            raise ValueError(msg)
    plan = MergePlan(master=master, others=others)
    updates = plan.master_updates

    collections = list(mdata.get("collections") or [])
    tags = list(mdata.get("tags") or [])
    relations = {k: v for k, v in (mdata.get("relations") or {}).items()}
    date_added = mdata.get("dateAdded") or ""
    master_uri = _item_uri(zot, master_key)
    master_children = zot.children(master_key)
    master_pdfs = {
        c["data"]["md5"]: c
        for c in master_children
        if c["data"].get("itemType") == "attachment"
        and c["data"].get("contentType") == "application/pdf"
        and c["data"].get("md5")
        and not c["data"].get("deleted")
    }
    master_web = [
        c
        for c in master_children
        if c["data"].get("linkMode") in WEB_LINK_MODES and not c["data"].get("deleted")
    ]
    master_att_relations: dict[str, dict[str, Any]] = {}
    merged_into: set[str] = set()

    for other in others:
        odata = other["data"]
        other_uri = _item_uri(zot, other["key"])
        for coll in odata.get("collections") or []:
            if coll not in collections:
                collections.append(coll)
        tags = _merge_tags(tags, odata.get("tags") or [])
        for pred, values in (odata.get("relations") or {}).items():
            if pred == REPLACES:
                # Merge history of the other item moves to the master.
                for v in _as_list(values):
                    _add_relation(relations, REPLACES, v)
                continue
            for v in _as_list(values):
                if v != master_uri:
                    _add_relation(relations, pred, v)
        _add_relation(relations, REPLACES, other_uri)
        if odata.get("dateAdded") and odata["dateAdded"] < date_added:
            date_added = odata["dateAdded"]
        if fill_empty:
            for field, value in odata.items():
                if field in SYSTEM_FIELDS or not value or field not in mdata:
                    continue
                if not mdata.get(field) and field not in updates:
                    updates[field] = value

        for child in zot.children(other["key"]):
            cdata = child["data"]
            twin = None
            if cdata.get("itemType") == "attachment" and not cdata.get("deleted"):
                if cdata.get("contentType") == "application/pdf" and cdata.get("md5"):
                    candidate = master_pdfs.get(cdata["md5"])
                    if (
                        candidate is not None
                        and candidate["key"] not in merged_into
                        and (candidate["data"].get("linkMode") == "linked_file")
                        == (cdata.get("linkMode") == "linked_file")
                    ):
                        twin = candidate
                elif cdata.get("linkMode") in WEB_LINK_MODES:
                    twin = next(
                        (
                            w
                            for w in master_web
                            if w["data"].get("title") == cdata.get("title")
                            and w["data"].get("linkMode") == cdata.get("linkMode")
                            and (
                                w["data"].get("url") == cdata.get("url")
                                or cdata.get("linkMode") != "linked_url"
                            )
                        ),
                        None,
                    )
            if twin is None:
                plan.moved_children.append(child["key"])
                plan.writes.append(
                    {
                        "key": child["key"],
                        "version": child["version"],
                        "parentItem": master_key,
                    }
                )
                continue
            # The twin replaces this attachment: move its annotations and
            # notes to the twin, and trash it.
            merged_into.add(twin["key"])
            plan.trashed_attachments.append(child["key"])
            for grandchild in zot.children(child["key"]):
                plan.moved_children.append(grandchild["key"])
                plan.writes.append(
                    {
                        "key": grandchild["key"],
                        "version": grandchild["version"],
                        "parentItem": twin["key"],
                    }
                )
            trels = master_att_relations.setdefault(
                twin["key"], dict(twin["data"].get("relations") or {})
            )
            _add_relation(trels, REPLACES, _item_uri(zot, child["key"]))
            plan.writes.append(
                {"key": child["key"], "version": child["version"], "deleted": 1}
            )
            if twin in master_web:
                master_web.remove(twin)

    for att_key, rels in master_att_relations.items():
        att = next(c for c in master_children if c["key"] == att_key)
        plan.writes.append(
            {"key": att_key, "version": att["version"], "relations": rels}
        )

    if collections != list(mdata.get("collections") or []):
        updates["collections"] = collections
    if tags != list(mdata.get("tags") or []):
        updates["tags"] = tags
    updates["relations"] = relations
    if date_added and date_added != mdata.get("dateAdded"):
        updates["dateAdded"] = date_added
    plan.writes.insert(0, {"key": master_key, "version": master["version"], **updates})

    # Items elsewhere in the library that point at a merged item (with a
    # relation other than RELATED) now point at the master. One pass over
    # the library's items.
    other_uris = {_item_uri(zot, o["key"]) for o in others}
    skip = {master_key, *other_keys}
    if library is None:
        library = zot.everything(zot.items(limit=100))
    for item in library:
        if item["key"] in skip:
            continue
        rels = dict(item["data"].get("relations") or {})
        changed = False
        for pred, values in list(rels.items()):
            # RELATED needs no repointing: the server adds the master to the
            # related item when the master gets the link, and removes the
            # trashed item when it drops its link.
            if pred in {REPLACES, RELATED}:
                continue
            for v in _as_list(values):
                if v in other_uris:
                    _remove_relation(rels, pred, v)
                    _add_relation(rels, pred, master_uri)
                    changed = True
        if changed:
            plan.repointed.append(item["key"])
            plan.writes.append(
                {"key": item["key"], "version": item["version"], "relations": rels}
            )

    for other in others:
        # The merge history now lives on the master: remove it here, so that
        # no deleted object has two subjects (as Zotero does). Related-item
        # links moved to the master too; on the trashed item they would keep
        # the related items linked to it (see RELATED).
        orels = {
            k: v
            for k, v in (other["data"].get("relations") or {}).items()
            if k not in {REPLACES, RELATED}
        }
        plan.writes.append(
            {
                "key": other["key"],
                "version": other["version"],
                "relations": orels,
                "deleted": 1,
            }
        )
    return plan


def apply_merge(zot: Zotero, plan: MergePlan) -> None:
    """Write a :func:`plan_merge` plan, one item at a time, master first.

    Each write is a PATCH of only the fields the plan changes, with the
    item's version from the plan. If the version is stale (HTTP 412), the
    write is retried once with the current version. A change to an item's
    children can raise its version, so the merge's own earlier writes can
    cause this. The retry can overwrite a concurrent edit of the same
    fields (relations, collections, tags) made between plan and apply.

    If a write fails, the earlier ones keep their changes. Plan and apply
    the merge again to finish: the plan is computed from the current data.
    """
    for payload in plan.writes:
        try:
            zot.update_item(payload)
        except ze.PreConditionFailedError:  # noqa: PERF203 -- one retry per write
            current = zot.item(payload["key"])
            zot.update_item({**payload, "version": current["version"]})
