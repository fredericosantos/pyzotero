"""Tests for duplicate detection and merging."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx2
import pytest

from pyzotero import errors as ze
from pyzotero import zotero
from pyzotero.duplicates import (
    REPLACES,
    apply_merge,
    find_duplicates,
    match_basis,
    normalize,
    plan_merge,
)

LIB = "http://zotero.org/users/42/items/"


def item(key, version=1, **data):
    data.setdefault("itemType", "journalArticle")
    data.setdefault("dateAdded", "2024-01-01T00:00:00Z")
    return {
        "key": key,
        "version": version,
        "data": {"key": key, "version": version, **data},
    }


def author(last, first="A"):
    return {"creatorType": "author", "lastName": last, "firstName": first}


def keys(groups):
    return [[i["key"] for i in g] for g in groups]


class TestFindDuplicates:
    def test_normalize(self):
        assert (
            normalize("Édge-Detect: Edge-centric  Network!")
            == "edge detect edge centric network"
        )

    def test_same_doi(self):
        items = [
            item("A", DOI="10.1/X", title="One"),
            item("B", DOI="10.1/x", title="Other"),
        ]
        assert keys(find_duplicates(items)) == [["A", "B"]]

    def test_title_needs_a_common_author(self):
        a = item("A", title="Same Title", creators=[author("Smith", "John")])
        b = item("B", title="Same title.", creators=[author("Smith", "J.")])
        c = item("C", title="Same title", creators=[author("Jones")])
        assert keys(find_duplicates([a, b, c])) == [["A", "B"]]

    def test_title_with_different_doi_or_far_years_is_not_a_duplicate(self):
        a = item("A", title="T", DOI="10.1/a", creators=[author("S")])
        b = item("B", title="T", DOI="10.1/b", creators=[author("S")])
        c = item("C", title="U", date="2010", creators=[author("S")])
        d = item("D", title="U", date="2013", creators=[author("S")])
        assert find_duplicates([a, b, c, d]) == []

    def test_isbn_10_and_13_match_for_books(self):
        a = item("A", itemType="book", title="X", ISBN="0-306-40615-2")
        b = item("B", itemType="book", title="Y", ISBN="978-0-306-40615-7")
        groups = find_duplicates([a, b])
        assert keys(groups) == [["A", "B"]]
        assert match_basis(groups[0]) == "isbn"

    def test_skips_attachments_notes_and_trash(self):
        items = [
            item("A", DOI="10.1/x"),
            item("B", DOI="10.1/x", deleted=1),
            item("C", itemType="attachment", title="paper.pdf"),
            item("D", itemType="attachment", title="paper.pdf"),
        ]
        assert find_duplicates(items) == []

    def test_oldest_first(self):
        a = item("A", DOI="10.1/x", dateAdded="2026-01-01T00:00:00Z")
        b = item("B", DOI="10.1/x", dateAdded="2023-01-01T00:00:00Z")
        assert keys(find_duplicates([a, b])) == [["B", "A"]]


def fake_zot(items, children):
    """Return a mock client over ``items`` (key -> item) and ``children``."""
    zot = MagicMock()
    zot.library_type, zot.library_id = "users", "42"
    zot.item.side_effect = lambda k: items[k]
    zot.children.side_effect = lambda k: [items[c] for c in children.get(k, [])]
    zot.everything.return_value = list(items.values())
    return zot


@pytest.fixture
def library():
    items = {
        "MASTER01": item(
            "MASTER01",
            version=10,
            title="Paper",
            collections=["C1"],
            tags=[{"tag": "keep"}, {"tag": "auto", "type": 1}],
            relations={},
            dateAdded="2024-05-01T00:00:00Z",
            publicationTitle="",
        ),
        "OTHER001": item(
            "OTHER001",
            version=20,
            title="Paper",
            collections=["C1", "C2"],
            tags=[{"tag": "auto"}, {"tag": "new"}],
            relations={"dc:relation": LIB + "RELATED1", REPLACES: LIB + "OLDMERGE"},
            dateAdded="2023-01-01T00:00:00Z",
            publicationTitle="Journal",
        ),
        "MPDF0001": item("MPDF0001", itemType="attachment", linkMode="imported_file", contentType="application/pdf", md5="a" * 32, parentItem="MASTER01"),
        "OPDF0001": item("OPDF0001", itemType="attachment", linkMode="imported_file", contentType="application/pdf", md5="a" * 32, parentItem="OTHER001"),
        "OPDF0002": item("OPDF0002", itemType="attachment", linkMode="imported_file", contentType="application/pdf", md5="b" * 32, parentItem="OTHER001"),
        "ONOTE001": item("ONOTE001", itemType="note", note="<p>hi</p>", parentItem="OTHER001"),
        "ANNOT001": item("ANNOT001", itemType="annotation", parentItem="OPDF0001"),
        "RELATED1": item("RELATED1", title="Related", relations={"dc:relation": LIB + "OTHER001"}),
    }  # fmt: skip
    children = {
        "MASTER01": ["MPDF0001"],
        "OTHER001": ["OPDF0001", "OPDF0002", "ONOTE001"],
        "OPDF0001": ["ANNOT001"],
    }
    return items, children


def writes_by_key(plan):
    out = {}
    for w in plan.writes:
        out.setdefault(w["key"], []).append(w)
    return out


class TestPlanMerge:
    def test_full_plan(self, library):
        zot = fake_zot(*library)
        plan = plan_merge(zot, "MASTER01", ["OTHER001"])
        w = writes_by_key(plan)

        master = w["MASTER01"][0]
        assert plan.writes[0] is master  # master first
        assert master["version"] == 10  # noqa: PLR2004
        assert master["collections"] == ["C1", "C2"]
        # The manual "auto" tag of the other item wins over the automatic one.
        assert {t["tag"]: t.get("type", 0) for t in master["tags"]} == {
            "keep": 0,
            "auto": 0,
            "new": 0,
        }
        assert master["relations"]["dc:relation"] == LIB + "RELATED1"
        assert set(master["relations"][REPLACES]) == {
            LIB + "OTHER001",
            LIB + "OLDMERGE",
        }
        assert master["dateAdded"] == "2023-01-01T00:00:00Z"
        assert "publicationTitle" not in master  # fill_empty is off

        # Identical PDF: trashed, its annotation moves to the master's copy,
        # which records the replacement.
        assert plan.trashed_attachments == ["OPDF0001"]
        assert w["OPDF0001"] == [{"key": "OPDF0001", "version": 1, "deleted": 1}]
        assert w["ANNOT001"] == [
            {"key": "ANNOT001", "version": 1, "parentItem": "MPDF0001"}
        ]
        assert w["MPDF0001"][0]["relations"] == {REPLACES: LIB + "OPDF0001"}
        # A different PDF and the note move to the master.
        assert w["OPDF0002"][0]["parentItem"] == "MASTER01"
        assert w["ONOTE001"][0]["parentItem"] == "MASTER01"

        # The server keeps related-item links symmetric, so the related
        # item is not written: the other item drops its link instead.
        assert "RELATED1" not in w
        # The other item is trashed last, without its merge history and
        # related-item links.
        assert plan.writes[-1] == {
            "key": "OTHER001",
            "version": 20,
            "relations": {},
            "deleted": 1,
        }

    def test_repoints_other_relations(self, library):
        items, children = library
        items["CITING01"] = item("CITING01", relations={"owl:sameAs": LIB + "OTHER001"})
        plan = plan_merge(fake_zot(items, children), "MASTER01", ["OTHER001"])
        assert plan.repointed == ["CITING01"]
        rels = writes_by_key(plan)["CITING01"][0]["relations"]
        assert rels == {"owl:sameAs": LIB + "MASTER01"}

    def test_fill_empty(self, library):
        plan = plan_merge(fake_zot(*library), "MASTER01", ["OTHER001"], fill_empty=True)
        assert plan.writes[0]["publicationTitle"] == "Journal"

    def test_rejects_bad_input(self, library):
        zot = fake_zot(*library)
        with pytest.raises(ValueError, match="all different"):
            plan_merge(zot, "MASTER01", ["MASTER01"])
        with pytest.raises(ValueError, match="only regular items"):
            plan_merge(zot, "MASTER01", ["ONOTE001"])

    def test_nothing_is_written(self, library):
        zot = fake_zot(*library)
        plan_merge(zot, "MASTER01", ["OTHER001"])
        zot.update_item.assert_not_called()


class TestApplyMerge:
    def test_retries_once_on_stale_version(self, library):
        items, children = library
        zot = fake_zot(items, children)
        plan = plan_merge(zot, "MASTER01", ["OTHER001"])
        calls = []

        def update(payload):
            calls.append(payload)
            if payload["key"] == "OTHER001" and payload["version"] == 20:  # noqa: PLR2004
                raise ze.PreConditionFailedError("412")

        zot.update_item.side_effect = update
        items["OTHER001"] = {**items["OTHER001"], "version": 21}
        apply_merge(zot, plan)
        assert [c["version"] for c in calls if c["key"] == "OTHER001"] == [20, 21]
        assert len(calls) == len(plan.writes) + 1


def test_update_item_raises_on_rejection():
    """apply_merge relies on update_item raising when Zotero rejects a write."""
    client = httpx2.Client(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(412, request=r))
    )
    zot = zotero.Zotero("42", "user", "key", client=client)
    zot.check_items = lambda items: items
    with pytest.raises(ze.PreConditionFailedError):
        zot.update_item({"key": "ABCD1234", "version": 1, "title": "x"})
