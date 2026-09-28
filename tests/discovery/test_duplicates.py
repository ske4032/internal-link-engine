"""Exact duplicate grouping and the canonical copy of each group."""

from __future__ import annotations

import random

import pytest

from linking_engine.discovery.duplicates import canonical_order, duplicate_groups
from linking_engine.models import DuplicateGroup, DuplicateInput

BODY = "a" * 64
OTHER_BODY = "b" * 64


def row(
    path: str,
    body: str = BODY,
    *,
    language: str | None = "en",
    inbound: int = 0,
    indexable: bool = True,
) -> DuplicateInput:
    return DuplicateInput(
        url=f"example.com{path}",
        body_hash=body,
        language=language,
        indexable=indexable,
        inbound=inbound,
    )


def url(path: str) -> str:
    return f"example.com{path}"


def test_pages_with_an_identical_body_form_one_group() -> None:
    groups = duplicate_groups(
        [row("/blog/post", inbound=3), row("/post"), row("/news/post"), row("/other", OTHER_BODY)]
    )

    assert groups == [
        DuplicateGroup(
            group_id=0, canonical=url("/blog/post"), copies=(url("/news/post"), url("/post"))
        )
    ]


def test_singletons_and_different_bodies_are_not_grouped() -> None:
    assert duplicate_groups([row("/a"), row("/b", OTHER_BODY), row("/c", "c" * 64)]) == []
    assert duplicate_groups([]) == []


def test_the_same_body_in_two_languages_makes_two_groups() -> None:
    groups = duplicate_groups(
        [
            row("/en/a", language="en"),
            row("/en/b", language="en"),
            row("/de/a", language="de"),
            row("/de/b", language="de"),
            row("/fr/a", language="fr"),
        ]
    )

    assert [(g.canonical, g.copies) for g in groups] == [
        (url("/de/a"), (url("/de/b"),)),
        (url("/en/a"), (url("/en/b"),)),
    ], "the lone French copy has no duplicate in its own language"


def test_no_language_is_a_language_of_its_own() -> None:
    groups = duplicate_groups(
        [row("/a", language=None), row("/b", language=None), row("/c", language="en")]
    )

    assert [(g.canonical, g.copies) for g in groups] == [(url("/a"), (url("/b"),))]


@pytest.mark.parametrize(
    ("pages", "canonical"),
    [
        pytest.param(
            [row("/a-long-url", inbound=2), row("/b", inbound=1)], "/a-long-url", id="inbound"
        ),
        pytest.param([row("/longer"), row("/short")], "/short", id="then-url-length"),
        pytest.param([row("/b"), row("/a")], "/a", id="then-url"),
        pytest.param(
            [row("/zz", inbound=1), row("/yy", inbound=1), row("/x")], "/yy", id="length-and-url"
        ),
        pytest.param(
            [row("/a", inbound=9, indexable=False), row("/longer-url"), row("/b", inbound=1)],
            "/b",
            id="indexable-before-inbound",
        ),
        pytest.param(
            [row("/a", indexable=False), row("/b", inbound=3, indexable=False), row("/c")],
            "/c",
            id="the-only-indexable-copy",
        ),
        pytest.param(
            [row("/a", indexable=False), row("/longer-b", inbound=2, indexable=False)],
            "/longer-b",
            id="none-indexable-most-inbound",
        ),
    ],
)
def test_the_canonical_copy_is_indexable_then_has_the_most_inbound_links_then_the_shortest_url(
    pages: list[DuplicateInput], canonical: str
) -> None:
    [group] = duplicate_groups(pages)

    assert group.canonical == url(canonical)
    assert sorted(pages, key=canonical_order)[0].url == url(canonical)


def test_group_ids_are_the_rank_of_the_canonical_url_whatever_the_input_order() -> None:
    pages = [
        row("/z1", "1" * 64, inbound=5),
        row("/a1", "1" * 64),
        row("/m2", "2" * 64),
        row("/m3", "2" * 64),
        row("/b3", "3" * 64, inbound=1),
        row("/c3", "3" * 64),
        row("/d3", "3" * 64),
        row("/alone", "4" * 64),
    ]
    expected = duplicate_groups(pages)

    for seed in range(5):
        shuffled = pages.copy()
        random.Random(seed).shuffle(shuffled)
        assert duplicate_groups(shuffled) == expected

    assert [(g.group_id, g.canonical) for g in expected] == [
        (0, url("/b3")),
        (1, url("/m2")),
        (2, url("/z1")),
    ]
    assert expected[0].copies == (url("/c3"), url("/d3"))


def test_a_non_indexable_copy_with_the_most_inbound_links_is_a_copy_not_the_canonical() -> None:
    [group] = duplicate_groups(
        [row("/blog/post", inbound=7, indexable=False), row("/post", inbound=1), row("/p")]
    )

    assert (group.canonical, group.copies) == (url("/post"), (url("/blog/post"), url("/p")))


def test_without_an_indexable_copy_the_most_linked_copy_is_still_canonical() -> None:
    [group] = duplicate_groups(
        [row("/blog/post", inbound=7, indexable=False), row("/post", inbound=1, indexable=False)]
    )

    assert (group.canonical, group.copies) == (url("/blog/post"), (url("/post"),))


def test_a_page_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        duplicate_groups([row("/a"), row("/a", OTHER_BODY)])
