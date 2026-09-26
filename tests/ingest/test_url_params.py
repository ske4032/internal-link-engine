from __future__ import annotations

from linking_engine.ingest.url_params import query_param_evidence
from linking_engine.urls import UrlRules, url_rules


def by_name(pages: list[tuple[str, str | None]]) -> dict[str, tuple[int, int, int, bool]]:
    return {
        e.name: (e.urls, e.content_changed, e.content_same, e.kept)
        for e in query_param_evidence(pages)
    }


def test_pagination_changes_content_and_tracking_does_not() -> None:
    evidence = by_name(
        [
            ("https://example.com/blog", "h-base"),
            ("https://example.com/blog?page=2", "h-p2"),
            ("https://example.com/blog?page=3", "h-p3"),
            ("https://example.com/post", "h-post"),
            ("https://example.com/post?utm_source=x", "h-post"),
        ]
    )
    assert evidence["page"] == (2, 3, 0, True)
    assert evidence["utm_source"] == (1, 0, 1, False)


def test_a_facet_that_changes_content_is_reported_as_stripped() -> None:
    evidence = by_name(
        [("https://example.com/cases", "all"), ("https://example.com/cases?industry=farm", "farm")]
    )
    assert evidence["industry"] == (1, 1, 0, False)


def test_pairs_differing_in_two_parameters_are_not_attributed() -> None:
    evidence = by_name(
        [
            ("https://example.com/a?page=2&sort=x", "one"),
            ("https://example.com/a?page=3&sort=y", "two"),
        ]
    )
    assert evidence["page"][1:3] == (0, 0)
    assert evidence["sort"][1:3] == (0, 0)


def test_urls_without_a_content_hash_are_counted_but_not_compared() -> None:
    evidence = by_name([("https://example.com/a?page=2", None), ("https://example.com/a", "h")])
    assert evidence["page"] == (1, 0, 0, True)


def test_kept_follows_the_active_tenant_rules() -> None:
    pages: list[tuple[str, str | None]] = [("https://example.com/n?announcement_pg=2", "x")]
    with url_rules(UrlRules(keep_params={"announcement_pg"})):
        assert by_name(pages)["announcement_pg"][3] is True
    assert by_name(pages)["announcement_pg"][3] is False
