from __future__ import annotations

from typing import Literal

from pydantic import HttpUrl

from linking_engine.ingest.markdown_clean import clean_page
from linking_engine.ingest.template_links import count_template_inlinks
from linking_engine.models import CleanedPage, TemplateInlinks, TemplateLink

Zone = Literal["menu", "footer"]


def page(path: str, *links: tuple[Zone, str]) -> tuple[str, CleanedPage]:
    return f"example.com{path}", CleanedPage(
        url=HttpUrl(f"https://example.com{path}"),
        body_text="Body.",
        template_links=tuple(
            TemplateLink(target_url=HttpUrl(target), zone=zone) for zone, target in links
        ),
    )


def counts(*pages: tuple[str, CleanedPage]) -> dict[str, tuple[int, int]]:
    return {
        item.url: (item.menu_inlinks, item.footer_inlinks) for item in count_template_inlinks(pages)
    }


def test_a_source_counts_once_per_zone_however_often_it_repeats() -> None:
    assert counts(
        page("/a", *[("menu", "https://example.com/t")] * 3, ("footer", "https://example.com/t")),
        page("/b", ("footer", "https://example.com/t"), ("footer", "https://www.example.com/t/")),
        page("/c", ("menu", "https://example.com/t")),
    ) == {"example.com/t": (2, 2)}


def test_self_links_never_count_in_any_url_form() -> None:
    assert counts(
        page("/a", ("menu", "https://www.example.com/a/"), ("footer", "http://example.com/a")),
        page("/b", ("menu", "https://example.com/a")),
    ) == {"example.com/a": (1, 0)}


def test_one_source_in_both_zones_counts_in_both() -> None:
    assert counts(
        page("/a", ("menu", "https://example.com/t"), ("footer", "https://example.com/t"))
    ) == {"example.com/t": (1, 1)}


def test_results_are_keys_sorted_and_include_targets_that_were_not_crawled() -> None:
    result = count_template_inlinks(
        [page("/a", ("footer", "https://example.com/z"), ("menu", "https://example.com/b"))]
    )
    assert result == [
        TemplateInlinks(url="example.com/b", menu_inlinks=1, footer_inlinks=0),
        TemplateInlinks(url="example.com/z", menu_inlinks=0, footer_inlinks=1),
    ]


def test_a_target_without_a_url_key_is_skipped() -> None:
    assert counts(page("/a", ("menu", "http://[::1]/x"), ("menu", "https://example.com/t"))) == {
        "example.com/t": (1, 0)
    }


def test_no_template_links_means_no_counts() -> None:
    assert count_template_inlinks([page("/a"), page("/b")]) == []


def test_counts_from_cleaned_pages() -> None:
    menu = "[Docs](/docs/)"
    footer = "[Pricing](/pricing/) [Docs](/docs/)"
    template = frozenset({menu, footer})
    pages = [
        (
            f"example.com{path}",
            clean_page(
                f"{menu}\nBody of {path}.\n{footer}",
                f"https://example.com{path}",
                boilerplate=template,
            ),
        )
        for path in ("/a", "/b", "/docs")
    ]
    assert counts(*pages) == {"example.com/docs": (2, 2), "example.com/pricing": (0, 3)}
