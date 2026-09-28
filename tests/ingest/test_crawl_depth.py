"""Crawl depth: breadth-first from the site root over body, menu and footer links."""

from __future__ import annotations

from typing import Any

from linking_engine.ingest.depth import crawl_depths
from linking_engine.ingest.prepare import prepare_corpus
from linking_engine.models import CrawlPage, LanguageRules

MENU = "- [Pricing](https://example.com/pricing)"
FOOTER = "- [Contact](https://example.com/contact)"


def crawl(path: str, body: str, **fields: Any) -> CrawlPage:
    values: dict[str, Any] = {
        "url": f"https://example.com{path}",
        "title": path,
        "content": f"{MENU}\n\n{body}\n\n{FOOTER}",
        "status_code": 200,
        "usable": True,
        **fields,
    }
    return CrawlPage(**values)


# Six or more pages, so the menu and footer lines are detected as template.
SITE = [
    crawl("/", "The home page points to [section a](https://example.com/a)."),
    crawl("/a", "Section a leads on to [page b](https://example.com/b)."),
    crawl("/b", "Page b is a leaf with nothing further to read."),
    crawl("/pricing", "Plans and prices for every team size."),
    crawl("/contact", "Write to us and we answer within a day."),
    crawl("/island", "Nothing links here, though it links to [page b](https://example.com/b)."),
]


def depths_of(
    pages: list[CrawlPage], rules: LanguageRules = LanguageRules()
) -> dict[str, int | None]:
    corpus = prepare_corpus(pages, source="s", language_rules=rules)
    return {str(r.url): r.crawl_depth for r in corpus.records}


def test_depth_is_the_fewest_links_from_a_root() -> None:
    edges = {"r": ["a", "b"], "a": ["c", "r"], "b": ["c"], "c": ["d"], "x": ["y"]}

    assert crawl_depths(edges, ["r"]) == {"r": 0, "a": 1, "b": 1, "c": 2, "d": 3}


def test_every_root_starts_at_zero_even_without_links() -> None:
    edges = {"r1": ["a"], "a": ["r2", "b"], "r2": ["b"]}

    assert crawl_depths(edges, ["r1", "r2", "alone"]) == {
        "r1": 0,
        "r2": 0,
        "alone": 0,
        "a": 1,
        "b": 1,
    }


def test_without_roots_nothing_is_reached() -> None:
    assert crawl_depths({"a": ["b"]}, []) == {}


def test_menu_and_footer_links_count_so_template_only_pages_are_shallow() -> None:
    assert depths_of(SITE) == {
        "example.com": 0,
        "example.com/a": 1,
        # Only the menu and the footer link here; neither is a body link.
        "example.com/pricing": 1,
        "example.com/contact": 1,
        "example.com/b": 2,
        "example.com/island": None,
    }


def test_a_crawl_without_a_page_at_the_site_root_has_no_depths() -> None:
    no_root = [page for page in SITE if str(page.url) != "https://example.com/"]
    no_root.append(crawl("/filler", "Another page so the template is still found."))

    assert set(depths_of(no_root).values()) == {None}


def test_language_homes_are_roots_so_a_redirecting_site_root_still_gets_depths() -> None:
    site = [
        crawl("/", "", status_code=301, content=None),
        crawl("/en", "Welcome, start with [the guide](https://example.com/en/a)."),
        crawl("/en/a", "The guide continues on [the next page](https://example.com/en/b)."),
        crawl("/en/b", "The last page of the English guide."),
        crawl("/de", "Willkommen, lies zuerst [die Anleitung](https://example.com/de/x)."),
        crawl("/de/x", "Die Anleitung endet hier."),
        crawl("/pricing", "Plans and prices for every team size."),
        crawl("/contact", "Write to us and we answer within a day."),
    ]
    rules = LanguageRules(prefixes=(("/en/", "en"), ("/de/", "de")))

    assert depths_of(site, rules) == {
        "example.com": 0,
        "example.com/en": 0,
        "example.com/de": 0,
        "example.com/en/a": 1,
        "example.com/de/x": 1,
        "example.com/pricing": 1,
        "example.com/contact": 1,
        "example.com/en/b": 2,
    }
    # Without the prefixes only the redirect is a root, and it links nowhere.
    assert {url: depth for url, depth in depths_of(site).items() if depth is not None} == {
        "example.com": 0
    }
