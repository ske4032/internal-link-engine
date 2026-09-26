"""Issue #2 Gotcha: `HttpUrl` normalises, but not the way a crawler needs.

It lowercases the scheme and host, and it leaves the path alone — so `/foo` and `/foo/`
are two different pages as far as the graph is concerned, and `page_url IS UNIQUE` will
happily hold both. Canonicalise before constructing, not after. This file is that rule
written as assertions instead of a comment.
"""

from __future__ import annotations

import pytest
from factories import LINK_SPEC, PAGE_SPEC
from pydantic import AnyUrl, ValidationError

INVALID_URLS = ["not a url", "/relative/path", "example.com/foo", "ftp://example.com/foo"]
URL_ERROR_TYPES = {"url_parsing", "url_scheme", "url_type", "url_syntax_violation"}


def test_a_trailing_slash_makes_a_different_url() -> None:
    bare = PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url="https://example.com/foo"))
    slashed = PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url="https://example.com/foo/"))

    assert bare.url != slashed.url, (
        "/foo and /foo/ must stay distinct — the model does not canonicalise for you, "
        "so an uncanonicalised crawl puts the same page in the graph twice"
    )
    assert str(bare.url) != str(slashed.url)
    assert bare != slashed
    assert str(bare.url).endswith("/foo")
    assert str(slashed.url).endswith("/foo/")


def test_scheme_and_host_are_normalised_but_the_path_is_not() -> None:
    page = PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url="HTTPS://Example.COM/Guides/Foo"))

    assert page.url.scheme == "https"
    assert page.url.host == "example.com"
    assert str(page.url).endswith("/Guides/Foo"), (
        "path case is significant to a web server and must survive validation"
    )


def test_url_fields_hold_parsed_urls_not_strings() -> None:
    page = PAGE_SPEC.build()
    assert isinstance(page.url, AnyUrl), "url must be HttpUrl, not str"
    assert not isinstance(page.url, str)
    assert page.url.scheme == "https"


def test_str_of_a_url_revalidates_to_the_same_url() -> None:
    """Why a json-mode round trip is safe: str -> HttpUrl is lossless here."""
    page = PAGE_SPEC.build()
    again = PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url=str(page.url)))
    assert again.url == page.url
    assert again == page


@pytest.mark.parametrize("value", INVALID_URLS)
def test_url_validation_rejects_non_http_urls(value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url=value))
    error = exc_info.value.errors()[0]
    assert error["loc"] == ("url",)
    assert error["type"] in URL_ERROR_TYPES, f"{value!r} failed for the wrong reason"


def test_link_endpoints_are_both_urls() -> None:
    link = LINK_SPEC.build()
    assert isinstance(link.source_url, AnyUrl)
    assert isinstance(link.target_url, AnyUrl)
    assert link.source_url != link.target_url
