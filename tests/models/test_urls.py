"""Every url field holds a normalised key, so one page has one identity across tenants."""

from __future__ import annotations

import pytest
from factories import LINK_SPEC, PAGE_SPEC
from pydantic import ValidationError


def test_url_variants_of_one_page_become_one_key() -> None:
    variants = [
        "https://example.com/foo",
        "https://example.com/foo/",
        "http://www.example.com/foo?utm_source=x",
        "HTTPS://Example.COM/Foo#section",
    ]
    keys = {PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url=v)).url for v in variants}
    assert keys == {"example.com/foo"}


def test_a_key_revalidates_to_itself() -> None:
    page = PAGE_SPEC.build()
    assert PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url=page.url)) == page


@pytest.mark.parametrize(
    "value", ["not a url", "/relative/path", "ftp://example.com/foo", "mailto:a@b.c"]
)
def test_non_http_urls_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        PAGE_SPEC.model(**PAGE_SPEC.kwargs_with(url=value))
    assert exc_info.value.errors()[0]["loc"] == ("url",)


def test_link_endpoints_are_keys() -> None:
    link = LINK_SPEC.build()
    assert link.source_url == "example.com/guides/trail-running-shoes"
    assert link.target_url == "example.com/shop/trail-shoes"
