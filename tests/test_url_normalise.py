from __future__ import annotations

import pytest

from linking_engine.urls import host_of, normalise_url


@pytest.mark.parametrize(
    ("url", "key"),
    [
        (
            "https://www.Action1.com/Patch-Management/?utm_source=x#top",
            "action1.com/patch-management",
        ),
        ("http://action1.com/patch-management", "action1.com/patch-management"),
        ("https://action1.com/", "action1.com"),
        ("action1.com", "action1.com"),
        ("//www.action1.com/a", "action1.com/a"),
        ("https://user:pass@www2.example.com:443/a//b/./c/../d/", "example.com/a/b/d"),
        ("http://example.com:80/a", "example.com/a"),
        ("https://example.com:8080/x", "example.com:8080/x"),
        ("https://example.com./x/index.html", "example.com/x"),
        ("https://example.com/blog/default.aspx", "example.com/blog"),
        ("https://example.com/shop/;jsessionid=ABC123/item", "example.com/shop/item"),
        ("https://example.com/caf%C3%A9", "example.com/caf%C3%A9"),
        ("https://example.com/café", "example.com/caf%C3%A9"),
        ("https://example.com/Caf%c3%a9/", "example.com/caf%C3%A9"),
        ("https://bücher.de/Über", "xn--bcher-kva.de/%C3%BCber"),
        ("https://www.example.co.uk/A", "example.co.uk/a"),
        ("www.com/a", "www.com/a"),
        ("  https://example.com/a  ", "example.com/a"),
    ],
)
def test_normalise_url(url: str, key: str) -> None:
    assert normalise_url(url) == key


@pytest.mark.parametrize(
    "url",
    [
        "",
        "mailto:a@b.c",
        "tel:+123",
        "javascript:void(0)",
        "ftp://example.com/a",
        "https:///no-host",
        "/relative/path",
        "not a url",
        "https://example.com:99999/a",
    ],
)
def test_non_web_or_malformed_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match=r"url|host|port"):
        normalise_url(url)


def test_normalising_a_key_returns_it_unchanged() -> None:
    urls = ["https://www.Example.com/A/b/?q=1", "https://bücher.de/Über", "https://e.com/a%2520b"]
    for key in map(normalise_url, urls):
        assert normalise_url(key) == key


def test_host_of() -> None:
    assert host_of("example.com:8080/a/b") == "example.com:8080"
    assert host_of("example.com") == "example.com"
