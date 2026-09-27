from __future__ import annotations

import pytest

from linking_engine.urls import UrlRules, host_of, is_kept_param, normalise_url, url_rules


@pytest.mark.parametrize(
    ("url", "key"),
    [
        (
            "https://www.example.com/Patch-Management/?utm_source=x#top",
            "example.com/patch-management",
        ),
        ("http://example.com/patch-management", "example.com/patch-management"),
        ("https://example.com/", "example.com"),
        ("example.com", "example.com"),
        ("//www.example.com/a", "example.com/a"),
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


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://www.example.com/blog?page=2", "example.com/blog?page=2"),
        ("https://www.example.com/blog?page=1", "example.com/blog"),
        ("https://example.com/blog/?page=0&utm_source=x", "example.com/blog"),
        ("https://example.com/news?utm_source=a&offset=20&sort=new", "example.com/news?offset=20"),
        ("https://example.com/list?start=0", "example.com/list"),
        ("https://example.com/list?page=abc", "example.com/list"),
        ("https://www.example.com/?page_id=12345", "example.com?page_id=12345"),
        ("https://example.com/post?p=0042&ref=x", "example.com/post?p=42"),
        (
            "https://example.com/shop?product_id=SKU-9&page=3",
            "example.com/shop?page=3&product_id=sku-9",
        ),
        ("https://example.com/news?announcement_pg=3", "example.com/news"),
        ("https://example.com/blog/page/2/", "example.com/blog/page/2"),
    ],
)
def test_only_pagination_and_document_ids_survive_in_the_query(url: str, key: str) -> None:
    assert normalise_url(url) == key


def test_tenant_rules_add_and_remove_parameters() -> None:
    rules = UrlRules(keep_params={"Announcement_PG"}, drop_params={"p"})
    with url_rules(rules):
        assert (
            normalise_url("https://example.com/news?announcement_pg=3")
            == "example.com/news?announcement_pg=3"
        )
        assert normalise_url("https://example.com/x?p=5") == "example.com/x"
        assert is_kept_param("announcement_pg")
        assert not is_kept_param("p")
    assert normalise_url("https://example.com/news?announcement_pg=3") == "example.com/news"
    assert is_kept_param("p")


def test_a_key_keeps_its_query_whatever_rules_are_active() -> None:
    with url_rules(UrlRules(keep_params={"announcement_pg"})):
        key = normalise_url("https://example.com/news?announcement_pg=3")
    assert normalise_url(key) == key


def test_host_of_a_key_with_a_query() -> None:
    assert host_of("example.com?page_id=12345") == "example.com"
