from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pymongo import AsyncMongoClient

from linking_engine.ingest.mongo_repo import CrawlSource
from linking_engine.ingest.prepare import page_language, prepare_corpus, prepare_tenant
from linking_engine.models import CrawlPage, LanguageRules
from linking_engine.urls import UrlRules

if TYPE_CHECKING:
    from linking_engine.ingest.mongo_repo import MongoRepo

NAV = "- [Pricing](https://example.com/pricing)"


def crawl(path: str, body: str = "", **fields: Any) -> CrawlPage:
    values: dict[str, Any] = {
        "url": f"https://example.com{path}",
        "title": path,
        "content": f"{NAV}\n\n{body or f'Body of {path} with enough words to count.'}",
        "status_code": 200,
        "usable": True,
        **fields,
    }
    return CrawlPage(**values)


# Six or more cleaned pages: with fewer, every line is on 20% of pages and counts as template.
USABLE = [
    crawl("/pricing", "Plans and prices for every team size."),
    crawl("/a", "Read [the guide](https://example.com/b) before you start."),
    crawl("/b", "The guide links back to [page a](https://example.com/a)."),
    *(crawl(f"/p{i}") for i in range(4)),
]
SITE = [
    *USABLE,
    crawl("/gone", status_code=404),
    crawl("/draft", usable=False),
    crawl("/blank", content=""),
]


def test_every_crawled_url_becomes_one_page_and_skips_are_counted() -> None:
    corpus = prepare_corpus(SITE, source="crawl_db.crawl_pages")
    assert len(corpus.records) == len(SITE)
    assert corpus.skipped == {"status 404": 1, "not usable": 1, "no content": 1}
    assert len(corpus.cleaned) == len(USABLE)
    assert {r.source for r in corpus.records} == {"crawl_db.crawl_pages"}


def test_template_lines_leave_the_body_and_count_as_menu_inlinks() -> None:
    corpus = prepare_corpus(SITE, source="s")
    pages = {str(r.url): r for r in corpus.records}
    assert "Pricing" not in pages["example.com/a"].body_text
    # Every other page with content carries the nav line, 404 and unusable ones included,
    # as their body links are edges too; a page does not link to itself.
    pricing = pages["example.com/pricing"]
    assert (pricing.menu_inlinks, pricing.footer_inlinks) == (len(SITE) - 2, 0)
    assert [(str(link.source_url), str(link.target_url)) for link in corpus.links] == [
        ("example.com/a", "example.com/b"),
        ("example.com/b", "example.com/a"),
    ]


def test_urls_sharing_a_key_keep_the_200_https_shortest_one() -> None:
    docs = [
        crawl("/a", status_code=404),
        crawl("/a/?utm_source=x"),
        crawl("/a", url="http://example.com/a"),
    ]
    corpus = prepare_corpus(docs, source="s")
    [record] = corpus.records
    assert (record.status_code, record.crawl_url) == (200, "https://example.com/a/?utm_source=x")
    assert len(corpus.merged) == 2


def test_a_page_takes_the_language_of_its_longest_matching_prefix() -> None:
    rules = LanguageRules(
        default_language="en", prefixes=(("/de/", "de"), ("/de/at/", "de-at"), ("/FR", "fr"))
    )

    assert page_language("example.com/de", rules) == "de"
    assert page_language("example.com/de/produkte", rules) == "de"
    assert page_language("example.com/de/at/produkte", rules) == "de-at"
    assert page_language("example.com/fr/produits?page=2", rules) == "fr"
    # Whole segments only, and the root belongs to no prefix.
    assert page_language("example.com/design", rules) == "en"
    assert page_language("example.com", rules) == "en"
    assert page_language("example.com?lang=de", rules) == "en"


def test_every_record_gets_a_language_english_without_rules() -> None:
    plain = prepare_corpus(SITE, source="s")
    german = prepare_corpus(
        SITE,
        source="s",
        language_rules=LanguageRules(default_language="de", prefixes=(("/a", "fr"),)),
    )

    assert {r.language for r in plain.records} == {"en"}
    languages = {str(r.url): r.language for r in german.records}
    assert languages.pop("example.com/a") == "fr"
    assert set(languages.values()) == {"de"}


@pytest.mark.integration
async def test_the_stage_reads_the_crawl_applies_tenant_rules_and_writes_only_when_asked(
    mongo: MongoRepo, mongo_uri: str, tenant: str
) -> None:
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    source_db = f"crawl_{tenant.replace('-', '_')}"
    news = [
        {
            "url": f"https://example.com/news?pg={n}",
            "title": "News",
            "content": f"{NAV}\n\nNews page {n} links [the pricing](https://example.com/pricing).",
            "statusCode": 200,
            "usable": True,
        }
        for n in (1, 2)
    ]
    stories = [
        {
            "url": f"https://example.com/story-{n}",
            "title": "Story",
            "content": f"{NAV}\n\nStory {n} is about something else entirely.",
            "statusCode": 200,
            "usable": True,
        }
        for n in range(5)
    ]
    await client[source_db]["crawl_pages"].insert_many([*news, *stories])
    await client.close()
    await mongo.set_url_rules(tenant, UrlRules(keep_params=frozenset({"pg"})))
    await mongo.set_language_rules(
        tenant, LanguageRules(default_language="fr", prefixes=(("/news", "de"),))
    )

    async with await CrawlSource.connect(mongo_uri, source_db, "crawl_pages") as source:
        _, dry = await prepare_tenant(source, mongo, tenant, source_name="s", write=False)
        assert [s async for batch in mongo.iter_page_summaries(tenant) for s in batch] == []
        corpus, report = await prepare_tenant(source, mongo, tenant, source_name="s")

    assert (dry.pages_written, dry.links_written) == (0, 0)
    # "pg" is kept for this tenant, so page 2 is its own page; page 1 is the default.
    assert sorted(str(r.url) for r in corpus.records if "news" in str(r.url)) == [
        "example.com/news",
        "example.com/news?pg=2",
    ]
    assert (report.documents, report.pages, report.links) == (7, 7, 2)
    assert (report.pages_written, report.links_written) == (7, 2)
    stored = [s async for batch in mongo.iter_page_summaries(tenant) for s in batch]
    assert len(stored) == 7
    assert {s.url: s.language for s in stored if s.language != "fr"} == {
        "example.com/news": "de",
        "example.com/news?pg=2": "de",
    }
    # /news is the home of its language prefix, so a root; nothing it links to was crawled.
    assert (dry.pages_with_depth, report.pages_with_depth) == (1, 1)
    assert {s.url: s.crawl_depth for s in stored if s.crawl_depth is not None} == {
        "example.com/news": 0
    }


async def test_the_stage_rejects_a_blank_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await prepare_tenant(None, None, " ", source_name="s")  # type: ignore[arg-type]
