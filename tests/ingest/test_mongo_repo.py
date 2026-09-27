from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseAuthError, DatabaseReadError, DatabaseUnavailableError
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.ingest.mongo_repo import INDEXES, CrawlSource, MongoRepo
from linking_engine.models import AnchorRules, Heading, LinkRecord, PageRecord
from linking_engine.urls import UrlRules

BASE = "example.com"
SCRAPED = datetime(2026, 9, 19, 15, 40, 30, tzinfo=UTC)


def url(path: str) -> str:
    return f"{BASE}{path}"


def record(path: str, links: int = 0, **fields: object) -> PageRecord:
    data: dict[str, object] = {
        "url": url(path),
        "status_code": 200,
        "usable": True,
        "meta_title": f"Title {path}",
        "meta_description": None,
        "h1": "Heading",
        "headings": (Heading(level=1, text="Heading"),),
        "body_text": "Body text.",
        "word_count": 2,
        "link_count": links,
        "content_hash": "abc",
        "scraped_at": SCRAPED,
        "source": "crawl.pages_v2",
        "crawl_url": url(path),
    }
    data.update(fields)
    data.setdefault("body_hash", body_hash(str(data["body_text"])))
    return PageRecord.model_validate(data)


def link_record(source: str, position: int, target: str, *, internal: bool = True) -> LinkRecord:
    return LinkRecord(
        source_url=str(url(source)),
        position=position,
        target_url=str(url(target) if internal else f"https://other.test{target}"),
        anchor_text=f"anchor {position}",
        surrounding_text="around the anchor",
        is_internal=internal,
    )


async def test_unreachable_server_raises_unavailable() -> None:
    with pytest.raises(DatabaseUnavailableError):
        await MongoRepo.connect("mongodb://127.0.0.1:1", "db", timeout_ms=200)


async def test_invalid_uri_raises_unavailable() -> None:
    with pytest.raises(DatabaseUnavailableError, match="invalid connection settings"):
        await MongoRepo.connect("http://example.com", "db")


@pytest.mark.integration
async def test_wrong_password_raises_auth_error(mongo_uri: str) -> None:
    bad = mongo_uri.replace("test:test@", "test:wrong@")
    with pytest.raises(DatabaseAuthError):
        await MongoRepo.connect(bad, "linking_engine_test")


@pytest.mark.integration
async def test_indexes_apply_twice_and_all_exist(mongo: MongoRepo, mongo_uri: str) -> None:
    await mongo.ensure_indexes()
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    db = client["linking_engine_test"]
    for name, models in INDEXES.items():
        info = await db[name].index_information()
        for model in models:
            spec = model.document
            assert spec["name"] in info, (name, spec["name"])
            assert info[spec["name"]]["key"] == list(spec["key"].items())
    ttl = (await db["recommendations"].index_information())["created_ttl"]
    assert ttl["expireAfterSeconds"] == 30 * 24 * 3600
    await client.close()


@pytest.mark.integration
async def test_write_and_read_pages_and_links(
    mongo: MongoRepo, tenant: str, mongo_uri: str
) -> None:
    pages = [record("/a", links=2), record("/b")]
    links = [link_record("/a", 0, "/b"), link_record("/a", 1, "/x", internal=False)]
    assert await mongo.write_pages(tenant, pages, links) == (2, 2, 0)
    assert await mongo.get_pages(tenant, [url("/b"), url("/a")]) == pages
    assert await mongo.links_for(tenant, [url("/a"), url("/b")]) == links

    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri, tz_aware=True)
    stored = await client["linking_engine_test"]["pages"].find_one(
        {"tenantId": tenant, "url": url("/a")}
    )
    assert stored is not None
    assert stored["scrapedAt"] == SCRAPED
    assert stored["headings"] == [{"level": 1, "text": "Heading"}]
    assert {"metaTitle", "bodyText", "linkCount", "preparedAt", "tenantId"} <= set(stored)
    await client.close()


@pytest.mark.integration
async def test_body_hash_round_trips_through_every_read(
    mongo: MongoRepo, tenant: str, mongo_uri: str
) -> None:
    bodies = {"/a": "Body text.", "/b": "Caf\u00e9 body, \u65e5\u672c."}
    await mongo.write_pages(
        tenant, [record(path, body_text=text) for path, text in bodies.items()], []
    )
    expected = {url(path): body_hash(text) for path, text in bodies.items()}
    assert len(set(expected.values())) == 2

    records = await mongo.get_pages(tenant, list(expected))
    assert {str(r.url): r.body_hash for r in records} == expected
    summaries = [s async for batch in mongo.iter_page_summaries(tenant) for s in batch]
    assert {str(s.url): s.body_hash for s in summaries} == expected

    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    stored = await client["linking_engine_test"]["pages"].find_one(
        {"tenantId": tenant, "url": url("/b")}
    )
    await client.close()
    assert stored is not None
    assert stored["bodyHash"] == expected[url("/b")]


@pytest.mark.integration
async def test_document_without_body_hash_fails_on_read(
    mongo: MongoRepo, tenant: str, mongo_uri: str
) -> None:
    document = {
        "tenantId": tenant,
        "url": url("/legacy"),
        "statusCode": 200,
        "usable": True,
        "metaTitle": None,
        "metaDescription": None,
        "h1": None,
        "headings": [],
        "bodyText": "Body text.",
        "wordCount": 2,
        "linkCount": 0,
        "contentHash": None,
        "scrapedAt": None,
        "source": "crawl.pages_v2",
    }
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    await client["linking_engine_test"]["pages"].insert_one(document)
    await client.close()
    with pytest.raises(DatabaseReadError, match=r"does not fit PageSummary(.|\n)*body_hash"):
        _ = [b async for b in mongo.iter_page_summaries(tenant)]
    with pytest.raises(DatabaseReadError, match=r"does not fit PageRecord(.|\n)*body_hash"):
        await mongo.get_pages(tenant, [url("/legacy")])


@pytest.mark.integration
async def test_rewrite_converges_and_deletes_stale_links(mongo: MongoRepo, tenant: str) -> None:
    await mongo.write_pages(
        tenant, [record("/a", links=3)], [link_record("/a", i, "/b") for i in range(3)]
    )
    assert await mongo.write_pages(
        tenant, [record("/a", links=1)], [link_record("/a", 0, "/c")]
    ) == (
        1,
        1,
        2,
    )
    [only] = await mongo.links_for(tenant, [url("/a")])
    assert (only.position, str(only.target_url)) == (0, url("/c"))


@pytest.mark.integration
async def test_write_pages_validates_link_counts(mongo: MongoRepo, tenant: str) -> None:
    with pytest.raises(ValueError, match="link_count 2, 1 links given"):
        await mongo.write_pages(tenant, [record("/a", links=2)], [link_record("/a", 0, "/b")])
    with pytest.raises(ValueError, match="whose page is not being written"):
        await mongo.write_pages(tenant, [record("/a")], [link_record("/z", 0, "/b")])


@pytest.mark.integration
async def test_batched_reads_do_not_skip_or_repeat(mongo: MongoRepo, tenant: str) -> None:
    pages = [record(f"/p{i}", links=3) for i in range(5)]
    links = [link_record(f"/p{i}", j, "/t") for i in range(5) for j in range(3)]
    await mongo.write_pages(tenant, pages, links, batch_size=4)

    batches = [b async for b in mongo.iter_page_summaries(tenant, batch_size=2)]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sorted(str(s.url) for b in batches for s in b) == sorted(str(p.url) for p in pages)

    got = await mongo.links_for(tenant, [str(p.url) for p in pages], batch_size=2)
    assert sorted((str(lk.source_url), lk.position) for lk in got) == sorted(
        (str(lk.source_url), lk.position) for lk in links
    )


@pytest.mark.integration
async def test_tenants_are_isolated(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await mongo.write_pages(t, [record("/a", links=1)], [link_record("/a", 0, "/b")])
    assert await mongo.delete_tenant(tenant) == 2
    assert await mongo.get_pages(tenant, [url("/a")]) == []
    assert len(await mongo.get_pages(other, [url("/a")])) == 1
    assert len(await mongo.links_for(other, [url("/a")])) == 1
    await mongo.delete_tenant(other)


@pytest.mark.integration
async def test_document_that_does_not_fit_raises_read_error(
    mongo: MongoRepo, tenant: str, mongo_uri: str
) -> None:
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    await client["linking_engine_test"]["pages"].insert_one(
        {"tenantId": tenant, "url": url("/bad"), "statusCode": 42, "wordCount": 1}
    )
    await client.close()
    with pytest.raises(DatabaseReadError, match="does not fit PageSummary"):
        _ = [b async for b in mongo.iter_page_summaries(tenant)]


@pytest.mark.integration
async def test_crawl_source_reads_the_crawler_shape(mongo_uri: str) -> None:
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    await client["crawl_test"]["pages_v2"].insert_many(
        [
            {
                "url": f"https://{BASE}/c{i}",
                "title": "T",
                "description": "D",
                "content": "# Hello",
                "statusCode": 200,
                "usable": True,
                "contentHash": "h",
                "scrapedAt": "2026-09-19T15:40:30+00:00",
                "internalLinks": [f"https://{BASE}/"],
                "source": "example.com",
            }
            for i in range(3)
        ]
    )
    await client.close()
    async with await CrawlSource.connect(mongo_uri, "crawl_test", "pages_v2") as source:
        pages = [p async for batch in source.iter_pages(batch_size=2) for p in batch]
    assert [str(p.url) for p in pages] == [f"https://{BASE}/c{i}" for i in range(3)]
    assert pages[0].scraped_at == datetime(2026, 9, 19, 15, 40, 30, tzinfo=UTC)
    assert pages[0].content == "# Hello"


@pytest.mark.integration
async def test_url_rules_round_trip_per_tenant(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    assert await mongo.get_url_rules(tenant) == UrlRules()
    rules = UrlRules(keep_params={"Announcement_PG"}, drop_params={"p"})
    await mongo.set_url_rules(tenant, rules)
    assert await mongo.get_url_rules(tenant) == UrlRules(
        keep_params={"announcement_pg"}, drop_params={"p"}
    )
    assert await mongo.get_url_rules(other) == UrlRules()
    await mongo.delete_tenant(tenant)
    assert await mongo.get_url_rules(tenant) == UrlRules()


@pytest.mark.integration
async def test_anchor_rules_round_trip_per_tenant(mongo: MongoRepo, tenant: str) -> None:
    assert await mongo.get_anchor_rules(tenant) == AnchorRules()
    rules = AnchorRules(generic_add={"download now"}, generic_remove={"details"})
    await mongo.set_anchor_rules(tenant, rules)
    assert await mongo.get_anchor_rules(tenant) == rules
    assert await mongo.get_anchor_rules(f"{tenant}-other") == AnchorRules()
    await mongo.delete_tenant(tenant)
    assert await mongo.get_anchor_rules(tenant) == AnchorRules()
