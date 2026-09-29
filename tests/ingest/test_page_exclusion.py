"""#100: sitemap pages, tenant-excluded paths and pages that are mostly link text stay out of
the pipeline, labelled with why."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pymongo import AsyncMongoClient

from linking_engine.ingest.graph_load import load_tenant_graph
from linking_engine.ingest.mongo_repo import CrawlSource
from linking_engine.ingest.prepare import (
    LINK_ONLY_MIN_LINKS,
    LINK_ONLY_SHARE,
    exclusion,
    prepare_corpus,
    prepare_tenant,
)
from linking_engine.models import (
    EXCLUSION_LABELS,
    CleanedPage,
    CrawlPage,
    ExcludedPage,
    ExclusionReason,
    ExtractedLink,
)
from linking_engine.urls import is_sitemap, normalise_path, under_paths

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

NAV = "- [Pricing](https://example.com/pricing)"


def crawl(path: str, body: str, **fields: Any) -> CrawlPage:
    values: dict[str, Any] = {
        "url": f"https://example.com{path}",
        "title": path,
        "content": f"{NAV}\n\n{body}",
        "status_code": 200,
        "usable": True,
        **fields,
    }
    return CrawlPage(**values)


def links(names: list[str]) -> str:
    return " ".join(f"[{name} guide](https://example.com/{name.lower()})" for name in names)


GUIDES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
SITE = [
    crawl("/pricing", "Plans and prices for every team size and budget."),
    crawl("/alpha", f"The alpha guide explains setup in detail. See {links(['Beta'])} next."),
    crawl("/beta", "The beta guide covers configuration and daily operation for teams."),
    *(crawl(f"/topic-{n}", f"Topic {n} is written as plain copy for readers.") for n in range(4)),
    crawl("/sitemap", f"Site map {links(GUIDES)}"),
    crawl("/guides-links", links(GUIDES)),
    # A listing: its links sit in real copy, so it keeps its place in the pipeline.
    crawl(
        "/guides",
        " ".join(
            f"{links([name])} walks through one task step by step with worked examples."
            for name in GUIDES
        ),
    ),
    crawl("/archive/2020", f"Old posts {links(GUIDES)} from the archive years ago."),
    crawl("/news", f"Company news: read {links(['Alpha'])} and our sitemap page."),
]


def cleaned(words: int, link_words: list[int]) -> CleanedPage:
    anchors = [" ".join(["w"] * n) for n in link_words]
    copy = ["c"] * (words - sum(link_words))
    return CleanedPage(
        url="https://example.com/p",
        body_text=" ".join([*anchors, *copy]),
        links=tuple(
            ExtractedLink(
                target_url=f"https://example.com/t{i}",
                anchor_text=anchor,
                surrounding_text=anchor,
                is_internal=True,
            )
            for i, anchor in enumerate(anchors)
        ),
    )


def test_the_rule_is_at_least_80_percent_link_text_over_at_least_5_links() -> None:
    assert (LINK_ONLY_SHARE, LINK_ONLY_MIN_LINKS) == (0.8, 5)
    at_line = exclusion("example.com/p", cleaned(10, [2, 2, 2, 1, 1]), frozenset())
    assert at_line is not None
    assert (at_line.reason, at_line.words, at_line.link_words, at_line.links) == (
        ExclusionReason.INSUFFICIENT_CONTENT,
        10,
        8,
        5,
    )
    assert at_line.label == "insufficient content: mostly links, please review"
    below_share = cleaned(10, [2, 2, 1, 1, 1])  # 70% link text
    assert exclusion("example.com/p", below_share, frozenset()) is None
    too_few_links = cleaned(8, [2, 2, 2, 2])  # 100% link text, 4 links
    assert exclusion("example.com/p", too_few_links, frozenset()) is None
    assert exclusion("example.com/p", cleaned(0, []), frozenset()) is None


def test_a_sitemap_or_an_excluded_path_goes_whatever_its_copy() -> None:
    copy = cleaned(40, [2])
    sitemap = exclusion("example.com/sitemap", copy, frozenset())
    assert sitemap is not None
    assert sitemap.reason is ExclusionReason.SITEMAP
    tenant = exclusion("example.com/archive/2020", copy, {normalise_path("/archive")})
    assert tenant is not None
    assert tenant.reason is ExclusionReason.TENANT_EXCLUDED
    # A sitemap is named as one even under an excluded path.
    both = exclusion("example.com/archive/sitemap", copy, {normalise_path("/archive")})
    assert both is not None
    assert both.reason is ExclusionReason.SITEMAP
    assert set(EXCLUSION_LABELS) == set(ExclusionReason)


def test_paths_match_whole_segments_case_insensitively_and_never_the_host() -> None:
    paths = {normalise_path(p) for p in ("/Archive/", "resources/links")}
    assert paths == {"/archive", "/resources/links"}
    assert under_paths("example.com/archive", paths)
    assert under_paths("example.com/ARCHIVE/2020/post", paths)
    assert under_paths("https://example.com/resources/links?x=1", paths)
    assert not under_paths("example.com/archives", paths)
    assert not under_paths("archive.example.com/post", paths)
    assert not under_paths("example.com/archive", frozenset())
    assert not is_sitemap("sitemap-tools.example/blog")


def test_prepare_keeps_excluded_pages_and_their_links_out_but_keeps_links_into_them() -> None:
    corpus = prepare_corpus(SITE, source="s", excluded_paths={normalise_path("/archive")})
    urls = {str(record.url) for record in corpus.records}
    reasons = {page.url: page.reason for page in corpus.excluded}
    assert reasons == {
        "example.com/sitemap": ExclusionReason.SITEMAP,
        "example.com/guides-links": ExclusionReason.INSUFFICIENT_CONTENT,
        "example.com/archive/2020": ExclusionReason.TENANT_EXCLUDED,
    }
    assert not urls & set(reasons)
    assert "example.com/guides" in urls, "a listing with copy stays"
    sources = {str(link.source_url) for link in corpus.links}
    assert not sources & set(reasons)
    unexcluded = prepare_corpus(SITE, source="s")
    assert {
        (str(link.source_url), str(link.target_url), link.anchor_text) for link in corpus.links
    } == {
        (str(link.source_url), str(link.target_url), link.anchor_text)
        for link in unexcluded.links
        if str(link.source_url) not in reasons
    }, "every other page's links are unchanged"
    # Other pages' links into an excluded page stay, pointing at a page no stage holds.
    assert all(page.label == EXCLUSION_LABELS[page.reason] for page in corpus.excluded)
    assert [page.url for page in corpus.excluded] == sorted(reasons)


def test_excluded_pages_count_no_template_inlinks_and_give_no_crawl_depth() -> None:
    corpus = prepare_corpus(SITE, source="s")
    assert "example.com/sitemap" not in corpus.inlinks
    records = {str(record.url): record for record in corpus.records}
    assert "example.com/sitemap" not in records
    assert all(record.crawl_depth is None or record.crawl_depth >= 0 for record in records.values())


@pytest.mark.integration
async def test_excluded_paths_are_normalised_refuse_the_root_and_stay_per_tenant(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    assert await mongo.get_excluded_paths(tenant) == frozenset()
    stored = await mongo.set_excluded_paths(tenant, ["/Archive/", " resources/links ", ""])
    assert stored == {"/archive", "/resources/links"}
    assert await mongo.get_excluded_paths(tenant) == stored
    assert await mongo.get_excluded_paths(other) == frozenset()
    with pytest.raises(ValueError, match="root"):
        await mongo.set_excluded_paths(tenant, ["/"])
    assert await mongo.get_excluded_paths(tenant) == stored


@pytest.mark.integration
async def test_the_stage_labels_excluded_pages_and_removes_earlier_copies(
    mongo: MongoRepo, graph: GraphRepo, mongo_uri: str, tenant: str
) -> None:
    other = f"{tenant}-other"
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    source_db = f"crawl_{tenant.replace('-', '_')}"
    documents = [
        {
            "url": str(doc.url),
            "title": doc.title,
            "content": doc.content,
            "statusCode": 200,
            "usable": True,
        }
        for doc in SITE
    ]
    await client[source_db]["crawl_pages"].insert_many(documents)
    await client.close()

    # A first preparation before the tenant excluded /archive keeps it.
    async with await CrawlSource.connect(mongo_uri, source_db, "crawl_pages") as source:
        first, _ = await prepare_tenant(source, mongo, tenant, source_name="s")
        await prepare_tenant(source, mongo, other, source_name="s")
        await load_tenant_graph(mongo, graph, tenant)
        await load_tenant_graph(mongo, graph, other)
        await mongo.set_excluded_paths(tenant, ["/archive"])
        corpus, report = await prepare_tenant(source, mongo, tenant, source_name="s")
    assert "example.com/archive/2020" in {str(r.url) for r in first.records}

    assert report.excluded == {
        ExclusionReason.SITEMAP: 1,
        ExclusionReason.INSUFFICIENT_CONTENT: 1,
        ExclusionReason.TENANT_EXCLUDED: 1,
    }
    stored_pages = {
        summary.url async for batch in mongo.iter_page_summaries(tenant) for summary in batch
    }
    assert "example.com/archive/2020" not in stored_pages, "an earlier copy is deleted"
    assert await mongo.links_for(tenant, ["example.com/archive/2020"]) == []
    labelled = await mongo.excluded_pages(tenant)
    assert labelled == corpus.excluded
    assert all(isinstance(page, ExcludedPage) for page in labelled)
    # The other tenant has no exclusion setting: its archive stays, and only its own
    # sitemap and link-only page are excluded.
    assert {page.url for page in await mongo.excluded_pages(other)} == {
        "example.com/sitemap",
        "example.com/guides-links",
    }

    loaded = await load_tenant_graph(mongo, graph, tenant)
    assert loaded.excluded_pages_removed == 1, "the archive page's earlier node"
    snapshot = await graph.link_graph(tenant)
    crawled = {url for url, ph in zip(snapshot.pages, snapshot.placeholders, strict=True) if not ph}
    assert not crawled & {page.url for page in labelled}
    kept = await graph.link_graph(other)
    assert "example.com/archive/2020" in {
        url for url, ph in zip(kept.pages, kept.placeholders, strict=True) if not ph
    }, "another tenant's page with the same url stays"
