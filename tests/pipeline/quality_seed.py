"""A planted tenant for the #74 quality evaluation.

Two topics of twelve pages whose content vectors cluster tightly around orthogonal centres.
Page i links to i+1, i+2 and i+3 of its own topic, so a hidden link's source is one of the
nine same-topic pages still eligible for its target, and those rank before every page of the
other topic: recall@10 of the hidden links is 1 by construction. That proves the plumbing,
not the method.

Each page's H1 is its keyword; the trail topic's first page resolves a two-keyword strategic
set instead. The keyword stage writes those keywords as the stored edges. The copy of page i
names page i+4's keyword verbatim, and page i+5's keyword word as four words after the topic.
Anchors: offset 1 names the target's keyword, offset 2 is generic, offset 3 names only the
topic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from voyage_fakes import MODEL, FakeVoyage

from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import LanguageRules, Link, Page, PageRecord
from linking_engine.pipeline.keywords import resolve_tenant_keywords

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TOPICS = ("trail", "tent")
WORDS = (
    "alder",
    "birch",
    "cedar",
    "dogwood",
    "elm",
    "fir",
    "ginkgo",
    "hazel",
    "ivy",
    "juniper",
    "larch",
    "maple",
)
SIZE = len(WORDS)
# The tenant's configured dimension: the graph's hub centroids are indexed at it.
DIMENSION = 2048
OFFSETS = (1, 2, 3)
GENERIC_ANCHOR = "click here"
STRATEGIC = ("Alder Trail Guide", "Trail Alder Boots")
# Pages whose target GSC totals exist: the first six of the trail topic.
GSC_PAGES = 6


def url(topic: str, index: int) -> str:
    return f"example.com/{topic}/{WORDS[index % SIZE]}"


def keyword(topic: str, index: int) -> str:
    return f"{topic.title()} {WORDS[index % SIZE].title()}"


STRATEGIC_URL = url("trail", 0)
URLS = tuple(url(topic, i) for topic in TOPICS for i in range(SIZE))
KEYWORDS = (*(keyword(topic, i) for topic in TOPICS for i in range(SIZE)), *STRATEGIC)


def centre(topic: str) -> np.ndarray:
    vector = np.zeros(DIMENSION)
    vector[TOPICS.index(topic) * (DIMENSION // 2)] = 1.0
    return vector


def page_vectors() -> dict[str, list[float]]:
    rng = np.random.default_rng(5)
    return {
        url(topic, i): (centre(topic) + 0.3 / DIMENSION**0.5 * rng.normal(size=DIMENSION)).tolist()
        for topic in TOPICS
        for i in range(SIZE)
    }


def keyword_vector(text: str) -> list[float]:
    """A keyword sits at its topic's centre, so it is close to its own page's vector."""
    for topic in TOPICS:
        if topic in text.casefold():
            return (centre(topic) * 3.0 + 0.01).tolist()
    return [1.0] * DIMENSION


def respond(texts: Sequence[str]) -> list[list[float]]:
    return [keyword_vector(text) for text in texts]


def voyage() -> FakeVoyage:
    """Voyage for the keywords, at the pages' dimension."""
    return FakeVoyage(dimension=DIMENSION, respond=respond)


def body(topic: str, index: int) -> str:
    near, far = WORDS[(index + 4) % SIZE], WORDS[(index + 5) % SIZE]
    return f"{keyword(topic, index)} guide. Compare {topic} {near} with {far} kit for any {topic}."


def links() -> list[Link]:
    anchors = {1: None, 2: GENERIC_ANCHOR, 3: None}
    found = []
    for topic in TOPICS:
        for i in range(SIZE):
            for offset in OFFSETS:
                target = i + offset
                anchor = anchors[offset] or (
                    keyword(topic, target) if offset == 1 else f"{topic} gear"
                )
                found.append(
                    Link(
                        source_url=url(topic, i),
                        target_url=url(topic, target),
                        position=offset - 1,
                        anchor_text=anchor,
                        surrounding_text="",
                    )
                )
    return found


def record(topic: str, index: int, *, keywords: bool) -> PageRecord:
    text = body(topic, index)
    return PageRecord.model_validate(
        {
            "url": url(topic, index),
            "status_code": 200,
            "usable": True,
            "meta_title": f"{keyword(topic, index)} | Acme" if keywords else None,
            "meta_description": None,
            "h1": keyword(topic, index) if keywords else None,
            "headings": (),
            "body_text": text,
            "word_count": len(text.split()),
            "link_count": 0,
            "content_hash": None,
            "body_hash": body_hash(text),
            "scraped_at": None,
            "source": "test",
            "crawl_url": f"https://{url(topic, index)}",
            "language": "en",
        }
    )


def context_score(source: int, offset: int) -> float:
    return round(0.3 + 0.05 * ((source + offset) % 12), 3)


async def seed_quality(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    *,
    gsc: bool = True,
    keywords: bool = True,
    scores: bool = True,
) -> None:
    """The planted tenant; ``gsc`` adds GSC totals and the strategic keywords, ``keywords``
    the H1s and titles that resolve every page, ``scores`` the stored link relevance. The
    keyword stage runs last, so the keyword edges are stored as production stores them."""
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    await mongo.write_pages(
        tenant, [record(t, i, keywords=keywords) for t in TOPICS for i in range(SIZE)], []
    )
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url(t, i),
                status_code=200,
                is_indexable=True,
                word_count=300,
                language="en",
                crawl_depth=1 + i % 3,
            )
            for t in TOPICS
            for i in range(SIZE)
        ],
    )
    await graph.replace_links(tenant, list(URLS), links())
    vectors = page_vectors()
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model, p.hubId = row.hub, "
        "p.isHubPillar = row.pillar, p.pageRankPercentile = row.pr",
        t=tenant,
        model=MODEL,
        rows=[
            {
                "url": url(t, i),
                "vec": vectors[url(t, i)],
                "hub": TOPICS.index(t),
                "pillar": i == 0,
                "pr": i / SIZE,
            }
            for t in TOPICS
            for i in range(SIZE)
        ],
    )
    if scores:
        await graph._auto(
            "UNWIND $rows AS row "
            "MATCH (:Page {tenantId: $t, url: row.s})-[r:LINKS_TO {position: row.position}]->"
            "(:Page {tenantId: $t}) "
            "SET r.contextRelevance = row.context, r.anchorTargetFit = row.fit, "
            "r.anchorGeneric = row.generic",
            t=tenant,
            rows=[
                {
                    "s": url(t, i),
                    "position": offset - 1,
                    "context": context_score(i, offset),
                    "fit": 0.8 if offset == 1 else None,
                    "generic": offset == 2,
                }
                for t in TOPICS
                for i in range(SIZE)
                for offset in OFFSETS
            ],
        )
    if gsc:
        await mongo._db["gsc_metrics"].insert_many(
            [
                {
                    "tenantId": tenant,
                    "url": url("trail", i),
                    "impressions_28d": 400 + 10 * i,
                    "clicks_28d": 20,
                    "avg_position": 4.0 + i,
                    "query_count": 1,
                }
                for i in range(GSC_PAGES)
            ]
        )
        await mongo._db["strategic_keywords"].insert_many(
            [
                {
                    "tenantId": tenant,
                    "url": STRATEGIC_URL,
                    "keyword": text,
                    "language": "en",
                    "priority": priority,
                    "isPrimary": primary,
                }
                for text, priority, primary in (
                    (STRATEGIC[0], 1, True),
                    (STRATEGIC[1], 2, False),
                )
            ]
        )
    await resolve_tenant_keywords(graph, mongo, tenant)
