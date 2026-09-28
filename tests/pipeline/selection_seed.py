"""A planted tenant for the #23 gate: fifty pages, each with one ranked keyword, where eleven
keywords (22%) are written in no other page's copy.

Page i's keyword is ``<tree> <object>``. Each of the other 39 keywords is written once, in page
i + 7's copy, verbatim, as a plural, or reordered around one stop word, so each lexical rung
resolves some. A missing keyword's tree word alone is written there instead, which no rung
may take for the keyword. Page 0's keyword is also written in page 20's copy, so its target
gets two anchors; page 30 already links to page 1 with page 1's keyword as the anchor. Voyage answers with a vector drawn from the text's hash, so no two
texts are close and the semantic rung matches nothing: the lexical rungs decide, as planted.
That proves the plumbing, not the method. With ``bare``, one more page has no keyword at all;
with ``textless``, that page has no stored record, so its copy cannot be searched.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from test_keyword_stage import page_record, url
from voyage_fakes import MODEL, FakeVoyage

from linking_engine.models import (
    KeywordRung,
    KeywordSource,
    KeywordTarget,
    LanguageRules,
    Link,
    Page,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TREES = (
    "alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel", "ivy", "juniper",
    "larch", "maple", "oak", "pine", "rowan", "spruce", "willow", "yew", "aspen", "beech",
    "cherry", "hawthorn", "holly", "laurel", "linden", "magnolia", "olive", "poplar", "sequoia",
    "sycamore", "walnut", "acacia", "bamboo", "cypress", "ebony", "fig", "hemlock", "hickory",
    "locust", "mahogany", "myrtle", "palm", "pear", "plum", "redwood", "sassafras", "teak",
    "tamarack", "catalpa", "mulberry",
)  # fmt: skip
OBJECTS = ("lantern", "kayak", "compass", "hammock", "stove")
PAGES = len(TREES)
# Keywords written in no other page's copy.
MISSING = frozenset({3, 7, 12, 16, 21, 25, 30, 34, 39, 43, 48})
# The page whose copy writes page i's keyword.
OFFSET = 7
# (page, the other page whose keyword its copy writes as well).
ECHO = (20, 0)
# (source, target) of the one existing link, anchored with the target's keyword.
EXISTING = (30, 1)
DIMENSION = 2048
BARE = "/gear/bare"
BARE_BODY = "Notes from the shed.\nNothing planted here."


def path(index: int) -> str:
    return f"/gear/{TREES[index]}"


def keyword(index: int) -> str:
    return f"{TREES[index]} {OBJECTS[index % len(OBJECTS)]}"


def mention(index: int) -> str:
    tree, thing = TREES[index], OBJECTS[index % len(OBJECTS)]
    if index in MISSING:
        return f"The {tree} grove is quiet."
    plural = f"{thing}es" if thing.endswith("s") else f"{thing}s"
    return (
        f"Try the {tree} {thing} today.",
        f"Two {tree} {plural} arrived.",
        f"The {thing} with {tree} trim is here.",
    )[index % 3]


def body(index: int) -> str:
    """The page's own topic, then the one planted mention it carries."""
    written = (index - OFFSET) % PAGES
    echo = f" {mention(ECHO[1])}" if index == ECHO[0] else ""
    return f"Notes from the {TREES[index]} workshop.\nOpening hours vary. {mention(written)}{echo}"


def resolved() -> frozenset[str]:
    """The urls of the pages whose keyword is written in another page's copy."""
    return frozenset(url(path(i)) for i in range(PAGES) if i not in MISSING)


def page_vectors() -> dict[str, list[float]]:
    rng = np.random.default_rng(23)
    return {url(path(i)): rng.normal(size=DIMENSION).tolist() for i in range(PAGES)}


def hashed(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    vector: list[float] = np.random.default_rng(seed).normal(size=DIMENSION).tolist()
    return vector


def respond(texts: Sequence[str]) -> list[list[float]]:
    return [hashed(text) for text in texts]


def voyage() -> FakeVoyage:
    return FakeVoyage(dimension=DIMENSION, respond=respond)


async def seed_selection(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    *,
    bare: bool = False,
    textless: int | None = None,
) -> None:
    paths = [path(i) for i in range(PAGES)] + ([BARE] if bare else [])
    vectors = page_vectors()
    if bare:
        vectors[url(BARE)] = hashed(BARE_BODY)
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    await mongo.write_pages(
        tenant,
        [
            page_record(path(i), 200, f"{keyword(i).title()} | Summit", None, body(i), "en")
            for i in range(PAGES)
            if i != textless
        ]
        + ([page_record(BARE, 200, None, None, BARE_BODY, "en")] if bare else []),
        [],
    )
    await graph.upsert_pages(
        tenant,
        [Page(url=url(p), status_code=200, is_indexable=True, language="en") for p in paths],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model",
        t=tenant,
        model=MODEL,
        rows=[{"url": page, "vec": vector} for page, vector in vectors.items()],
    )
    source, target = EXISTING
    await graph.replace_links(
        tenant,
        [url(path(source))],
        [
            Link(
                source_url=url(path(source)),
                target_url=url(path(target)),
                position=0,
                anchor_text=keyword(target),
                surrounding_text=f"See the {keyword(target)} range.",
            )
        ],
    )
    strategic = KeywordSource.CLIENT_STRATEGIC
    await graph.replace_keyword_targets(
        tenant,
        strategic,
        [
            KeywordTarget(
                url=url(path(i)),
                text=keyword(i),
                language="en",
                source=strategic,
                rung=KeywordRung.STRATEGIC,
            )
            for i in range(PAGES)
        ],
    )
