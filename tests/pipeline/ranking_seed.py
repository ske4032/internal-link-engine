"""A planted tenant for the #24-#27 ranker, and #24's graded labels from its planted truth.

Three topics of 24 pages: trail and tent sit close in content, kayak far from both. Page i of a
topic links to pages i+1 to i+6 of its own topic, each link anchored with its target's keyword
inside a sentence of page i's copy, so hiding a link frees its span on the held-out view and
the exact rung finds it there. Page i's copy also names page i+8's keyword without a link, and
each trail page names the tent page of its index: no body link crosses trail and tent, the
planted bridge gap. Two strategic pages, one trail and one kayak, have no inbound body link
(orphans); pages 3 and 9 of their topic name them. Four noise pages are reached from the menu
template only, with vectors in no topic. Voyage answers with a vector drawn from the text's
hash, so the semantic rung matches nothing. That proves the plumbing, not the method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas
from selection_seed import respond
from test_keyword_stage import page_record, url
from voyage_fakes import MODEL, FakeVoyage

from linking_engine.discovery.features import FEATURE_COLUMNS
from linking_engine.graph.algorithms import NOISE
from linking_engine.models import LanguageRules, Link, Page
from linking_engine.pipeline.keywords import resolve_tenant_keywords

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TOPICS: Final = ("trail", "tent", "kayak")
# Close in content, never linked to each other.
BRIDGE: Final = frozenset({"trail", "tent"})
WORDS: Final = (
    "alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel", "ivy", "juniper",
    "larch", "maple", "oak", "pine", "rowan", "spruce", "willow", "yew", "aspen", "beech",
    "cherry", "holly", "laurel", "linden", "walnut",
)  # fmt: skip
SIZE: Final = 24
OFFSETS: Final = (1, 2, 3, 4, 5, 6)
# Named in the copy without a link.
MENTIONED: Final = 8
ORPHAN_INDEX: Final = SIZE
ORPHAN_TOPICS: Final = ("trail", "kayak")
ORPHAN_MENTIONS: Final = (3, 9)
NOISE_WORDS: Final = {
    "contact": "Desk",
    "privacy": "Charter",
    "careers": "Board",
    "press": "Archive",
}
NOISE_NAMES: Final = tuple(NOISE_WORDS)
DIMENSION: Final = 2048
# Cosine of the trail and tent centres.
BRIDGE_COSINE: Final = 0.6


def path(topic: str, index: int) -> str:
    return f"/{topic}/{WORDS[index % len(WORDS)]}"


def page_url(topic: str, index: int) -> str:
    return url(path(topic, index))


def keyword(topic: str, index: int) -> str:
    return f"{topic.title()} {WORDS[index].title()}"


def phrase(topic: str, index: int) -> str:
    """The keyword as the copy writes it."""
    return keyword(topic, index).lower()


def noise_url(name: str) -> str:
    return url(f"/about/{name}")


def noise_keyword(name: str) -> str:
    """Two words no other page's copy or keyword shares, so no rung matches it elsewhere and
    no column name or MLflow text holds it by chance."""
    return f"{name.title()} {NOISE_WORDS[name]}"


TOPIC_PAGES: Final = tuple(page_url(t, i) for t in TOPICS for i in range(SIZE))
ORPHANS: Final = frozenset(page_url(t, ORPHAN_INDEX) for t in ORPHAN_TOPICS)
NOISE_URLS: Final = frozenset(noise_url(name) for name in NOISE_NAMES)
URLS: Final = (*TOPIC_PAGES, *sorted(ORPHANS), *sorted(NOISE_URLS))
TOPIC_OF: Final = {
    **{page_url(t, i): t for t in TOPICS for i in range(SIZE)},
    **{page_url(t, ORPHAN_INDEX): t for t in ORPHAN_TOPICS},
}


def linked(topic: str, index: int) -> str:
    return f"Pack the {phrase(topic, index)} before the weekend."


def body(topic: str, index: int) -> str:
    sentences = [f"{keyword(topic, index)} guide."]
    sentences += [linked(topic, (index + k) % SIZE) for k in OFFSETS]
    sentences.append(f"Compare the {phrase(topic, (index + MENTIONED) % SIZE)} at the shop.")
    if topic == "trail":
        sentences.append(f"Pair it with the {phrase('tent', index)} tonight.")
    if topic in ORPHAN_TOPICS and index in ORPHAN_MENTIONS:
        sentences.append(f"New this season: the {phrase(topic, ORPHAN_INDEX)} range.")
    return " ".join(sentences)


def orphan_body(topic: str) -> str:
    return f"{keyword(topic, ORPHAN_INDEX)} guide. Everything new for the {topic} season."


def noise_body(name: str) -> str:
    return f"{noise_keyword(name)} at Acme. Reach the Acme team on any working day."


# Every page's copy and keyword, by url.
COPY: Final = {
    **{page_url(t, i): body(t, i) for t in TOPICS for i in range(SIZE)},
    **{page_url(t, ORPHAN_INDEX): orphan_body(t) for t in ORPHAN_TOPICS},
    **{noise_url(name): noise_body(name) for name in NOISE_NAMES},
}
KEYWORD_OF: Final = {
    **{page_url(t, i): keyword(t, i) for t in TOPICS for i in range(SIZE)},
    **{page_url(t, ORPHAN_INDEX): keyword(t, ORPHAN_INDEX) for t in ORPHAN_TOPICS},
    **{noise_url(name): noise_keyword(name) for name in NOISE_NAMES},
}
KEYWORDS: Final = tuple(KEYWORD_OF.values())


def names(source: str, target: str) -> bool:
    """Whether the source's copy writes the target's keyword."""
    return KEYWORD_OF[target].casefold() in COPY[source].casefold()


def links() -> list[Link]:
    """Page i to pages i+1 to i+6 of its topic, anchored where its copy names them."""
    return [
        Link(
            source_url=page_url(t, i),
            target_url=page_url(t, (i + k) % SIZE),
            position=k - 1,
            anchor_text=phrase(t, (i + k) % SIZE),
            surrounding_text=linked(t, (i + k) % SIZE),
        )
        for t in TOPICS
        for i in range(SIZE)
        for k in OFFSETS
    ]


def body_links() -> frozenset[tuple[str, str]]:
    return frozenset((link.source_url, link.target_url) for link in links())


def inbound() -> dict[str, int]:
    """Stored inbound body links per page."""
    counts = dict.fromkeys(URLS, 0)
    for _, target in body_links():
        counts[target] += 1
    return counts


def voyage() -> FakeVoyage:
    return FakeVoyage(dimension=DIMENSION, respond=respond)


def centre(topic: str) -> npt.NDArray[np.float64]:
    vector = np.zeros(DIMENSION)
    if topic == "tent":
        vector[0], vector[1] = BRIDGE_COSINE, (1 - BRIDGE_COSINE**2) ** 0.5
    else:
        vector[TOPICS.index(topic) * (DIMENSION // 4)] = 1.0
    return vector


def page_vectors() -> dict[str, list[float]]:
    rng = np.random.default_rng(24)
    spread = 0.3 / DIMENSION**0.5
    vectors = {
        page: (centre(topic) + spread * rng.normal(size=DIMENSION)).tolist()
        for page, topic in TOPIC_OF.items()
    }
    vectors.update({page: rng.normal(size=DIMENSION).tolist() for page in sorted(NOISE_URLS)})
    return vectors


def grade(source: str, target: str) -> int:
    """#24's graded label of one pair from the planted truth alone: 3 across the bridge gap or
    into an orphan strategic page, 2 within a planted topic, 1 into a noise page, else 0."""
    if source == target:
        raise ValueError("a pair joins two pages")
    first, second = TOPIC_OF.get(source), TOPIC_OF.get(target)
    if target in ORPHANS or {first, second} == BRIDGE:
        return 3
    if first is not None and first == second:
        return 2
    if target in NOISE_URLS:
        return 1
    return 0


def planted_grades(pairs: Iterable[tuple[str, str]]) -> npt.NDArray[np.int8]:
    return np.fromiter((grade(source, target) for source, target in pairs), dtype=np.int8)


def graded_frame(*, seed: int = 24, noise: float = 0.35) -> pandas.DataFrame:
    """Every ordered pair of planted pages as round 0, labelled by `planted_grades`. A few
    feature columns carry the planted truth blurred by ``noise``, so one tree cannot fit the
    grades and boosting has something to add; every other feature column is NaN."""
    rng = np.random.default_rng(seed)
    pairs = [(s, t) for s in URLS for t in URLS if s != t]
    sources, targets = (np.array(side) for side in zip(*pairs, strict=True))
    vectors = {page: np.asarray(v) / np.linalg.norm(v) for page, v in page_vectors().items()}
    n = len(pairs)

    def flip(share: float) -> npt.NDArray[np.bool_]:
        return rng.random(n) < share

    same = np.array([TOPIC_OF.get(s, s) == TOPIC_OF.get(t) for s, t in pairs])
    orphan = np.isin(targets, sorted(ORPHANS))
    in_hub = np.isin(targets, sorted(TOPIC_OF))
    counts = inbound()
    frame = pandas.DataFrame(np.nan, index=range(n), columns=list(FEATURE_COLUMNS))
    frame["content_cosine"] = [float(vectors[s] @ vectors[t]) for s, t in pairs] + rng.normal(
        0, noise, n
    )
    frame["same_hub"] = (same ^ flip(noise / 2)).astype(float)
    frame["target_is_orphan"] = (orphan ^ flip(noise / 3)).astype(float)
    frame["target_hub_size"] = np.where(in_hub ^ flip(noise / 3), SIZE, np.nan)
    frame["target_inbound_count"] = [counts[t] for t in targets] + rng.poisson(noise * 4, n)
    frame.insert(0, "label", planted_grades(pairs))
    frame.insert(0, "target_url", targets)
    frame.insert(0, "source_url", sources)
    frame.insert(0, "round", np.zeros(n, dtype=np.int16))
    return frame


async def seed_ranking(graph: GraphRepo, mongo: MongoRepo, tenant: str) -> None:
    """The planted tenant in both stores, as the stages before the ranker leave it; the
    keyword stage runs last so the keyword edges are stored as production stores them."""
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    records = [
        page_record(path(t, i), 200, f"{keyword(t, i)} | Acme", keyword(t, i), body(t, i), "en")
        for t in TOPICS
        for i in range(SIZE)
    ]
    records += [
        page_record(
            path(t, ORPHAN_INDEX),
            200,
            f"{keyword(t, ORPHAN_INDEX)} | Acme",
            keyword(t, ORPHAN_INDEX),
            orphan_body(t),
            "en",
        )
        for t in ORPHAN_TOPICS
    ]
    records += [
        page_record(
            f"/about/{name}",
            200,
            f"{noise_keyword(name)} | Acme",
            noise_keyword(name),
            noise_body(name),
            "en",
        )
        for name in NOISE_NAMES
    ]
    await mongo.write_pages(tenant, records, [])
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=page,
                status_code=200,
                is_indexable=True,
                word_count=300,
                language="en",
                crawl_depth=None if page in ORPHANS else 1 + i % 3,
                menu_inlinks=len(TOPIC_PAGES) if page in NOISE_URLS else 0,
            )
            for i, page in enumerate(URLS)
        ],
    )
    await graph.replace_links(tenant, list(TOPIC_PAGES), links())
    vectors = page_vectors()
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model, p.hubId = row.hub, "
        "p.isHubPillar = row.pillar, p.pageRankPercentile = row.pr",
        t=tenant,
        model=MODEL,
        rows=[
            {
                "url": page,
                "vec": vectors[page],
                "hub": TOPICS.index(TOPIC_OF[page]) if page in TOPIC_OF else NOISE,
                "pillar": page in {page_url(t, 0) for t in TOPICS},
                "pr": i / len(URLS),
            }
            for i, page in enumerate(URLS)
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row "
        "MATCH (:Page {tenantId: $t, url: row.s})-[r:LINKS_TO {position: row.position}]->"
        "(:Page {tenantId: $t}) "
        "SET r.contextRelevance = row.context, r.anchorTargetFit = 0.8, r.anchorGeneric = false",
        t=tenant,
        rows=[
            {"s": link.source_url, "position": link.position, "context": 0.6 + 0.05 * link.position}
            for link in links()
        ],
    )
    await mongo._db["strategic_keywords"].insert_many(
        [
            {
                "tenantId": tenant,
                "url": page_url(t, ORPHAN_INDEX),
                "keyword": keyword(t, ORPHAN_INDEX),
                "language": "en",
                "priority": 1,
                "isPrimary": True,
            }
            for t in ORPHAN_TOPICS
        ]
    )
    await resolve_tenant_keywords(graph, mongo, tenant)
