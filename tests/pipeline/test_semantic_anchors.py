"""The semantic rung over a tenant's pairs (#22) with Voyage faked: the planted match, the
derived and overridden threshold, the per-tenant caches, no key and an outage, and the two
placement features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from structlog.testing import capture_logs
from voyage_fakes import FakeVoyage, client, settings
from voyageai.error import AuthenticationError, ServiceUnavailableError

from linking_engine.anchor.extraction import SourceIndex, Stems, extract
from linking_engine.anchor.semantic import DEFAULT_SEMANTIC_THRESHOLD, MIN_NEGATIVES
from linking_engine.models import (
    AnchorRung,
    ExtractionSettings,
    KeywordSource,
    PageStructure,
)
from linking_engine.pipeline.semantic_anchors import (
    NO_KEY,
    AnchorVectors,
    SemanticRun,
    placement_features,
    semantic_rung,
)
from linking_engine.pipeline.text_vectors import cache_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


TENANT = "test-semantic"
DIM = 64
# A toy embedding: each word is one concept, synonyms share one. Stop words carry nothing.
SYNONYMS = {
    "shoes": "shoe",
    "footwear": "shoe",
    "boots": "shoe",
    "paths": "trail",
    "rock": "rocky",
    "shelter": "tents",
    "camping": "camp",
    "pegs": "stakes",
}
STOP = {"for", "with", "on", "before", "a", "the"}


def words(text: str) -> list[str]:
    return [w for w in text.casefold().replace(".", " ").split() if w not in STOP]


def concept_vector(text: str) -> list[float]:
    vector = np.zeros(DIM)
    for word in words(text):
        vector[CONCEPTS[SYNONYMS.get(word, word)]] += 1.0
    vector[DIM - 1] = 1e-3
    return vector.tolist()


def respond(texts: Sequence[str]) -> list[list[float]]:
    return [concept_vector(text) for text in texts]


def voyage(**options: object) -> FakeVoyage:
    return FakeVoyage(dimension=DIM, respond=respond, **options)  # type: ignore[arg-type]


def url(name: str) -> str:
    return f"example.com/{name}"


BODIES = {
    "a0": "Good footwear for rocky paths matters. Pack water for long days.",
    "a1": "Trail shoes guide for beginners.",
    "a2": "Boots with grip help on wet rock. Rest often on steep climbs.",
    "b0": "Winter tents need strong stakes. Camp early before dark.",
    "b1": "A warm shelter keeps camping fun. Bring spare pegs always.",
    "b2": "Winter tents buyer notes.",
}
HUB = {name: 0 if name.startswith("a") else 1 for name in BODIES}
KEYWORDS = {
    url("a1"): [
        (1, "trail shoes", KeywordSource.CLIENT_STRATEGIC),
        (2, "hiking boots", KeywordSource.GSC_OBSERVED),
    ],
    url("b2"): [(1, "winter tents", KeywordSource.INFERRED)],
}
# Same-topic pairs have high content cosines and cross-topic ones low, so the pool's median
# splits them: only the cross-topic pairs are unrelated.
ALL_PAIRS = [
    (url("a0"), url("a1")),
    (url("a2"), url("a1")),
    (url("b0"), url("b2")),
    (url("b1"), url("b2")),
    (url("b0"), url("a1")),
    (url("b1"), url("a1")),
    (url("a0"), url("b2")),
    (url("a2"), url("b2")),
]
# The one pair no lexical rung matches: a0 says "footwear for rocky paths", not "trail shoes".
PAIRS = [(url("a0"), url("a1"))]
INBOUND = {url("a1"): ["trail shoes"], url("b2"): ["spare pegs"]}


# For the guards: c0 names widget 11 where c1's keyword names widget 10, and a3's keyword
# is closer than a1's to every phrase a0 offers a1.
GUARDED_BODIES = {"c0": "The widget 11 upgrade ships soon."}
GUARDED_KEYWORDS = {
    url("c1"): [(1, "widget 10 upgrade", KeywordSource.INFERRED)],
    url("a3"): [(1, "rocky footwear paths", KeywordSource.INFERRED)],
}
TOPIC_WORDS_ALL = (
    "footwear rocky paths boots grip wet steep climbs water long days good matters pack rest often",
    "tents stakes winter camp warm shelter pegs fun strong need early dark keeps bring spare always",
)
# One dimension per concept of the corpus, so no two concepts collide.
CONCEPTS = {
    concept: i
    for i, concept in enumerate(
        sorted(
            {
                SYNONYMS.get(word, word)
                for text in [
                    *BODIES.values(),
                    *(k for ranked in KEYWORDS.values() for _, k, _ in ranked),
                    *(a for anchors in INBOUND.values() for a in anchors),
                    *GUARDED_BODIES.values(),
                    *TOPIC_WORDS_ALL,
                    *(k for ranked in GUARDED_KEYWORDS.values() for _, k, _ in ranked),
                ]
                for word in words(text)
            }
        )
    )
}
assert len(CONCEPTS) < DIM


def content() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    topic = {0: np.asarray(concept_vector("trail shoe")), 1: np.asarray(concept_vector("tents"))}
    return {
        url(name): (topic[HUB[name]] + 0.05 * rng.random(DIM)).astype(np.float32) for name in BODIES
    }


def indexes() -> dict[str, SourceIndex]:
    stems = Stems("en")
    return {url(name): SourceIndex(url(name), body, (), stems) for name, body in BODIES.items()}


class FakeGraph:
    """Only the page structure the negative sample reads its hubs from."""

    def __init__(self) -> None:
        self.calls = 0

    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        self.calls += 1
        return [
            PageStructure(url=url(name), inbound=0, outbound=0, hub_id=HUB[name]) for name in BODIES
        ]


async def run(
    vectors: AnchorVectors, *, override: float | None = None, graph: FakeGraph | None = None
) -> SemanticRun:
    return await semantic_rung(
        graph or FakeGraph(),  # type: ignore[arg-type]
        vectors,
        TENANT,
        pairs=PAIRS,
        all_pairs=ALL_PAIRS,
        indexes=indexes(),
        keywords=KEYWORDS,
        existing={},
        inbound=INBOUND,
        settings=ExtractionSettings(semantic_threshold=override),
    )


MODEL = "voyage-4-large"


def anchor_vectors(fake: FakeVoyage | None, cache: Path, **options: object) -> AnchorVectors:
    found = None if fake is None else client(fake, settings(**options) if options else None)
    return AnchorVectors(found, TENANT, content(), page_models=[MODEL], cache_dir=cache)


def entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_the_planted_pair_has_no_lexical_match() -> None:
    source = indexes()[url("a0")]
    assert extract(source, url("a1"), KEYWORDS[url("a1")])[0] == []


async def test_a_phrase_in_other_words_becomes_a_semantic_match(tmp_path: Path) -> None:
    fake = voyage()
    vectors = anchor_vectors(fake, tmp_path)

    with capture_logs() as logs:
        found = await run(vectors)

    [match] = found.matches.values()
    assert list(found.matches) == PAIRS
    assert (match.rung, match.phrase, match.keyword, match.keyword_rank) == (
        AnchorRung.SEMANTIC,
        "footwear for rocky paths",
        "trail shoes",
        1,
    )
    assert match.keyword_source is KeywordSource.CLIENT_STRATEGIC
    # {shoe, trail, rocky} against {trail, shoe}: 2 / (sqrt 3 x sqrt 2), float16 rounding aside.
    assert match.semantic_similarity == pytest.approx(2 / 6**0.5, abs=2e-3)
    assert BODIES["a0"][match.start : match.end] == match.phrase
    assert (match.sentence, match.sentence_index, match.sentence_start) == (
        "Good footwear for rocky paths matters.",
        0,
        0,
    )
    assert (found.skipped_reason, found.invocations) == (None, len(PAIRS))
    assert found.zero_overlap == 1, "footwear, rocky and paths share no stem with trail shoes"
    [line] = [entry for entry in logs if entry["event"] == "anchors.semantic"]
    assert (line["stage"], line["tenant_id"], line["matched"], line["zero_overlap"]) == (
        "anchor-selection",
        TENANT,
        1,
        1,
    )
    [debug] = [entry for entry in logs if entry["event"] == "anchors.semantic_zero_overlap"]
    assert (debug["keyword_rank"], debug["phrase_words"]) == (1, 4)
    logged = " ".join(str(value) for entry in logs for value in entry.values())
    assert [u for u in map(url, BODIES) if u in logged] == [], "urls in the logs"
    words = ["footwear", "trail shoes", "hiking boots", "winter tents", "spare pegs"]
    assert [w for w in words if w in logged] == [], "texts in the logs"


async def test_a_small_tenant_has_too_few_distinct_negatives_and_falls_back(
    tmp_path: Path,
) -> None:
    graph = FakeGraph()

    found = await run(anchor_vectors(voyage(), tmp_path), graph=graph)

    threshold = found.threshold
    assert graph.calls == 1
    assert 0 < threshold.negatives < MIN_NEGATIVES
    assert (threshold.value, threshold.fallback, threshold.overridden) == (
        DEFAULT_SEMANTIC_THRESHOLD,
        True,
        False,
    )
    # "trail shoes" into a1 reaches the value; "spare pegs" into b2 does not.
    assert (threshold.positives, threshold.positive_recall) == (2, 0.5)


# Twelve pages a topic, four six-word sentences each from the topic's words: enough distinct
# phrases for a derived threshold.
TOPIC_WORDS = {
    0: "footwear rocky paths boots grip wet steep climbs water long days good matters pack rest often",
    1: "tents stakes winter camp warm shelter pegs fun strong need early dark keeps bring spare always",
}


def big_corpus() -> tuple[dict[str, str], dict[str, int]]:
    bodies: dict[str, str] = {}
    hubs: dict[str, int] = {}
    for topic, vocabulary in TOPIC_WORDS.items():
        pool = vocabulary.split()
        for i in range(12):
            name = f"big{topic}-{i}"
            bodies[name] = " ".join(
                " ".join(pool[(3 * i + 5 * j + 7 * k) % len(pool)] for k in range(6)).capitalize()
                + "."
                for j in range(4)
            )
            hubs[name] = topic
    return bodies, hubs


class BigGraph:
    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        _, hubs = big_corpus()
        return [PageStructure(url=url(n), inbound=0, outbound=0, hub_id=h) for n, h in hubs.items()]


async def test_the_threshold_is_derived_from_distinct_unrelated_phrases_and_bounded(
    tmp_path: Path,
) -> None:
    bodies, hubs = big_corpus()
    stems = Stems("en")
    centre = {0: np.asarray(concept_vector("trail shoe")), 1: np.asarray(concept_vector("tents"))}
    rng = np.random.default_rng(2)
    pages = {
        url(name): (centre[hubs[name]] + 0.05 * rng.random(DIM)).astype(np.float32)
        for name in bodies
    }
    targets = {
        url("big0-0"): "trail shoes",
        url("big0-1"): "steep climbs",
        url("big1-0"): "winter tents",
        url("big1-1"): "warm shelter",
    }
    vectors = AnchorVectors(
        client(voyage()), TENANT, pages, page_models=[MODEL], cache_dir=tmp_path
    )

    found = await semantic_rung(
        BigGraph(),  # type: ignore[arg-type]
        vectors,
        TENANT,
        pairs=[],
        all_pairs=[(u, t) for u in pages for t in targets if u != t],
        indexes={url(n): SourceIndex(url(n), body, (), stems) for n, body in bodies.items()},
        keywords={t: [(1, k, KeywordSource.INFERRED)] for t, k in targets.items()},
        existing={},
        inbound={},
        settings=ExtractionSettings(),
    )

    threshold = found.threshold
    # Only cross-topic phrases are negatives, and they share no concept with the keywords: the
    # quantile is near 0, so the value is the lower bound. One same-topic phrase would lift it.
    assert threshold.negatives >= MIN_NEGATIVES
    assert (threshold.value, threshold.bounded, threshold.fallback) == (0.35, True, False)
    assert (threshold.positives, threshold.positive_recall) == (0, None)


async def test_an_override_is_used_as_is_and_skips_the_negatives(tmp_path: Path) -> None:
    strict = await run(anchor_vectors(voyage(), tmp_path), override=0.9)
    loose = await run(anchor_vectors(voyage(), tmp_path), override=0.8)

    assert strict.matches == {}, "0.816 is below the override"
    assert (strict.threshold.value, strict.threshold.negatives, strict.threshold.overridden) == (
        0.9,
        0,
        True,
    )
    assert (strict.threshold.bounded, strict.threshold.fallback) == (False, False)
    assert (strict.threshold.positives, strict.threshold.positive_recall) == (2, 0.5)
    assert list(loose.matches) == PAIRS


async def test_every_vector_is_cached_per_tenant_so_a_rerun_embeds_nothing(tmp_path: Path) -> None:
    first_vectors = anchor_vectors(voyage(), tmp_path)
    first = await run(first_vectors)
    fake = voyage()
    again_vectors = anchor_vectors(fake, tmp_path)

    again = await run(again_vectors)

    assert fake.call_count == 0
    assert again.matches == first.matches
    assert again.threshold == first.threshold
    for kind in ("keywords", "sentences", "phrases"):
        embedded, cached = first_vectors.counts(kind)
        assert embedded > 0, kind
        assert again_vectors.counts(kind) == (0, embedded + cached), kind
        assert cache_path(tmp_path, TENANT, kind).is_file(), kind
    assert entries(tmp_path) == [TENANT]
    other = AnchorVectors(
        client(voyage()), "test-other", content(), page_models=[MODEL], cache_dir=tmp_path
    )
    await other.ensure("phrases", ["footwear for rocky paths"])
    assert other.counts("phrases") == (1, 0), "another tenant reused this tenant's vectors"


async def test_without_a_key_the_rung_is_skipped_with_its_reason(tmp_path: Path) -> None:
    default = await run(anchor_vectors(None, tmp_path))
    overridden = await run(anchor_vectors(None, tmp_path), override=0.5)

    assert (default.matches, default.skipped_reason, default.zero_overlap) == ({}, NO_KEY, 0)
    assert default.invocations == 0, "a skipped rung examined no pair"
    assert (default.threshold.value, default.threshold.fallback) == (
        DEFAULT_SEMANTIC_THRESHOLD,
        True,
    )
    assert (overridden.threshold.value, overridden.threshold.fallback) == (0.5, False)
    assert entries(tmp_path) == []


async def test_a_warm_cache_runs_the_rung_without_voyage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await run(anchor_vectors(voyage(), tmp_path))
    # The cache is read at the tenant's configured model and dimension.
    monkeypatch.setenv("TENANT_EMBEDDING_DIMENSIONS", str(DIM))
    offline = anchor_vectors(None, tmp_path)

    again = await run(offline)

    assert (again.skipped_reason, again.invocations) == (None, len(PAIRS))
    assert (again.matches, again.threshold, again.zero_overlap) == (
        first.matches,
        first.threshold,
        first.zero_overlap,
    )
    assert all(offline.counts(kind)[0] == 0 for kind in ("keywords", "sentences", "phrases"))


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        pytest.param(
            ServiceUnavailableError("down", http_status=503),
            "Voyage unavailable after retries (ServiceUnavailableError)",
            id="outage",
        ),
        pytest.param(
            AuthenticationError("Provided API key is invalid.", http_status=401),
            "Voyage refused the key (AuthenticationError)",
            id="bad-key",
        ),
    ],
)
async def test_a_voyage_failure_skips_the_rung_once_with_its_reason(
    tmp_path: Path, failure: Exception, reason: str
) -> None:
    fake = voyage(failures=[failure])
    vectors = anchor_vectors(fake, tmp_path, max_attempts=1)

    with capture_logs() as logs:
        found = await run(vectors)

    assert (found.matches, found.skipped_reason, found.invocations) == ({}, reason, 0)
    assert vectors.skipped_reason() == reason
    assert fake.call_count == 1, "Voyage was called again after it failed"
    [warning] = [entry for entry in logs if entry["event"] == "anchors.semantic_skipped"]
    assert (warning["stage"], warning["tenant_id"], warning["reason"]) == (
        "anchor-selection",
        TENANT,
        reason,
    )


async def test_after_an_outage_cached_vectors_are_still_read(tmp_path: Path) -> None:
    warm = anchor_vectors(voyage(), tmp_path)
    await warm.ensure("phrases", ["footwear for rocky paths"])
    fake = voyage(failures=[ServiceUnavailableError("down", http_status=503)])
    vectors = anchor_vectors(fake, tmp_path, max_attempts=1)

    await vectors.ensure("phrases", ["footwear for rocky paths", "rocky paths"])

    assert vectors.skipped_reason() is not None
    assert vectors.vector("phrases", "footwear for rocky paths") is not None
    assert vectors.vector("phrases", "rocky paths") is None
    assert vectors.counts("phrases") == (0, 1)


# ── placement features ──────────────────────────────────────────────────────


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


async def test_placement_features_are_the_sentence_and_phrase_against_the_target_page(
    tmp_path: Path,
) -> None:
    vectors = anchor_vectors(voyage(), tmp_path)
    [match] = (await run(vectors)).matches.values()
    page = content()[url("a1")]

    context, fit = placement_features(vectors, match)

    assert context is not None
    assert fit is not None
    assert context == pytest.approx(
        (1 + cosine(concept_vector(match.sentence), page)) / 2, abs=2e-3
    )
    assert fit == pytest.approx((1 + cosine(concept_vector(match.phrase), page)) / 2, abs=2e-3)
    assert 0 <= context <= 1
    assert 0 <= fit <= 1


async def test_each_placement_feature_is_missing_without_its_vectors(tmp_path: Path) -> None:
    vectors = anchor_vectors(voyage(), tmp_path)
    [match] = (await run(vectors)).matches.values()
    unused = voyage()

    assert placement_features(anchor_vectors(None, tmp_path / "empty"), match) == (None, None)
    # Every vector below comes from the cache the run filled; Voyage is never called.
    no_page = AnchorVectors(client(unused), TENANT, {}, page_models=[], cache_dir=tmp_path)
    await no_page.ensure("sentences", [match.sentence])
    await no_page.ensure("phrases", [match.phrase])
    assert no_page.vector("phrases", match.phrase) is not None
    assert placement_features(no_page, match) == (None, None), "the target has no page vector"
    sentence_only = AnchorVectors(
        client(unused), TENANT, content(), page_models=[MODEL], cache_dir=tmp_path
    )
    await sentence_only.ensure("sentences", [match.sentence])
    context, fit = placement_features(sentence_only, match)
    assert context is not None
    assert fit is None
    assert unused.call_count == 0


async def test_identifier_and_other_target_refusals_are_counted_per_pair(tmp_path: Path) -> None:
    stems = Stems("en")
    guarded = {
        **indexes(),
        url("c0"): SourceIndex(url("c0"), GUARDED_BODIES["c0"], (), stems),
    }
    pairs = [(url("a0"), url("a1")), (url("c0"), url("c1"))]

    with capture_logs() as logs:
        found = await semantic_rung(
            FakeGraph(),  # type: ignore[arg-type]
            anchor_vectors(voyage(), tmp_path),
            TENANT,
            pairs=pairs,
            all_pairs=[*ALL_PAIRS, pairs[1]],
            indexes=guarded,
            keywords={**KEYWORDS, **GUARDED_KEYWORDS},
            existing={},
            inbound=INBOUND,
            settings=ExtractionSettings(semantic_threshold=0.6),
        )

    # "widget 11 upgrade" reaches 2/3 on "widget 10 upgrade" but names another version; every
    # phrase a0 offers a1 is closer to a3's "rocky footwear paths".
    assert found.matches == {}
    assert (found.rejected_identifier, found.rejected_other_target) == (1, 1)
    assert found.invocations == 2
    [line] = [entry for entry in logs if entry["event"] == "anchors.semantic"]
    assert (line["rejected_identifier"], line["rejected_other_target"]) == (1, 1)


async def test_without_the_rival_target_the_pair_matches_again(tmp_path: Path) -> None:
    found = await semantic_rung(
        FakeGraph(),  # type: ignore[arg-type]
        anchor_vectors(voyage(), tmp_path),
        TENANT,
        pairs=PAIRS,
        all_pairs=ALL_PAIRS,
        indexes=indexes(),
        keywords={**KEYWORDS, url("c1"): GUARDED_KEYWORDS[url("c1")]},
        existing={},
        inbound=INBOUND,
        settings=ExtractionSettings(semantic_threshold=0.6),
    )

    assert list(found.matches) == PAIRS
    assert (found.rejected_identifier, found.rejected_other_target) == (0, 0)


# ── page vectors of another model or dimension ──────────────────────────────


@pytest.mark.parametrize(
    ("page_models", "dimension", "reason"),
    [
        pytest.param(
            ["voyage-3-large"],
            DIM,
            "page vectors are from voyage-3-large, not voyage-4-large",
            id="other-model",
        ),
        pytest.param(
            [None], DIM, "page vectors are from <no model>, not voyage-4-large", id="no-model"
        ),
        pytest.param(
            [MODEL, "voyage-3-large"],
            DIM,
            "page vectors are from voyage-3-large, voyage-4-large, not voyage-4-large",
            id="mixed",
        ),
        pytest.param(
            [MODEL], 8, f"page vectors have 8 dimensions, not {DIM}", id="other-dimension"
        ),
    ],
)
async def test_page_vectors_that_do_not_match_the_phrases_are_left_out_with_a_reason(
    tmp_path: Path, page_models: list[str | None], dimension: int, reason: str
) -> None:
    pages = {u: vector[:dimension] for u, vector in content().items()}

    with capture_logs() as logs:
        vectors = AnchorVectors(
            client(voyage()), TENANT, pages, page_models=page_models, cache_dir=tmp_path
        )
    found = await run(vectors)

    assert vectors.content(url("a1")) is None
    assert vectors.embedding_skipped_reason() == reason
    assert vectors.skipped_reason() is None, "Voyage itself is fine"
    [warning] = [entry for entry in logs if entry["event"] == "anchors.page_vectors_skipped"]
    assert (warning["tenant_id"], warning["reason"]) == (TENANT, reason)
    assert found.matches, "the rung still matches; only the page-vector features are missing"
    for match in found.matches.values():
        await vectors.ensure("sentences", [match.sentence])
        assert placement_features(vectors, match) == (None, None)
        assert vectors.phrase_page(match.phrase, match.target_url) is None


async def test_matching_page_vectors_and_no_page_vectors_are_no_mismatch(tmp_path: Path) -> None:
    kept = AnchorVectors(
        client(voyage()), TENANT, content(), page_models=[MODEL], cache_dir=tmp_path
    )
    empty = AnchorVectors(client(voyage()), TENANT, {}, page_models=[], cache_dir=tmp_path)

    assert kept.content(url("a1")) is not None
    assert (kept.embedding_skipped_reason(), empty.embedding_skipped_reason()) == (None, None)


def test_both_reasons_are_joined_when_voyage_and_the_pages_are_unusable(tmp_path: Path) -> None:
    no_key = AnchorVectors(None, TENANT, {}, page_models=[], cache_dir=tmp_path)
    both = AnchorVectors(
        None, TENANT, content(), page_models=["voyage-3-large"], cache_dir=tmp_path
    )

    assert no_key.embedding_skipped_reason() == NO_KEY
    assert both.embedding_skipped_reason() == (
        f"{NO_KEY}; page vectors are from voyage-3-large, not voyage-4-large"
    )
