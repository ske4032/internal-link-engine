"""The semantic rung over a tenant's pairs (#22), and the vectors anchor selection compares.

Keyword, sentence and phrase vectors come from the tenant's own text-vector caches and are
embedded only when missing. Without Voyage, or after an outage, the rung is skipped with a
reason and every vector the cache does not hold is missing; nothing fails.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import structlog

from linking_engine.anchor.semantic import (
    best_other_cosines,
    candidate_phrases,
    cosines,
    derive_threshold,
    eligible_phrases,
    negative_phrases,
    relevance,
    sample,
    semantic_match,
    shares_stem,
    top_sentences,
)
from linking_engine.errors import EmbeddingAuthError, EmbeddingUnavailableError
from linking_engine.graph.algorithms import NOISE
from linking_engine.models import TenantConfig
from linking_engine.pipeline.embed import NO_MODEL
from linking_engine.pipeline.text_vectors import cached_text_vectors, text_vectors

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from linking_engine.anchor.extraction import SourceIndex
    from linking_engine.anchor.semantic import Phrase
    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import (
        AnchorMatch,
        ExtractionSettings,
        KeywordSource,
        PageStructure,
        SemanticThreshold,
    )
    from linking_engine.pipeline.text_vectors import Kind

log = structlog.get_logger(__name__)

STAGE: Final = "anchor-selection"
NO_KEY: Final = "no Voyage API key"
_KINDS: Final[tuple[Kind, ...]] = ("keywords", "sentences", "phrases")


class AnchorVectors:
    """One run's vectors for one tenant: the pages' content vectors, and keyword, sentence and
    phrase vectors from the tenant's caches, embedded when missing. Once Voyage is missing or
    down, only cached vectors are read and ``skipped_reason()`` says why. Page vectors from
    another model or dimension than the phrases' are left out, so no cosine mixes two models."""

    def __init__(
        self,
        voyage: VoyageClient | None,
        tenant_id: str,
        content: Mapping[str, npt.NDArray[np.float32]],
        *,
        page_models: Collection[str | None],
        cache_dir: Path,
    ) -> None:
        """``page_models`` are the embedding models the content vectors were stored with."""
        tenant = TenantConfig(tenant_id=tenant_id)
        self._voyage = voyage
        self._tenant_id = tenant_id
        self._cache_dir = cache_dir
        self.model = voyage.model if voyage is not None else tenant.embedding_model
        self.dimension = voyage.dimension if voyage is not None else tenant.embedding_dimensions
        self._pages: str | None = None
        if content:
            models = {model or NO_MODEL for model in page_models}
            dimensions = {len(vector) for vector in content.values()}
            if models != {self.model}:
                self._pages = f"page vectors are from {', '.join(sorted(models))}, not {self.model}"
            elif dimensions != {self.dimension}:
                self._pages = (
                    f"page vectors have {', '.join(map(str, sorted(dimensions)))} dimensions, "
                    f"not {self.dimension}"
                )
        if self._pages is not None:
            log.warning(
                "anchors.page_vectors_skipped", stage=STAGE, tenant_id=tenant_id, reason=self._pages
            )
        self._content = (
            {}
            if self._pages is not None
            else {url: _unit(vector) for url, vector in content.items() if vector.any()}
        )
        self._vectors: dict[Kind, dict[str, npt.NDArray[np.float64]]] = {k: {} for k in _KINDS}
        self._counts: dict[Kind, list[int]] = {kind: [0, 0] for kind in _KINDS}
        self._skipped: str | None = NO_KEY if voyage is None else None

    @classmethod
    async def load(
        cls, graph: GraphRepo, voyage: VoyageClient | None, tenant_id: str, *, cache_dir: Path
    ) -> AnchorVectors:
        """The run's vectors, with the tenant's content vectors and their models from the graph."""
        return cls(
            voyage,
            tenant_id,
            await graph.content_vectors(tenant_id),
            page_models=[row.embedding_model for row in await graph.embedding_models(tenant_id)],
            cache_dir=cache_dir,
        )

    async def ensure(self, kind: Kind, texts: Iterable[str]) -> None:
        """Hold a vector for every text: from the cache, else embedded while Voyage works."""
        wanted = set(texts) - self._vectors[kind].keys()
        if not wanted:
            return
        found: Mapping[str, npt.NDArray[np.float32]] = {}
        if self._voyage is not None and self._skipped is None:
            try:
                embedded = await text_vectors(
                    self._voyage,
                    self._tenant_id,
                    kind,
                    wanted,
                    cache_dir=self._cache_dir,
                    stage=STAGE,
                )
            except EmbeddingAuthError as error:
                self._skip(f"Voyage refused the key ({error.error_type})")
            except EmbeddingUnavailableError as error:
                self._skip(f"Voyage unavailable after retries ({error.error_type})")
            else:
                self._counts[kind][0] += embedded.embedded
                self._counts[kind][1] += embedded.cached
                found = embedded.vectors
        if self._skipped is not None:
            found = await cached_text_vectors(
                self._tenant_id,
                kind,
                wanted,
                model=self.model,
                dimension=self.dimension,
                cache_dir=self._cache_dir,
                stage=STAGE,
            )
            self._counts[kind][1] += len(found)
        self._vectors[kind].update((text, _unit(vector)) for text, vector in found.items())

    def _skip(self, reason: str) -> None:
        self._skipped = reason
        log.warning(
            "anchors.semantic_skipped", stage=STAGE, tenant_id=self._tenant_id, reason=reason
        )

    def skipped_reason(self) -> str | None:
        """Why Voyage is not used any more in this run: no key, or down or refusing the key."""
        return self._skipped

    def embedding_skipped_reason(self) -> str | None:
        """Why some vectors are missing: Voyage not used, or page vectors of another model or
        dimension, which leave the page cosines and the placement features empty."""
        return "; ".join(reason for reason in (self._skipped, self._pages) if reason) or None

    def missing(self, kind: Kind, texts: Iterable[str]) -> int:
        """How many distinct texts of ``kind`` have no vector."""
        return len(set(texts) - self._vectors[kind].keys())

    def counts(self, kind: Kind) -> tuple[int, int]:
        """Texts of ``kind`` embedded in this run, and read back from the cache."""
        embedded, cached = self._counts[kind]
        return embedded, cached

    def vector(self, kind: Kind, text: str) -> npt.NDArray[np.float64] | None:
        return self._vectors[kind].get(text)

    def content(self, url: str) -> npt.NDArray[np.float64] | None:
        return self._content.get(url)

    def phrase_page(self, phrase: str, url: str) -> float | None:
        """Cosine of a phrase to a page's content vector; None when either vector is missing."""
        return _cosine(self.vector("phrases", phrase), self.content(url))

    def phrase_keyword(self, phrase: str, keyword: str) -> float | None:
        """Cosine of a phrase to a keyword; None when either vector is missing."""
        return _cosine(self.vector("phrases", phrase), self.vector("keywords", keyword))


def _unit(vector: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    values = np.asarray(vector, dtype=np.float64)
    unit: npt.NDArray[np.float64] = values / np.linalg.norm(values)
    return unit


def _cosine(
    first: npt.NDArray[np.float64] | None, second: npt.NDArray[np.float64] | None
) -> float | None:
    if first is None or second is None:
        return None
    return min(1.0, max(-1.0, float(first @ second)))


def placement_features(
    vectors: AnchorVectors, match: AnchorMatch
) -> tuple[float | None, float | None]:
    """(context relevance, anchor-target fit) of a chosen anchor: its sentence and its phrase
    against the target's content vector, as (1 + cosine) / 2 like the existing links' scores;
    each None when a vector is missing."""
    target = vectors.content(match.target_url)
    return (
        relevance(vectors.vector("sentences", match.sentence), target),
        relevance(vectors.vector("phrases", match.phrase), target),
    )


@dataclass(frozen=True, slots=True)
class SemanticRun:
    # (source, target) to the semantic match of each pair the rung matched.
    matches: dict[tuple[str, str], AnchorMatch]
    threshold: SemanticThreshold
    skipped_reason: str | None
    zero_overlap: int
    # Pairs the rung searched: all it was given when it ran, 0 when skipped.
    invocations: int = 0
    # Pairs with a phrase at or above the threshold but no match: every such phrase held other
    # identifiers than its keyword, or was closer to another target's keywords.
    rejected_identifier: int = 0
    rejected_other_target: int = 0


def _other_topic(first: PageStructure | None, second: PageStructure | None) -> bool:
    """Whether two pages sit in different hubs, or in different content communities when
    either has no hub; False when that cannot be told."""
    if first is None or second is None:
        return False
    hubs = (first.hub_id, second.hub_id)
    if all(hub is not None and hub != NOISE for hub in hubs):
        return hubs[0] != hubs[1]
    communities = (first.content_community_id, second.content_community_id)
    return None not in communities and communities[0] != communities[1]


def _keyword_matrix(
    vectors: AnchorVectors, ranked: Sequence[tuple[int, str, KeywordSource]]
) -> tuple[list[tuple[int, str, KeywordSource]], npt.NDArray[np.float64] | None]:
    """The target's keywords that have a vector, and their vectors as rows."""
    kept = [entry for entry in ranked if vectors.vector("keywords", entry[1]) is not None]
    if not kept:
        return kept, None
    rows = [vectors.vector("keywords", text) for _, text, _ in kept]
    return kept, np.stack([row for row in rows if row is not None])


def _best_to_keywords(
    vectors: AnchorVectors,
    phrases: Sequence[tuple[str, str]],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
) -> list[float]:
    """Each (phrase, target)'s highest cosine to one of the target's keywords."""
    found: list[float] = []
    for phrase, target in phrases:
        vector = vectors.vector("phrases", phrase)
        _, matrix = _keyword_matrix(vectors, keywords.get(target, ()))
        if vector is not None and matrix is not None:
            found.append(float(cosines(vector, matrix).max()))
    return found


async def _threshold(
    vectors: AnchorVectors,
    structure: Mapping[str, PageStructure],
    *,
    all_pairs: Sequence[tuple[str, str]],
    indexes: Mapping[str, SourceIndex],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    inbound: Mapping[str, Sequence[str]],
    override: float | None,
) -> SemanticThreshold | None:
    """The tenant's cut-off: derived from phrases of source pages unrelated to a target, or the
    override; with the recall of the target's descriptive existing anchors either way. None
    when Voyage is unavailable and a phrase it needs is not cached."""
    targets = sorted({target for _, target in all_pairs if keywords.get(target)})
    positives = sample(
        [(anchor, target) for target in targets for anchor in inbound.get(target, ())]
    )
    negatives: list[tuple[str, str]] = []
    if override is None:
        similarities = [
            float(source @ target)
            for source, target in ((vectors.content(s), vectors.content(t)) for s, t in all_pairs)
            if source is not None and target is not None
        ]
        median = statistics.median(similarities) if similarities else 0.0

        def unrelated(source_url: str, target_url: str) -> bool:
            source, target = vectors.content(source_url), vectors.content(target_url)
            return (
                source is not None
                and target is not None
                and _other_topic(structure.get(source_url), structure.get(target_url))
                and float(source @ target) < median
            )

        negatives = negative_phrases([indexes[url] for url in sorted(indexes)], targets, unrelated)
    texts = [phrase for phrase, _ in positives + negatives]
    await vectors.ensure("phrases", texts)
    if _unavailable(vectors, "phrases", texts):
        return None
    return derive_threshold(
        _best_to_keywords(vectors, negatives, keywords),
        _best_to_keywords(vectors, positives, keywords),
        override=override,
    )


@dataclass(frozen=True, slots=True)
class _Rivals:
    """The keyword vectors of one language's pages, as rows."""

    matrix: npt.NDArray[np.float64]
    # Per page, the rows of its keywords no other page of the language has.
    sole: dict[str, list[int]]


def _rivals(
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    structure: Mapping[str, PageStructure],
    vectors: AnchorVectors,
) -> dict[str | None, _Rivals]:
    """Every language's keyword vectors, each row a distinct keyword text of its pages."""
    owners: defaultdict[str | None, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for url, ranked in keywords.items():
        page = structure.get(url)
        for _, text, _ in ranked:
            if vectors.vector("keywords", text) is not None:
                owners[page.language if page else None][text].add(url)
    found: dict[str | None, _Rivals] = {}
    for language, by_text in owners.items():
        texts = sorted(by_text)
        rows = [vectors.vector("keywords", text) for text in texts]
        sole: defaultdict[str, list[int]] = defaultdict(list)
        for column, text in enumerate(texts):
            if len(by_text[text]) == 1:
                sole[next(iter(by_text[text]))].append(column)
        found[language] = _Rivals(np.stack([row for row in rows if row is not None]), dict(sole))
    return found


def _unavailable(vectors: AnchorVectors, kind: Kind, texts: Sequence[str]) -> bool:
    """Whether a needed vector is missing and Voyage cannot provide it."""
    return vectors.skipped_reason() is not None and vectors.missing(kind, texts) > 0


async def semantic_rung(
    graph: GraphRepo,
    vectors: AnchorVectors,
    tenant_id: str,
    *,
    pairs: Sequence[tuple[str, str]],
    all_pairs: Sequence[tuple[str, str]],
    indexes: Mapping[str, SourceIndex],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    existing: Mapping[str, Sequence[tuple[int, int]]],
    inbound: Mapping[str, Sequence[str]],
    settings: ExtractionSettings,
) -> SemanticRun:
    """The semantic match, if any, of each of ``pairs``: the pairs whose target has keywords
    and no lexical match, in processing order. ``all_pairs`` is the whole pair pool,
    ``indexes`` every source page's index, ``existing`` each source's existing anchor spans
    and ``inbound`` each target's descriptive existing anchor texts. With a warm cache it runs
    without Voyage; skipped, with the default or the override as its threshold, when Voyage is
    missing or down and a vector it needs is not cached."""
    override = settings.semantic_threshold
    skipped = derive_threshold([], [], override=override)
    # Every keyword of the tenant: a phrase is checked against other targets' keywords too.
    texts = [text for url in sorted(keywords) for _, text, _ in keywords[url]]
    await vectors.ensure("keywords", texts)
    if _unavailable(vectors, "keywords", texts):
        return SemanticRun({}, skipped, vectors.skipped_reason(), 0)
    structure = {page.url: page for page in await graph.page_structure(tenant_id)}
    threshold = await _threshold(
        vectors,
        structure,
        all_pairs=all_pairs,
        indexes=indexes,
        keywords=keywords,
        inbound=inbound,
        override=override,
    )
    if threshold is None:
        return SemanticRun({}, skipped, vectors.skipped_reason(), 0)

    sources = sorted({source for source, _ in pairs if source in indexes})
    texts = [sentence.text for url in sources for sentence in indexes[url].sentences]
    await vectors.ensure("sentences", texts)
    if _unavailable(vectors, "sentences", texts):
        return SemanticRun({}, threshold, vectors.skipped_reason(), 0)
    proposed: dict[tuple[str, str], list[Phrase]] = {}
    for source, target in pairs:
        index = indexes.get(source)
        ranked, matrix = _keyword_matrix(vectors, keywords.get(target, ()))
        if index is None or not index.sentences or matrix is None:
            continue
        rows = [vectors.vector("sentences", sentence.text) for sentence in index.sentences]
        if any(row is None for row in rows):
            continue
        top = top_sentences(np.stack([row for row in rows if row is not None]), matrix)
        proposed[source, target] = [
            phrase
            for position in sorted(top)
            for phrase in candidate_phrases(index, position, existing=existing.get(source, ()))
        ]
    texts = [phrase.text for phrases in proposed.values() for phrase in phrases]
    await vectors.ensure("phrases", texts)
    if _unavailable(vectors, "phrases", texts):
        return SemanticRun({}, threshold, vectors.skipped_reason(), 0)

    prepared: dict[
        tuple[str, str],
        tuple[
            list[Phrase],
            npt.NDArray[np.float64],
            list[tuple[int, str, KeywordSource]],
            npt.NDArray[np.float64],
        ],
    ] = {}
    checks: defaultdict[str | None, list[tuple[tuple[str, str], int]]] = defaultdict(list)
    for (source, target), phrases in proposed.items():
        ranked, matrix = _keyword_matrix(vectors, keywords.get(target, ()))
        kept = [phrase for phrase in phrases if vectors.vector("phrases", phrase.text) is not None]
        if not kept or matrix is None:
            continue
        rows = [vectors.vector("phrases", phrase.text) for phrase in kept]
        found = np.stack([row for row in rows if row is not None])
        prepared[source, target] = (kept, found, ranked, matrix)
        page = structure.get(target)
        checks[page.language if page else None].extend(
            ((source, target), i)
            for i in eligible_phrases(
                kept,
                found,
                ranked,
                matrix,
                threshold=threshold.value,
                stems=indexes[source].stems,
                brand=indexes[source].brand,
            )
        )
    rivals = _rivals(keywords, structure, vectors)
    other_best: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for language, entries in checks.items():
        rival = rivals.get(language)
        if rival is None or not entries:
            continue
        best = await asyncio.to_thread(
            best_other_cosines,
            np.stack([prepared[pair][1][i] for pair, i in entries]),
            rival.matrix,
            [rival.sole.get(pair[1], []) for pair, _ in entries],
        )
        for (pair, i), value in zip(entries, best.tolist(), strict=True):
            other_best[pair][i] = value

    matches: dict[tuple[str, str], AnchorMatch] = {}
    rejected: Counter[str] = Counter()
    zero_overlap = 0
    by_rank: defaultdict[int, int] = defaultdict(int)
    for (source, target), (kept, found, ranked, matrix) in prepared.items():
        index = indexes[source]
        outcome = semantic_match(
            index,
            target,
            kept,
            found,
            ranked,
            matrix,
            threshold=threshold.value,
            other_best=other_best.get((source, target), {}),
        )
        if outcome.rejected is not None:
            rejected[outcome.rejected] += 1
        match = outcome.match
        if match is None:
            continue
        matches[source, target] = match
        by_rank[match.keyword_rank] += 1
        if not shares_stem(match.phrase, match.keyword, index.stems):
            zero_overlap += 1
            log.debug(
                "anchors.semantic_zero_overlap",
                stage=STAGE,
                tenant_id=tenant_id,
                keyword_rank=match.keyword_rank,
                phrase_words=len(match.phrase.split()),
                similarity=round(match.semantic_similarity or 0.0, 4),
            )
    log.info(
        "anchors.semantic",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=len(pairs),
        proposed=len(proposed),
        matched=len(matches),
        rejected_identifier=rejected["identifier"],
        rejected_other_target=rejected["other_target"],
        zero_overlap=zero_overlap,
        by_keyword_rank=dict(sorted(by_rank.items())),
        threshold=round(threshold.value, 4),
        negatives=threshold.negatives,
        positives=threshold.positives,
        positive_recall=threshold.positive_recall,
        bounded=threshold.bounded,
        fallback=threshold.fallback,
    )
    return SemanticRun(
        matches,
        threshold,
        None,
        zero_overlap,
        invocations=len(pairs),
        rejected_identifier=rejected["identifier"],
        rejected_other_target=rejected["other_target"],
    )
