"""The semantic rung over a tenant's pairs (#22), and the vectors anchor selection compares.

Keyword, sentence and phrase vectors come from the tenant's own text-vector caches and are
embedded only when missing. Without Voyage, or after an outage, the rung is skipped with a
reason and every vector the cache does not hold is missing; nothing fails. Each text's vector
is held once, at the cache's precision, and cosines are float32 products of per-source blocks.
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
    OTHER_TARGET_CHUNK,
    best_other_by_similarity,
    candidate_phrases,
    derive_threshold,
    eligible_phrases_by_similarity,
    negative_phrases,
    relevance,
    sample,
    semantic_match_by_similarity,
    shares_stem,
    top_sentences_by_similarity,
)
from linking_engine.errors import EmbeddingAuthError, EmbeddingUnavailableError
from linking_engine.graph.algorithms import NOISE
from linking_engine.models import TenantConfig
from linking_engine.pipeline.embed import NO_MODEL
from linking_engine.pipeline.text_vectors import cached_text_rows, text_rows

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from linking_engine.anchor.extraction import SourceIndex
    from linking_engine.anchor.semantic import Phrase, SemanticOutcome
    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import (
        AnchorMatch,
        ExtractionSettings,
        KeywordSource,
        PageStructure,
        SemanticThreshold,
    )
    from linking_engine.pipeline.text_vectors import Kind, Rows

log = structlog.get_logger(__name__)

STAGE: Final = "anchor-selection"
NO_KEY: Final = "no Voyage API key"
_KINDS: Final[tuple[Kind, ...]] = ("keywords", "sentences", "phrases")
# A phrase's margin between another target's keywords and its own target's this close to a tie
# is decided again in float64, where float32 error cannot break the tie.
NEAR_TIE: Final = 1e-5


class _Store:
    """One kind's vectors: a row per text, in blocks at the cache's precision."""

    __slots__ = ("_blocks", "_size", "_starts", "rows")

    def __init__(self) -> None:
        self.rows: dict[str, int] = {}
        self._blocks: list[npt.NDArray[np.floating]] = []
        self._starts: list[int] = []
        self._size = 0

    def add(self, found: Rows) -> None:
        if not found.texts:
            return
        self._blocks.append(found.matrix)
        self._starts.append(self._size)
        self.rows.update(
            zip(found.texts, range(self._size, self._size + len(found.texts)), strict=True)
        )
        self._size += len(found.texts)

    def units(self, rows: npt.NDArray[np.intp], dimension: int) -> npt.NDArray[np.float32]:
        """The rows as float32 unit vectors."""
        found = np.empty((len(rows), dimension), dtype=np.float32)
        self._fill(rows, found)
        return found

    def exact_units(self, rows: npt.NDArray[np.intp], dimension: int) -> npt.NDArray[np.float64]:
        """The rows as float64 unit vectors, normalised in float64."""
        found = np.empty((len(rows), dimension), dtype=np.float64)
        self._fill(rows, found)
        return found

    def _fill(self, rows: npt.NDArray[np.intp], found: npt.NDArray[np.floating]) -> None:
        blocks = np.searchsorted(self._starts, rows, side="right") - 1
        for block in np.unique(blocks).tolist():
            chosen = blocks == block
            found[chosen] = self._blocks[block][rows[chosen] - self._starts[block]]
        norms = np.linalg.norm(found, axis=1, keepdims=True)
        np.divide(found, norms, out=found, where=norms > 0)


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
        self._stores: dict[Kind, _Store] = {kind: _Store() for kind in _KINDS}
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
        store = self._stores[kind]
        wanted = set(texts) - store.rows.keys()
        if not wanted:
            return
        if self._voyage is not None and self._skipped is None:
            try:
                found = await text_rows(
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
                self._counts[kind][0] += len(found.embedded.texts)
                self._counts[kind][1] += len(found.cached.texts)
                store.add(found.cached)
                store.add(found.embedded)
        if self._skipped is not None:
            cached = await cached_text_rows(
                self._tenant_id,
                kind,
                wanted,
                model=self.model,
                dimension=self.dimension,
                cache_dir=self._cache_dir,
                stage=STAGE,
            )
            self._counts[kind][1] += len(cached.texts)
            store.add(cached)

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
        return len(set(texts) - self._stores[kind].rows.keys())

    def counts(self, kind: Kind) -> tuple[int, int]:
        """Texts of ``kind`` embedded in this run, and read back from the cache."""
        embedded, cached = self._counts[kind]
        return embedded, cached

    def index(self, kind: Kind, text: str) -> int | None:
        """The row of the text's vector among those of ``kind``; None when it has none."""
        return self._stores[kind].rows.get(text)

    def rows(self, kind: Kind, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """The texts' vectors as float32 unit rows, in order; every text must have one."""
        store = self._stores[kind]
        found = np.fromiter((store.rows[text] for text in texts), dtype=np.intp, count=len(texts))
        return store.units(found, self.dimension)

    def exact_rows(self, kind: Kind, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """The texts' vectors as float64 unit rows, for the few cosines a near tie needs."""
        store = self._stores[kind]
        found = np.fromiter((store.rows[text] for text in texts), dtype=np.intp, count=len(texts))
        return store.exact_units(found, self.dimension)

    def vector(self, kind: Kind, text: str) -> npt.NDArray[np.float32] | None:
        """The text's vector as a float32 unit vector; None when it has none."""
        row = self.index(kind, text)
        if row is None:
            return None
        found: npt.NDArray[np.float32] = self._stores[kind].units(
            np.asarray([row], dtype=np.intp), self.dimension
        )[0]
        return found

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
    first: npt.NDArray[np.floating] | None, second: npt.NDArray[np.floating] | None
) -> float | None:
    if first is None or second is None:
        return None
    return min(1.0, max(-1.0, float(first @ second)))


def _cosines(
    rows: npt.NDArray[np.float32], columns: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Cosine of every unit row to every unit column, clipped to [-1, 1]."""
    found: npt.NDArray[np.float32] = np.clip(rows @ columns.T, -1.0, 1.0)
    return found


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


def _with_vectors(
    vectors: AnchorVectors, keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]]
) -> dict[str, list[tuple[int, str, KeywordSource]]]:
    """Each page's ranked keywords that have a vector."""
    return {
        url: [entry for entry in ranked if vectors.index("keywords", entry[1]) is not None]
        for url, ranked in keywords.items()
    }


def _keyword_rows(
    vectors: AnchorVectors, keywords: Iterable[Sequence[tuple[int, str, KeywordSource]]]
) -> tuple[dict[str, int], npt.NDArray[np.float32]]:
    """The distinct keywords of the ranked lists as unit rows, and each keyword's row."""
    row = {
        text: i
        for i, text in enumerate(
            dict.fromkeys(text for ranked in keywords for _, text, _ in ranked)
        )
    }
    return row, vectors.rows("keywords", list(row))


def _best_to_keywords(
    vectors: AnchorVectors,
    phrases: Sequence[tuple[str, str]],
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
) -> list[float]:
    """Each (phrase, target)'s highest cosine to one of the target's keywords."""
    by_target: defaultdict[str, list[str]] = defaultdict(list)
    for phrase, target in phrases:
        if ranked.get(target) and vectors.index("phrases", phrase) is not None:
            by_target[target].append(phrase)
    found: list[float] = []
    for target, texts in by_target.items():
        _, keyword_rows = _keyword_rows(vectors, [ranked[target]])
        found.extend(_cosines(vectors.rows("phrases", texts), keyword_rows).max(axis=1).tolist())
    return found


async def _threshold(
    vectors: AnchorVectors,
    structure: Mapping[str, PageStructure],
    *,
    all_pairs: Sequence[tuple[str, str]],
    indexes: Mapping[str, SourceIndex],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
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
        _best_to_keywords(vectors, negatives, ranked),
        _best_to_keywords(vectors, positives, ranked),
        override=override,
    )


@dataclass(frozen=True, slots=True)
class _Rivals:
    """One language's distinct keyword texts with a vector."""

    texts: list[str]
    # Per page, the rows of its own keywords: never its rivals, also when another page shares
    # one, since a shared keyword cannot be closer than the page's own best.
    own: dict[str, list[int]]


def _rivals(
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    structure: Mapping[str, PageStructure],
) -> dict[str | None, _Rivals]:
    """Every language's keywords, each row a distinct keyword text of its pages."""
    owners: defaultdict[str | None, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for url, entries in ranked.items():
        page = structure.get(url)
        for _, text, _ in entries:
            owners[page.language if page else None][text].add(url)
    found: dict[str | None, _Rivals] = {}
    for language, by_text in owners.items():
        texts = sorted(by_text)
        own: defaultdict[str, list[int]] = defaultdict(list)
        for column, text in enumerate(texts):
            for url in by_text[text]:
                own[url].append(column)
        found[language] = _Rivals(texts, dict(own))
    return found


def _rival_margins(
    vectors: AnchorVectors,
    rival: _Rivals,
    entries: Sequence[tuple[tuple[str, str], int, str]],
) -> list[float]:
    """Each (pair, phrase, text) entry's margin: the phrase's highest cosine to the language's
    keywords other than its target's own, less its highest to its target's own; -inf when no
    other keyword remains. Both come from one product, each distinct phrase once and
    OTHER_TARGET_CHUNK phrases per product; a margin within NEAR_TIE of zero is recomputed
    from float64 cosines."""
    keyword_rows = vectors.rows("keywords", rival.texts)
    texts = list(dict.fromkeys(text for _, _, text in entries))
    row = {text: i for i, text in enumerate(texts)}
    members: list[list[int]] = [[] for _ in texts]
    for position, (_, _, text) in enumerate(entries):
        members[row[text]].append(position)
    found = np.full(len(entries), -np.inf)
    for start in range(0, len(texts), OTHER_TARGET_CHUNK):
        chunk = texts[start : start + OTHER_TARGET_CHUNK]
        similarity = _cosines(vectors.rows("phrases", chunk), keyword_rows)
        chosen = [position for i in range(start, start + len(chunk)) for position in members[i]]
        rows = [row[entries[position][2]] - start for position in chosen]
        own = [rival.own[entries[position][0][1]] for position in chosen]
        others = best_other_by_similarity(similarity, rows, own)
        for at, (position, line, columns) in enumerate(zip(chosen, rows, own, strict=True)):
            margin = others[at] - float(similarity[line, columns].max())
            if abs(margin) <= NEAR_TIE:
                close = np.flatnonzero(similarity[line] >= others[at] - NEAR_TIE).tolist()
                margin = _exact_margin(
                    vectors, entries[position][2], rival, sorted(set(close) - set(columns)), columns
                )
            found[position] = margin
    margins: list[float] = found.tolist()
    return margins


def _exact_margin(
    vectors: AnchorVectors,
    phrase: str,
    rival: _Rivals,
    others: Sequence[int],
    own: Sequence[int],
) -> float:
    """The phrase's highest float64 cosine to the ``others`` keyword columns, less its highest
    to the ``own`` ones."""
    vector = vectors.exact_rows("phrases", [phrase])[0]

    def best(columns: Sequence[int]) -> float:
        found = vectors.exact_rows("keywords", [rival.texts[column] for column in columns])
        return float(np.clip(found @ vector, -1.0, 1.0).max())

    return best(others) - best(own)


def _top_sentences(
    vectors: AnchorVectors,
    index: SourceIndex,
    targets: Sequence[str],
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
) -> dict[str, list[int]]:
    """Per target, the source's sentences closest to its keywords, in text order: one product
    of the source's sentences and its targets' distinct keywords."""
    column, keyword_rows = _keyword_rows(vectors, [ranked[target] for target in targets])
    similarity = _cosines(
        vectors.rows("sentences", [sentence.text for sentence in index.sentences]), keyword_rows
    )
    return {
        target: sorted(
            top_sentences_by_similarity(
                similarity[:, [column[text] for _, text, _ in ranked[target]]]
            )
        )
        for target in targets
    }


def _pair_similarities(
    vectors: AnchorVectors,
    phrases: Mapping[str, Sequence[Phrase]],
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
) -> dict[str, npt.NDArray[np.float32]]:
    """Per target, its phrases' cosines to its keywords: one product of the source's distinct
    phrases and its targets' distinct keywords, of which each pair keeps its own cells."""
    row = {
        text: i
        for i, text in enumerate(
            dict.fromkeys(phrase.text for found in phrases.values() for phrase in found)
        )
    }
    column, keyword_rows = _keyword_rows(vectors, [ranked[target] for target in phrases])
    similarity = _cosines(vectors.rows("phrases", list(row)), keyword_rows)
    return {
        target: similarity[
            np.ix_(
                [row[phrase.text] for phrase in found],
                [column[text] for _, text, _ in ranked[target]],
            )
        ]
        for target, found in phrases.items()
    }


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
    missing or down and a vector it needs is not cached. Sources are taken one at a time, so
    beside the vectors only one source's cosines are held."""
    override = settings.semantic_threshold
    skipped = derive_threshold([], [], override=override)
    # Every keyword of the tenant: a phrase is checked against other targets' keywords too.
    texts = [text for url in sorted(keywords) for _, text, _ in keywords[url]]
    await vectors.ensure("keywords", texts)
    if _unavailable(vectors, "keywords", texts):
        return SemanticRun({}, skipped, vectors.skipped_reason(), 0)
    ranked = _with_vectors(vectors, keywords)
    structure = {page.url: page for page in await graph.page_structure(tenant_id)}
    threshold = await _threshold(
        vectors,
        structure,
        all_pairs=all_pairs,
        indexes=indexes,
        keywords=keywords,
        ranked=ranked,
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
    targets_of: defaultdict[str, list[str]] = defaultdict(list)
    for source, target in dict.fromkeys(pairs):
        index = indexes.get(source)
        if index is not None and index.sentences and ranked.get(target):
            targets_of[source].append(target)
    tops: dict[tuple[str, str], list[int]] = {}
    phrases_at: defaultdict[str, dict[int, list[Phrase]]] = defaultdict(dict)
    for source, targets in targets_of.items():
        index = indexes[source]
        if vectors.missing("sentences", [sentence.text for sentence in index.sentences]):
            continue
        for target, top in _top_sentences(vectors, index, targets, ranked).items():
            tops[source, target] = top
            for position in top:
                if position not in phrases_at[source]:
                    phrases_at[source][position] = candidate_phrases(
                        index, position, existing=existing.get(source, ())
                    )
    texts = [
        phrase.text for at in phrases_at.values() for phrases in at.values() for phrase in phrases
    ]
    await vectors.ensure("phrases", texts)
    if _unavailable(vectors, "phrases", texts):
        return SemanticRun({}, threshold, vectors.skipped_reason(), 0)

    outcomes: dict[tuple[str, str], SemanticOutcome] = {}
    pending: dict[tuple[str, str], tuple[list[Phrase], npt.NDArray[np.float32]]] = {}
    checks: defaultdict[str | None, list[tuple[tuple[str, str], int, str]]] = defaultdict(list)
    for source, targets in targets_of.items():
        index = indexes[source]
        kept: dict[str, list[Phrase]] = {}
        for target in targets:
            if (source, target) not in tops:
                continue
            proposed = [
                phrase
                for position in tops[source, target]
                for phrase in phrases_at[source][position]
                if vectors.index("phrases", phrase.text) is not None
            ]
            if proposed:
                kept[target] = proposed
        if not kept:
            continue
        for target, similarity in _pair_similarities(vectors, kept, ranked).items():
            eligible = eligible_phrases_by_similarity(
                kept[target],
                ranked[target],
                similarity,
                threshold=threshold.value,
                stems=index.stems,
                brand=index.brand,
            )
            if not eligible:
                outcomes[source, target] = semantic_match_by_similarity(
                    index,
                    target,
                    kept[target],
                    ranked[target],
                    similarity,
                    threshold=threshold.value,
                )
                continue
            pending[source, target] = (kept[target], similarity)
            page = structure.get(target)
            checks[page.language if page else None].extend(
                ((source, target), i, kept[target][i].text) for i in eligible
            )
    rivals = _rivals(ranked, structure)
    other_best: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for language, entries in checks.items():
        rival = rivals.get(language)
        if rival is None or not entries:
            continue
        margins = await asyncio.to_thread(_rival_margins, vectors, rival, entries)
        for (pair, i, _), margin in zip(entries, margins, strict=True):
            # The rival's cosine, as its margin over the pair's own best: both sides of the
            # comparison come from one product, so a tie stays a tie.
            other_best[pair][i] = float(pending[pair][1][i].max()) + margin
    for (source, target), (phrases, similarity) in pending.items():
        outcomes[source, target] = semantic_match_by_similarity(
            indexes[source],
            target,
            phrases,
            ranked[target],
            similarity,
            threshold=threshold.value,
            other_best=other_best.get((source, target), {}),
        )

    matches: dict[tuple[str, str], AnchorMatch] = {}
    rejected: Counter[str] = Counter()
    zero_overlap = 0
    by_rank: defaultdict[int, int] = defaultdict(int)
    for pair in dict.fromkeys(pairs):
        outcome = outcomes.get(pair)
        if outcome is None:
            continue
        if outcome.rejected is not None:
            rejected[outcome.rejected] += 1
        match = outcome.match
        if match is None:
            continue
        matches[pair] = match
        by_rank[match.keyword_rank] += 1
        if not shares_stem(match.phrase, match.keyword, indexes[pair[0]].stems):
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
        proposed=len(tops),
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
