"""Quality evaluation stage: every check of one tenant as one report, read-only against both
stores. Held-out links live in memory only, so no held-out feature ever reaches the feature
cache; the only file written is the tenant's keyword vector cache."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas
import structlog

from linking_engine.anchor.extraction import (
    DEFAULT_THRESHOLD,
    SourceIndex,
    Stems,
    extract,
    keyword_tokens,
)
from linking_engine.anchor.generic import is_generic
from linking_engine.audit.relevance import score_distribution
from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import (
    CHUNK_PAIRS,
    FEATURE_COLUMNS,
    code_digest,
    feature_chunks,
    feature_report,
    missing_pages,
    page_contexts,
    to_frame,
)
from linking_engine.discovery.scoring import default_weights, score_frame, weights_hash
from linking_engine.discovery.signals import build_page_signals
from linking_engine.errors import DatabaseReadError, EmbeddingAuthError, EmbeddingUnavailableError
from linking_engine.graph.algorithms import (
    NOISE,
    Partition,
    build_link_graphs,
    link_pass_graph,
    link_states,
    page_rank,
    partition,
    percentile_rank,
    pillars,
)
from linking_engine.gsc import normalise_term
from linking_engine.ml.quality import (
    ALERT_BAND,
    HIDE_SEED,
    HIDE_SHARE,
    KEYWORD_ORIGINS,
    LENGTH_BINS,
    LINK_DERIVED_COLUMNS,
    QUALITY_STAGE,
    RECALL_KS,
    SIGNAL_MARGIN,
    alerts,
    anchor_matches,
    auc,
    copy_words,
    feature_auc,
    has_signal,
    hide_links,
    keyword_origin,
    keyword_words,
    length_bin,
    quality_metrics,
    random_recall,
    recall_at,
    relevance_groups,
)
from linking_engine.models import (
    AnchorMatchCheck,
    AnchorRung,
    CoverageCheck,
    ExtractionSettings,
    FeatureSignalCheck,
    KeywordCheck,
    KeywordExtractability,
    KeywordRelevance,
    KeywordRung,
    LinkGraphSnapshot,
    LinkRelevanceCheck,
    QualityReport,
    QualityVersions,
    RankRelevance,
    RecallAtK,
    RetrievalCheck,
    ScorerCheck,
    ScorerWeights,
    SourceExtractability,
)
from linking_engine.models.scoring import SCORE_HISTOGRAM_BINS
from linking_engine.pipeline.embed import NO_MODEL
from linking_engine.pipeline.keywords import plan_keywords
from linking_engine.pipeline.text_vectors import text_vectors

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence, Set

    import numpy.typing as npt

    from linking_engine.discovery.features import PageContext
    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import (
        CandidateSet,
        CheckName,
        CommunityContext,
        FeatureReport,
        KeywordSource,
        KeywordTarget,
        LinkRelevance,
        LinkText,
        PageStructure,
        QualityAlert,
        QualityBaseline,
        RelevanceGroup,
        ResolvedKeyword,
        TargetCandidates,
    )
    from linking_engine.pipeline.keywords import KeywordPlan

log = structlog.get_logger(__name__)

STAGE: Final = QUALITY_STAGE
UNKNOWN_SHA: Final = "unknown"
_RUNGS: Final = tuple(AnchorRung)
_GIT_TIMEOUT_S: Final = 5


def git_sha() -> str:
    """``GIT_SHA`` when set, else the checkout's HEAD, else "unknown"."""
    found = os.environ.get("GIT_SHA", "").strip()
    if found:
        return found
    git = shutil.which("git")
    if git is None:
        return UNKNOWN_SHA
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else UNKNOWN_SHA


@dataclass(frozen=True, slots=True)
class HeldOutView:
    """The tenant's graph as it would be without the hidden links."""

    links: tuple[tuple[str, str], ...]
    structure: tuple[PageStructure, ...]
    context: tuple[CommunityContext, ...]


def held_out_view(
    snapshot: LinkGraphSnapshot,
    structure: Sequence[PageStructure],
    context: Sequence[CommunityContext],
    vectors: Mapping[str, npt.NDArray[np.float32]],
    hidden: Set[tuple[str, str]],
) -> HeldOutView:
    """Every link-derived page value recomputed without ``hidden`` by the production
    functions: inbound and outbound counts, PageRank percentile, link community, hub pillar
    and orphan state. Hubs and the keyword and content communities come from content, not
    links, so they stay as stored."""
    links = tuple(link for link in snapshot.links if link not in hidden)
    graphs = build_link_graphs(snapshot.model_copy(update={"links": links}))
    crawled = graphs.crawled
    vertex_of = {graphs.url_of(v): v for v in crawled}
    unknown = sorted(
        ({page.url for page in structure} | {c.url for c in context}) - vertex_of.keys()
    )
    if unknown:
        raise DatabaseReadError(
            "neo4j",
            f"{len(unknown)} pages of {snapshot.tenant_id!r} are not crawled pages of its link "
            f"graph, first {unknown[0]!r}",
        )
    ranks = page_rank(graphs)
    percentiles = dict(
        zip(
            (graphs.url_of(v) for v in crawled),
            percentile_rank(ranks[list(crawled)]).tolist(),
            strict=True,
        )
    )
    link_graph, link_vertices = link_pass_graph(graphs)
    communities = partition(link_graph, link_vertices, weighted=False).labels()
    no_inbound, _ = link_states(graphs)
    orphan = {graphs.url_of(v): bool(flag) for v, flag in zip(crawled, no_inbound, strict=True)}
    clustered = [
        (vertex_of[page.url], page.hub_id)
        for page in structure
        if page.hub_id is not None and page.hub_id != NOISE
    ]
    hub_pillars = pillars(
        Partition(tuple(v for v, _ in clustered), tuple(h for _, h in clustered), 0.0, 0),
        {vertex_of[url]: vector for url, vector in vectors.items() if url in vertex_of},
        ranks,
    )
    inbound: Counter[str] = Counter()
    outbound: Counter[str] = Counter()
    for source, target in set(links):
        if source != target and source in vertex_of and target in vertex_of:
            inbound[target] += 1
            outbound[source] += 1
    return HeldOutView(
        links=links,
        structure=tuple(
            page.model_copy(
                update={
                    "inbound": inbound[page.url],
                    "outbound": outbound[page.url],
                    "is_orphan": orphan[page.url],
                    "page_rank_percentile": percentiles[page.url],
                    "link_community_id": communities.get(vertex_of[page.url]),
                    "is_hub_pillar": vertex_of[page.url] in hub_pillars,
                }
            )
            for page in structure
        ),
        context=tuple(
            item.model_copy(update={"link_community_id": communities.get(vertex_of[item.url])})
            for item in context
        ),
    )


def retrieval_check(
    body_link_pairs: int,
    hidden: Set[tuple[str, str]],
    found: CandidateSet,
    pool: Set[str],
    languages: Mapping[str, str | None],
) -> RetrievalCheck | None:
    """Recall of the hidden links a retrieval over the held-out view could return: the
    target is a retrieval target and the source is in the pool in the target's language.
    None when no hidden link can be recovered."""
    by_target = {entry.target_url: entry for entry in found.targets}
    ranks: list[int | None] = []
    eligible: list[int] = []
    for source, target in sorted(hidden):
        entry = by_target.get(target)
        if entry is None or source not in pool or languages.get(source) != languages.get(target):
            continue
        rank = entry.sources.index(source) + 1 if source in entry.sources else None
        ranks.append(rank)
        eligible.append(entry.eligible)
    if not ranks:
        return None
    return RetrievalCheck(
        hide_share=HIDE_SHARE,
        seed=HIDE_SEED,
        body_link_pairs=body_link_pairs,
        hidden=len(hidden),
        recoverable=len(ranks),
        candidates=found.report.candidates,
        recall=tuple(
            RecallAtK(k=k, recall=recall_at(ranks, k), random=random_recall(eligible, k))
            for k in RECALL_KS
        ),
    )


def pair_checks(
    tenant_id: str,
    targets: Sequence[TargetCandidates],
    pages: Mapping[str, PageContext],
    hidden: Set[tuple[str, str]],
    weights: ScorerWeights,
    *,
    chunk_pairs: int = CHUNK_PAIRS,
) -> tuple[FeatureSignalCheck | None, ScorerCheck | None, FeatureReport]:
    """Feature signal and the scorer over the held-out candidate pairs, labelled by whether
    each pair is a hidden link, and the data gaps of their features. The matrix is built a
    chunk at a time and kept as float32 only."""
    total = sum(len(entry.sources) for entry in targets)
    values = np.empty((total, len(FEATURE_COLUMNS)), dtype=np.float32)

    def frames() -> Iterator[pandas.DataFrame]:
        row = 0
        for chunk in feature_chunks(targets, pages, chunk_pairs=chunk_pairs):
            frame = to_frame(chunk)
            values[row : row + len(frame)] = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(
                dtype=np.float32
            )
            row += len(frame)
            yield frame

    gaps = feature_report(
        tenant_id, frames(), cache_key=STAGE, cache_hit=False, started=time.perf_counter()
    )
    labels = np.fromiter(
        ((source, entry.target_url) in hidden for entry in targets for source in entry.sources),
        dtype=bool,
        count=total,
    )
    positives = int(labels.sum())
    if not positives or positives == total:
        return None, None, gaps
    columns = tuple(
        feature_auc(name, labels, values[:, i]) for i, name in enumerate(FEATURE_COLUMNS)
    )
    signal = FeatureSignalCheck(
        pairs=total,
        positives=positives,
        margin=SIGNAL_MARGIN,
        columns=columns,
        features_with_signal=sum(1 for entry in columns if has_signal(entry)),
    )
    position = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    frame = pandas.DataFrame(
        {
            "source_url": [source for entry in targets for source in entry.sources],
            "target_url": [entry.target_url for entry in targets for _ in entry.sources],
            **{f.column: values[:, position[f.column]] for f in weights.features},
        }
    )
    scores = score_frame(frame, weights)["score"].to_numpy(dtype=np.float64)
    score_auc = auc(labels, scores)
    unmoved = tuple(f for f in weights.features if f.column not in LINK_DERIVED_COLUMNS)
    score_auc_unmoved = None
    if unmoved:
        restricted = ScorerWeights(
            version=weights.version, features=unmoved, tier_shares=weights.tier_shares
        )
        score_auc_unmoved = auc(
            labels, score_frame(frame, restricted)["score"].to_numpy(dtype=np.float64)
        )
    if score_auc is None or (unmoved and score_auc_unmoved is None):
        raise AssertionError("both classes are present")
    total_weight = sum(f.weight for f in weights.features)
    best = max(columns, key=lambda entry: entry.ranker_auc)
    best_unmoved = max(
        (entry for entry in columns if entry.column not in LINK_DERIVED_COLUMNS),
        key=lambda entry: entry.ranker_auc,
    )
    hidden_counts, _ = np.histogram(scores[labels], bins=SCORE_HISTOGRAM_BINS, range=(0.0, 100.0))
    other_counts, _ = np.histogram(scores[~labels], bins=SCORE_HISTOGRAM_BINS, range=(0.0, 100.0))
    scorer = ScorerCheck(
        weights_version=weights.version,
        score_auc=score_auc,
        best_feature=best.column,
        best_feature_auc=best.ranker_auc,
        score_auc_lift=score_auc - best.ranker_auc,
        link_derived_weight_share=1 - sum(f.weight for f in unmoved) / total_weight,
        score_auc_excl_link_counts=score_auc_unmoved,
        best_feature_excl_link_counts=best_unmoved.column,
        best_feature_auc_excl_link_counts=best_unmoved.ranker_auc,
        score_auc_lift_excl_link_counts=(
            None if score_auc_unmoved is None else score_auc_unmoved - best_unmoved.ranker_auc
        ),
        hidden_histogram=tuple(int(n) for n in hidden_counts),
        other_histogram=tuple(int(n) for n in other_counts),
    )
    return signal, scorer, gaps


def _ranked(plan: KeywordPlan) -> dict[str, list[KeywordTarget]]:
    ranked: defaultdict[str, list[KeywordTarget]] = defaultdict(list)
    for targets in plan.by_source.values():
        for target in targets:
            if target.rank is not None:
                ranked[target.url].append(target)
    return {
        url: sorted(targets, key=lambda target: (target.rank or 0, target.text))
        for url, targets in ranked.items()
    }


def keyword_sets(plan: KeywordPlan) -> dict[str, tuple[str, ...]]:
    """Each resolved page's ranked keyword set, rank 1 first."""
    return {url: tuple(t.text for t in targets) for url, targets in _ranked(plan).items()}


def keyword_origins(plan: KeywordPlan) -> dict[str, tuple[str, ...]]:
    """The origin of every keyword of ``keyword_sets``, in the same order."""
    return {
        url: tuple(keyword_origin(t.rank or 0, t.rung, t.source) for t in targets)
        for url, targets in _ranked(plan).items()
    }


def unique_share(resolved: Mapping[str, ResolvedKeyword]) -> float | None:
    """Share of the distinct resolved keywords, by normalised text and language, that exactly
    one page resolved; None without any."""
    if not resolved:
        return None
    counts = Counter((normalise_term(k.text), k.language) for k in resolved.values())
    return sum(1 for n in counts.values() if n == 1) / len(counts)


@dataclass(frozen=True, slots=True)
class _SourcePage:
    """What the extraction ladder reads of one source page."""

    body: str
    headings: tuple[str, ...]
    language: str | None


def _extractability(
    wanted: Mapping[str, Sequence[str]],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    sources: Mapping[str, _SourcePage],
    threshold: float,
    brand: frozenset[str],
    rungs: Mapping[str, KeywordRung],
) -> KeywordExtractability | None:
    """The ladder over every (source, target) pair in ``wanted``, source to its targets, also
    split by the rung in ``rungs`` the target's primary keyword was resolved at; a source
    without a stored body carries nothing."""
    stems: dict[str | None, Stems] = {}
    words_of: dict[str, tuple[frozenset[str], ...]] = {}
    best: Counter[AnchorRung] = Counter()
    pairs = found_primary = found_set = words_primary = words_set = 0
    parts: defaultdict[KeywordRung, Counter[str]] = defaultdict(Counter)
    for source_url, targets in wanted.items():
        page = sources.get(source_url)
        index = None
        present: frozenset[str] = frozenset()
        if page is not None and page.body.strip():
            if page.language not in stems:
                stems[page.language] = Stems(page.language)
            index = SourceIndex(
                source_url, page.body, page.headings, stems[page.language], brand=brand
            )
            present = copy_words(page.body)
        for target in targets:
            ranked = keywords[target]
            pairs += 1
            if target not in words_of:
                words_of[target] = tuple(keyword_words(text) for _, text, _ in ranked)
            words = words_of[target]
            has_words = bool(words[0]) and words[0] <= present
            words_primary += has_words
            words_set += any(bool(found) and found <= present for found in words)
            part = parts[rungs[target]] if target in rungs else Counter[str]()
            part["pairs"] += 1
            part["words_primary"] += has_words
            if index is None:
                continue
            matches, _, _ = extract(index, target, ranked, threshold=threshold)
            if not matches:
                continue
            primary = any(match.keyword_rank == ranked[0][0] for match in matches)
            found_set += 1
            found_primary += primary
            part["found_set"] += 1
            part["found_primary"] += primary
            best[min((match.rung for match in matches), key=_RUNGS.index)] += 1
    if not pairs:
        return None
    return KeywordExtractability(
        pairs=pairs,
        found_primary=found_primary / pairs,
        found_set=found_set / pairs,
        exact_set=best[AnchorRung.EXACT] / pairs,
        stemmed_set=best[AnchorRung.STEMMED] / pairs,
        stem_set_set=best[AnchorRung.STEM_SET] / pairs,
        words_primary=words_primary / pairs,
        words_set=words_set / pairs,
        stem_set_threshold=threshold,
        by_source={
            rung: SourceExtractability(
                pairs=part["pairs"],
                found_primary=part["found_primary"] / part["pairs"],
                found_set=part["found_set"] / part["pairs"],
                words_primary=part["words_primary"] / part["pairs"],
            )
            for rung, part in sorted(parts.items())
        },
    )


async def keyword_extractability(
    mongo: MongoRepo,
    tenant_id: str,
    targets: Sequence[TargetCandidates],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    brand: frozenset[str] = frozenset(),
    rungs: Mapping[str, KeywordRung] | None = None,
) -> KeywordExtractability | None:
    """Over the candidate pairs whose target has a ranked keyword set, as (rank, text, source)
    rank ascending: does the extraction ladder find the target's primary keyword, or any
    keyword of its set, in the source copy? Existing anchors are not passed, so this asks
    whether the copy carries the keyword at all; ``brand`` tokens are no identifiers, as in
    anchor selection. None without such pairs."""
    wanted: defaultdict[str, list[str]] = defaultdict(list)
    for entry in targets:
        if keywords.get(entry.target_url):
            for source in entry.sources:
                wanted[source].append(entry.target_url)
    if not wanted:
        return None
    sources: dict[str, _SourcePage] = {}
    async for records in mongo.iter_page_records(tenant_id, urls=wanted):
        for record in records:
            sources[record.url] = _SourcePage(
                record.body_text,
                tuple(heading.text for heading in record.headings),
                record.language,
            )
    return await asyncio.to_thread(
        _extractability, wanted, keywords, sources, threshold, brand, rungs or {}
    )


def _anchor_counts(
    texts: Sequence[LinkText],
    target_of: Mapping[tuple[str, int], str],
    wanted: Mapping[str, Sequence[frozenset[str]]],
    generic_add: frozenset[str],
    generic_remove: frozenset[str],
) -> tuple[int, int, int]:
    """Descriptive anchors judged, those matching the primary keyword, those matching any."""
    anchors = primary = any_rank = 0
    for text in texts:
        target = target_of.get((text.source_url, text.position))
        if target is None or is_generic(text.anchor_text, add=generic_add, remove=generic_remove):
            continue
        words = keyword_words(text.anchor_text)
        if not words:
            continue
        hits = [anchor_matches(words, keyword) for keyword in wanted[target]]
        anchors += 1
        primary += hits[0]
        any_rank += any(hits)
    return anchors, primary, any_rank


async def anchor_match(
    graph: GraphRepo,
    tenant_id: str,
    links: Sequence[LinkRelevance],
    sets: Mapping[str, Sequence[str]],
    *,
    generic_add: frozenset[str] = frozenset(),
    generic_remove: frozenset[str] = frozenset(),
) -> AnchorMatchCheck | None:
    """Over the descriptive anchors of existing links into pages with a keyword set: does the
    anchor match the target's primary keyword, or any keyword of its set? ``links`` come from
    ``graph.link_relevance``, so only links between crawled pages whose target has a content
    vector are judged. None without such anchors."""
    target_of = {
        (link.source_url, link.position): link.target_url
        for link in links
        if link.target_url in sets
    }
    wanted = {
        url: tuple(keyword_words(keyword) for keyword in sets[url])
        for url in set(target_of.values())
    }
    totals = np.zeros(3, dtype=np.int64)
    if target_of:
        async for batch in graph.iter_link_texts(tenant_id):
            totals += await asyncio.to_thread(
                _anchor_counts, batch, target_of, wanted, generic_add, generic_remove
            )
    anchors, primary, any_rank = totals.tolist()
    if not anchors:
        return None
    return AnchorMatchCheck(anchors=anchors, primary=primary / anchors, any_rank=any_rank / anchors)


def _relevance(
    pages: Mapping[str, Sequence[str]],
    origins: Mapping[str, Sequence[str]],
    vectors: Mapping[str, npt.NDArray[np.float32]],
    keywords: Mapping[str, npt.NDArray[np.float32]],
) -> tuple[tuple[RankRelevance, ...], tuple[RelevanceGroup, ...], tuple[RelevanceGroup, ...]]:
    """Every ranked keyword's cosine to its page's content vector, grouped by rank, origin
    and length; empty groups without a page vector of non-zero length."""
    by_rank: defaultdict[int, list[float]] = defaultdict(list)
    by_origin: defaultdict[str, list[float]] = defaultdict(list)
    by_length: defaultdict[str, list[float]] = defaultdict(list)
    for url, texts in pages.items():
        page = vectors[url].astype(np.float64)
        norm = float(np.linalg.norm(page))
        if not norm:
            continue
        for rank, (text, origin) in enumerate(zip(texts, origins[url], strict=True), 1):
            vector = keywords[text].astype(np.float64)
            cosine = min(
                1.0, max(-1.0, float(vector @ page) / norm / float(np.linalg.norm(vector)))
            )
            by_rank[rank].append(cosine)
            by_origin[origin].append(cosine)
            by_length[length_bin(text)].append(cosine)
    ranks = []
    for rank, cosines in sorted(by_rank.items()):
        p10, p50, p90 = (float(p) for p in np.percentile(cosines, [10, 50, 90]))
        ranks.append(
            RankRelevance(
                rank=rank,
                keywords=len(cosines),
                mean=min(1.0, max(-1.0, float(np.mean(cosines)))),
                p10=p10,
                p50=p50,
                p90=p90,
            )
        )
    return (
        tuple(ranks),
        relevance_groups(by_origin, KEYWORD_ORIGINS),
        relevance_groups(by_length, [label for label, _, _ in LENGTH_BINS]),
    )


async def keyword_relevance(
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    sets: Mapping[str, Sequence[str]],
    vectors: Mapping[str, npt.NDArray[np.float32]],
    *,
    origins: Mapping[str, Sequence[str]],
    cache_dir: Path,
) -> tuple[KeywordRelevance | None, str | None]:
    """The cosine of each ranked keyword to its page's content vector, summarised per rank,
    per origin (``origins`` aligned with ``sets``) and per length in words, with None as the
    reason; or None and why not: no keyworded page with a vector, page vectors from another
    model or dimension than ``voyage``'s, or Voyage down or refusing the key after retries."""
    pages = {url: keywords for url, keywords in sets.items() if url in vectors}
    if not pages:
        return None, "no keyworded page has a content vector"
    models = {row.embedding_model or NO_MODEL for row in await graph.embedding_models(tenant_id)}
    dimensions = {len(vectors[url]) for url in pages}
    reason = None
    if models != {voyage.model}:
        reason = f"page vectors are from {', '.join(sorted(models))}, not {voyage.model}"
    elif dimensions != {voyage.dimension}:
        reason = (
            f"page vectors have {', '.join(map(str, sorted(dimensions)))} dimensions, "
            f"not {voyage.dimension}"
        )
    if reason is None:
        try:
            found = await text_vectors(
                voyage,
                tenant_id,
                "keywords",
                (keyword for keywords in pages.values() for keyword in keywords),
                cache_dir=cache_dir,
                stage=STAGE,
            )
        except EmbeddingAuthError as error:
            reason = f"Voyage refused the key ({error.error_type})"
        except EmbeddingUnavailableError as error:
            reason = f"Voyage unavailable after retries ({error.error_type})"
        else:
            ranks, by_origin, by_length = await asyncio.to_thread(
                _relevance, pages, origins, vectors, found.vectors
            )
            if ranks:
                relevance = KeywordRelevance(
                    ranks=ranks,
                    by_origin=by_origin,
                    by_length=by_length,
                    texts=len(found.vectors),
                    embedded=found.embedded,
                    cached=found.cached,
                    api_tokens=found.api_tokens,
                )
                return relevance, None
            reason = "no keyworded page has a non-zero content vector"
    log.warning(
        "quality.keyword_relevance_skipped", stage=STAGE, tenant_id=tenant_id, reason=reason
    )
    return None, reason


def link_relevance_check(links: Sequence[LinkRelevance]) -> LinkRelevanceCheck | None:
    """The spread of the stored scores; None when no link has a context score."""
    context = score_distribution(
        [link.context_relevance for link in links if link.context_relevance is not None]
    )
    if context is None:
        return None
    return LinkRelevanceCheck(
        links=len(links),
        context=context,
        anchor=score_distribution(
            [link.anchor_target_fit for link in links if link.anchor_target_fit is not None]
        ),
    )


async def evaluate_quality(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    voyage: VoyageClient | None = None,
    baseline: QualityBaseline | None = None,
) -> QualityReport:
    """Every quality check of the tenant, read-only against both stores, with alerts against
    ``baseline``, the tenant's previous run. Keyword relevance needs ``voyage``; its vectors
    are cached at ``<cache_dir>/<tenant>/``."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    started = time.perf_counter()
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    unknown = [f.column for f in weights.features if f.column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"scorer weights {weights.version!r} of {tenant_id!r} name columns the feature "
            f"matrix does not have: {', '.join(unknown)}"
        )
    snapshot = await graph.link_graph(tenant_id)
    structure = await graph.page_structure(tenant_id)
    context = await graph.community_context(tenant_id)
    copies = await graph.non_canonical_copies(tenant_id)
    vectors = await graph.content_vectors(tenant_id)

    plan = await plan_keywords(mongo, tenant_id)
    sets = keyword_sets(plan)
    relevance: KeywordRelevance | None = None
    relevance_reason: str | None = "no ranked keyword sets"
    if voyage is None:
        relevance_reason = "no Voyage API key"
    elif sets:
        relevance, relevance_reason = await keyword_relevance(
            graph,
            voyage,
            tenant_id,
            sets,
            vectors,
            origins=keyword_origins(plan),
            cache_dir=cache_dir,
        )

    crawled = {
        url
        for url, placeholder in zip(snapshot.pages, snapshot.placeholders, strict=True)
        if not placeholder
    }
    body_links = {(s, t) for s, t in snapshot.links if s != t and s in crawled and t in crawled}
    hidden = hide_links(body_links)
    body_link_pairs = len(body_links)
    del body_links
    view = await asyncio.to_thread(held_out_view, snapshot, structure, context, vectors, hidden)
    pool = vectors.keys() - copies

    strategic = await mongo.strategic_keywords(tenant_id)
    signals = build_page_signals(
        view.context,
        await mongo.gsc_queries(tenant_id),
        [(row.url, row.keyword) for row in strategic],
    )
    try:
        pages = page_contexts(
            view.structure,
            signals,
            view.links,
            await mongo.gsc_metrics(tenant_id),
            strategic,
            plan.curve,
        )
    except ValueError as error:
        raise DatabaseReadError("neo4j", f"feature inputs of {tenant_id!r}: {error}") from error
    held = await retrieve_candidates(
        graph, tenant_id, links=view.links, vectors=vectors, stage=STAGE
    )
    missing = missing_pages(held.targets, pages)
    if missing:
        raise DatabaseReadError(
            "neo4j",
            f"{len(missing)} candidate pages of {tenant_id!r} are no longer crawled pages, "
            f"first {missing[0]!r}",
        )
    languages = {page.url: page.language for page in structure}
    retrieval = retrieval_check(body_link_pairs, hidden, held, pool, languages)
    signal, scorer, gaps = await asyncio.to_thread(
        pair_checks, tenant_id, held.targets, pages, hidden, weights
    )
    del pages, held

    extract = None
    ranked = await graph.ranked_keywords(tenant_id)
    if ranked:
        settings = await mongo.get_extraction_settings(tenant_id) or ExtractionSettings()
        production = await retrieve_candidates(
            graph, tenant_id, links=snapshot.links, vectors=vectors, stage=STAGE
        )
        brand = frozenset(
            token
            for affix in (plan.brand_prefix, plan.brand_suffix)
            if affix
            for token in keyword_tokens(affix)
        )
        extract = await keyword_extractability(
            mongo,
            tenant_id,
            production.targets,
            ranked,
            threshold=settings.stem_set_threshold,
            brand=brand,
            rungs={url: keyword.rung for url, keyword in plan.resolved.items()},
        )
        del production
    del vectors
    links = await graph.link_relevance(tenant_id)
    anchors = await anchor_match(
        graph,
        tenant_id,
        links,
        sets,
        generic_add=plan.generic_add,
        generic_remove=plan.generic_remove,
    )
    link_relevance = link_relevance_check(links)

    keywords = None
    if plan.pages:
        rungs = Counter(keyword.rung for keyword in plan.resolved.values())
        keywords = KeywordCheck(
            pages=plan.pages,
            resolved=len(plan.resolved),
            by_rung={rung: rungs[rung] for rung in KeywordRung},
            fallbacks_rejected=dict(sorted(plan.fallbacks_rejected.items())),
            unique_share=unique_share(plan.resolved),
            extractability=extract,
            anchors=anchors,
            relevance=relevance,
            relevance_reason=relevance_reason if relevance is None else None,
        )
    sections: dict[CheckName, object] = {
        "retrieval": retrieval,
        "feature_signal": signal,
        "scorer": scorer,
        "keywords": keywords,
    }
    if keywords is not None:
        sections.update(
            {
                "keyword_uniqueness": keywords.unique_share,
                "keyword_extractability": keywords.extractability,
                "anchor_match": keywords.anchors,
                "keyword_relevance": keywords.relevance,
            }
        )
    sections["link_relevance"] = link_relevance
    sha = await asyncio.to_thread(git_sha)
    report = QualityReport(
        tenant_id=tenant_id,
        versions=QualityVersions(
            git_sha=sha,
            feature_digest=code_digest(),
            weights_version=weights.version,
            weights_hash=weights_hash(weights),
        ),
        retrieval=retrieval,
        feature_signal=signal,
        scorer=scorer,
        keywords=keywords,
        link_relevance=link_relevance,
        coverage=CoverageCheck(
            pairs=gaps.pairs,
            gsc_pair_share=gaps.has_gsc_data_share,
            keyword_page_share=len(plan.resolved) / plan.pages if plan.pages else None,
            all_null_columns=gaps.all_null_columns,
            constant_columns=gaps.constant_columns,
        ),
        not_applicable=tuple(name for name, section in sections.items() if section is None),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    if baseline is not None:
        report = QualityReport.model_validate(
            {
                **report.model_dump(),
                "baseline_run_id": baseline.run_id,
                "alerts": alerts(quality_metrics(report), baseline.metrics),
            }
        )
    log.info(
        "quality.evaluated",
        stage=STAGE,
        tenant_id=tenant_id,
        not_applicable=list(report.not_applicable),
        alerts=[alert.metric for alert in report.alerts],
        baseline_run_id=report.baseline_run_id,
        hidden_links=len(hidden),
        seconds=report.seconds,
    )
    return report


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _group(entry: RelevanceGroup) -> str:
    return f"{entry.group} {entry.mean:.3f} (median {entry.p50:.3f}, n={entry.keywords})"


def _change(alert: QualityAlert) -> str:
    if alert.change is None:
        return "from 0"
    return f"{alert.change:+.1%}" if alert.relative else f"{alert.change:+.4g}"


def summarise_quality(report: QualityReport) -> str:
    """What was checked and what moved, for the MLflow run description; no page urls or
    keyword texts."""
    versions = report.versions
    lines = [
        f"Quality evaluation for tenant {report.tenant_id}: git {versions.git_sha[:12]}, "
        f"feature code {versions.feature_digest[:12]}, weights {versions.weights_version} "
        f"({versions.weights_hash[:12]}); read-only, {report.seconds:.1f} s.",
    ]
    retrieval = report.retrieval
    if retrieval is None:
        lines.append(
            "Retrieval: not applicable, no hidden body link between crawled pages could be "
            "recovered."
        )
    else:
        recall = ", ".join(f"{r.recall:.1%} at {r.k}" for r in retrieval.recall)
        random = ", ".join(f"{r.random:.1%}" for r in retrieval.recall)
        lines.append(
            f"Retrieval: {retrieval.hidden} of {retrieval.body_link_pairs} body links hidden "
            f"({retrieval.hide_share:.0%}, seed {retrieval.seed}), {retrieval.recoverable} "
            "recoverable; link counts, PageRank, link communities, hub pillars and orphan state "
            f"recomputed without them. Recall {recall} (random {random})."
        )
    signal = report.feature_signal
    if signal is None:
        lines.append("Feature signal and scorer: not applicable, no hidden link among the pairs.")
    else:
        strongest = sorted(
            (entry for entry in signal.columns if entry.auc is not None),
            key=lambda entry: -abs((entry.auc or 0.5) - 0.5),
        )[:5]
        lines.append(
            f"Feature signal: {signal.features_with_signal} of {len(signal.columns)} columns "
            f"separate the {signal.positives} hidden links from the other "
            f"{signal.pairs - signal.positives} candidate pairs "
            f"by at least {signal.margin} AUC; strongest "
            + ", ".join(f"{entry.column} {entry.auc:.3f}" for entry in strongest)
            + "."
        )
    if report.scorer is not None:
        scorer = report.scorer
        lines.append(
            f"Scorer ({scorer.weights_version}): AUC {scorer.score_auc:.3f} against the best "
            f"single feature, {scorer.best_feature} {scorer.best_feature_auc:.3f} (lift "
            f"{scorer.score_auc_lift:+.3f}). This AUC carries the protocol inflation: "
            f"{scorer.link_derived_weight_share:.0%} of the scorer's weight sits on link-derived "
            "columns, which hiding a link moves by itself (its source's outbound and its "
            "target's inbound count)."
        )
        lines.append(
            "Like for like, without the link-derived columns: "
            + (
                f"scorer AUC {scorer.score_auc_excl_link_counts:.3f}, weights renormalised, "
                f"against {scorer.best_feature_excl_link_counts} "
                f"{scorer.best_feature_auc_excl_link_counts:.3f} (lift "
                f"{scorer.score_auc_lift_excl_link_counts:+.3f}). This is the bar the ranker "
                "(#25) has to beat."
                if scorer.score_auc_excl_link_counts is not None
                and scorer.score_auc_lift_excl_link_counts is not None
                else "none, every weighted column is link-derived; the best such feature is "
                f"{scorer.best_feature_excl_link_counts} "
                f"{scorer.best_feature_auc_excl_link_counts:.3f}."
            )
        )
    keywords = report.keywords
    if keywords is None:
        lines.append("Keywords: not applicable, no crawled 2xx pages.")
    else:
        rungs = ", ".join(f"{n} {rung.value}" for rung, n in keywords.by_rung.items())
        rejected = ", ".join(f"{n} {reason}" for reason, n in keywords.fallbacks_rejected.items())
        lines.append(
            f"Keywords: {keywords.resolved} of {keywords.pages} crawled 2xx pages resolved "
            f"({rungs}); fallbacks rejected: {rejected or 'none'}; "
            f"{_share(keywords.unique_share)} of the resolved keywords belong to one page only."
        )
        extract = keywords.extractability
        lines.append(
            "Extractability: not applicable, no candidate pair with a keyworded target."
            if extract is None
            else f"Extractability over {extract.pairs} candidate pairs, by the extraction ladder "
            f"(stem set threshold {extract.stem_set_threshold:g}), existing anchors aside: the "
            f"primary keyword is found in the source copy for {extract.found_primary:.1%}, some "
            f"keyword of the set for {extract.found_set:.1%} (best rung exact "
            f"{extract.exact_set:.1%}, stemmed {extract.stemmed_set:.1%}, stem set "
            f"{extract.stem_set_set:.1%}); every word anywhere, the literal upper bound of the "
            f"exact rung: {extract.words_primary:.1%} primary, {extract.words_set:.1%} set."
            + "".join(
                f" {rung.value} keywords: {part.found_primary:.1%} found, "
                f"{part.words_primary:.1%} ceiling, {part.pairs} pairs."
                for rung, part in extract.by_source.items()
            )
        )
        anchors = keywords.anchors
        lines.append(
            "Existing anchors: not applicable, no descriptive anchor into a keyworded page."
            if anchors is None
            else f"Existing anchors: of {anchors.anchors} descriptive anchors, {anchors.primary:.1%} "
            f"match the target's primary keyword, {anchors.any_rank:.1%} any of its set."
        )
        relevance = keywords.relevance
        lines.append(
            f"Keyword relevance: not applicable, {keywords.relevance_reason}."
            if relevance is None
            else "Keyword relevance, cosine to the page: "
            + ", ".join(
                f"rank {entry.rank} {entry.mean:.3f} (n={entry.keywords})"
                for entry in relevance.ranks
            )
            + f"; {relevance.embedded} keywords embedded, {relevance.cached} cached."
        )
        if relevance is not None:
            lines.append(
                "Rank averages mix keyword origins and lengths, which a cosine to a whole-page "
                "vector does not score alike, so compare within a group. By origin: "
                + ", ".join(_group(entry) for entry in relevance.by_origin)
                + "; by words: "
                + ", ".join(_group(entry) for entry in relevance.by_length)
                + "."
            )
    links = report.link_relevance
    if links is None:
        lines.append("Link relevance: not applicable, no stored context score.")
    else:
        context = links.context
        split = (
            f"split at {context.split:.3f}, {context.low_share:.1%} below"
            if context.split is not None and context.low_share is not None
            else "no split"
        )
        anchor = (
            f"; anchor-target fit mean {links.anchor.mean:.3f} over {links.anchor.count}"
            if links.anchor is not None
            else ""
        )
        lines.append(
            f"Link relevance over {links.links} links: context mean {context.mean:.3f}, "
            f"median {context.p50:.3f}, {split}{anchor}."
        )
    coverage = report.coverage
    lines.append(
        f"Coverage over {coverage.pairs} pairs: GSC data for the target of "
        f"{_share(coverage.gsc_pair_share)}, keywords on {_share(coverage.keyword_page_share)} "
        "of the pages; all null: "
        + (", ".join(coverage.all_null_columns) or "none")
        + "; constant: "
        + (", ".join(coverage.constant_columns) or "none")
        + "."
    )
    lines.append("Not applicable: " + (", ".join(report.not_applicable) or "none") + ".")
    if report.baseline_run_id is None:
        lines.append("First quality run of this tenant: no baseline, so no alerts.")
    elif not report.alerts:
        lines.append(
            f"Against run {report.baseline_run_id}: no headline metric moved beyond its band "
            f"(default {ALERT_BAND:.0%} relative)."
        )
    else:
        lines.append(f"Moved since run {report.baseline_run_id}:")
        lines.extend(
            f"- {alert.metric}: {alert.previous:.4g} -> {alert.current:.4g} "
            f"({_change(alert)}, band {alert.band:g}{' relative' if alert.relative else ''})"
            for alert in report.alerts
        )
    return "\n".join(lines)
