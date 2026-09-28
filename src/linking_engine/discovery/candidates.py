"""Candidate retrieval: exact cosine search of every target against the tenant's own pages,
keeping the nearest eligible sources per target only. Non-canonical duplicate copies are
left out of the pool, so they are neither targets nor sources."""

from __future__ import annotations

import asyncio
import statistics
import time
from datetime import UTC, datetime
from itertools import batched, pairwise
from typing import TYPE_CHECKING, Final

import numpy as np
import structlog

from linking_engine.errors import DatabaseReadError
from linking_engine.models import CandidateReport, CandidateSet, TargetCandidates

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import TargetSelection, VectorIndex

log = structlog.get_logger(__name__)

PER_TARGET: Final = 50
# A chunk's scores take TARGET_CHUNK x pool pages x 4 bytes, about three times that at peak.
TARGET_CHUNK: Final = 512
STAGE: Final = "candidate-retrieval"


def nearest_eligible(
    urls: Sequence[str],
    vectors: npt.ArrayLike,
    targets: Sequence[str],
    links: Iterable[tuple[str, str]],
    *,
    per_target: int = PER_TARGET,
    chunk_size: int = TARGET_CHUNK,
    languages: Mapping[str, str | None] | None = None,
) -> tuple[TargetCandidates, ...]:
    """The ``per_target`` nearest eligible sources of each target, in ``targets`` order.

    ``urls`` is the pool, ascending, one row of ``vectors`` each; every target is in it.
    ``links`` are (source, target) pairs. A source must share the target's language in
    ``languages`` (a url without an entry has none, and none matches none); pages of another
    language are neither eligible nor counted as linked. Targets are scored ``chunk_size`` at
    a time against the whole pool, so memory grows with the chunk, not with the square of
    the pool.
    """
    if per_target < 1:
        raise ValueError("per_target must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    pool = np.asarray(vectors, dtype=np.float32)
    if pool.ndim != 2 or len(pool) != len(urls) or (len(pool) and not pool.shape[1]):
        raise ValueError("vectors must be a matrix with one non-empty row per url")
    if any(a >= b for a, b in pairwise(urls)):
        raise ValueError("urls must be unique and ascending")
    position = {url: i for i, url in enumerate(urls)}
    missing = [url for url in targets if url not in position]
    if missing:
        raise ValueError(f"{len(missing)} targets are not in the pool, first {missing[0]!r}")
    if not targets:
        return ()

    codes = _language_codes(urls, languages or {})
    # Pool pages sharing each language, the page itself included.
    peers = np.bincount(codes) if codes is not None else np.array([len(urls)])
    wanted = set(targets)
    linking: dict[str, set[int]] = {}
    for source, target in links:
        if (
            target in wanted
            and source != target
            and source in position
            and (codes is None or codes[position[source]] == codes[position[target]])
        ):
            linking.setdefault(target, set()).add(position[source])
    linked = {url: np.array(sorted(found), dtype=np.intp) for url, found in linking.items()}
    no_links = np.empty(0, dtype=np.intp)

    norms = np.linalg.norm(pool, axis=1)
    # A zero vector scores 0 against every page.
    norms[norms == 0] = 1
    # BLAS can round identical columns differently; every duplicate vector takes the score of
    # its first occurrence, so identical pages tie and break by url.
    _, first, inverse = np.unique(pool, axis=0, return_index=True, return_inverse=True)
    copy_of = first[inverse.reshape(-1)]
    copies = np.flatnonzero(copy_of != np.arange(len(pool)))
    results: list[TargetCandidates] = []
    for chunk in batched(targets, chunk_size):
        rows = np.array([position[url] for url in chunk], dtype=np.intp)
        scores = pool[rows] @ pool.T
        scores /= norms[rows, None]
        scores /= norms
        scores[:, copies] = scores[:, copy_of[copies]]
        np.clip(scores, -1.0, 1.0, out=scores)
        excluded = [linked.get(url, no_links) for url in chunk]
        excluded_scores = [scores[i, cols] for i, cols in enumerate(excluded)]
        for i, cols in enumerate(excluded):
            scores[i, cols] = -np.inf
        scores[np.arange(len(chunk)), rows] = -np.inf
        if codes is not None:
            scores[codes[rows, None] != codes] = -np.inf
        for i, (url, (kept, similarities)) in enumerate(
            zip(chunk, _best(scores, per_target), strict=True)
        ):
            cols = excluded[i]
            if len(kept) < per_target:
                nearer = len(cols)
            else:
                last, floor = kept[-1], similarities[-1]
                ahead = excluded_scores[i]
                nearer = int(np.count_nonzero((ahead > floor) | ((ahead == floor) & (cols < last))))
            results.append(
                TargetCandidates(
                    target_url=url,
                    sources=tuple(urls[j] for j in kept.tolist()),
                    similarities=tuple(similarities.tolist()),
                    eligible=int(peers[codes[position[url]] if codes is not None else 0])
                    - 1
                    - len(cols),
                    linked=len(cols),
                    linked_nearer=nearer,
                )
            )
    return tuple(results)


def _language_codes(
    urls: Sequence[str], languages: Mapping[str, str | None]
) -> npt.NDArray[np.intp] | None:
    """One small integer per pool page for its language; None when every page shares one, so
    single-language tenants skip the constraint."""
    code_of: dict[str | None, int] = {}
    codes = np.array(
        [code_of.setdefault(languages.get(url), len(code_of)) for url in urls], dtype=np.intp
    )
    return codes if len(code_of) > 1 else None


def _best(
    scores: npt.NDArray[np.float32], per_target: int
) -> list[tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]]:
    """Per row, the columns and scores of the best ``per_target`` finite scores, best first,
    column ascending on ties; masked entries are -inf."""
    k = min(per_target, scores.shape[1])
    kth = -np.partition(-scores, k - 1, axis=1)[:, k - 1]
    # Every cosine is at least -1, so the floor also drops the masked entries; ties at the
    # floor can admit more than k, which the sort below trims.
    rows, cols = np.nonzero(scores >= np.maximum(kth, -1.0)[:, None])
    values = scores[rows, cols]
    order = np.lexsort((cols, -values, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    counts = np.bincount(rows, minlength=len(scores))
    starts = np.cumsum(counts) - counts
    return [
        (cols[start : start + min(count, k)], values[start : start + min(count, k)])
        for start, count in zip(starts.tolist(), counts.tolist(), strict=True)
    ]


def candidate_report(
    tenant_id: str,
    index: VectorIndex,
    per_target: int,
    chunk_size: int,
    selection: TargetSelection,
    source_pages: int,
    targets: Sequence[TargetCandidates],
    *,
    non_canonical: Set[str] = frozenset(),
    load_seconds: float,
    search_seconds: float,
    seconds: float,
) -> CandidateReport:
    """``non_canonical`` are the duplicate copies left out of the pool; the selection's targets
    among them are not targets."""
    kept = [t for t in selection.targets if t.url not in non_canonical]
    if sorted(t.target_url for t in targets) != sorted(t.url for t in kept):
        raise ValueError("targets must be exactly the selection's targets that are not copies")
    counts = [len(t.sources) for t in targets]
    candidates = sum(counts)
    linked_nearer = sum(t.linked_nearer for t in targets)
    return CandidateReport(
        tenant_id=tenant_id,
        index=index,
        per_target=per_target,
        chunk_size=chunk_size,
        crawled_pages=selection.crawled_pages,
        not_indexable=selection.not_indexable,
        without_vector=selection.without_vector,
        targets=len(targets),
        indexable_assumed=sum(t.indexable_assumed for t in kept),
        source_pages=source_pages,
        non_canonical_excluded=len(non_canonical),
        candidates=candidates,
        full_targets=sum(1 for count in counts if count >= per_target),
        short_targets=sum(1 for count in counts if 0 < count < per_target),
        empty_targets=sum(1 for count in counts if count == 0),
        min_per_target=min(counts) if counts else None,
        median_per_target=float(statistics.median(counts)) if counts else None,
        max_per_target=max(counts) if counts else None,
        linked_pairs=sum(t.linked for t in targets),
        linked_nearer=linked_nearer,
        drop_rate=(
            linked_nearer / (candidates + linked_nearer) if candidates + linked_nearer else None
        ),
        load_seconds=load_seconds,
        search_seconds=search_seconds,
        seconds=seconds,
        finished_at=datetime.now(UTC),
    )


def _pool(
    vectors: Mapping[str, npt.NDArray[np.float32]], tenant_id: str, index: VectorIndex
) -> tuple[list[str], npt.NDArray[np.float32]]:
    urls = sorted(vectors)
    if not urls:
        return urls, np.empty((0, 0), dtype=np.float32)
    shapes = {vectors[url].shape for url in urls}
    if len(shapes) > 1 or len(shape := next(iter(shapes))) != 1 or not shape[0]:
        raise DatabaseReadError(
            "neo4j",
            f"{index} vectors of {tenant_id!r} must share one non-zero length, found {sorted(shapes)}",
        )
    pool = np.stack([vectors[url] for url in urls]).astype(np.float32, copy=False)
    if not np.isfinite(pool).all():
        raise DatabaseReadError("neo4j", f"{index} vectors of {tenant_id!r} hold non-finite values")
    return urls, pool


async def retrieve_candidates(
    graph: GraphRepo,
    tenant_id: str,
    *,
    index: VectorIndex = "page_content",
    per_target: int = PER_TARGET,
    chunk_size: int = TARGET_CHUNK,
) -> CandidateSet:
    """Every target's nearest eligible sources among the tenant's own crawled pages of the
    target's language."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if per_target < 1:
        raise ValueError("per_target must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    started = time.perf_counter()
    selection = await graph.candidate_targets(tenant_id, index=index)
    vectors = await graph.page_vectors(tenant_id, index=index)
    copies = await graph.non_canonical_copies(tenant_id)
    languages = await graph.page_languages(tenant_id)
    snapshot = await graph.link_graph(tenant_id)
    loaded = time.perf_counter()

    # Separate reads, so a page can lose its vector between them.
    missing = [t.url for t in selection.targets if t.url not in vectors]
    if missing:
        raise DatabaseReadError(
            "neo4j",
            f"{len(missing)} targets of {tenant_id!r} have no vector in {index}, "
            f"first {missing[0]!r}",
        )
    dropped = vectors.keys() & copies
    urls, pool = _pool(
        {url: vector for url, vector in vectors.items() if url not in dropped}, tenant_id, index
    )
    targets = await asyncio.to_thread(
        nearest_eligible,
        urls,
        pool,
        sorted(t.url for t in selection.targets if t.url not in dropped),
        snapshot.links,
        per_target=per_target,
        chunk_size=chunk_size,
        languages=languages,
    )
    searched = time.perf_counter()
    report = candidate_report(
        tenant_id,
        index,
        per_target,
        chunk_size,
        selection,
        len(urls),
        targets,
        non_canonical=dropped,
        load_seconds=round(loaded - started, 3),
        search_seconds=round(searched - loaded, 3),
        seconds=round(time.perf_counter() - started, 3),
    )
    log.info("candidates.retrieved", stage=STAGE, **report.model_dump(mode="json"))
    return CandidateSet(report=report, targets=targets)


def summarise_candidates(report: CandidateReport) -> str:
    """A short prose record of one candidate retrieval run, for the MLflow run description."""
    scope = (
        f"Candidate retrieval for tenant {report.tenant_id}, exact search over the "
        f"{report.index} vectors: {report.targets} targets and {report.source_pages} source "
        f"pages of {report.crawled_pages} crawled ({report.not_indexable} not indexable, "
        f"{report.without_vector} indexable without a vector, {report.non_canonical_excluded} "
        f"non-canonical duplicate copies left out; {report.indexable_assumed} of the targets "
        "assumed indexable from a 2xx status)."
    )
    timing = (
        f"{report.seconds:.1f} s: {report.load_seconds:.1f} s loading, "
        f"{report.search_seconds:.1f} s searching in chunks of {report.chunk_size} targets."
    )
    if not report.targets:
        return "\n".join([scope, "No targets, so no candidates.", timing])
    drop_rate = f"{report.drop_rate:.1%}" if report.drop_rate is not None else "n/a"
    return "\n".join(
        [
            scope,
            f"{report.candidates} candidates, at most {report.per_target} per target: "
            f"{report.full_targets} targets full, {report.short_targets} short, "
            f"{report.empty_targets} empty; per target min {report.min_per_target}, "
            f"median {report.median_per_target:g}, max {report.max_per_target}.",
            f"{report.linked_pairs} existing links from scored pages excluded, "
            f"{report.linked_nearer} of them ranked above a target's last kept candidate; "
            f"drop rate {drop_rate} (pooled over all targets).",
            timing,
        ]
    )
