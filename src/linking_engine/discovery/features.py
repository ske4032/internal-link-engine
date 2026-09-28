"""Feature assembly: the context features of every candidate pair, as the ranker sees them.

Page-level values are derived once per run; a pair only combines its two pages. The matrix
is built one chunk at a time and never held whole. Its columns keep a fixed order, because
the ranker relies on column position between training and prediction.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import statistics
import sys
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np
import pandas
from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.discovery.signals import HUB_NOISE, pair_signals
from linking_engine.gsc import ctr_at
from linking_engine.models import (
    ClusterAgreement,
    FeatureReport,
    PageSignals,
    PageStructure,
    PairFeatures,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from linking_engine.models import CtrCurve, GscMetrics, StrategicKeyword, TargetCandidates

STAGE: Final = "feature-assembly"
# Pairs built, converted and written together; memory grows with this, not with the tenant.
CHUNK_PAIRS: Final = 50_000
# Upper bounds of the position bands 1-3, 4-10, 11-20 and 21-50; anything lower is the last band.
POSITION_BANDS: Final = (3, 10, 20, 50)

KEY_COLUMNS: Final = ("source_url", "target_url")
# PairFeatures fields in matrix order. Cluster ids are labels of one run, so they stay on
# PairFeatures and out of the matrix.
_MATRIX_FIELDS: Final = (
    "source_impressions_log",
    "target_impressions_log",
    "target_position_band",
    "target_ctr_gap",
    "target_query_count",
    "has_gsc_data",
    "pair_query_overlap",
    "pair_kw_overlap",
    "target_kw_count",
    "target_max_priority",
    "target_keyword_gap",
    "target_inbound_count",
    "target_is_orphan",
    "target_crawl_depth",
    "target_page_rank_percentile",
    "target_saturation_ratio",
    "source_outbound_count",
    "source_outbound_density",
    "source_link_equity_share",
    "same_hub",
    "source_is_hub_pillar",
    "target_is_hub_pillar",
    "source_hub_size",
    "target_hub_size",
    "target_hub_coverage",
    "same_link_community",
    "same_keyword_community",
    "same_content_community",
    "cluster_agreement",
    "content_agreement",
    "content_cosine",
    "context_relevance",
    "anchor_target_fit",
)
# Categoricals are one-hot, one 0/1 column per level; a missing position is a level of its own.
_LEVELS: Final[Mapping[str, tuple[object, ...]]] = {
    "target_position_band": (*range(len(POSITION_BANDS) + 1), None),
    "cluster_agreement": tuple(ClusterAgreement),
    "content_agreement": tuple(ClusterAgreement),
}


def _columns(field: str) -> tuple[str, ...]:
    levels = _LEVELS.get(field)
    if levels is None:
        return (field,)
    return tuple(f"{field}_{'null' if level is None else str(level).lower()}" for level in levels)


FEATURE_COLUMNS: Final = tuple(column for field in _MATRIX_FIELDS for column in _columns(field))


class PageContext(BaseModel):
    """What one crawled page brings to every pair it is part of, derived once per run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    structure: PageStructure
    signals: PageSignals
    has_gsc_data: bool
    impressions_log: float | None = Field(default=None, ge=0)
    position_band: int | None = Field(default=None, ge=0, le=len(POSITION_BANDS))
    ctr_gap: float | None = Field(default=None, ge=-1, le=1)
    query_count: int | None = Field(default=None, ge=0)
    max_priority: int | None = Field(default=None, ge=1, le=5)
    is_orphan: bool
    saturation_ratio: float = Field(ge=0)
    outbound_density: float = Field(ge=0)
    link_equity_share: float = Field(gt=0, le=1)
    hub_size: int | None = Field(default=None, ge=1)
    hub_coverage: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _one_page(self) -> Self:
        if self.structure.url != self.signals.url:
            raise ValueError("structure and signals must describe the same page")
        return self


def position_band(position: float | None) -> int | None:
    """The band of a GSC average position, rounded like the CTR curve; None stays None."""
    if position is None:
        return None
    if not position >= 1:
        raise ValueError(f"a position is at least 1, got {position}")
    return bisect_left(POSITION_BANDS, int(position + 0.5))


def _in_hub(hub_id: int | None) -> bool:
    return hub_id is not None and hub_id != HUB_NOISE


def _decile(percentile: float | None) -> int | None:
    return None if percentile is None else min(int(percentile * 10), 9)


def _check_reads(pages: Mapping[str, PageStructure], signals: Mapping[str, PageSignals]) -> None:
    """Structure and signals are separate reads; both must describe the same graph."""
    differ = sorted(pages.keys() ^ signals.keys())
    if differ:
        raise ValueError(
            f"page structure and signals differ on {len(differ)} pages, first {differ[0]!r}"
        )
    for url, page in pages.items():
        found = signals[url]
        if (
            page.link_community_id,
            page.keyword_community_id,
            page.content_community_id,
            page.hub_id,
        ) != (
            found.link_community_id,
            found.keyword_community_id,
            found.content_community_id,
            found.hub_id,
        ):
            raise ValueError(f"page structure and signals disagree on the clusters of {url!r}")


def page_contexts(
    structure: Sequence[PageStructure],
    signals: Mapping[str, PageSignals],
    links: Iterable[tuple[str, str]],
    metrics: Iterable[GscMetrics],
    keywords: Iterable[StrategicKeyword],
    curve: CtrCurve | None,
) -> dict[str, PageContext]:
    """Every crawled page's context. ``links`` are body links as (source, target); links,
    metrics and keywords of urls that are not crawled pages are ignored. The CTR gap needs
    the tenant's curve, so it is None for every page without one."""
    pages = {page.url: page for page in structure}
    if len(pages) != len(structure):
        raise ValueError("duplicate page urls")
    _check_reads(pages, signals)

    gsc: dict[str, GscMetrics] = {}
    for totals in metrics:
        if totals.url in gsc:
            raise ValueError(f"duplicate GSC metrics for {totals.url!r}")
        if totals.url in pages:
            gsc[totals.url] = totals
    priorities: dict[str, int] = {}
    for keyword in keywords:
        if keyword.url in pages and keyword.priority is not None:
            priorities[keyword.url] = max(priorities.get(keyword.url, 0), keyword.priority)

    by_decile: defaultdict[int | None, list[int]] = defaultdict(list)
    for page in structure:
        by_decile[_decile(page.page_rank_percentile)].append(page.inbound)
    medians = {decile: statistics.median(inbound) for decile, inbound in by_decile.items()}

    hub_sizes = Counter(page.hub_id for page in structure if _in_hub(page.hub_id))
    hub_sources: defaultdict[str, set[str]] = defaultdict(set)
    for source_url, target_url in links:
        source, target = pages.get(source_url), pages.get(target_url)
        if (
            source is not None
            and target is not None
            and source_url != target_url
            and _in_hub(target.hub_id)
            and source.hub_id == target.hub_id
        ):
            hub_sources[target_url].add(source_url)

    contexts: dict[str, PageContext] = {}
    for url, page in pages.items():
        row = gsc.get(url)
        median = medians[_decile(page.page_rank_percentile)]
        hub_size = hub_sizes[page.hub_id] if _in_hub(page.hub_id) else None
        others = hub_size - 1 if hub_size is not None else 0
        contexts[url] = PageContext(
            structure=page,
            signals=signals[url],
            has_gsc_data=row is not None,
            impressions_log=math.log1p(row.impressions_28d) if row is not None else None,
            position_band=position_band(row.avg_position if row is not None else None),
            ctr_gap=_ctr_gap(row, curve),
            query_count=row.query_count if row is not None else None,
            max_priority=priorities.get(url),
            # Before graph analytics has run, an orphan is a page no crawled page links to.
            is_orphan=page.is_orphan if page.is_orphan is not None else page.inbound == 0,
            saturation_ratio=page.inbound / median if median else float(page.inbound),
            # An empty page counts as one word.
            outbound_density=page.outbound * 1000 / max(page.word_count, 1),
            link_equity_share=1 / (page.outbound + 1),
            hub_size=hub_size,
            # A hub of one has no other member that could link to the page.
            hub_coverage=(
                (len(hub_sources[url]) / others if others else 0.0)
                if hub_size is not None
                else None
            ),
        )
    return contexts


def _ctr_gap(row: GscMetrics | None, curve: CtrCurve | None) -> float | None:
    """Actual CTR minus the tenant's expected CTR at the page's position."""
    if row is None or curve is None or row.avg_position is None or not row.impressions_28d:
        return None
    actual = min(row.clicks_28d / row.impressions_28d, 1.0)
    return actual - ctr_at(curve, row.avg_position)


def pair_features(source: PageContext, target: PageContext, similarity: float) -> PairFeatures:
    """The features of one source -> target pair; ``similarity`` is their content cosine."""
    found = pair_signals(source.signals, target.signals)
    s, t = source.structure, target.structure
    return PairFeatures(
        source_url=s.url,
        target_url=t.url,
        source_impressions_log=source.impressions_log,
        target_impressions_log=target.impressions_log,
        target_position_band=target.position_band,
        target_ctr_gap=target.ctr_gap,
        target_query_count=target.query_count,
        has_gsc_data=target.has_gsc_data,
        pair_query_overlap=found.query_overlap,
        pair_kw_overlap=found.keyword_overlap,
        target_kw_count=len(target.signals.keywords),
        target_max_priority=target.max_priority,
        target_keyword_gap=target.signals.keyword_gap,
        target_inbound_count=t.inbound,
        target_is_orphan=target.is_orphan,
        target_crawl_depth=t.crawl_depth,
        target_page_rank_percentile=t.page_rank_percentile,
        target_saturation_ratio=target.saturation_ratio,
        source_outbound_count=s.outbound,
        source_outbound_density=source.outbound_density,
        source_link_equity_share=source.link_equity_share,
        same_hub=found.same_hub,
        source_is_hub_pillar=s.is_hub_pillar,
        target_is_hub_pillar=t.is_hub_pillar,
        source_hub_size=source.hub_size,
        target_hub_size=target.hub_size,
        target_hub_coverage=target.hub_coverage,
        source_link_community_id=s.link_community_id,
        target_link_community_id=t.link_community_id,
        source_keyword_community_id=s.keyword_community_id,
        target_keyword_community_id=t.keyword_community_id,
        source_content_community_id=s.content_community_id,
        target_content_community_id=t.content_community_id,
        source_hub_id=s.hub_id,
        target_hub_id=t.hub_id,
        same_link_community=found.same_link_community,
        same_keyword_community=found.same_keyword_community,
        same_content_community=found.same_content_community,
        cluster_agreement=found.cluster_agreement,
        content_agreement=found.content_agreement,
        content_cosine=similarity,
    )


def missing_pages(
    targets: Iterable[TargetCandidates], pages: Mapping[str, PageContext]
) -> list[str]:
    """Candidate urls without a page context, ascending."""
    urls = {url for entry in targets for url in (entry.target_url, *entry.sources)}
    return sorted(urls - pages.keys())


def feature_chunks(
    targets: Iterable[TargetCandidates],
    pages: Mapping[str, PageContext],
    *,
    chunk_pairs: int = CHUNK_PAIRS,
) -> Iterator[list[PairFeatures]]:
    """Every candidate pair's features in candidate order, ``chunk_pairs`` at a time."""
    if chunk_pairs < 1:
        raise ValueError("chunk_pairs must be at least 1")
    chunk: list[PairFeatures] = []
    for entry in targets:
        target = _context(pages, entry.target_url)
        for source_url, similarity in zip(entry.sources, entry.similarities, strict=True):
            chunk.append(pair_features(_context(pages, source_url), target, similarity))
            if len(chunk) == chunk_pairs:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _context(pages: Mapping[str, PageContext], url: str) -> PageContext:
    try:
        return pages[url]
    except KeyError:
        raise ValueError(f"no page context for candidate url {url!r}") from None


# The modules whose code decides the matrix values.
_CODE_MODULES: Final = (__name__, "linking_engine.discovery.signals", "linking_engine.gsc")


@functools.cache
def code_digest() -> str:
    """sha256 over the files of the modules that compute the features, so any change to
    that code invalidates every cached matrix."""
    digest = hashlib.sha256()
    for name in _CODE_MODULES:
        path = sys.modules[name].__file__
        if path is None:
            raise RuntimeError(f"module {name} has no file to hash")
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def cache_key(
    tenant_id: str, targets: Iterable[TargetCandidates], pages: Mapping[str, PageContext]
) -> str:
    """sha256 over the tenant, the feature code, the matrix columns, the candidate pairs in
    order and every page context; sets are hashed sorted, so the key does not depend on the
    process."""
    digest = hashlib.sha256()

    def add(value: object) -> None:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")

    add([tenant_id, code_digest(), KEY_COLUMNS, FEATURE_COLUMNS])
    for entry in targets:
        add([entry.target_url, entry.sources, entry.similarities])
    for url in sorted(pages):
        page = pages[url]
        add(
            [
                page.model_dump(mode="json", exclude={"signals": {"queries", "keywords"}}),
                sorted(page.signals.queries),
                sorted(page.signals.keywords),
            ]
        )
    return digest.hexdigest()


def to_frame(rows: Sequence[PairFeatures]) -> pandas.DataFrame:
    """The key columns, then FEATURE_COLUMNS as float64: booleans 0/1, None as NaN, and
    categoricals one-hot."""
    data: dict[str, object] = {key: [getattr(row, key) for row in rows] for key in KEY_COLUMNS}
    for field in _MATRIX_FIELDS:
        values = [getattr(row, field) for row in rows]
        levels = _LEVELS.get(field)
        if levels is None:
            data[field] = np.array(values, dtype=np.float64)
            continue
        for level, column in zip(levels, _columns(field), strict=True):
            data[column] = np.array([value == level for value in values], dtype=np.float64)
    return pandas.DataFrame(data, columns=[*KEY_COLUMNS, *FEATURE_COLUMNS])


def feature_report(
    tenant_id: str,
    frames: Iterable[pandas.DataFrame],
    *,
    cache_key: str,
    cache_hit: bool,
    started: float,
) -> FeatureReport:
    """The data gaps of the matrix, summarised one frame at a time so it is never held whole.

    ``started`` is the run's ``time.perf_counter()`` start. A column is constant when every
    row holds the same value; one with nulls and a single value still splits on its nulls.
    """
    width = len(FEATURE_COLUMNS)
    gsc_column = FEATURE_COLUMNS.index("has_gsc_data")
    rows = chunks = 0
    with_gsc = 0.0
    nulls = np.zeros(width, dtype=np.int64)
    low = np.full(width, np.nan)
    high = np.full(width, np.nan)
    for frame in frames:
        chunks += 1
        matrix = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        if not len(matrix):
            continue
        rows += len(matrix)
        nulls += np.isnan(matrix).sum(axis=0)
        # fmin and fmax skip NaN, and leave a column NaN only when it has no value at all.
        low = np.fmin(low, np.fmin.reduce(matrix, axis=0))
        high = np.fmax(high, np.fmax.reduce(matrix, axis=0))
        with_gsc += float(matrix[:, gsc_column].sum())
    counts = dict(zip(FEATURE_COLUMNS, nulls.tolist(), strict=True))
    return FeatureReport(
        tenant_id=tenant_id,
        pairs=rows,
        columns=FEATURE_COLUMNS,
        chunks=chunks,
        all_null_columns=tuple(name for name, count in counts.items() if rows and count == rows),
        constant_columns=tuple(
            name
            for i, name in enumerate(FEATURE_COLUMNS)
            if rows and not counts[name] and low[i] == high[i]
        ),
        null_share={name: count / rows for name, count in counts.items()} if rows else {},
        has_gsc_data_share=with_gsc / rows if rows else None,
        cache_key=cache_key,
        cache_hit=cache_hit,
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )


def summarise_features(report: FeatureReport) -> str:
    """A short prose record of one feature assembly run, for the MLflow run description."""
    source = "read from the cache" if report.cache_hit else "built"
    scope = (
        f"Feature assembly for tenant {report.tenant_id}: {report.pairs} candidate pairs, "
        f"{len(report.columns)} matrix columns, {source} in {report.chunks} chunks "
        f"(cache key {report.cache_key[:12]})."
    )
    timing = f"{report.seconds:.1f} s."
    if report.has_gsc_data_share is None:
        return "\n".join([scope, "No candidate pairs, so an empty matrix.", timing])
    gaps = [
        f"GSC data for the target of {report.has_gsc_data_share:.1%} of the pairs.",
        "All null: " + (", ".join(report.all_null_columns) or "none") + ".",
        "Constant: " + (", ".join(report.constant_columns) or "none") + ".",
    ]
    partly = sorted(
        ((share, name) for name, share in report.null_share.items() if 0 < share < 1),
        reverse=True,
    )
    if partly:
        gaps.append(
            "Partly null: " + ", ".join(f"{name} {share:.1%}" for share, name in partly) + "."
        )
    return "\n".join([scope, *gaps, timing])
