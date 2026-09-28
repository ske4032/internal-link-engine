"""Assemble the feature matrix of a tenant's candidate pairs as a Parquet file per tenant,
keyed by the contents of everything it is built from, so an unchanged input reuses it."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import structlog

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import (
    CHUNK_PAIRS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    STAGE,
    cache_key,
    feature_chunks,
    feature_report,
    missing_pages,
    page_contexts,
    to_frame,
)
from linking_engine.errors import DatabaseReadError
from linking_engine.gsc import fit_ctr_curve
from linking_engine.pipeline.signals import load_page_signals

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import pandas

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import (
        CtrCurve,
        FeatureReport,
        GscMetrics,
        PageSignals,
        PageStructure,
        StrategicKeyword,
        TargetCandidates,
    )

log = structlog.get_logger(__name__)

CACHE_DIR: Final = Path(".cache/features")
# Written by anchor selection beside the matrix; its rank-1 rows fill the placement features.
ANCHOR_CHOICES_FILE: Final = "anchor_choices.parquet"
_PLACEMENT_COLUMNS: Final = (
    "source_url",
    "target_url",
    "rank",
    "context_relevance",
    "anchor_target_fit",
)


@dataclass(frozen=True, slots=True)
class AnchorPlacements:
    # (source, target) of each pair with a chosen anchor, to its context relevance and
    # anchor-target fit; either is None when its vector was missing.
    features: dict[tuple[str, str], tuple[float | None, float | None]]
    # sha256 of the choices file they came from; None without a usable one.
    digest: str | None
    # Rank-1 rows read from it, those without any feature included.
    rows: int = 0


def anchor_placements(cache_dir: Path, tenant_id: str) -> AnchorPlacements:
    """The placement features of every pair's chosen anchor, from the tenant's anchor choices
    file; none when there is no file, or it is unreadable or inconsistent. The digest is of the
    bytes read, so it always matches the features."""
    path = cache_dir / tenant_id / ANCHOR_CHOICES_FILE
    if not path.is_file():
        return AnchorPlacements({}, None)
    try:
        data = path.read_bytes()
        table = pq.read_table(pa.BufferReader(data), columns=list(_PLACEMENT_COLUMNS))
    except (OSError, pa.ArrowException) as error:
        log.warning(
            "features.anchor_choices_unreadable",
            stage=STAGE,
            tenant_id=tenant_id,
            error=type(error).__name__,
        )
        return AnchorPlacements({}, None)
    features: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    seen: set[tuple[str, str]] = set()
    rows = table.filter(pc.equal(table["rank"], 1)).to_pylist()
    for row in rows:
        pair = (row["source_url"], row["target_url"])
        placement = (row["context_relevance"], row["anchor_target_fit"])
        values = [float(value) for value in placement if value is not None]
        if pair in seen or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            log.warning("features.anchor_choices_invalid", stage=STAGE, tenant_id=tenant_id)
            return AnchorPlacements({}, None)
        seen.add(pair)
        if values:
            features[pair] = placement
    return AnchorPlacements(features, hashlib.sha256(data).hexdigest(), rows=len(rows))


async def assemble_features(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    chunk_pairs: int = CHUNK_PAIRS,
) -> tuple[FeatureReport, Path]:
    """The features of every candidate pair of the tenant, at
    ``<cache_dir>/<tenant>/<cache_key>.parquet``, and the report of the run."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    if chunk_pairs < 1:
        raise ValueError("chunk_pairs must be at least 1")

    started = time.perf_counter()
    candidates = await retrieve_candidates(graph, tenant_id)
    signals, _ = await load_page_signals(graph, mongo, tenant_id)
    structure = await graph.page_structure(tenant_id)
    snapshot = await graph.link_graph(tenant_id)
    metrics = await mongo.gsc_metrics(tenant_id)
    keywords = await mongo.strategic_keywords(tenant_id)
    curve = fit_ctr_curve(await mongo.gsc_query_stats(tenant_id))
    placements = await asyncio.to_thread(anchor_placements, cache_dir, tenant_id)
    report, path = await asyncio.to_thread(
        _build,
        tenant_id,
        candidates.targets,
        structure,
        signals,
        snapshot.links,
        metrics,
        keywords,
        curve,
        placements,
        cache_dir=cache_dir,
        chunk_pairs=chunk_pairs,
        started=started,
    )
    log.info(
        "features.assembled",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=report.pairs,
        chunks=report.chunks,
        columns=len(report.columns),
        all_null_columns=report.all_null_columns,
        constant_columns=report.constant_columns,
        has_gsc_data_share=report.has_gsc_data_share,
        gsc_curve=curve is not None,
        cache_key=report.cache_key,
        cache_hit=report.cache_hit,
        anchor_choices_read=placements.rows,
        placements_filled=sum(
            1
            for entry in candidates.targets
            for source in entry.sources
            if (source, entry.target_url) in placements.features
        ),
        anchor_choices_digest=report.anchor_choices_digest,
        seconds=report.seconds,
    )
    return report, path


def _build(
    tenant_id: str,
    targets: Sequence[TargetCandidates],
    structure: Sequence[PageStructure],
    signals: Mapping[str, PageSignals],
    links: Iterable[tuple[str, str]],
    metrics: Sequence[GscMetrics],
    keywords: Sequence[StrategicKeyword],
    curve: CtrCurve | None,
    placements: AnchorPlacements,
    *,
    cache_dir: Path,
    chunk_pairs: int,
    started: float,
) -> tuple[FeatureReport, Path]:
    # Every input is a separate read, so the graph can change between them.
    try:
        pages = page_contexts(structure, signals, links, metrics, keywords, curve)
    except ValueError as error:
        raise DatabaseReadError("neo4j", f"feature inputs of {tenant_id!r}: {error}") from error
    missing = missing_pages(targets, pages)
    if missing:
        raise DatabaseReadError(
            "neo4j",
            f"{len(missing)} candidate pages of {tenant_id!r} are no longer crawled pages, "
            f"first {missing[0]!r}",
        )

    key = cache_key(tenant_id, targets, pages, anchor_choices_digest=placements.digest)
    path = cache_dir / tenant_id / f"{key}.parquet"
    schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in KEY_COLUMNS]
        + [pa.field(name, pa.float64()) for name in FEATURE_COLUMNS],
        metadata={
            "feature_columns": json.dumps(FEATURE_COLUMNS),
            "key_columns": json.dumps(KEY_COLUMNS),
            "cache_key": key,
            "tenant_id": tenant_id,
        },
    )
    if path.is_file():
        cached = _cached(
            tenant_id,
            path,
            schema,
            key,
            chunk_pairs=chunk_pairs,
            started=started,
            anchor_choices_digest=placements.digest,
        )
        if cached is not None:
            return cached, path

    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the cache path and renamed, so a file at that path is always complete.
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{key}.", suffix=".tmp")
    os.close(handle)
    temp = Path(name)
    try:
        with pq.ParquetWriter(temp, schema) as writer:
            frames = (
                to_frame(chunk)
                for chunk in feature_chunks(
                    targets, pages, chunk_pairs=chunk_pairs, placements=placements.features
                )
            )
            report = feature_report(
                tenant_id,
                _written(frames, writer, schema, chunk_pairs),
                cache_key=key,
                cache_hit=False,
                started=started,
                anchor_choices_digest=placements.digest,
            )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return report, path


def _written(
    frames: Iterable[pandas.DataFrame], writer: pq.ParquetWriter, schema: pa.Schema, rows: int
) -> Iterator[pandas.DataFrame]:
    """Each frame, once written as one row group."""
    for frame in frames:
        writer.write_table(
            pa.Table.from_pandas(frame, schema=schema, preserve_index=False),
            row_group_size=rows,
        )
        yield frame


def _cached(
    tenant_id: str,
    path: Path,
    schema: pa.Schema,
    key: str,
    *,
    chunk_pairs: int,
    started: float,
    anchor_choices_digest: str | None,
) -> FeatureReport | None:
    """The report of the cached matrix, streamed from its file; None when the file is not
    this key's matrix and has to be rebuilt."""
    try:
        with pq.ParquetFile(path) as cached:
            if not cached.schema_arrow.equals(schema, check_metadata=True):
                log.warning(
                    "features.cache_invalid", stage=STAGE, tenant_id=tenant_id, cache_key=key
                )
                return None
            return feature_report(
                tenant_id,
                (
                    batch.to_pandas()
                    for batch in cached.iter_batches(
                        batch_size=chunk_pairs, columns=list(FEATURE_COLUMNS)
                    )
                ),
                cache_key=key,
                cache_hit=True,
                started=started,
                anchor_choices_digest=anchor_choices_digest,
            )
    except (OSError, pa.ArrowException) as error:
        log.warning(
            "features.cache_unreadable",
            stage=STAGE,
            tenant_id=tenant_id,
            cache_key=key,
            error=type(error).__name__,
        )
        return None
