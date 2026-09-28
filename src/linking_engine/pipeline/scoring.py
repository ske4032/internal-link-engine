"""Score a tenant's candidate pairs with the baseline scorer over its cached feature matrix,
and persist the scores beside it for later comparison against learned rankers."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from linking_engine.discovery.features import FEATURE_COLUMNS, KEY_COLUMNS
from linking_engine.discovery.scoring import (
    STAGE,
    TOP_CONTRIBUTIONS,
    default_weights,
    score_frame,
    score_report,
    weights_hash,
)
from linking_engine.pipeline.features import assemble_features

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import ScoreReport, ScorerWeights

log = structlog.get_logger(__name__)


async def score_pairs(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> tuple[ScoreReport, Path]:
    """Every candidate pair's baseline score, tier and top contributions, at
    ``<cache_dir>/<tenant>/<feature_cache_key>.<weights_hash[:12]>.scores.parquet``, and the
    report of the run. The tenant's stored weights override the packaged default."""
    started = time.perf_counter()
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    unknown = [f.column for f in weights.features if f.column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"scorer weights {weights.version!r} of {tenant_id!r} name columns the feature "
            f"matrix does not have: {', '.join(unknown)}"
        )
    features, matrix = await assemble_features(graph, mongo, tenant_id, cache_dir=cache_dir)
    path = (
        cache_dir / tenant_id / f"{features.cache_key}.{weights_hash(weights)[:12]}.scores.parquet"
    )
    report = await asyncio.to_thread(
        _score, tenant_id, matrix, path, weights, features.cache_key, started
    )
    log.info(
        "scores.computed",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=report.pairs,
        weights_version=report.weights.version,
        weights_hash=report.weights_hash,
        tiers=report.tiers,
        score_p10=report.score_p10,
        score_p50=report.score_p50,
        score_p90=report.score_p90,
        feature_cache_key=report.feature_cache_key,
        feature_cache_hit=features.cache_hit,
        seconds=report.seconds,
    )
    return report, path


def _score(
    tenant_id: str,
    matrix: Path,
    path: Path,
    weights: ScorerWeights,
    feature_cache_key: str,
    started: float,
) -> ScoreReport:
    columns = [feature.column for feature in weights.features]
    frame = pq.read_table(matrix, columns=[*KEY_COLUMNS, *columns]).to_pandas()
    scores = score_frame(frame, weights)
    schema = pa.schema(
        [
            pa.field("source_url", pa.string(), nullable=False),
            pa.field("target_url", pa.string(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("tier", pa.int8(), nullable=False),
            pa.field("raw_score", pa.float64()),
        ]
        + [
            field
            for rank in range(1, TOP_CONTRIBUTIONS + 1)
            for field in (
                pa.field(f"top{rank}_feature", pa.string()),
                pa.field(f"top{rank}_contribution", pa.float64()),
                pa.field(f"top{rank}_value", pa.float64()),
            )
        ],
        metadata={
            "weights": weights.model_dump_json(),
            "weights_hash": weights_hash(weights),
            "feature_cache_key": feature_cache_key,
            "tenant_id": tenant_id,
        },
    )
    table = pa.Table.from_pandas(scores, schema=schema, preserve_index=False)
    # Written beside the target and renamed, so a file at that path is always complete.
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temp = Path(name)
    try:
        pq.write_table(table, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return score_report(
        tenant_id, frame, scores, weights, feature_cache_key=feature_cache_key, started=started
    )
