"""Find the bridges that keep a tenant's hubs connected and write them, with every scored hub
pair, as Parquet files per tenant. Read-only against both stores."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from linking_engine.discovery.bridges import (
    STAGE,
    bridge_links,
    bridge_report,
    covered_pairs,
    hub_pairs,
)
from linking_engine.errors import DatabaseReadError
from linking_engine.models import HubPair
from linking_engine.pipeline.signals import load_page_signals

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import BridgeLink, BridgeReport, KeywordRung, PageStructure

log = structlog.get_logger(__name__)

BRIDGES_FILE: Final = "bridges.parquet"
HUB_PAIRS_FILE: Final = "hub_pairs.parquet"

_PAIR_SCHEMA: Final = pa.schema(
    [
        pa.field("language", pa.string()),
        pa.field("hub_a", pa.int64(), nullable=False),
        pa.field("hub_b", pa.int64(), nullable=False),
        pa.field("size_a", pa.int64(), nullable=False),
        pa.field("size_b", pa.int64(), nullable=False),
        pa.field("pages_ab", pa.int64(), nullable=False),
        pa.field("pages_ba", pa.int64(), nullable=False),
        pa.field("link_density", pa.float64(), nullable=False),
        pa.field("centroid_cosine", pa.float64(), nullable=False),
        pa.field("query_jaccard", pa.float64()),
        pa.field("shared_queries", pa.list_(pa.string()), nullable=False),
        pa.field("bridge_gap", pa.float64(), nullable=False),
        # BridgeReason values; empty for a pair no reason covers.
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
    ]
)
_LINK_SCHEMA: Final = pa.schema(
    [
        pa.field("language", pa.string()),
        pa.field("hub_from", pa.int64(), nullable=False),
        pa.field("hub_to", pa.int64(), nullable=False),
        pa.field("slot", pa.int64(), nullable=False),
        pa.field("rank", pa.int64(), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("target_url", pa.string(), nullable=False),
        pa.field("similarity", pa.float64(), nullable=False),
        pa.field("source_page_rank_percentile", pa.float64()),
        pa.field("anchor_keyword", pa.string()),
        pa.field("anchor_rung", pa.string()),
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
    ]
)


async def find_bridges(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> tuple[BridgeReport, Path]:
    """The tenant's bridge links at ``<cache_dir>/<tenant>/bridges.parquet``, every scored hub
    pair beside it in ``hub_pairs.parquet``, and the report of the run."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")

    started = time.perf_counter()
    pages = await graph.page_structure(tenant_id)
    centroids, _ = await graph.stored_hubs(tenant_id)
    snapshot = await graph.link_graph(tenant_id)
    vectors = await graph.content_vectors(tenant_id)
    selection = await graph.candidate_targets(tenant_id)
    copies = await graph.non_canonical_copies(tenant_id)
    keywords = await graph.resolved_keywords(tenant_id)
    signals, _ = await load_page_signals(graph, mongo, tenant_id)
    queries = {url: page.queries for url, page in signals.items()}
    # Without a single GSC query on a crawled page, query overlap says nothing.
    gsc_used = any(queries.values())
    report, path = await asyncio.to_thread(
        _find,
        tenant_id,
        pages,
        centroids,
        snapshot.links,
        vectors,
        frozenset(target.url for target in selection.targets),
        copies,
        keywords,
        queries if gsc_used else None,
        folder=cache_dir / tenant_id,
        started=started,
    )
    log.info("graph.bridges", stage=STAGE, **report.model_dump(mode="json"))
    return report, path


def _find(
    tenant_id: str,
    pages: Sequence[PageStructure],
    centroids: Mapping[int, npt.NDArray[np.float32]],
    links: Sequence[tuple[str, str]],
    vectors: Mapping[str, npt.NDArray[np.float32]],
    targets: frozenset[str],
    copies: frozenset[str],
    keywords: Mapping[str, tuple[str, KeywordRung]],
    queries: Mapping[str, frozenset[str]] | None,
    *,
    folder: Path,
    started: float,
) -> tuple[BridgeReport, Path]:
    # Every input is a separate read, so hubs, pages and centroids can disagree.
    try:
        pairs = covered_pairs(hub_pairs(pages, centroids, links, queries))
    except ValueError as error:
        raise DatabaseReadError("neo4j", f"hub graph of {tenant_id!r}: {error}") from error
    proposed = bridge_links(pairs, pages, links, vectors, targets, copies, keywords)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / BRIDGES_FILE
    _write_together(
        (
            pa.Table.from_pylist([_row(pair) for pair in pairs], schema=_PAIR_SCHEMA),
            folder / HUB_PAIRS_FILE,
        ),
        (pa.Table.from_pylist([_row(link) for link in proposed], schema=_LINK_SCHEMA), path),
    )
    report = bridge_report(
        tenant_id, pages, pairs, proposed, gsc_used=queries is not None, started=started
    )
    return report, path


def _row(model: HubPair | BridgeLink) -> dict[str, object]:
    return model.model_dump(mode="json")


def _write_together(*tables: tuple[pa.Table, Path]) -> None:
    """Every table is written beside its target before any is renamed, so a failed run leaves
    the previous files in place together."""
    temps: list[Path] = []
    try:
        for table, path in tables:
            handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            os.close(handle)
            temps.append(Path(name))
            pq.write_table(table, temps[-1])
        for temp, (_, path) in zip(temps, tables, strict=True):
            temp.replace(path)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)


def read_hub_pairs(path: Path) -> list[HubPair]:
    """The hub pairs of a ``hub_pairs.parquet`` file."""
    return [HubPair.model_validate(row) for row in pq.read_table(path).to_pylist()]
