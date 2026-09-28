"""Extract anchor phrases for a tenant's candidate pairs and hub bridges from the source pages'
own copy, and write them as a Parquet file per tenant. Read-only against both stores."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from linking_engine.anchor.extraction import (
    STAGE,
    SourceIndex,
    Stems,
    anchor_report,
    extract,
    locate_anchors,
)
from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.models import ExtractionSettings
from linking_engine.pipeline.bridges import BRIDGES_FILE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import AnchorMatch, AnchorReport, KeywordSource

log = structlog.get_logger(__name__)

ANCHORS_FILE: Final = "anchors.parquet"

_SCHEMA: Final = pa.schema(
    [
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("target_url", pa.string(), nullable=False),
        pa.field("keyword", pa.string(), nullable=False),
        pa.field("keyword_rank", pa.int64(), nullable=False),
        pa.field("keyword_source", pa.string(), nullable=False),
        pa.field("rung", pa.string(), nullable=False),
        pa.field("phrase", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("sentence", pa.string(), nullable=False),
        pa.field("sentence_index", pa.int64(), nullable=False),
        pa.field("sentence_start", pa.int64(), nullable=False),
        pa.field("stem_jaccard", pa.float64()),
    ]
)


@dataclass(frozen=True, slots=True)
class _Source:
    """What extraction reads of one source page."""

    body: str
    headings: tuple[str, ...]
    language: str | None


async def extract_anchors(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> tuple[AnchorReport, Path]:
    """Every candidate pair's and hub bridge's anchor phrases found in the source copy, at
    ``<cache_dir>/<tenant>/anchors.parquet``, and the report of the run."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")

    started = time.perf_counter()
    folder = cache_dir / tenant_id
    settings = await mongo.get_extraction_settings(tenant_id) or ExtractionSettings()
    candidates = await retrieve_candidates(graph, tenant_id)
    bridges = await asyncio.to_thread(_bridge_pairs, folder / BRIDGES_FILE)
    pairs = list(
        dict.fromkeys(
            [(source, entry.target_url) for entry in candidates.targets for source in entry.sources]
            + bridges
        )
    )
    keywords = await graph.ranked_keywords(tenant_id)

    wanted = {source for source, _ in pairs}
    sources: dict[str, _Source] = {}
    async for records in mongo.iter_page_records(tenant_id, urls=wanted):
        for record in records:
            sources[record.url] = _Source(
                record.body_text,
                tuple(heading.text for heading in record.headings),
                record.language,
            )
    links: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    async for texts in graph.iter_link_texts(tenant_id):
        for text in texts:
            if text.source_url in wanted:
                links[text.source_url].append((text.anchor_text, text.surrounding_text))

    report, path = await asyncio.to_thread(
        _extract,
        tenant_id,
        pairs,
        len(set(bridges)),
        keywords,
        sources,
        links,
        settings,
        folder=folder,
        started=started,
    )
    log.info(
        "anchors.extracted",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=report.pairs,
        bridge_pairs=report.bridge_pairs,
        pairs_with_keywords=report.pairs_with_keywords,
        pairs_matched=report.pairs_matched,
        primary_matched=report.primary_matched,
        matches=report.matches,
        by_rung={rung.value: count for rung, count in report.by_rung.items()},
        overlapping_existing_anchors=report.overlapping_existing_anchors,
        existing_anchors_located=report.existing_anchors_located,
        existing_anchors_unlocated=report.existing_anchors_unlocated,
        source_pages=report.source_pages,
        sources_without_body=report.sources_without_body,
        seconds=report.seconds,
    )
    return report, path


def _bridge_pairs(path: Path) -> list[tuple[str, str]]:
    """(source, target) of every bridge link, alternatives included; none without the file."""
    if not path.is_file():
        return []
    table = pq.read_table(path, columns=["source_url", "target_url"])
    return list(
        zip(
            table.column("source_url").to_pylist(),
            table.column("target_url").to_pylist(),
            strict=True,
        )
    )


def _extract(
    tenant_id: str,
    pairs: Sequence[tuple[str, str]],
    bridge_pairs: int,
    keywords: Mapping[str, list[tuple[int, str, KeywordSource]]],
    sources: Mapping[str, _Source],
    links: Mapping[str, Sequence[tuple[str, str]]],
    settings: ExtractionSettings,
    *,
    folder: Path,
    started: float,
) -> tuple[AnchorReport, Path]:
    targets: defaultdict[str, list[str]] = defaultdict(list)
    for pair_source, pair_target in pairs:
        targets[pair_source].append(pair_target)
    stems: dict[str | None, Stems] = {}
    matches: list[AnchorMatch] = []
    overlapping = without_body = missing = located = unlocated = 0
    for source_url, source_targets in targets.items():
        source = sources.get(source_url)
        if source is None or not source.body.strip():
            without_body += 1
            missing += source is None
            continue
        language = source.language
        if language not in stems:
            stems[language] = Stems(language)
        index = SourceIndex(source_url, source.body, source.headings, stems[language])
        source_links = links.get(source_url, ())
        spans, lost = locate_anchors(source.body, source_links)
        located += len(source_links) - lost
        unlocated += lost
        blocking: set[tuple[int, int]] = set()
        for target_url in source_targets:
            ranked = keywords.get(target_url)
            if not ranked:
                continue
            found, blocked = extract(
                index, target_url, ranked, existing=spans, threshold=settings.stem_set_threshold
            )
            matches.extend(found)
            blocking |= blocked
        overlapping += len(blocking)
    matches.sort(key=lambda match: (match.source_url, match.target_url, match.keyword_rank))

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ANCHORS_FILE
    _write(
        pa.Table.from_pylist([match.model_dump(mode="json") for match in matches], schema=_SCHEMA),
        path,
    )
    report = anchor_report(
        tenant_id,
        matches,
        threshold=settings.stem_set_threshold,
        pairs=len(pairs),
        bridge_pairs=bridge_pairs,
        pairs_with_keywords=sum(1 for _, target in pairs if keywords.get(target)),
        overlapping=overlapping,
        located_anchors=located,
        unlocated_anchors=unlocated,
        keywords={
            (text, page.language)
            for source_url, target_url in pairs
            if (page := sources.get(source_url)) is not None and page.body.strip()
            for _, text, _ in keywords.get(target_url, ())
        },
        source_languages={url: page.language for url in targets if (page := sources.get(url))},
        sources_without_body=without_body,
        missing_sources=missing,
        started=started,
    )
    return report, path


def _write(table: pa.Table, path: Path) -> None:
    """Written beside the target and renamed, so a file at that path is always complete."""
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temp = Path(name)
    try:
        pq.write_table(table, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
