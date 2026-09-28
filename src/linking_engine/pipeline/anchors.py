"""Extract anchor phrases for a tenant's candidate pairs and hub bridges from the source pages'
own copy, and write them as a Parquet file per tenant. Read-only against both stores.

`lexical_run` does the reads and the ladder's rungs 1 to 2.5 in memory, so anchor selection
reuses them; `extract_anchors` writes their matches.
"""

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
from linking_engine.anchor.scoring import Brand, brand_tokens, brand_words
from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.models import ExtractionSettings
from linking_engine.pipeline.bridges import BRIDGES_FILE

if TYPE_CHECKING:
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
        pa.field("semantic_similarity", pa.float64()),
    ]
)


@dataclass(frozen=True, slots=True)
class AnchorPair:
    source_url: str
    target_url: str
    # The source's retrieval cosine to the target: the candidate's, or the bridge link's.
    similarity: float
    bridge: bool


@dataclass(frozen=True, slots=True)
class SourcePage:
    """What extraction reads of one source page."""

    body: str
    headings: tuple[str, ...]
    language: str | None


@dataclass(frozen=True, slots=True)
class LexicalRun:
    """A tenant's pairs, their source pages and keywords, and the matches of rungs 1 to 2.5."""

    tenant_id: str
    settings: ExtractionSettings
    # Candidates, then bridges not among them; per target, retrieval similarity descending.
    pairs: tuple[AnchorPair, ...]
    # Distinct bridge links read, candidates among them included.
    bridge_pairs: int
    keywords: dict[str, list[tuple[int, str, KeywordSource]]]
    # The tenant's brand affixes as token sequences; their tokens are never identifiers.
    brand: Brand
    # Stored source pages, and the index of each with body text.
    sources: dict[str, SourcePage]
    indexes: dict[str, SourceIndex]
    # One per source language.
    stems: dict[str | None, Stems]
    # Located existing link anchors per indexed source page, which no phrase may overlap.
    existing: dict[str, list[tuple[int, int]]]
    matches: dict[tuple[str, str], list[AnchorMatch]]
    overlapping: int
    # Distinct stemmed or stem set places per source refused because their identifiers
    # disagreed with the keyword's.
    identifier_mismatches: int
    located: int
    unlocated: int
    without_body: int
    missing: int


def cache_folder(cache_dir: Path, tenant_id: str) -> Path:
    """The tenant's folder under ``cache_dir``; the tenant id must be usable as its name."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    return cache_dir / tenant_id


async def lexical_run(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> LexicalRun:
    """The reads of both anchor stages and every pair's lexical matches."""
    folder = cache_folder(cache_dir, tenant_id)
    settings = await mongo.get_extraction_settings(tenant_id) or ExtractionSettings()
    candidates = await retrieve_candidates(graph, tenant_id)
    bridges = await asyncio.to_thread(_bridge_pairs, folder / BRIDGES_FILE)
    found: dict[tuple[str, str], AnchorPair] = {}
    for entry in candidates.targets:
        for source, similarity in zip(entry.sources, entry.similarities, strict=True):
            found[(source, entry.target_url)] = AnchorPair(
                source, entry.target_url, similarity, bridge=False
            )
    for source, target, similarity in bridges:
        found.setdefault((source, target), AnchorPair(source, target, similarity, bridge=True))
    pairs = tuple(sorted(found.values(), key=lambda p: (p.target_url, -p.similarity, p.source_url)))
    keywords = await graph.ranked_keywords(tenant_id)
    brand = brand_tokens(await mongo.page_titles(tenant_id))

    wanted = {pair.source_url for pair in pairs}
    sources: dict[str, SourcePage] = {}
    async for records in mongo.iter_page_records(tenant_id, urls=wanted):
        for record in records:
            sources[record.url] = SourcePage(
                record.body_text,
                tuple(heading.text for heading in record.headings),
                record.language,
            )
    links: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    async for texts in graph.iter_link_texts(tenant_id):
        for text in texts:
            if text.source_url in wanted:
                links[text.source_url].append((text.anchor_text, text.surrounding_text))

    return await asyncio.to_thread(
        _lexical,
        tenant_id,
        settings,
        pairs,
        len({(source, target) for source, target, _ in bridges}),
        keywords,
        brand,
        sources,
        links,
    )


def _bridge_pairs(path: Path) -> list[tuple[str, str, float]]:
    """(source, target, similarity) of every bridge link, alternatives included; none without
    the file."""
    if not path.is_file():
        return []
    table = pq.read_table(path, columns=["source_url", "target_url", "similarity"])
    return list(
        zip(
            table.column("source_url").to_pylist(),
            table.column("target_url").to_pylist(),
            table.column("similarity").to_pylist(),
            strict=True,
        )
    )


def _lexical(
    tenant_id: str,
    settings: ExtractionSettings,
    pairs: tuple[AnchorPair, ...],
    bridge_pairs: int,
    keywords: dict[str, list[tuple[int, str, KeywordSource]]],
    brand: Brand,
    sources: dict[str, SourcePage],
    links: dict[str, list[tuple[str, str]]],
) -> LexicalRun:
    targets: defaultdict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        targets[pair.source_url].append(pair.target_url)
    words = brand_words(brand)
    stems: dict[str | None, Stems] = {}
    indexes: dict[str, SourceIndex] = {}
    existing: dict[str, list[tuple[int, int]]] = {}
    matches: dict[tuple[str, str], list[AnchorMatch]] = {}
    overlapping = mismatched = without_body = missing = located = unlocated = 0
    for source_url, source_targets in targets.items():
        source = sources.get(source_url)
        if source is None or not source.body.strip():
            without_body += 1
            missing += source is None
            continue
        language = source.language
        if language not in stems:
            stems[language] = Stems(language)
        index = indexes[source_url] = SourceIndex(
            source_url, source.body, source.headings, stems[language], brand=words
        )
        source_links = links.get(source_url, [])
        spans, lost = locate_anchors(source.body, source_links)
        existing[source_url] = spans
        located += len(source_links) - lost
        unlocated += lost
        blocking: set[tuple[int, int]] = set()
        refused: set[tuple[int, int]] = set()
        for target_url in source_targets:
            ranked = keywords.get(target_url)
            if not ranked:
                continue
            found, blocked, disagreed = extract(
                index, target_url, ranked, existing=spans, threshold=settings.stem_set_threshold
            )
            if found:
                matches[(source_url, target_url)] = found
            blocking |= blocked
            refused |= disagreed
        overlapping += len(blocking)
        mismatched += len(refused)
    return LexicalRun(
        tenant_id=tenant_id,
        settings=settings,
        pairs=pairs,
        bridge_pairs=bridge_pairs,
        keywords=keywords,
        brand=brand,
        sources=sources,
        indexes=indexes,
        stems=stems,
        existing=existing,
        matches=matches,
        overlapping=overlapping,
        identifier_mismatches=mismatched,
        located=located,
        unlocated=unlocated,
        without_body=without_body,
        missing=missing,
    )


async def extract_anchors(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> tuple[AnchorReport, Path]:
    """Every candidate pair's and hub bridge's anchor phrases found in the source copy, at
    ``<cache_dir>/<tenant>/anchors.parquet``, and the report of the run."""
    started = time.perf_counter()
    run = await lexical_run(graph, mongo, tenant_id, cache_dir=cache_dir)
    report, path = await asyncio.to_thread(
        _write_matches, run, cache_folder(cache_dir, tenant_id), started
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
        identifier_mismatches=report.identifier_mismatches,
        existing_anchors_located=report.existing_anchors_located,
        existing_anchors_unlocated=report.existing_anchors_unlocated,
        source_pages=report.source_pages,
        sources_without_body=report.sources_without_body,
        seconds=report.seconds,
    )
    return report, path


def _write_matches(run: LexicalRun, folder: Path, started: float) -> tuple[AnchorReport, Path]:
    matches = sorted(
        (match for found in run.matches.values() for match in found),
        key=lambda match: (match.source_url, match.target_url, match.keyword_rank),
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ANCHORS_FILE
    write_atomically(
        pa.Table.from_pylist([match.model_dump(mode="json") for match in matches], schema=_SCHEMA),
        path,
    )
    sources = {pair.source_url for pair in run.pairs}
    report = anchor_report(
        run.tenant_id,
        matches,
        threshold=run.settings.stem_set_threshold,
        pairs=len(run.pairs),
        bridge_pairs=run.bridge_pairs,
        pairs_with_keywords=sum(1 for pair in run.pairs if run.keywords.get(pair.target_url)),
        overlapping=run.overlapping,
        identifier_mismatches=run.identifier_mismatches,
        located_anchors=run.located,
        unlocated_anchors=run.unlocated,
        keywords={
            (text, run.sources[pair.source_url].language)
            for pair in run.pairs
            if pair.source_url in run.indexes
            for _, text, _ in run.keywords.get(pair.target_url, ())
        },
        source_languages={url: page.language for url in sources if (page := run.sources.get(url))},
        sources_without_body=run.without_body,
        missing_sources=run.missing,
        started=started,
    )
    return report, path


def write_atomically(table: pa.Table, path: Path) -> None:
    """Written beside the target and renamed, so a file at that path is always complete."""
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temp = Path(name)
    try:
        pq.write_table(table, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
