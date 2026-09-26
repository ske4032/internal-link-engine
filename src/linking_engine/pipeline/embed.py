"""Embed a tenant's new and changed pages into Neo4j, one committed flush at a time."""

from __future__ import annotations

import gc
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched
from typing import TYPE_CHECKING, Final

import numpy as np
import structlog
from structlog.contextvars import bound_contextvars

from linking_engine.errors import (
    DatabaseError,
    DatabaseReadError,
    DatabaseWriteError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingResponseError,
    SchemaError,
)
from linking_engine.graph.repo import VECTOR_DIMENSIONS
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import EmbedRunReport, PageText

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from structlog.typing import FilteringBoundLogger

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import EmbeddingModelCount, EmbeddingTarget

log = structlog.get_logger(__name__)

FLUSH_SIZE: Final = 1000
STAGE: Final = "embedding"
SAMPLE_URLS: Final = 5
NO_MODEL: Final = "<no model>"


@dataclass(slots=True)
class _Tally:
    embedded: int = 0
    not_usable: int = 0
    empty_body: int = 0
    missing: int = 0
    hash_mismatch: int = 0
    api_tokens: int = 0
    tokens: int = 0
    truncated: int = 0

    @property
    def skipped(self) -> int:
        return self.not_usable + self.empty_body + self.missing + self.hash_mismatch

    def add(self, other: _Tally) -> None:
        self.embedded += other.embedded
        self.not_usable += other.not_usable
        self.empty_body += other.empty_body
        self.missing += other.missing
        self.hash_mismatch += other.hash_mismatch
        self.api_tokens += other.api_tokens
        self.tokens += other.tokens
        self.truncated += other.truncated


def check_models(found: Sequence[EmbeddingModelCount], configured: str, *, tenant_id: str) -> None:
    """Raise unless every stored vector carries the configured model; no stored vectors passes."""
    if not configured.strip():
        raise ValueError("configured model must be a non-empty string")
    if not found:
        return
    models = {row.embedding_model for row in found}
    listing = ", ".join(f"{row.embedding_model or NO_MODEL} ({row.vectors})" for row in found)
    if None in models:
        problem = "some stored vectors have no embeddingModel"
    elif len(models) > 1:
        problem = f"stored vectors mix {len(models)} embedding models"
    elif configured not in models:
        problem = "stored vectors use a different model than configured"
    else:
        return
    raise EmbeddingModelMismatchError(
        f"tenant {tenant_id}: {problem}; found {listing}; configured {configured}. "
        "Vectors from different models are not comparable"
    )


async def embed_tenant(
    mongo: MongoRepo,
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    *,
    flush_size: int = FLUSH_SIZE,
) -> EmbedRunReport:
    """Embed every selected page of one tenant; earlier flushes stay committed if a later one fails."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if flush_size < 1:
        raise ValueError("flush_size must be at least 1")
    if voyage.dimension != VECTOR_DIMENSIONS:
        raise SchemaError(
            "neo4j",
            f"embedding dimension {voyage.dimension} does not match "
            f"the {VECTOR_DIMENSIONS}d vector index",
        )

    started = time.perf_counter()
    run_log: FilteringBoundLogger = log.bind(tenant_id=tenant_id, stage=STAGE)
    with bound_contextvars(tenant_id=tenant_id, stage=STAGE):
        total = _Tally()
        part = _Tally()
        number = processed = 0
        try:
            check_models(await graph.embedding_models(tenant_id), voyage.model, tenant_id=tenant_id)
            selection = await graph.embedding_selection(tenant_id)
            targets = selection.targets
            unhashed = [target.url for target in targets if target.body_hash is None]
            if unhashed:
                raise DatabaseReadError(
                    "neo4j",
                    f"{len(unhashed)} selected pages have no bodyHash (e.g. {unhashed[0]}); "
                    "run load_graph after prepare_corpus",
                )
            flushes = -(-len(targets) // flush_size)
            run_log.info(
                "embedding.run.start",
                model=voyage.model,
                selected=len(targets),
                up_to_date=selection.up_to_date,
                placeholders=selection.placeholders,
                non_2xx=selection.non_2xx,
                flush_size=flush_size,
                flushes=flushes,
            )
            for number, chunk in enumerate(batched(targets, flush_size), start=1):
                await _flush(
                    mongo,
                    graph,
                    voyage,
                    tenant_id,
                    chunk,
                    number=number,
                    flush_size=flush_size,
                    tally=part,
                    run_log=run_log,
                )
                # The driver keeps the last write's parameters in a reference cycle until GC.
                gc.collect()
                total.add(part)
                processed += len(chunk)
                run_log.info(
                    "embedding.flush",
                    flush=number,
                    flushes=flushes,
                    written=part.embedded,
                    skipped=part.skipped,
                    not_usable=part.not_usable,
                    empty_body=part.empty_body,
                    missing=part.missing,
                    hash_mismatch=part.hash_mismatch,
                    pages_done=total.embedded,
                    remaining=len(targets) - processed,
                    api_tokens=total.api_tokens,
                    elapsed_s=round(time.perf_counter() - started, 3),
                )
                # Merged; on failure only an unfinished flush is added to total.
                part = _Tally()
        except (EmbeddingError, DatabaseError) as error:
            so_far = _Tally()
            so_far.add(total)
            so_far.add(part)
            run_log.error(
                "embedding.run.failed",
                flush=number,
                pages_done=so_far.embedded,
                embedded=so_far.embedded,
                not_usable=so_far.not_usable,
                empty_body=so_far.empty_body,
                missing=so_far.missing,
                hash_mismatch=so_far.hash_mismatch,
                api_tokens=so_far.api_tokens,
                tokens=so_far.tokens,
                elapsed_s=round(time.perf_counter() - started, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        report = EmbedRunReport(
            tenant_id=tenant_id,
            embedding_model=voyage.model,
            dimensions=voyage.dimension,
            selected=len(targets),
            embedded=total.embedded,
            skipped_not_usable=total.not_usable,
            skipped_empty_body=total.empty_body,
            skipped_missing=total.missing,
            skipped_hash_mismatch=total.hash_mismatch,
            up_to_date=selection.up_to_date,
            placeholders=selection.placeholders,
            non_2xx=selection.non_2xx,
            flushes=number,
            api_tokens=total.api_tokens,
            tokens=total.tokens,
            truncated=total.truncated,
            elapsed_s=round(time.perf_counter() - started, 3),
            finished_at=datetime.now(UTC),
        )
        run_log.info("embedding.run.done", **report.model_dump(mode="json", exclude={"tenant_id"}))
    return report


async def _flush(
    mongo: MongoRepo,
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    chunk: Sequence[EmbeddingTarget],
    *,
    number: int,
    flush_size: int,
    tally: _Tally,
    run_log: FilteringBoundLogger,
) -> None:
    """Classify, embed and write one flush into ``tally``; its data is released on return."""
    records = {
        str(record.url): record
        for record in await mongo.get_pages(
            tenant_id, [target.url for target in chunk], batch_size=flush_size
        )
    }
    pages: list[PageText] = []
    hashes: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for target in chunk:
        record = records.get(target.url)
        if record is None:
            missing.append(target.url)
        elif record.usable is False:
            tally.not_usable += 1
        elif not record.body_text.strip():
            tally.empty_body += 1
        else:
            digest = body_hash(record.body_text)
            if digest != target.body_hash:
                mismatched.append(target.url)
            else:
                pages.append(PageText(url=target.url, text=record.body_text))
                hashes.append(digest)
    del records
    tally.missing = len(missing)
    tally.hash_mismatch = len(mismatched)
    if missing:
        run_log.warning(
            "embedding.flush.missing",
            flush=number,
            count=len(missing),
            sample_urls=missing[:SAMPLE_URLS],
        )
    if mismatched:
        run_log.warning(
            "embedding.flush.hash_mismatch",
            flush=number,
            count=len(mismatched),
            sample_urls=mismatched[:SAMPLE_URLS],
            hint="graph bodyHash is stale; run load_graph after prepare_corpus",
        )

    count = len(pages)
    if count + tally.skipped != len(chunk):
        raise AssertionError(
            f"flush {number}: {count} to embed and {tally.skipped} skipped "
            f"do not account for {len(chunk)} targets"
        )
    if not count:
        return

    vectors = np.empty((count, voyage.dimension), dtype=np.float32)
    filled = await _fill(voyage, pages, vectors, tally)
    urls = [page.url for page in pages]
    del pages
    if not filled == count == len(urls) == len(hashes):
        raise EmbeddingResponseError(
            f"flush {number}: {filled} vectors for {count} pages, "
            f"{len(urls)} urls and {len(hashes)} hashes"
        )
    written = await graph.write_embeddings(
        tenant_id, urls, hashes, vectors, model=voyage.model, dimensions=voyage.dimension
    )
    if written != count:
        raise DatabaseWriteError("neo4j", f"flush {number}: wrote {written} of {count} embeddings")
    tally.embedded = written


async def _fill(
    voyage: VoyageClient,
    pages: Sequence[PageText],
    vectors: npt.NDArray[np.float32],
    tally: _Tally,
) -> int:
    """Copy each vector into its row as its request returns; returns the rows filled."""
    filled = 0
    async with aclosing(voyage.iter_embed(pages)) as batches:
        async for batch in batches:
            for embedding in batch.embeddings:
                expected = pages[filled].url if filled < len(pages) else None
                if embedding.url != expected:
                    raise EmbeddingResponseError(
                        f"vector {filled} is for {embedding.url}, expected {expected or 'no more pages'}"
                    )
                vectors[filled] = embedding.vector
                filled += 1
                tally.tokens += embedding.tokens
                tally.truncated += embedding.truncated
            tally.api_tokens += batch.api_tokens
    return filled
