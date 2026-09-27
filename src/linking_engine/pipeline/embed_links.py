"""Embed a tenant's distinct anchors and surrounding sentences into Neo4j, one committed flush at a time."""

from __future__ import annotations

import gc
import hashlib
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import batched
from typing import TYPE_CHECKING, Final

import numpy as np
import structlog
from structlog.contextvars import bound_contextvars

from linking_engine.anchor.generic import generic_overrides, is_generic, normalise_anchor
from linking_engine.errors import (
    DatabaseError,
    DatabaseReadError,
    DatabaseWriteError,
    EmbeddingError,
    EmbeddingResponseError,
    SchemaError,
)
from linking_engine.graph.repo import ANCHOR_KEY_MAX_BYTES, VECTOR_DIMENSIONS
from linking_engine.models import (
    AnchorKeyUpdate,
    AnchorRules,
    EdgeRef,
    LinkEmbedReport,
    PageText,
    SentenceTarget,
)
from linking_engine.pipeline.embed import FLUSH_SIZE, check_models

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from structlog.typing import FilteringBoundLogger

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import LinkText

log = structlog.get_logger(__name__)

STAGE: Final = "embedding.links"


def normalise_sentence(text: str) -> str:
    """Collapse whitespace and strip; case is kept because it carries meaning in running text."""
    return " ".join(text.split())


def sentence_hash(text: str) -> str:
    """sha256 hex of the normalised sentence; the edge's surroundingEmbeddedHash."""
    return _digest(normalise_sentence(text))


def _digest(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def _fit_key(key: str) -> str:
    """Cut a key to the Anchor index limit at a character boundary."""
    encoded = key.encode("utf-8")
    if len(encoded) <= ANCHOR_KEY_MAX_BYTES:
        return key
    # A cut through a multi-byte character leaves an invalid tail, which "ignore" drops.
    return encoded[:ANCHOR_KEY_MAX_BYTES].decode("utf-8", errors="ignore").rstrip()


@dataclass(slots=True)
class _Tally:
    keys_written: int = 0
    surrounding_cleared: int = 0
    anchors_embedded: int = 0
    sentences_embedded: int = 0
    sentences_reused: int = 0
    surrounding_edges_written: int = 0
    anchor_flushes: int = 0
    sentence_flushes: int = 0
    api_tokens: int = 0
    tokens: int = 0
    truncated: int = 0


@dataclass(slots=True)
class _Sentence:
    text: str
    # (source_url, position); EdgeRef models are built only for the flush being written.
    edges: list[tuple[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class _Scan:
    edges: int = 0
    empty_anchors: int = 0
    generic_edges: int = 0
    empty_sentences: int = 0
    # Edges whose key was cut to ANCHOR_KEY_MAX_BYTES.
    long_anchors: int = 0
    # Distinct non-empty key -> generic, in first-seen order.
    keys: dict[str, bool] = field(default_factory=dict)
    hashes: set[str] = field(default_factory=set)
    # Only hashes with at least one edge lacking this hash and model.
    pending: dict[str, _Sentence] = field(default_factory=dict)
    # The current page's writes; drained after every iter_link_texts page.
    key_updates: list[AnchorKeyUpdate] = field(default_factory=list)
    stale: list[EdgeRef] = field(default_factory=list)
    # Normalised tenant overrides of the generic dictionary.
    generic_add: frozenset[str] = frozenset()
    generic_remove: frozenset[str] = frozenset()

    def add(self, link: LinkText, model: str) -> None:
        """Account for one edge, queueing its key update or stale-vector removal for this page."""
        self.edges += 1
        self._add_sentence(link, model)
        key = normalise_anchor(link.anchor_text)
        if key:
            fitted = _fit_key(key)
            if fitted != key:
                self.long_anchors += 1
            key = fitted
            generic = self.keys.get(key)
            if generic is None:
                # Keys cut to the same prefix share the first edge's flag; generic phrases are short.
                generic = is_generic(
                    link.anchor_text, add=self.generic_add, remove=self.generic_remove
                )
                self.keys[key] = generic
            if generic:
                self.generic_edges += 1
        else:
            generic = False
            self.empty_anchors += 1
        if (link.anchor_key, link.anchor_generic) != (key or None, generic):
            self.key_updates.append(
                AnchorKeyUpdate(
                    source_url=link.source_url,
                    position=link.position,
                    anchor_key=key or None,
                    anchor_generic=generic,
                )
            )

    def _add_sentence(self, link: LinkText, model: str) -> None:
        sentence = normalise_sentence(link.surrounding_text)
        if not sentence:
            self.empty_sentences += 1
            if link.surrounding_embedded_hash is not None:
                self.stale.append(EdgeRef(source_url=link.source_url, position=link.position))
            return
        digest = _digest(sentence)
        self.hashes.add(digest)
        if link.surrounding_embedded_hash == digest and link.surrounding_embedding_model == model:
            return
        pending = self.pending.get(digest)
        if pending is None:
            pending = self.pending[digest] = _Sentence(sentence)
        pending.edges.append((link.source_url, link.position))


async def embed_links(
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    *,
    flush_size: int = FLUSH_SIZE,
    rules: AnchorRules | None = None,
) -> LinkEmbedReport:
    """Embed a tenant's new anchors and changed sentences; earlier flushes stay committed on failure."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if flush_size < 1:
        raise ValueError("flush_size must be at least 1")
    if voyage.dimension != VECTOR_DIMENSIONS:
        raise SchemaError(
            "neo4j",
            f"embedding dimension {voyage.dimension} does not match "
            f"the {VECTOR_DIMENSIONS}d link vectors",
        )

    started = time.perf_counter()
    run_log: FilteringBoundLogger = log.bind(tenant_id=tenant_id, stage=STAGE)
    with bound_contextvars(tenant_id=tenant_id, stage=STAGE):
        tally = _Tally()
        step = "anchor_models"
        number = 0
        try:
            check_models(
                await graph.anchor_embedding_models(tenant_id), voyage.model, tenant_id=tenant_id
            )
            step = "sentence_models"
            check_models(
                await graph.surrounding_embedding_models(tenant_id),
                voyage.model,
                tenant_id=tenant_id,
            )

            step = "scan"
            scan = await _scan(graph, voyage, tenant_id, tally, rules or AnchorRules())
            step = "anchor"
            embeddable = [key for key, generic in scan.keys.items() if not generic]
            pending_keys = await _keys_to_embed(graph, voyage, tenant_id, embeddable)
            step = "sentence_reuse"
            reusable = await graph.surrounding_vectors(
                tenant_id, list(scan.pending), model=voyage.model
            )
            reused = [(digest, scan.pending.pop(digest)) for digest in list(reusable)]
            step = "anchor"
            anchor_flushes = -(-len(pending_keys) // flush_size)
            sentence_flushes = -(-len(scan.pending) // flush_size)
            run_log.info(
                "embedding.links.start",
                model=voyage.model,
                edges=scan.edges,
                keys_written=tally.keys_written,
                empty_anchors=scan.empty_anchors,
                unique_anchors=len(scan.keys),
                generic_anchors=len(scan.keys) - len(embeddable),
                anchors_pending=len(pending_keys),
                empty_sentences=scan.empty_sentences,
                surrounding_cleared=tally.surrounding_cleared,
                unique_sentences=len(scan.hashes),
                sentences_reusable=len(reused),
                sentences_pending=len(scan.pending),
                flush_size=flush_size,
                anchor_flushes=anchor_flushes,
                sentence_flushes=sentence_flushes,
            )
            if scan.long_anchors:
                run_log.warning(
                    "embedding.links.long_anchors",
                    edges=scan.long_anchors,
                    max_bytes=ANCHOR_KEY_MAX_BYTES,
                    hint="anchor keys cut at a character boundary to fit the Anchor index",
                )

            for number, keys in enumerate(batched(pending_keys, flush_size), start=1):
                await _flush_anchors(graph, voyage, tenant_id, keys, number=number, tally=tally)
                # The driver keeps the last write's parameters in a reference cycle until GC.
                gc.collect()
                tally.anchor_flushes = number
                run_log.info(
                    "embedding.links.flush",
                    kind="anchor",
                    flush=number,
                    flushes=anchor_flushes,
                    written=len(keys),
                    api_tokens=tally.api_tokens,
                    elapsed_s=round(time.perf_counter() - started, 3),
                )

            for number, items in enumerate(batched(reused, flush_size), start=1):
                await _write_reused(
                    graph, voyage, tenant_id, items, reusable, number=number, tally=tally
                )
                gc.collect()
            del reusable, reused

            step = "sentence"
            number = 0
            for number, items in enumerate(batched(scan.pending.items(), flush_size), start=1):
                edges_written = await _flush_sentences(
                    graph, voyage, tenant_id, items, number=number, tally=tally
                )
                gc.collect()
                tally.sentence_flushes = number
                run_log.info(
                    "embedding.links.flush",
                    kind="sentence",
                    flush=number,
                    flushes=sentence_flushes,
                    written=len(items),
                    edges_written=edges_written,
                    api_tokens=tally.api_tokens,
                    elapsed_s=round(time.perf_counter() - started, 3),
                )
        except (EmbeddingError, DatabaseError) as error:
            run_log.error(
                "embedding.links.failed",
                step=step,
                flush=number,
                keys_written=tally.keys_written,
                surrounding_cleared=tally.surrounding_cleared,
                anchors_embedded=tally.anchors_embedded,
                sentences_embedded=tally.sentences_embedded,
                sentences_reused=tally.sentences_reused,
                surrounding_edges_written=tally.surrounding_edges_written,
                api_tokens=tally.api_tokens,
                tokens=tally.tokens,
                elapsed_s=round(time.perf_counter() - started, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

        unique_anchors = len(scan.keys)
        unique_sentences = len(scan.hashes)
        report = LinkEmbedReport(
            tenant_id=tenant_id,
            embedding_model=voyage.model,
            dimensions=voyage.dimension,
            edges=scan.edges,
            keys_written=tally.keys_written,
            empty_anchors=scan.empty_anchors,
            unique_anchors=unique_anchors,
            anchor_dedupe_ratio=(
                (scan.edges - scan.empty_anchors) / unique_anchors if unique_anchors else None
            ),
            generic_anchors=unique_anchors - len(embeddable),
            generic_edges=scan.generic_edges,
            anchors_cached=len(embeddable) - len(pending_keys),
            anchors_embedded=tally.anchors_embedded,
            empty_sentences=scan.empty_sentences,
            surrounding_cleared=tally.surrounding_cleared,
            unique_sentences=unique_sentences,
            sentence_dedupe_ratio=(
                (scan.edges - scan.empty_sentences) / unique_sentences if unique_sentences else None
            ),
            sentences_cached=unique_sentences - len(scan.pending) - tally.sentences_reused,
            sentences_reused=tally.sentences_reused,
            sentences_embedded=tally.sentences_embedded,
            surrounding_edges_written=tally.surrounding_edges_written,
            anchor_flushes=tally.anchor_flushes,
            sentence_flushes=tally.sentence_flushes,
            api_tokens=tally.api_tokens,
            tokens=tally.tokens,
            truncated=tally.truncated,
            elapsed_s=round(time.perf_counter() - started, 3),
            finished_at=datetime.now(UTC),
        )
        run_log.info(
            "embedding.links.done", **report.model_dump(mode="json", exclude={"tenant_id"})
        )
    return report


async def _scan(
    graph: GraphRepo, voyage: VoyageClient, tenant_id: str, tally: _Tally, rules: AnchorRules
) -> _Scan:
    """Read every edge once; changed keys and stale vectors are written page by page."""
    add, remove = generic_overrides(rules.generic_add, rules.generic_remove)
    scan = _Scan(generic_add=add, generic_remove=remove)
    # Neither write touches (url, position), so the keyset paging stays exact.
    async with aclosing(graph.iter_link_texts(tenant_id)) as pages:
        async for links in pages:
            for link in links:
                scan.add(link, voyage.model)
            if scan.key_updates:
                written = await graph.set_anchor_keys(tenant_id, scan.key_updates)
                if written != len(scan.key_updates):
                    raise DatabaseWriteError(
                        "neo4j", f"wrote {written} of {len(scan.key_updates)} anchor keys"
                    )
                tally.keys_written += written
                scan.key_updates.clear()
            if scan.stale:
                cleared = await graph.clear_surrounding_embeddings(tenant_id, scan.stale)
                if cleared != len(scan.stale):
                    raise DatabaseWriteError(
                        "neo4j", f"cleared {cleared} of {len(scan.stale)} stale surrounding vectors"
                    )
                tally.surrounding_cleared += cleared
                scan.stale.clear()
    return scan


async def _keys_to_embed(
    graph: GraphRepo, voyage: VoyageClient, tenant_id: str, keys: Sequence[str]
) -> tuple[str, ...]:
    """Keys without a vector from the configured model, checked to be a distinct subset of ``keys``."""
    if not keys:
        return ()
    pending = await graph.anchor_keys_to_embed(tenant_id, keys, model=voyage.model)
    if len(set(pending)) != len(pending) or not set(pending) <= set(keys):
        raise DatabaseReadError(
            "neo4j", f"{len(pending)} anchor keys to embed are not a distinct subset of {len(keys)}"
        )
    return pending


async def _flush_anchors(
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    keys: Sequence[str],
    *,
    number: int,
    tally: _Tally,
) -> None:
    """Embed and write one flush of anchor keys; its buffer is released on return."""
    # PageText.url is only the id Voyage echoes back per vector; the key fills it.
    texts = [PageText(url=key, text=key) for key in keys]
    vectors = np.empty((len(texts), voyage.dimension), dtype=np.float32)
    filled = await _fill(voyage, texts, vectors, tally)
    del texts
    if filled != len(keys):
        raise EmbeddingResponseError(
            f"anchor flush {number}: {filled} vectors for {len(keys)} keys"
        )
    written = await graph.write_anchor_embeddings(
        tenant_id, keys, vectors, model=voyage.model, dimensions=voyage.dimension
    )
    if written != len(keys):
        raise DatabaseWriteError(
            "neo4j", f"anchor flush {number}: wrote {written} of {len(keys)} anchors"
        )
    tally.anchors_embedded += written


async def _flush_sentences(
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    items: Sequence[tuple[str, _Sentence]],
    *,
    number: int,
    tally: _Tally,
) -> int:
    """Embed one flush of unique sentences and write each vector to all of its pending edges."""
    # PageText.url is only the id Voyage echoes back per vector; the sentence hash fills it.
    texts = [PageText(url=digest, text=sentence.text) for digest, sentence in items]
    targets = [
        SentenceTarget(
            sentence_hash=digest,
            edges=tuple(EdgeRef(source_url=url, position=pos) for url, pos in sentence.edges),
        )
        for digest, sentence in items
    ]
    expected = sum(len(target.edges) for target in targets)
    vectors = np.empty((len(texts), voyage.dimension), dtype=np.float32)
    filled = await _fill(voyage, texts, vectors, tally)
    del texts
    if filled != len(targets):
        raise EmbeddingResponseError(
            f"sentence flush {number}: {filled} vectors for {len(targets)} sentences"
        )
    written = await graph.write_surrounding_embeddings(
        tenant_id, targets, vectors, model=voyage.model, dimensions=voyage.dimension
    )
    if written != expected:
        raise DatabaseWriteError(
            "neo4j", f"sentence flush {number}: wrote {written} of {expected} edges"
        )
    tally.sentences_embedded += len(targets)
    tally.surrounding_edges_written += written
    return written


async def _write_reused(
    graph: GraphRepo,
    voyage: VoyageClient,
    tenant_id: str,
    items: Sequence[tuple[str, _Sentence]],
    found: dict[str, npt.NDArray[np.float32]],
    *,
    number: int,
    tally: _Tally,
) -> None:
    """Write already-paid-for vectors of this tenant to edges that lack them; no Voyage call."""
    targets = [
        SentenceTarget(
            sentence_hash=digest,
            edges=tuple(EdgeRef(source_url=url, position=pos) for url, pos in sentence.edges),
        )
        for digest, sentence in items
    ]
    expected = sum(len(target.edges) for target in targets)
    vectors = np.stack([found[digest] for digest, _ in items]).astype(np.float32, copy=False)
    written = await graph.write_surrounding_embeddings(
        tenant_id, targets, vectors, model=voyage.model, dimensions=voyage.dimension
    )
    if written != expected:
        raise DatabaseWriteError(
            "neo4j", f"reuse flush {number}: wrote {written} of {expected} edges"
        )
    tally.sentences_reused += len(targets)
    tally.surrounding_edges_written += written


async def _fill(
    voyage: VoyageClient,
    texts: Sequence[PageText],
    vectors: npt.NDArray[np.float32],
    tally: _Tally,
) -> int:
    """Copy each vector into its row as its request returns; returns the rows filled."""
    filled = 0
    async with aclosing(voyage.iter_embed(texts)) as batches:
        async for batch in batches:
            for embedding in batch.embeddings:
                expected = texts[filled].url if filled < len(texts) else None
                if embedding.url != expected:
                    raise EmbeddingResponseError(
                        f"vector {filled} is for {embedding.url}, expected {expected or 'no more texts'}"
                    )
                vectors[filled] = embedding.vector
                filled += 1
                tally.tokens += embedding.tokens
                tally.truncated += embedding.truncated
            tally.api_tokens += batch.api_tokens
    return filled
