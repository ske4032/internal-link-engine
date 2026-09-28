"""A tenant's text vectors, cached as Parquet so each text is embedded once per model.

One cache per tenant and kind, keyed by the model and the sha256 of the text; the texts
themselves are never stored, and tenants never share a cache, even for identical text. Every
cache only grows, so no text is paid for twice, also when quality evaluation and anchor selection
ask for different keywords. Keywords keep the quality evaluation's float32 file; sentences and
phrases are float16 under ``text_vectors/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import structlog

from linking_engine.models import PageText

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    import numpy.typing as npt

    from linking_engine.embedding.voyage_client import VoyageClient

log = structlog.get_logger(__name__)

Kind = Literal["keywords", "sentences", "phrases"]
# Newly embedded texts added to the cache per write, so an outage keeps what was paid for.
FLUSH_TEXTS: Final = 10_000


@dataclass(frozen=True, slots=True)
class _Layout:
    # Relative to the tenant's cache folder.
    parts: tuple[str, ...]
    dtype: type[np.floating]


_LAYOUTS: Final[Mapping[Kind, _Layout]] = {
    "keywords": _Layout(("keyword_vectors.parquet",), np.float32),
    "sentences": _Layout(("text_vectors", "sentences.parquet"), np.float16),
    "phrases": _Layout(("text_vectors", "phrases.parquet"), np.float16),
}


@dataclass(frozen=True, slots=True)
class TextVectors:
    # Text to its vector as stored: float32 values, float16-rounded for sentences and phrases.
    vectors: dict[str, npt.NDArray[np.float32]]
    embedded: int
    cached: int
    api_tokens: int


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_path(cache_dir: Path, tenant_id: str, kind: Kind) -> Path:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    return cache_dir.joinpath(tenant_id, *_LAYOUTS[kind].parts)


def _schema(tenant_id: str, kind: Kind, dimension: int) -> pa.Schema:
    value = pa.from_numpy_dtype(_LAYOUTS[kind].dtype)
    return pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("text_sha256", pa.string(), nullable=False),
            # Parquet names a list's child "element", so the schema round-trips as written.
            pa.field("vector", pa.list_(pa.field("element", value), dimension), nullable=False),
        ],
        metadata={"tenant_id": tenant_id, "dimension": str(dimension)},
    )


async def text_vectors(
    voyage: VoyageClient,
    tenant_id: str,
    kind: Kind,
    texts: Iterable[str],
    *,
    cache_dir: Path,
    stage: str,
) -> TextVectors:
    """A vector per distinct text from ``voyage``'s model, embedding only the texts the
    tenant's cache of ``kind`` lacks; new vectors are added to the cache every FLUSH_TEXTS
    texts. Embedding errors propagate, and the texts embedded before one stay cached."""
    path = cache_path(cache_dir, tenant_id, kind)
    wanted = {text_key(text): text for text in sorted(set(texts))}
    schema = _schema(tenant_id, kind, voyage.dimension)
    cached, table = await asyncio.to_thread(
        _read, tenant_id, kind, path, schema, voyage.model, wanted.keys(), stage=stage
    )
    missing = [key for key in wanted if key not in cached]
    dtype = _LAYOUTS[kind].dtype
    fresh: dict[str, npt.NDArray[np.float32]] = {}
    pending: dict[str, npt.NDArray[np.float32]] = {}
    api_tokens = 0
    try:
        if missing:
            async for batch in voyage.iter_embed(
                [PageText(url=key, text=wanted[key]) for key in missing]
            ):
                api_tokens += batch.api_tokens
                found = {
                    item.url: np.asarray(item.vector, dtype=dtype).astype(np.float32)
                    for item in batch.embeddings
                }
                fresh.update(found)
                pending.update(found)
                if len(pending) >= FLUSH_TEXTS:
                    table = await asyncio.to_thread(
                        _write, kind, path, schema, voyage.model, pending, table
                    )
                    pending = {}
    finally:
        # Also on an error, so what was paid for is kept.
        if pending:
            await asyncio.to_thread(_write, kind, path, schema, voyage.model, pending, table)
    log.info(
        f"{kind}.vectors",
        stage=stage,
        tenant_id=tenant_id,
        model=voyage.model,
        texts=len(wanted),
        cached=len(cached),
        embedded=len(fresh),
        api_tokens=api_tokens,
    )
    return TextVectors(
        vectors={wanted[key]: vector for key, vector in {**cached, **fresh}.items()},
        embedded=len(fresh),
        cached=len(cached),
        api_tokens=api_tokens,
    )


async def cached_text_vectors(
    tenant_id: str,
    kind: Kind,
    texts: Iterable[str],
    *,
    model: str,
    dimension: int,
    cache_dir: Path,
    stage: str,
) -> dict[str, npt.NDArray[np.float32]]:
    """The vectors of ``model`` the tenant's cache of ``kind`` holds for the texts, embedding
    nothing; a text not cached is left out."""
    path = cache_path(cache_dir, tenant_id, kind)
    wanted = {text_key(text): text for text in set(texts)}
    cached, _ = await asyncio.to_thread(
        _read,
        tenant_id,
        kind,
        path,
        _schema(tenant_id, kind, dimension),
        model,
        wanted.keys(),
        stage=stage,
    )
    return {wanted[key]: vector for key, vector in cached.items()}


def _read(
    tenant_id: str,
    kind: Kind,
    path: Path,
    schema: pa.Schema,
    model: str,
    keys: Collection[str],
    *,
    stage: str,
) -> tuple[dict[str, npt.NDArray[np.float32]], pa.Table | None]:
    """The cached vectors of ``keys`` from ``model``, and the table to append to; nothing when
    the file is missing, belongs to another tenant or dimension, or cannot be read. A wanted
    row that is not a usable vector is left out of both, so it is embedded again."""
    if not path.is_file():
        return {}, None
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        log.warning(
            f"{kind}.vectors_cache_unreadable",
            stage=stage,
            tenant_id=tenant_id,
            path=str(path),
            error=type(error).__name__,
        )
        return {}, None
    if not table.schema.equals(schema, check_metadata=True):
        log.warning(
            f"{kind}.vectors_cache_invalid", stage=stage, tenant_id=tenant_id, path=str(path)
        )
        return {}, None
    wanted = pa.array(list(keys), type=pa.string())
    found = table.filter(
        pc.and_(pc.equal(table["model"], model), pc.is_in(table["text_sha256"], value_set=wanted))
    )
    dimension = schema.field("vector").type.list_size
    matrix = (
        found["vector"]
        .combine_chunks()
        .flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(-1, dimension)
    ).astype(np.float32)
    usable = np.isfinite(matrix).all(axis=1) & (np.linalg.norm(matrix, axis=1) > 0)
    found_keys = found["text_sha256"].to_pylist()
    if not usable.all():
        broken = pa.array([key for key, ok in zip(found_keys, usable, strict=True) if not ok])
        log.warning(
            f"{kind}.vectors_cache_invalid",
            stage=stage,
            tenant_id=tenant_id,
            path=str(path),
            rows=len(broken),
        )
        table = table.filter(
            pc.invert(
                pc.and_(
                    pc.equal(table["model"], model),
                    pc.is_in(table["text_sha256"], value_set=broken),
                )
            )
        )
    return {
        key: vector for key, vector, ok in zip(found_keys, matrix, usable, strict=True) if ok
    }, table


def _write(
    kind: Kind,
    path: Path,
    schema: pa.Schema,
    model: str,
    vectors: Mapping[str, npt.NDArray[np.float32]],
    kept: pa.Table | None,
) -> pa.Table:
    """``vectors`` as rows of ``model``, after the rows of ``kept`` when given; returns the
    table written."""
    keys = sorted(vectors)
    dtype = _LAYOUTS[kind].dtype
    flat = np.concatenate([vectors[key] for key in keys]) if keys else np.empty(0, dtype)
    table = pa.table(
        {
            "model": pa.array([model] * len(keys), type=pa.string()),
            "text_sha256": pa.array(keys, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(flat.astype(dtype)), type=schema.field("vector").type
            ),
        },
        schema=schema,
    )
    if kept is not None:
        table = pa.concat_tables([kept, table])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the cache path and renamed, so a file at that path is always complete.
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temp = Path(name)
    try:
        pq.write_table(table, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return table
