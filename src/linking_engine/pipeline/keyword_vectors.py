"""A tenant's keyword vectors, cached as Parquet so each text is embedded once per model.

The cache is per tenant, at ``<cache_dir>/<tenant>/keyword_vectors.parquet``, keyed by the model
and the sha256 of the text; the texts themselves are never stored. It holds the texts of the
latest run only, and tenants never share it, even for identical text.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import structlog

from linking_engine.ml.quality import QUALITY_STAGE
from linking_engine.models import PageText

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

    from linking_engine.embedding.voyage_client import VoyageClient

log = structlog.get_logger(__name__)

FILE: Final = "keyword_vectors.parquet"


@dataclass(frozen=True, slots=True)
class KeywordVectors:
    # Keyword text to its vector, as Voyage returned it.
    vectors: dict[str, npt.NDArray[np.float32]]
    embedded: int
    cached: int
    api_tokens: int


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _schema(tenant_id: str, dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("text_sha256", pa.string(), nullable=False),
            # Parquet names a list's child "element", so the schema round-trips as written.
            pa.field(
                "vector", pa.list_(pa.field("element", pa.float32()), dimension), nullable=False
            ),
        ],
        metadata={"tenant_id": tenant_id, "dimension": str(dimension)},
    )


async def keyword_vectors(
    voyage: VoyageClient, tenant_id: str, texts: Iterable[str], *, cache_dir: Path
) -> KeywordVectors:
    """A vector per distinct text from ``voyage``'s model, embedding only the texts the
    tenant's cache lacks; the cache is rewritten when anything was embedded."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    wanted = {text_key(text): text for text in sorted(set(texts))}
    path = cache_dir / tenant_id / FILE
    schema = _schema(tenant_id, voyage.dimension)
    cached = await asyncio.to_thread(_read, tenant_id, path, schema, voyage.model, wanted.keys())
    missing = [key for key in wanted if key not in cached]
    fresh: dict[str, npt.NDArray[np.float32]] = {}
    api_tokens = 0
    if missing:
        async for batch in voyage.iter_embed(
            [PageText(url=key, text=wanted[key]) for key in missing]
        ):
            api_tokens += batch.api_tokens
            fresh.update(
                (item.url, np.asarray(item.vector, dtype=np.float32)) for item in batch.embeddings
            )
        await asyncio.to_thread(_write, path, schema, voyage.model, {**cached, **fresh})
    log.info(
        "keywords.vectors",
        stage=QUALITY_STAGE,
        tenant_id=tenant_id,
        model=voyage.model,
        texts=len(wanted),
        cached=len(cached),
        embedded=len(fresh),
        api_tokens=api_tokens,
    )
    return KeywordVectors(
        vectors={wanted[key]: vector for key, vector in {**cached, **fresh}.items()},
        embedded=len(fresh),
        cached=len(cached),
        api_tokens=api_tokens,
    )


def _read(
    tenant_id: str, path: Path, schema: pa.Schema, model: str, keys: Iterable[str]
) -> dict[str, npt.NDArray[np.float32]]:
    """The cached vectors of ``keys`` from ``model``; empty when the file is missing, belongs
    to another tenant or dimension, or cannot be read."""
    if not path.is_file():
        return {}
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        log.warning(
            "keywords.vectors_cache_unreadable",
            stage=QUALITY_STAGE,
            tenant_id=tenant_id,
            path=str(path),
            error=type(error).__name__,
        )
        return {}
    if not table.schema.equals(schema, check_metadata=True):
        log.warning(
            "keywords.vectors_cache_invalid",
            stage=QUALITY_STAGE,
            tenant_id=tenant_id,
            path=str(path),
        )
        return {}
    wanted = pa.array(list(keys), type=pa.string())
    table = table.filter(
        pc.and_(pc.equal(table["model"], model), pc.is_in(table["text_sha256"], value_set=wanted))
    )
    dimension = schema.field("vector").type.list_size
    matrix = (table["vector"].combine_chunks().flatten().to_numpy(zero_copy_only=False)).reshape(
        -1, dimension
    )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or (norms == 0).any():
        log.warning(
            "keywords.vectors_cache_invalid",
            stage=QUALITY_STAGE,
            tenant_id=tenant_id,
            path=str(path),
        )
        return {}
    return dict(zip(table["text_sha256"].to_pylist(), matrix.astype(np.float32), strict=True))


def _write(
    path: Path, schema: pa.Schema, model: str, vectors: Mapping[str, npt.NDArray[np.float32]]
) -> None:
    keys = sorted(vectors)
    flat = np.concatenate([vectors[key] for key in keys]) if keys else np.empty(0, np.float32)
    table = pa.table(
        {
            "model": pa.array([model] * len(keys), type=pa.string()),
            "text_sha256": pa.array(keys, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(flat.astype(np.float32), type=pa.float32()),
                type=schema.field("vector").type,
            ),
        },
        schema=schema,
    )
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
