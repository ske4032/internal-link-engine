"""A tenant's text vectors, cached as Parquet so each text is embedded once per model.

One cache per tenant and kind, keyed by the model and the sha256 of the text; the texts
themselves are never stored, and tenants never share a cache, even for identical text. Every
cache only grows, so no text is paid for twice, also when quality evaluation and anchor selection
ask for different keywords. Keywords keep the quality evaluation's float32 file; sentences and
phrases are float16 under ``text_vectors/``. A cache is read and rewritten in batches, so only
the wanted rows are ever held, never the whole file. Every write holds the tenant's lock and
merges into the file as it is then, so stages writing at once never lose each other's rows.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import tempfile
import weakref
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
    from collections.abc import Iterable, Iterator, Mapping

    import numpy.typing as npt

    from linking_engine.embedding.voyage_client import VoyageClient

log = structlog.get_logger(__name__)

Kind = Literal["keywords", "sentences", "phrases"]
# Newly embedded texts added to the cache per write, so an outage keeps what was paid for.
FLUSH_TEXTS: Final = 10_000
# Rows per batch when a cache is read or rewritten, and the bytes read at a time: with one
# thread and no pre-buffering a file is read page by page, never a whole column chunk at once.
BATCH_ROWS: Final = 1024
_BUFFER_BYTES: Final = 1 << 20
# A tenant's writers queue on its lock file across processes, and a loop's coroutines on an
# asyncio lock first, so none of them holds a thread while it waits.
_LOCK_NAME: Final = ".lock"
_LOOP_LOCKS: Final[
    weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]]
] = weakref.WeakKeyDictionary()


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
class Rows:
    # Texts and their vectors, one row each, at the cache's precision.
    texts: list[str]
    matrix: npt.NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class TextRows:
    cached: Rows
    embedded: Rows
    api_tokens: int


@dataclass(frozen=True, slots=True)
class TextVectors:
    # Text to its vector as stored: float32 values, float16-rounded for sentences and phrases.
    vectors: dict[str, npt.NDArray[np.float32]]
    embedded: int
    cached: int
    api_tokens: int


@dataclass(frozen=True, slots=True)
class _Cached:
    rows: Rows
    keys: frozenset[str]
    # Rows of the model that are no usable vector, dropped at the next write.
    broken: frozenset[str]


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_path(cache_dir: Path, tenant_id: str, kind: Kind) -> Path:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    return cache_dir.joinpath(tenant_id, *_LAYOUTS[kind].parts)


def _lock_path(cache_dir: Path, tenant_id: str) -> Path:
    # One lock for every kind of the tenant's caches.
    return cache_path(cache_dir, tenant_id, "sentences").with_name(_LOCK_NAME)


def _loop_lock(lock: Path) -> asyncio.Lock:
    return _LOOP_LOCKS.setdefault(asyncio.get_running_loop(), {}).setdefault(lock, asyncio.Lock())


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


async def text_rows(
    voyage: VoyageClient,
    tenant_id: str,
    kind: Kind,
    texts: Iterable[str],
    *,
    cache_dir: Path,
    stage: str,
) -> TextRows:
    """A row per distinct text from ``voyage``'s model at the cache's precision, embedding only
    the texts the tenant's cache of ``kind`` lacks; new rows are added to the cache every
    FLUSH_TEXTS texts. Embedding errors propagate, and the texts embedded before one stay
    cached."""
    path = cache_path(cache_dir, tenant_id, kind)
    lock = _lock_path(cache_dir, tenant_id)
    wanted = {text_key(text): text for text in sorted(set(texts))}
    schema = _schema(tenant_id, kind, voyage.dimension)
    cached = await asyncio.to_thread(
        _read, tenant_id, kind, path, schema, voyage.model, wanted, stage=stage
    )
    missing = [key for key in wanted if key not in cached.keys]
    fresh = np.empty((len(missing), voyage.dimension), _LAYOUTS[kind].dtype)
    keys: list[str] = []
    written = 0
    drop = cached.broken
    api_tokens = 0

    async def flush() -> None:
        nonlocal written, drop
        async with _loop_lock(lock):
            await asyncio.to_thread(
                _write,
                lock,
                path,
                schema,
                voyage.model,
                keys[written:],
                fresh[written : len(keys)],
                drop=drop,
            )
        written, drop = len(keys), frozenset()

    try:
        if missing:
            async for batch in voyage.iter_embed(
                [PageText(url=key, text=wanted[key]) for key in missing]
            ):
                api_tokens += batch.api_tokens
                for item in batch.embeddings:
                    fresh[len(keys)] = item.vector
                    keys.append(item.url)
                if len(keys) - written >= FLUSH_TEXTS:
                    await flush()
    finally:
        # Also on an error, so what was paid for is kept.
        if len(keys) > written:
            await flush()
    log.info(
        f"{kind}.vectors",
        stage=stage,
        tenant_id=tenant_id,
        model=voyage.model,
        texts=len(wanted),
        cached=len(cached.rows.texts),
        embedded=len(keys),
        api_tokens=api_tokens,
    )
    return TextRows(
        cached=cached.rows,
        embedded=Rows([wanted[key] for key in keys], fresh[: len(keys)]),
        api_tokens=api_tokens,
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
    found = await text_rows(voyage, tenant_id, kind, texts, cache_dir=cache_dir, stage=stage)
    return TextVectors(
        vectors={**_vectors(found.cached), **_vectors(found.embedded)},
        embedded=len(found.embedded.texts),
        cached=len(found.cached.texts),
        api_tokens=found.api_tokens,
    )


async def cached_text_rows(
    tenant_id: str,
    kind: Kind,
    texts: Iterable[str],
    *,
    model: str,
    dimension: int,
    cache_dir: Path,
    stage: str,
) -> Rows:
    """The rows of ``model`` the tenant's cache of ``kind`` holds for the texts, at the cache's
    precision, embedding nothing; a text not cached is left out."""
    path = cache_path(cache_dir, tenant_id, kind)
    cached = await asyncio.to_thread(
        _read,
        tenant_id,
        kind,
        path,
        _schema(tenant_id, kind, dimension),
        model,
        {text_key(text): text for text in set(texts)},
        stage=stage,
    )
    return cached.rows


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
    return _vectors(
        await cached_text_rows(
            tenant_id,
            kind,
            texts,
            model=model,
            dimension=dimension,
            cache_dir=cache_dir,
            stage=stage,
        )
    )


def _vectors(rows: Rows) -> dict[str, npt.NDArray[np.float32]]:
    return {
        text: np.asarray(row, dtype=np.float32)
        for text, row in zip(rows.texts, rows.matrix, strict=True)
    }


def _read(
    tenant_id: str,
    kind: Kind,
    path: Path,
    schema: pa.Schema,
    model: str,
    wanted: Mapping[str, str],
    *,
    stage: str,
) -> _Cached:
    """The cached rows of the ``wanted`` keys (key to text) from ``model``; none when the file is
    missing, belongs to another tenant or dimension, or cannot be read. A wanted row that is not
    a usable vector is left out and dropped at the next write, so it is embedded again."""
    dimension = schema.field("vector").type.list_size
    nothing = _Cached(
        Rows([], np.empty((0, dimension), _LAYOUTS[kind].dtype)), frozenset(), frozenset()
    )
    if not path.is_file():
        return nothing
    try:
        with _open(path) as file:
            if not file.schema_arrow.equals(schema, check_metadata=True):
                log.warning(
                    f"{kind}.vectors_cache_invalid",
                    stage=stage,
                    tenant_id=tenant_id,
                    path=str(path),
                )
                return nothing
            positions, keys = _positions(file, model, wanted.keys())
            matrix, usable = _gather(file, positions, dimension, _LAYOUTS[kind].dtype)
    except (OSError, pa.ArrowException) as error:
        log.warning(
            f"{kind}.vectors_cache_unreadable",
            stage=stage,
            tenant_id=tenant_id,
            path=str(path),
            error=type(error).__name__,
        )
        return nothing
    broken = frozenset(key for key, ok in zip(keys, usable, strict=True) if not ok)
    if broken:
        log.warning(
            f"{kind}.vectors_cache_invalid",
            stage=stage,
            tenant_id=tenant_id,
            path=str(path),
            rows=len(broken),
        )
    held = [key for key, ok in zip(keys, usable, strict=True) if ok]
    return _Cached(Rows([wanted[key] for key in held], matrix), frozenset(held), broken)


def _open(path: Path) -> pq.ParquetFile:
    return pq.ParquetFile(path, buffer_size=_BUFFER_BYTES, pre_buffer=False)


def _batches(file: pq.ParquetFile, columns: list[str] | None) -> Iterator[pa.RecordBatch]:
    yield from file.iter_batches(batch_size=BATCH_ROWS, columns=columns, use_threads=False)


def _positions(
    file: pq.ParquetFile, model: str, keys: Iterable[str]
) -> tuple[npt.NDArray[np.intp], list[str]]:
    """The file's rows of ``model`` holding one of ``keys``, the first of each key only, and
    their keys; read from the key columns alone."""
    value_set = pa.array(list(keys), type=pa.string())
    positions: list[int] = []
    found: list[str] = []
    seen: set[str] = set()
    start = 0
    for batch in _batches(file, ["model", "text_sha256"]):
        keys_column = batch.column("text_sha256")
        mask = pc.and_(
            pc.equal(batch.column("model"), model), pc.is_in(keys_column, value_set=value_set)
        )
        rows = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        for row, key in zip(rows.tolist(), keys_column.filter(mask).to_pylist(), strict=True):
            if key not in seen:
                seen.add(key)
                positions.append(start + row)
                found.append(key)
        start += batch.num_rows
    return np.asarray(positions, dtype=np.intp), found


def _gather(
    file: pq.ParquetFile,
    positions: npt.NDArray[np.intp],
    dimension: int,
    dtype: type[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """The vectors at the ascending ``positions`` as rows at the file's precision, the usable
    ones only (finite and non-zero), and which of the positions were usable."""
    matrix = np.empty((len(positions), dimension), dtype)
    usable = np.zeros(len(positions), dtype=np.bool_)
    if not len(positions):
        return matrix, usable
    done = filled = start = 0
    for batch in _batches(file, ["vector"]):
        end = start + batch.num_rows
        upto = int(np.searchsorted(positions, end))
        if upto > done:
            values = batch.column("vector").flatten().to_numpy(zero_copy_only=False)
            rows = values.reshape(-1, dimension)[positions[done:upto] - start]
            check = rows.astype(np.float32)
            ok = np.isfinite(check).all(axis=1) & (np.linalg.norm(check, axis=1) > 0)
            usable[done:upto] = ok
            matrix[filled : filled + int(ok.sum())] = rows[ok]
            filled += int(ok.sum())
            done = upto
        start = end
        if done == len(positions):
            break
    return matrix[:filled], usable


def _write(
    lock: Path,
    path: Path,
    schema: pa.Schema,
    model: str,
    keys: list[str],
    matrix: npt.NDArray[np.floating],
    *,
    drop: frozenset[str],
) -> None:
    """``matrix`` as rows of ``model`` keyed by ``keys``, merged under the tenant's ``lock``
    into the file as it is now: its rows stay when it holds ``schema``, less the rows of
    ``model`` whose key is in ``drop``, and a key it already holds for ``model`` is not added
    again. The file is streamed, never loaded."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The lock is released when its file is closed.
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        # Written beside the cache path and renamed, so a file at that path is always complete.
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(handle)
        temp = Path(name)
        try:
            new = pa.array(keys, type=pa.string())
            dropped = pa.array(sorted(drop), type=pa.string())
            present: set[str] = set()
            # Only the model repeats; keys and vectors gain nothing from a dictionary.
            with pq.ParquetWriter(temp, schema, use_dictionary=["model"]) as writer:
                for batch in _current(path, schema):
                    if drop:
                        batch = batch.filter(pc.invert(_keyed(batch, model, dropped)))
                    keyed = _keyed(batch, model, new)
                    present.update(batch.column("text_sha256").filter(keyed).to_pylist())
                    writer.write_batch(batch)
                order = sorted(
                    (i for i, key in enumerate(keys) if key not in present), key=keys.__getitem__
                )
                writer.write_table(
                    pa.table(
                        {
                            "model": pa.array([model] * len(order), type=pa.string()),
                            "text_sha256": pa.array([keys[i] for i in order], type=pa.string()),
                            "vector": pa.FixedSizeListArray.from_arrays(
                                pa.array(matrix[order].reshape(-1)),
                                type=schema.field("vector").type,
                            ),
                        },
                        schema=schema,
                    )
                )
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def _current(path: Path, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """The rows of the file at ``path`` when it holds ``schema``; none when it is missing, holds
    another tenant's or dimension's rows, or cannot be opened. A file that opens but fails while
    it is read raises, naming the file and how to rebuild it."""
    try:
        file = _open(path)
    except (OSError, pa.ArrowException):
        return
    with file:
        if file.schema_arrow.equals(schema, check_metadata=True):
            try:
                yield from _batches(file, None)
            except (OSError, pa.ArrowException) as error:
                raise ValueError(
                    f"vector cache {path} is unreadable ({type(error).__name__}); "
                    f"delete {path} to rebuild it"
                ) from error


def _keyed(batch: pa.RecordBatch, model: str, keys: pa.Array) -> pa.Array:
    """Which rows of ``batch`` are of ``model`` and keyed by one of ``keys``."""
    return pc.and_(
        pc.equal(batch.column("model"), model),
        pc.is_in(batch.column("text_sha256"), value_set=keys),
    )
