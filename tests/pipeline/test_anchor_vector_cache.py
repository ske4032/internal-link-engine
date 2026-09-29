"""AnchorVectors over the tenant's text-vector caches (#91): files written before the memory
rework are read as stored, never embedded again or rewritten; new rows are appended in the
stored precision, atomically, and another tenant's cache is never touched. Reads and appends
stream, so a large cache is never held in memory whole."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_semantic_anchors import MODEL
from voyage_fakes import FakeVoyage, client

from linking_engine.pipeline.semantic_anchors import AnchorVectors
from linking_engine.pipeline.text_vectors import (
    cache_path,
    cached_text_rows,
    text_key,
    text_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.pipeline.text_vectors import Kind

TENANT = "test-cache-a"
OTHER = "test-cache-b"
STAGE = "anchor-selection"
DIM = 16
KINDS: tuple[Kind, ...] = ("keywords", "sentences", "phrases")
TEXTS: dict[Kind, tuple[str, ...]] = {
    "keywords": ("trail shoes", "rain jackets", "four season tents"),
    "sentences": ("Trail shoes grip wet rock.", "Pack a rain jacket.", "Pitch the tent early."),
    "phrases": ("wet rock", "rain jacket", "tent early"),
}
LATER: dict[Kind, tuple[str, ...]] = {
    "keywords": ("camp stove",),
    "sentences": ("Boil water on the stove.",),
    "phrases": ("camp stove",),
}


def layout(tenant: str, kind: Kind, dimension: int = DIM) -> pa.Schema:
    """The cache file layout before #91: keywords float32, sentences and phrases float16."""
    value = pa.float32() if kind == "keywords" else pa.float16()
    return pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("text_sha256", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.field("element", value), dimension), nullable=False),
        ],
        metadata={"tenant_id": tenant, "dimension": str(dimension)},
    )


def write_cache(cache_dir: Path, kind: Kind, keys: Sequence[str], rows: np.ndarray) -> Path:
    """A cache file of ``kind`` as the code before #91 wrote it: one ``pq.write_table``."""
    schema = layout(TENANT, kind, rows.shape[1])
    table = pa.table(
        {
            "model": pa.array([MODEL] * len(rows), type=pa.string()),
            "text_sha256": pa.array(keys, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(rows.reshape(-1)), type=schema.field("vector").type
            ),
        },
        schema=schema,
    )
    path = cache_path(cache_dir, TENANT, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_existing(cache_dir: Path, kind: Kind) -> dict[str, np.ndarray]:
    """A small cache file of ``kind``, and its rows as stored."""
    dtype = np.float32 if kind == "keywords" else np.float16
    rows = np.random.default_rng(91).normal(size=(len(TEXTS[kind]), DIM)).astype(dtype)
    write_cache(cache_dir, kind, [text_key(text) for text in TEXTS[kind]], rows)
    return dict(zip(TEXTS[kind], rows, strict=True))


def fake() -> FakeVoyage:
    def respond(texts: Sequence[str]) -> list[list[float]]:
        return [
            np.random.default_rng(int(text_key(text)[:16], 16)).normal(size=DIM).tolist()
            for text in texts
        ]

    return FakeVoyage(dimension=DIM, respond=respond)


def vectors(voyage: FakeVoyage | None, cache_dir: Path, tenant: str = TENANT) -> AnchorVectors:
    found = None if voyage is None else client(voyage)
    return AnchorVectors(found, tenant, {}, page_models=[], cache_dir=cache_dir)


def unit(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float64)
    found: np.ndarray = values / np.linalg.norm(values)
    return found


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leftovers(cache_dir: Path) -> list[str]:
    # The tenant's lock file stays by design; anything else is a temp file left behind.
    return sorted(
        p.name
        for p in cache_dir.rglob("*")
        if p.is_file() and p.suffix != ".parquet" and p.name != ".lock"
    )


@pytest.mark.parametrize("kind", KINDS)
async def test_existing_cache_files_read_unchanged(
    tmp_path: Path, kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = write_existing(tmp_path, kind)
    path = cache_path(tmp_path, TENANT, kind)
    before = digest(path)
    voyage = fake()
    online = vectors(voyage, tmp_path)
    await online.ensure(kind, stored)
    # Offline, the cache is read at the tenant's configured model and dimension.
    monkeypatch.setenv("TENANT_EMBEDDING_DIMENSIONS", str(DIM))
    offline = vectors(None, tmp_path)
    await offline.ensure(kind, stored)

    assert voyage.call_count == 0, "a stored vector was embedded again"
    assert (online.counts(kind), offline.counts(kind)) == ((0, len(stored)), (0, len(stored)))
    for found in (online, offline):
        assert found.missing(kind, stored) == 0
        for text, row in stored.items():
            vector = found.vector(kind, text)
            assert vector is not None
            assert vector.dtype == np.float32, f"{kind} rows are computed in float32"
            assert vector.shape == (DIM,), "vectors are never shortened"
            # The stored row's direction, float16 rounding included, not a re-embedded one.
            np.testing.assert_allclose(unit(vector), unit(row), rtol=0, atol=1e-6)
    assert digest(path) == before, "reading rewrote the cache file"
    assert pq.read_schema(path).equals(layout(TENANT, kind), check_metadata=True)


@pytest.mark.parametrize("kind", KINDS)
async def test_text_rows_come_at_the_cache_precision_as_cached_and_embedded_blocks(
    tmp_path: Path, kind: Kind
) -> None:
    stored = write_existing(tmp_path, kind)
    dtype = np.float32 if kind == "keywords" else np.float16
    voyage = fake()
    texts = [*TEXTS[kind], *LATER[kind], TEXTS[kind][0]]

    found = await text_rows(client(voyage), TENANT, kind, texts, cache_dir=tmp_path, stage=STAGE)
    offline = await cached_text_rows(
        TENANT, kind, texts, model=MODEL, dimension=DIM, cache_dir=tmp_path, stage=STAGE
    )

    assert sorted(found.cached.texts) == sorted(TEXTS[kind])
    assert found.embedded.texts == list(LATER[kind])
    assert sorted(offline.texts) == sorted([*TEXTS[kind], *LATER[kind]]), "the append is read"
    for rows in (found.cached, found.embedded, offline):
        assert (rows.matrix.dtype, rows.matrix.shape) == (dtype, (len(rows.texts), DIM))
    for text, row in zip(found.cached.texts, found.cached.matrix, strict=True):
        np.testing.assert_array_equal(row, stored[text])
    respond = fake().respond
    assert respond is not None
    # Voyage's vector, rounded to the cache's precision.
    np.testing.assert_allclose(
        unit(found.embedded.matrix[0]), unit(np.asarray(respond(LATER[kind])[0])), atol=2e-3
    )
    assert voyage.call_count == 1
    assert found.api_tokens > 0


async def test_rows_gather_float32_unit_copies_across_blocks_in_the_order_asked(
    tmp_path: Path,
) -> None:
    stored = write_existing(tmp_path, "phrases")
    found = vectors(fake(), tmp_path)
    # One block read from the cache, one embedded later.
    await found.ensure("phrases", stored)
    await found.ensure("phrases", LATER["phrases"])
    first, second, _ = TEXTS["phrases"]
    later = LATER["phrases"][0]
    texts = [later, second, first, later]

    rows = found.rows("phrases", texts)

    assert (rows.dtype, rows.shape) == (np.float32, (len(texts), DIM))
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, rtol=0, atol=1e-6)
    for row, text in zip(rows, texts, strict=True):
        vector = found.vector("phrases", text)
        assert vector is not None
        np.testing.assert_allclose(row, vector, rtol=0, atol=1e-7)
    np.testing.assert_allclose(unit(rows[1]), unit(stored[second]), rtol=0, atol=1e-6)
    exact = found.exact_rows("phrases", texts)
    assert (exact.dtype, exact.shape) == (np.float64, rows.shape)
    np.testing.assert_allclose(np.linalg.norm(exact, axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(exact, rows, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(exact[0], exact[3])
    rows[:] = 0
    kept = found.vector("phrases", later)
    assert kept is not None
    assert kept.any(), "rows handed out the store itself"
    places = [found.index("phrases", text) for text in (*TEXTS["phrases"], later)]
    assert all(isinstance(place, int) for place in places)
    assert len(set(places)) == len(places), "two texts share a row"
    assert found.index("phrases", "never ensured") is None
    assert found.index("keywords", first) is None, "each kind holds its own rows"
    with pytest.raises(KeyError):
        found.rows("phrases", [first, "never ensured"])


@pytest.mark.parametrize("kind", KINDS)
async def test_cache_append_only_atomic_and_per_tenant(
    tmp_path: Path, kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_existing(tmp_path, kind)
    path = cache_path(tmp_path, TENANT, kind)
    first = pq.read_table(path)
    other_voyage = fake()
    await vectors(other_voyage, tmp_path, OTHER).ensure(kind, TEXTS[kind])
    other_path = cache_path(tmp_path, OTHER, kind)
    other_state = (digest(other_path), other_path.stat().st_mtime_ns)
    voyage = fake()

    grown = vectors(voyage, tmp_path)
    await grown.ensure(kind, TEXTS[kind] + LATER[kind])

    assert other_voyage.call_count > 0, "the other tenant reused this tenant's vectors"
    assert grown.counts(kind) == (len(LATER[kind]), len(TEXTS[kind]))
    assert [text for call in voyage.calls for text in call.texts] == list(LATER[kind])
    table = pq.read_table(path)
    assert table.schema.equals(layout(TENANT, kind), check_metadata=True), "precision as stored"
    assert table.slice(0, first.num_rows).equals(first), "a stored row changed"
    assert table["text_sha256"].to_pylist()[first.num_rows :] == [
        text_key(text) for text in LATER[kind]
    ]
    assert (digest(other_path), other_path.stat().st_mtime_ns) == other_state
    assert leftovers(tmp_path) == []

    # A write that cannot be moved into place leaves the complete previous file, no partial one.
    grown_state = digest(path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", refuse)
        patched.setattr(os, "rename", refuse)
        with contextlib.suppress(OSError):
            await vectors(fake(), tmp_path).ensure(kind, ["an extra text"])

    assert digest(path) == grown_state
    assert pq.read_table(path).equals(table)
    assert leftovers(tmp_path) == []
    assert (digest(other_path), other_path.stat().st_mtime_ns) == other_state


# ── a large cache: reads and appends stream ─────────────────────────────────

# Two phrase caches, one twice the other: 32 and 64 MiB of float16 rows decoded. Peak memory
# is compared between them, not to a fixed figure: a streamed read or append needs about the
# same on both, one that holds the file grows by the rows added.
ROWS, LARGE_DIM = (65_536, 131_072), 256
ADDED = (ROWS[1] - ROWS[0]) * LARGE_DIM * 2
WANTED = ("wet rock", "rain jacket", "tent early")
EMBEDDING = Path(__file__).resolve().parent.parent / "embedding"
# The child's own peak resident memory: bytes on macOS, KiB on Linux.
PEAK = """
import resource, sys
def peak():
    found = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return found if sys.platform == "darwin" else found * 1024
"""
READ = (
    PEAK
    + """
import asyncio
from pathlib import Path
from linking_engine.pipeline.semantic_anchors import AnchorVectors
cache_dir, tenant, *texts = sys.argv[1:]
vectors = AnchorVectors(None, tenant, {}, page_models=[], cache_dir=Path(cache_dir))
before = peak()
asyncio.run(vectors.ensure("phrases", texts))
print(peak() - before, vectors.missing("phrases", texts))
"""
)
APPEND = (
    PEAK
    + """
import asyncio
from pathlib import Path
embedding, cache_dir, tenant, dimension, text = sys.argv[1:]
sys.path.insert(0, embedding)
from voyage_fakes import FakeVoyage, client
from linking_engine.pipeline.semantic_anchors import AnchorVectors
size = int(dimension)
fake = FakeVoyage(dimension=size, respond=lambda texts: [[1.0] * size for _ in texts])
vectors = AnchorVectors(client(fake), tenant, {}, page_models=[], cache_dir=Path(cache_dir))
before = peak()
asyncio.run(vectors.ensure("phrases", [text]))
print(peak() - before, fake.call_count, *vectors.counts("phrases"))
"""
)


@pytest.fixture(scope="module")
def large_caches(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """Per size of ROWS, a phrase cache in one row group, as the code before #91 wrote it,
    holding WANTED at its first, middle and last rows."""
    found = {}
    for size in ROWS:
        cache_dir = tmp_path_factory.mktemp(f"rows-{size}")
        keys = [f"{i:064x}" for i in range(size)]
        for row, text in zip((0, size // 2, size - 1), WANTED, strict=True):
            keys[row] = text_key(text)
        rows = np.random.default_rng(7).standard_normal((size, LARGE_DIM), dtype=np.float32)
        path = write_cache(cache_dir, "phrases", keys, rows.astype(np.float16))
        assert pq.ParquetFile(path).metadata.num_row_groups == 1
        found[size] = cache_dir
    return found


def child(script: str, cwd: Path, *args: str) -> list[int]:
    """Run ``script`` in a fresh interpreter; its printed integers."""
    env = {**os.environ, "TENANT_EMBEDDING_DIMENSIONS": str(LARGE_DIM)}
    done = subprocess.run(  # noqa: S603 - this interpreter, fixed scripts, no shell
        [sys.executable, "-c", script, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return [int(value) for value in done.stdout.split()[-4:] if value.lstrip("-").isdigit()]


def test_reading_a_few_rows_needs_no_more_memory_from_a_cache_twice_as_large(
    large_caches: dict[int, Path], tmp_path: Path
) -> None:
    grown = []
    for size in ROWS:
        peak, missing = child(READ, tmp_path, str(large_caches[size]), TENANT, *WANTED)
        assert missing == 0, f"a wanted row of the {size}-row cache was not read"
        grown.append(peak)

    small, large = grown
    assert large - small < ADDED / 2, (
        f"reading {len(WANTED)} rows grew the peak by {small / 2**20:.1f} MiB from {ROWS[0]} rows "
        f"and {large / 2**20:.1f} MiB from {ROWS[1]}; the added rows decode to "
        f"{ADDED / 2**20:.0f} MiB"
    )


def test_an_append_streams_the_old_rows_and_keeps_them_as_stored(
    large_caches: dict[int, Path], tmp_path: Path
) -> None:
    grown = []
    for size in ROWS:
        cache_dir = tmp_path / f"rows-{size}"
        shutil.copytree(large_caches[size], cache_dir)
        path = cache_path(cache_dir, TENANT, "phrases")
        before = pq.read_table(path)

        peak, calls, embedded, cached = child(
            APPEND, tmp_path, str(EMBEDDING), str(cache_dir), TENANT, str(LARGE_DIM), "camp stove"
        )

        assert (calls, embedded, cached) == (1, 1, 0)
        after = pq.read_table(path)
        assert after.schema.equals(layout(TENANT, "phrases", LARGE_DIM), check_metadata=True)
        assert after.num_rows == size + 1
        assert after.slice(0, size).equals(before), "an old row changed"
        assert after["text_sha256"][size].as_py() == text_key("camp stove")
        assert leftovers(cache_dir) == []
        grown.append(peak)

    small, large = grown
    assert large - small < ADDED / 2, (
        f"appending one row grew the peak by {small / 2**20:.1f} MiB to {ROWS[0]} rows and "
        f"{large / 2**20:.1f} MiB to {ROWS[1]}; the added old rows decode to "
        f"{ADDED / 2**20:.0f} MiB"
    )
