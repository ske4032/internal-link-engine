"""The per-tenant text-vector caches: keywords in #74's file, sentences and phrases under
``text_vectors/``. Only new texts are embedded, texts are never stored, and no tenant ever reads
another's vectors, even for identical text."""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs
from voyage_fakes import DIMENSION, FakeResult, FakeVoyage, client, settings
from voyageai.error import ServiceUnavailableError

from linking_engine.embedding.voyage_client import VoyageClient
from linking_engine.errors import EmbeddingUnavailableError
from linking_engine.pipeline import text_vectors as text_vectors_module
from linking_engine.pipeline.text_vectors import (
    cache_path,
    cached_text_vectors,
    text_key,
    text_vectors,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from linking_engine.pipeline.text_vectors import Kind, TextVectors

TENANT = "test-kw-a"
OTHER = "test-kw-b"
STAGE = "quality-eval"
TEXTS = ("trail shoes", "rain jackets", "four season tents")
KINDS: tuple[Kind, ...] = ("keywords", "sentences", "phrases")
# #74's keyword file stays where it was; sentences and phrases get their own folder.
PATHS = {
    "keywords": ("keyword_vectors.parquet",),
    "sentences": ("text_vectors", "sentences.parquet"),
    "phrases": ("text_vectors", "phrases.parquet"),
}


def spike(text: str) -> list[float]:
    """A vector per text: a spike at a position derived from the text, so each is distinct."""
    vector = [1.0] * DIMENSION
    vector[sum(map(ord, text)) % DIMENSION] = 40.0 + len(text) / 7
    return vector


def fake(dimension: int = DIMENSION) -> FakeVoyage:
    def respond(texts: Sequence[str]) -> list[list[float]]:
        return [spike(text)[:dimension] for text in texts]

    return FakeVoyage(dimension=dimension, respond=respond)


def entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def cache_files(directory: Path) -> list[str]:
    """The files beside a cache, less the tenant's lock file, which stays by design."""
    return sorted(p.name for p in directory.iterdir() if p.is_file() and p.name != ".lock")


def embedded_texts(voyage: FakeVoyage) -> list[str]:
    return sorted(text for call in voyage.calls for text in call.texts)


def unit(text: str) -> np.ndarray:
    vector = np.asarray(spike(text), dtype=np.float64)
    return vector / np.linalg.norm(vector)


async def embed(
    voyage: FakeVoyage | VoyageClient,
    texts: Sequence[str],
    cache: Path,
    *,
    kind: Kind = "keywords",
    tenant: str = TENANT,
) -> TextVectors:
    found = voyage if isinstance(voyage, VoyageClient) else client(voyage)
    return await text_vectors(found, tenant, kind, texts, cache_dir=cache, stage=STAGE)


def stored(path: Path) -> pa.Table:
    return pq.read_table(path)


# ── every kind ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_has_its_own_file_in_the_tenants_folder(tmp_path: Path, kind: Kind) -> None:
    assert cache_path(tmp_path, TENANT, kind) == tmp_path.joinpath(TENANT, *PATHS[kind])


@pytest.mark.parametrize("kind", KINDS)
async def test_the_first_run_embeds_every_distinct_text_and_stores_no_text(
    tmp_path: Path, kind: Kind
) -> None:
    voyage = fake()

    found = await embed(voyage, [*TEXTS, TEXTS[0]], tmp_path, kind=kind)

    assert (found.embedded, found.cached) == (3, 0)
    assert found.api_tokens > 0
    assert embedded_texts(voyage) == sorted(TEXTS), "a repeated text is embedded once"
    assert sorted(found.vectors) == sorted(TEXTS)
    for text in TEXTS:
        assert found.vectors[text] == pytest.approx(unit(text), abs=1e-3)
    path = cache_path(tmp_path, TENANT, kind)
    table = stored(path)
    assert table.column_names == ["model", "text_sha256", "vector"]
    assert sorted(table["text_sha256"].to_pylist()) == sorted(map(text_key, TEXTS))
    assert set(table["model"].to_pylist()) == {"voyage-4-large"}
    assert table.schema.metadata[b"tenant_id"] == TENANT.encode()
    assert [text for text in TEXTS if text.encode() in path.read_bytes()] == []


@pytest.mark.parametrize(
    ("kind", "dtype"),
    [("keywords", pa.float32()), ("sentences", pa.float16()), ("phrases", pa.float16())],
)
async def test_keywords_are_stored_as_float32_sentences_and_phrases_as_float16(
    tmp_path: Path, kind: Kind, dtype: pa.DataType
) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind)

    vector = stored(cache_path(tmp_path, TENANT, kind)).schema.field("vector").type
    assert (vector.value_type, vector.list_size) == (dtype, DIMENSION)


@pytest.mark.parametrize("kind", KINDS)
async def test_a_second_run_embeds_nothing_returns_the_same_vectors_and_leaves_the_file(
    tmp_path: Path, kind: Kind
) -> None:
    first = await embed(fake(), TEXTS, tmp_path, kind=kind)
    path = cache_path(tmp_path, TENANT, kind)
    written = path.stat().st_mtime_ns
    voyage = fake()

    again = await embed(voyage, TEXTS, tmp_path, kind=kind)

    assert (again.embedded, again.cached, again.api_tokens) == (0, 3, 0)
    assert voyage.call_count == 0
    assert path.stat().st_mtime_ns == written, "a full cache hit must not rewrite the file"
    for text in TEXTS:
        # Fresh float16 vectors come back rounded as stored, so a cached run equals a fresh one.
        np.testing.assert_array_equal(again.vectors[text], first.vectors[text])


@pytest.mark.parametrize("kind", KINDS)
async def test_another_tenant_with_identical_texts_embeds_its_own(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind)
    voyage = fake()

    theirs = await embed(voyage, TEXTS, tmp_path, kind=kind, tenant=OTHER)

    assert (theirs.embedded, theirs.cached) == (3, 0), "one tenant reused another's vectors"
    assert embedded_texts(voyage) == sorted(TEXTS)
    assert entries(tmp_path) == sorted([TENANT, OTHER])
    assert stored(cache_path(tmp_path, OTHER, kind)).schema.metadata[b"tenant_id"] == b"test-kw-b"


@pytest.mark.parametrize("kind", KINDS)
async def test_another_tenants_file_in_this_tenants_place_is_ignored_and_rebuilt(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind, tenant=OTHER)
    mine = cache_path(tmp_path, TENANT, kind)
    mine.parent.mkdir(parents=True)
    shutil.copy(cache_path(tmp_path, OTHER, kind), mine)

    with capture_logs() as logs:
        found = await embed(fake(), TEXTS, tmp_path, kind=kind)

    assert (found.embedded, found.cached) == (3, 0)
    [warning] = [e for e in logs if e["event"] == f"{kind}.vectors_cache_invalid"]
    assert (warning["stage"], warning["tenant_id"]) == (STAGE, TENANT)
    assert stored(mine).schema.metadata[b"tenant_id"] == TENANT.encode()


@pytest.mark.parametrize("kind", KINDS)
async def test_an_unreadable_file_is_rebuilt(tmp_path: Path, kind: Kind) -> None:
    path = cache_path(tmp_path, TENANT, kind)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")

    with capture_logs() as logs:
        found = await embed(fake(), TEXTS, tmp_path, kind=kind)

    assert (found.embedded, found.cached) == (3, 0)
    [warning] = [e for e in logs if e["event"] == f"{kind}.vectors_cache_unreadable"]
    assert (warning["stage"], warning["tenant_id"]) == (STAGE, TENANT)
    assert pq.ParquetFile(path).metadata.num_rows == 3
    assert cache_files(path.parent) == [path.name], "temp file left"


@pytest.mark.parametrize("kind", KINDS)
async def test_vectors_of_another_model_are_not_reused(tmp_path: Path, kind: Kind) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind)
    voyage = fake()
    other_model = VoyageClient(settings(), model="voyage-3-large", dimension=DIMENSION, sdk=voyage)

    found = await embed(other_model, TEXTS, tmp_path, kind=kind)

    assert (found.embedded, found.cached) == (3, 0)
    assert {call.model for call in voyage.calls} == {"voyage-3-large"}


@pytest.mark.parametrize("kind", KINDS)
async def test_a_cache_of_another_dimension_is_ignored(tmp_path: Path, kind: Kind) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind)

    with capture_logs() as logs:
        found = await embed(fake(8), TEXTS, tmp_path, kind=kind)

    assert (found.embedded, found.cached) == (3, 0)
    assert {len(vector) for vector in found.vectors.values()} == {8}
    assert f"{kind}.vectors_cache_invalid" in [entry["event"] for entry in logs]


@pytest.mark.parametrize("kind", KINDS)
async def test_an_outage_propagates_and_leaves_the_cache_as_it_was(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEXTS[:2], tmp_path, kind=kind)
    path = cache_path(tmp_path, TENANT, kind)
    before = path.read_bytes()
    down = fake()
    down.failures = [ServiceUnavailableError("down", http_status=503)]

    with pytest.raises(EmbeddingUnavailableError):
        await embed(client(down, settings(max_attempts=1)), TEXTS, tmp_path, kind=kind)

    assert path.read_bytes() == before
    assert cache_files(path.parent) == [path.name]


@pytest.mark.parametrize("kind", KINDS)
async def test_no_texts_embed_nothing_and_write_nothing(tmp_path: Path, kind: Kind) -> None:
    voyage = fake()

    found = await embed(voyage, [], tmp_path, kind=kind)

    assert (found.vectors, found.embedded, found.cached) == ({}, 0, 0)
    assert voyage.call_count == 0
    assert entries(tmp_path) == []


@pytest.mark.parametrize("kind", KINDS)
async def test_the_log_line_names_counts_never_texts(tmp_path: Path, kind: Kind) -> None:
    with capture_logs() as logs:
        await embed(fake(), TEXTS, tmp_path, kind=kind)

    [line] = [entry for entry in logs if entry["event"] == f"{kind}.vectors"]
    assert (line["stage"], line["tenant_id"], line["texts"], line["embedded"]) == (
        STAGE,
        TENANT,
        3,
        3,
    )
    logged = " ".join(str(value) for value in line.values())
    assert [text for text in TEXTS if text in logged] == []


@pytest.mark.parametrize("tenant", [" ", "..", "../escape", "a/b"])
@pytest.mark.parametrize("kind", KINDS)
async def test_a_tenant_that_is_not_a_directory_name_is_refused(
    tmp_path: Path, tenant: str, kind: Kind
) -> None:
    voyage = fake()
    with pytest.raises(ValueError, match="tenant_id"):
        await embed(voyage, TEXTS, tmp_path, kind=kind, tenant=tenant)
    assert voyage.call_count == 0
    assert entries(tmp_path) == []


# ── what each kind keeps ────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
async def test_every_cache_only_grows_so_no_text_is_paid_for_twice(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEXTS, tmp_path, kind=kind)
    await embed(fake(), ["camp stoves"], tmp_path, kind=kind)
    voyage = fake()

    again = await embed(voyage, [*TEXTS, "camp stoves"], tmp_path, kind=kind)

    assert voyage.call_count == 0
    assert (again.embedded, again.cached) == (0, 4)
    kept = stored(cache_path(tmp_path, TENANT, kind))["text_sha256"].to_pylist()
    assert sorted(kept) == sorted(map(text_key, [*TEXTS, "camp stoves"])), "a row was lost"
    assert len(kept) == len(set(kept)), "a text was stored twice"


async def test_rows_of_another_model_stay_beside_the_new_ones(tmp_path: Path) -> None:
    await embed(fake(), TEXTS, tmp_path)
    other_model = VoyageClient(settings(), model="voyage-3-large", dimension=DIMENSION, sdk=fake())
    await embed(other_model, TEXTS[:1], tmp_path)
    voyage = fake()

    again = await embed(voyage, TEXTS, tmp_path)

    assert voyage.call_count == 0, "switching models evicted the first model's vectors"
    assert again.cached == 3
    models = stored(cache_path(tmp_path, TENANT, "keywords"))["model"].to_pylist()
    assert sorted(models) == ["voyage-3-large", *(["voyage-4-large"] * 3)]


async def test_the_quality_eval_and_anchor_selection_never_evict_each_others_keywords(
    tmp_path: Path,
) -> None:
    """The two stages share the tenant's keyword file with different keyword lists."""
    evaluated = ["trail shoes", "rain jackets", "four season tents"]
    selected = ["rain jackets", "camp stoves", "hiking boots"]
    runs: list[tuple[str, int, int]] = []
    for stage, keywords in (
        ("quality-eval", evaluated),
        ("anchor-selection", selected),
        ("quality-eval", evaluated),
        ("anchor-selection", selected),
    ):
        voyage = fake()
        found = await text_vectors(
            client(voyage), TENANT, "keywords", keywords, cache_dir=tmp_path, stage=stage
        )
        runs.append((stage, found.embedded, voyage.call_count))

    assert runs == [
        ("quality-eval", 3, 1),
        ("anchor-selection", 2, 1),
        ("quality-eval", 0, 0),
        ("anchor-selection", 0, 0),
    ]


# ── reading without Voyage ──────────────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
async def test_the_read_only_lookup_returns_only_cached_texts_and_embeds_nothing(
    tmp_path: Path, kind: Kind
) -> None:
    first = await embed(fake(), TEXTS[:2], tmp_path, kind=kind)
    path = cache_path(tmp_path, TENANT, kind)
    written = path.stat().st_mtime_ns

    found = await cached_text_vectors(
        TENANT,
        kind,
        TEXTS,
        model="voyage-4-large",
        dimension=DIMENSION,
        cache_dir=tmp_path,
        stage=STAGE,
    )

    assert sorted(found) == sorted(TEXTS[:2])
    for text in TEXTS[:2]:
        np.testing.assert_array_equal(found[text], first.vectors[text])
    assert path.stat().st_mtime_ns == written
    other = await cached_text_vectors(
        TENANT,
        kind,
        TEXTS,
        model="voyage-3-large",
        dimension=DIMENSION,
        cache_dir=tmp_path,
        stage=STAGE,
    )
    assert other == {}, "vectors of another model were returned"


async def test_the_read_only_lookup_of_a_missing_cache_is_empty(tmp_path: Path) -> None:
    found = await cached_text_vectors(
        TENANT,
        "phrases",
        TEXTS,
        model="voyage-4-large",
        dimension=DIMENSION,
        cache_dir=tmp_path,
        stage=STAGE,
    )

    assert found == {}
    assert entries(tmp_path) == []


# ── partial writes ──────────────────────────────────────────────────────────

TEN = [f"camp item {i}" for i in range(10)]


@pytest.mark.parametrize("kind", KINDS)
async def test_texts_embedded_before_an_outage_stay_cached(
    tmp_path: Path, kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_vectors_module, "FLUSH_TEXTS", 2)
    down = fake()
    # Ten texts are three requests of 4, 4 and 2; the third fails after its one attempt.
    down.fail_on = {3: ServiceUnavailableError("down", http_status=503)}

    with pytest.raises(EmbeddingUnavailableError):
        await embed(client(down, settings(max_attempts=1)), TEN, tmp_path, kind=kind)

    path = cache_path(tmp_path, TENANT, kind)
    assert stored(path).num_rows == 8, "the first two requests were paid for and kept"
    voyage = fake()
    again = await embed(voyage, TEN, tmp_path, kind=kind)
    assert (again.embedded, again.cached) == (2, 8)
    assert stored(path).num_rows == 10


@pytest.mark.parametrize("kind", KINDS)
async def test_a_broken_row_is_embedded_again_and_the_others_stay(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEN, tmp_path, kind=kind)
    path = cache_path(tmp_path, TENANT, kind)
    table = stored(path)
    broken = table["text_sha256"].to_pylist().index(text_key(TEN[0]))
    vectors = table["vector"].to_pylist()
    vectors[broken] = [float("nan")] * DIMENSION
    patched = table.set_column(
        2, table.schema.field("vector"), pa.array(vectors, type=table.schema.field("vector").type)
    )
    pq.write_table(patched.replace_schema_metadata(table.schema.metadata), path)
    voyage = fake()

    with capture_logs() as logs:
        found = await embed(voyage, [TEN[0], TEN[1]], tmp_path, kind=kind)

    assert embedded_texts(voyage) == [TEN[0]]
    assert (found.embedded, found.cached) == (1, 1)
    assert np.isfinite(found.vectors[TEN[0]]).all()
    [warning] = [e for e in logs if e["event"] == f"{kind}.vectors_cache_invalid"]
    assert warning["rows"] == 1
    kept = stored(path)
    assert kept.num_rows == 10, "the other rows stay and the broken one is replaced"
    assert sorted(kept["text_sha256"].to_pylist()) == sorted(map(text_key, TEN))


@pytest.mark.parametrize("kind", KINDS)
async def test_a_file_failing_while_it_is_copied_names_itself_and_how_to_rebuild_it(
    tmp_path: Path, kind: Kind
) -> None:
    await embed(fake(), TEN, tmp_path, kind=kind)
    path = cache_path(tmp_path, TENANT, kind)
    # The keys still read, so the new text is embedded; the vectors fail once they are copied.
    vectors = pq.ParquetFile(path).metadata.row_group(0).column(2)
    with path.open("r+b") as file:
        file.seek(vectors.data_page_offset)
        file.write(b"\xff" * vectors.total_compressed_size)
    before = path.read_bytes()

    with pytest.raises(ValueError, match=f"delete {re.escape(str(path))} to rebuild it"):
        await embed(fake(), ["camp stoves"], tmp_path, kind=kind)

    assert path.read_bytes() == before
    assert cache_files(path.parent) == [path.name], "temp file left"


# ── stages writing at once ──────────────────────────────────────────────────


@dataclass
class HeldVoyage(FakeVoyage):
    """Holds every request until ``go`` is set; the first request sets ``asked``."""

    asked: asyncio.Event = field(default_factory=asyncio.Event)
    go: asyncio.Event = field(default_factory=asyncio.Event)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int,
        truncation: bool,
    ) -> FakeResult:
        self.asked.set()
        await self.go.wait()
        return await super().embed(
            texts,
            model=model,
            input_type=input_type,
            output_dimension=output_dimension,
            truncation=truncation,
        )


@pytest.mark.parametrize("kind", KINDS)
async def test_a_write_keeps_the_rows_another_stage_added_after_this_one_read(
    tmp_path: Path, kind: Kind
) -> None:
    """The lost update of the pipeline: a stage that found no cache wrote only its own rows."""
    held = HeldVoyage(respond=fake().respond)
    first = asyncio.create_task(embed(held, ["trail shoes"], tmp_path, kind=kind))
    await held.asked.wait()

    await embed(fake(), ["rain jackets"], tmp_path, kind=kind)
    held.go.set()
    await first

    kept = stored(cache_path(tmp_path, TENANT, kind))["text_sha256"].to_pylist()
    assert sorted(kept) == sorted(map(text_key, ["trail shoes", "rain jackets"])), "a row was lost"


@pytest.mark.parametrize("kind", KINDS)
async def test_a_text_two_stages_embedded_at_once_is_stored_once(
    tmp_path: Path, kind: Kind
) -> None:
    held = HeldVoyage(respond=fake().respond)
    first = asyncio.create_task(embed(held, TEXTS[:2], tmp_path, kind=kind))
    await held.asked.wait()

    await embed(fake(), TEXTS[1:], tmp_path, kind=kind)
    held.go.set()
    await first

    kept = stored(cache_path(tmp_path, TENANT, kind))["text_sha256"].to_pylist()
    assert sorted(kept) == sorted(map(text_key, TEXTS)), "a text was stored twice"


@pytest.mark.parametrize("kind", KINDS)
def test_writers_in_other_processes_keep_every_row(
    tmp_path: Path, kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers with their own thread and event loop, as two processes have, each rewriting
    the cache when the other could: only the lock file keeps them from copying the same rows."""
    asyncio.run(embed(fake(), ["trail shoes"], tmp_path, kind=kind))
    both_copying = threading.Barrier(2, timeout=0.5)
    batches = text_vectors_module._batches

    def copying(file: pq.ParquetFile, columns: list[str] | None) -> Iterator[pa.RecordBatch]:
        # A rewrite reads every column: it waits here for the other writer to copy too.
        if columns is None:
            with contextlib.suppress(threading.BrokenBarrierError):
                both_copying.wait()
        yield from batches(file, columns)

    monkeypatch.setattr(text_vectors_module, "_batches", copying)
    texts = ["rain jackets", "camp stoves"]

    with ThreadPoolExecutor(len(texts)) as pool:
        list(pool.map(lambda text: asyncio.run(embed(fake(), [text], tmp_path, kind=kind)), texts))

    kept = stored(cache_path(tmp_path, TENANT, kind))["text_sha256"].to_pylist()
    assert sorted(kept) == sorted(map(text_key, ["trail shoes", *texts])), "a row was lost"
    assert (tmp_path / TENANT / "text_vectors" / ".lock").is_file()
    assert cache_files(cache_path(tmp_path, TENANT, kind).parent) == [
        cache_path(tmp_path, TENANT, kind).name
    ]
