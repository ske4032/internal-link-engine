"""The per-tenant keyword vector cache: only new texts are embedded, texts are never stored,
and no tenant ever reads another's vectors, even for identical text."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs
from voyage_fakes import DIMENSION, FakeVoyage, client, settings

from linking_engine.embedding.voyage_client import VoyageClient
from linking_engine.pipeline.keyword_vectors import FILE, keyword_vectors, text_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

TENANT = "test-kw-a"
OTHER = "test-kw-b"
TEXTS = ("trail shoes", "rain jackets", "four season tents")


def spike(text: str) -> list[float]:
    """A vector per text: a spike at a position derived from the text, so each is distinct."""
    vector = [1.0] * DIMENSION
    vector[sum(map(ord, text)) % DIMENSION] = 40.0
    return vector


def fake(dimension: int = DIMENSION) -> FakeVoyage:
    def respond(texts: Sequence[str]) -> list[list[float]]:
        return [spike(text)[:dimension] for text in texts]

    return FakeVoyage(dimension=dimension, respond=respond)


def entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def embedded_texts(voyage: FakeVoyage) -> list[str]:
    return sorted(text for call in voyage.calls for text in call.texts)


def unit(text: str) -> np.ndarray:
    vector = np.asarray(spike(text), dtype=np.float64)
    return vector / np.linalg.norm(vector)


async def test_the_first_run_embeds_every_distinct_text_and_caches_them_without_the_text(
    tmp_path: Path,
) -> None:
    voyage = fake()

    found = await keyword_vectors(client(voyage), TENANT, [*TEXTS, TEXTS[0]], cache_dir=tmp_path)

    assert (found.embedded, found.cached) == (3, 0)
    assert found.api_tokens > 0
    assert embedded_texts(voyage) == sorted(TEXTS), "a repeated text is embedded once"
    assert sorted(found.vectors) == sorted(TEXTS)
    for text in TEXTS:
        assert found.vectors[text] == pytest.approx(unit(text), abs=1e-6)
    path = tmp_path / TENANT / FILE
    table = pq.read_table(path)
    assert table.column_names == ["model", "text_sha256", "vector"]
    assert sorted(table["text_sha256"].to_pylist()) == sorted(map(text_key, TEXTS))
    assert set(table["model"].to_pylist()) == {"voyage-4-large"}
    assert table.schema.metadata[b"tenant_id"] == TENANT.encode()
    stored = path.read_bytes()
    assert [text for text in TEXTS if text.encode() in stored] == [], "texts reached the file"


async def test_a_second_run_embeds_nothing_and_leaves_the_file_alone(tmp_path: Path) -> None:
    first = await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)
    path = tmp_path / TENANT / FILE
    written = path.stat().st_mtime_ns
    voyage = fake()

    again = await keyword_vectors(client(voyage), TENANT, TEXTS, cache_dir=tmp_path)

    assert (again.embedded, again.cached, again.api_tokens) == (0, 3, 0)
    assert voyage.call_count == 0
    assert path.stat().st_mtime_ns == written, "a full cache hit must not rewrite the file"
    for text in TEXTS:
        np.testing.assert_array_equal(again.vectors[text], first.vectors[text])


async def test_only_new_texts_are_embedded_and_the_cache_keeps_the_current_texts(
    tmp_path: Path,
) -> None:
    await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)
    voyage = fake()

    found = await keyword_vectors(
        client(voyage), TENANT, [TEXTS[0], TEXTS[1], "camp stoves"], cache_dir=tmp_path
    )

    assert embedded_texts(voyage) == ["camp stoves"]
    assert (found.embedded, found.cached) == (1, 2)
    stored = pq.read_table(tmp_path / TENANT / FILE)["text_sha256"].to_pylist()
    assert sorted(stored) == sorted(map(text_key, [TEXTS[0], TEXTS[1], "camp stoves"]))


async def test_another_tenant_with_identical_texts_embeds_its_own(tmp_path: Path) -> None:
    await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)
    voyage = fake()

    theirs = await keyword_vectors(client(voyage), OTHER, TEXTS, cache_dir=tmp_path)

    assert (theirs.embedded, theirs.cached) == (3, 0), "one tenant reused another's vectors"
    assert embedded_texts(voyage) == sorted(TEXTS)
    assert entries(tmp_path) == sorted([TENANT, OTHER])
    assert pq.read_table(tmp_path / OTHER / FILE).schema.metadata[b"tenant_id"] == OTHER.encode()


async def test_another_tenants_file_in_this_tenants_place_is_ignored_and_rebuilt(
    tmp_path: Path,
) -> None:
    await keyword_vectors(client(fake()), OTHER, TEXTS, cache_dir=tmp_path)
    (tmp_path / TENANT).mkdir()
    shutil.copy(tmp_path / OTHER / FILE, tmp_path / TENANT / FILE)
    voyage = fake()

    with capture_logs() as logs:
        found = await keyword_vectors(client(voyage), TENANT, TEXTS, cache_dir=tmp_path)

    assert (found.embedded, found.cached) == (3, 0)
    [warning] = [e for e in logs if e["event"] == "keywords.vectors_cache_invalid"]
    assert (warning["stage"], warning["tenant_id"]) == ("quality-eval", TENANT)
    rebuilt = pq.read_table(tmp_path / TENANT / FILE)
    assert rebuilt.schema.metadata[b"tenant_id"] == TENANT.encode()


async def test_an_unreadable_file_is_rebuilt(tmp_path: Path) -> None:
    (tmp_path / TENANT).mkdir()
    (tmp_path / TENANT / FILE).write_bytes(b"not parquet")

    with capture_logs() as logs:
        found = await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)

    assert (found.embedded, found.cached) == (3, 0)
    [warning] = [e for e in logs if e["event"] == "keywords.vectors_cache_unreadable"]
    assert (warning["stage"], warning["tenant_id"]) == ("quality-eval", TENANT)
    assert pq.ParquetFile(tmp_path / TENANT / FILE).metadata.num_rows == 3
    assert entries(tmp_path / TENANT) == [FILE], "temp file left"


async def test_vectors_of_another_model_are_not_reused(tmp_path: Path) -> None:
    await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)
    voyage = fake()
    other_model = VoyageClient(settings(), model="voyage-3-large", dimension=DIMENSION, sdk=voyage)

    found = await keyword_vectors(other_model, TENANT, TEXTS, cache_dir=tmp_path)

    assert (found.embedded, found.cached) == (3, 0)
    assert {call.model for call in voyage.calls} == {"voyage-3-large"}
    assert set(pq.read_table(tmp_path / TENANT / FILE)["model"].to_pylist()) == {"voyage-3-large"}


async def test_a_cache_of_another_dimension_is_ignored(tmp_path: Path) -> None:
    await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)

    with capture_logs() as logs:
        found = await keyword_vectors(client(fake(8)), TENANT, TEXTS, cache_dir=tmp_path)

    assert (found.embedded, found.cached) == (3, 0)
    assert {len(vector) for vector in found.vectors.values()} == {8}
    assert "keywords.vectors_cache_invalid" in [entry["event"] for entry in logs]


async def test_no_texts_embed_nothing_and_write_nothing(tmp_path: Path) -> None:
    voyage = fake()

    found = await keyword_vectors(client(voyage), TENANT, [], cache_dir=tmp_path)

    assert (found.vectors, found.embedded, found.cached) == ({}, 0, 0)
    assert voyage.call_count == 0
    assert entries(tmp_path) == []


async def test_the_log_line_names_counts_never_texts(tmp_path: Path) -> None:
    with capture_logs() as logs:
        await keyword_vectors(client(fake()), TENANT, TEXTS, cache_dir=tmp_path)

    [line] = [entry for entry in logs if entry["event"] == "keywords.vectors"]
    assert (line["stage"], line["tenant_id"], line["texts"], line["embedded"]) == (
        "quality-eval",
        TENANT,
        3,
        3,
    )
    logged = " ".join(str(value) for value in line.values())
    assert [text for text in TEXTS if text in logged] == []


@pytest.mark.parametrize("tenant", [" ", "..", "../escape", "a/b"])
async def test_a_tenant_that_is_not_a_directory_name_is_refused(
    tmp_path: Path, tenant: str
) -> None:
    voyage = fake()
    with pytest.raises(ValueError, match="tenant_id"):
        await keyword_vectors(client(voyage), tenant, TEXTS, cache_dir=tmp_path)
    assert voyage.call_count == 0
    assert entries(tmp_path) == []
