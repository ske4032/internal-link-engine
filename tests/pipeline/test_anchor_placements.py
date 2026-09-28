"""The two placement features in the feature matrix (#22): read from the tenant's anchor choices
file, rank 1 only, and the file's digest in the feature cache key."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs
from test_feature_stage import seed as seed_features
from test_feature_stage import url

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.pipeline.anchor_selection import CHOICES_SCHEMA
from linking_engine.pipeline.features import (
    ANCHOR_CHOICES_FILE,
    anchor_placements,
    assemble_features,
)

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TENANT = "test-placements"


def choice(
    source: str,
    target: str,
    *,
    rank: int = 1,
    context: float | None = 0.8,
    fit: float | None = 0.7,
) -> dict[str, object]:
    """One row of the anchor choices file as anchor selection writes it."""
    sentence = "Our trail shoes guide covers wet rock."
    return {
        "source_url": source,
        "target_url": target,
        "rank": rank,
        "anchor_type": "EXACT",
        "keyword": "trail shoes",
        "keyword_rank": 1,
        "keyword_source": "CLIENT_STRATEGIC",
        "rung": "EXACT",
        "phrase": "trail shoes",
        "start": 4,
        "end": 15,
        "sentence": sentence,
        "sentence_index": 0,
        "sentence_start": 0,
        "stem_jaccard": None,
        "semantic_similarity": None,
        "score_semantic": 0.9,
        "score_keyword": 1.0,
        "score_diversity": 1.0,
        "score_length": 1.0,
        "score_rank_weight": 1.0,
        "score_profile_bonus": 0.0,
        "score_total": 0.95,
        "context_relevance": context,
        "anchor_target_fit": fit,
    }


def write_choices(cache_dir: Path, tenant: str, rows: list[dict[str, object]]) -> Path:
    path = cache_dir / tenant / ANCHOR_CHOICES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CHOICES_SCHEMA), path)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── reading the choices ─────────────────────────────────────────────────────


def test_without_a_choices_file_there_are_no_placements(tmp_path: Path) -> None:
    found = anchor_placements(tmp_path, TENANT)

    assert (found.features, found.digest) == ({}, None)


def test_rank_one_rows_give_the_features_and_the_digest_is_of_the_file(tmp_path: Path) -> None:
    path = write_choices(
        tmp_path,
        TENANT,
        [
            choice("s1", "t1", context=0.8, fit=0.7),
            choice("s1", "t1", rank=2, context=0.1, fit=0.1),
            choice("s2", "t1", context=0.6, fit=None),
            choice("s3", "t1", context=None, fit=None),
        ],
    )

    found = anchor_placements(tmp_path, TENANT)

    # An alternative never fills the features; a choice without either vector adds nothing.
    assert found.features == {("s1", "t1"): (0.8, 0.7), ("s2", "t1"): (0.6, None)}
    assert found.digest == sha(path)
    assert found.rows == 3, "every rank-1 row read, the one without features included"


def test_another_tenants_choices_are_never_read(tmp_path: Path) -> None:
    write_choices(tmp_path, "test-other", [choice("s1", "t1")])

    assert anchor_placements(tmp_path, TENANT).features == {}


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([choice("s1", "t1"), choice("s1", "t1", context=0.5)], id="duplicate-pair"),
        pytest.param(
            [choice("s1", "t1", context=None, fit=None), choice("s1", "t1")],
            id="duplicate-after-an-empty-row",
        ),
        pytest.param([choice("s1", "t1", context=1.2)], id="above-one"),
        pytest.param([choice("s1", "t1", fit=-0.1)], id="below-zero"),
        pytest.param([choice("s1", "t1", fit=float("nan"))], id="nan"),
    ],
)
def test_an_inconsistent_file_is_ignored_with_a_warning(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    write_choices(tmp_path, TENANT, rows)

    with capture_logs() as logs:
        found = anchor_placements(tmp_path, TENANT)

    assert (found.features, found.digest, found.rows) == ({}, None, 0)
    [warning] = [e for e in logs if e["event"] == "features.anchor_choices_invalid"]
    assert warning["tenant_id"] == TENANT


@pytest.mark.parametrize("content", [b"not parquet", None], ids=["garbage", "missing-columns"])
def test_an_unreadable_file_is_ignored_with_a_warning(
    tmp_path: Path, content: bytes | None
) -> None:
    path = tmp_path / TENANT / ANCHOR_CHOICES_FILE
    path.parent.mkdir(parents=True)
    if content is None:
        pq.write_table(pa.table({"source_url": ["s1"], "target_url": ["t1"]}), path)
    else:
        path.write_bytes(content)

    with capture_logs() as logs:
        found = anchor_placements(tmp_path, TENANT)

    assert (found.features, found.digest) == ({}, None)
    [warning] = [e for e in logs if e["event"] == "features.anchor_choices_unreadable"]
    assert warning["tenant_id"] == TENANT


# ── the matrix ──────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_a_chosen_anchor_fills_its_pairs_two_columns_and_changes_the_cache_key(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_features(graph, mongo, tenant)
    found = await retrieve_candidates(graph, tenant)
    pairs = [(source, entry.target_url) for entry in found.targets for source in entry.sources]
    chosen, other = pairs[0], pairs[1]
    plain, plain_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    path = write_choices(
        tmp_path,
        tenant,
        [
            choice(*chosen, context=0.8, fit=0.7),
            choice(*chosen, rank=2, context=0.2, fit=0.2),
            choice(*other, context=0.6, fit=None),
            # Not a candidate pair: nothing to fill.
            choice(url("a"), url("zzz"), context=0.1, fit=0.1),
        ],
    )

    with capture_logs() as logs:
        placed, placed_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)

    assert plain.anchor_choices_digest is None
    assert placed.anchor_choices_digest == sha(path)
    assert placed.cache_key != plain.cache_key, "the choices file did not change the key"
    assert (placed.cache_hit, placed_path != plain_path) == (False, True)
    assert plain_path.is_file(), "the matrix without choices is kept"
    frame = pq.read_table(placed_path).to_pandas().set_index(["source_url", "target_url"])
    assert tuple(frame.loc[chosen, ["context_relevance", "anchor_target_fit"]]) == (0.8, 0.7)
    assert frame.loc[other, "context_relevance"] == 0.6
    assert frame.loc[other, ["anchor_target_fit"]].isna().all()
    rest = frame.drop(index=[chosen, other])
    assert rest[["context_relevance", "anchor_target_fit"]].isna().all().all()
    assert {"context_relevance", "anchor_target_fit"} <= set(plain.all_null_columns)
    assert "context_relevance" not in placed.all_null_columns
    [line] = [entry for entry in logs if entry["event"] == "features.assembled"]
    # Three rank-1 rows were read, the one outside the candidate pairs too; two pairs filled.
    assert (line["anchor_choices_read"], line["placements_filled"]) == (3, 2)
    assert line["anchor_choices_digest"] == sha(path)

    again, again_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    assert (again.cache_hit, again.cache_key, again_path) == (True, placed.cache_key, placed_path)

    write_choices(tmp_path, tenant, [choice(*chosen, context=0.3, fit=0.4)])
    changed, changed_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    assert changed.cache_key not in {plain.cache_key, placed.cache_key}
    frame = pq.read_table(changed_path).to_pandas().set_index(["source_url", "target_url"])
    assert tuple(frame.loc[chosen, ["context_relevance", "anchor_target_fit"]]) == (0.3, 0.4)


@pytest.mark.integration
async def test_removing_the_choices_file_returns_to_the_plain_matrix(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_features(graph, mongo, tenant)
    plain, plain_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    found = await retrieve_candidates(graph, tenant)
    path = write_choices(
        tmp_path, tenant, [choice(found.targets[0].sources[0], found.targets[0].target_url)]
    )
    await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    path.unlink()

    back, back_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)

    assert (back.cache_key, back_path, back.cache_hit) == (plain.cache_key, plain_path, True)
    assert back.anchor_choices_digest is None
