"""score_pairs end to end: the tenant's feature matrix scored with the default or the tenant's
own weights, written as a Parquet file named by both the matrix and the weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs
from test_feature_stage import NAMES, entries, seed, url

from linking_engine.discovery.features import KEY_COLUMNS
from linking_engine.discovery.scoring import default_weights, weights_hash
from linking_engine.errors import DatabaseReadError
from linking_engine.models import FeatureWeight, ScorerWeights
from linking_engine.pipeline.features import assemble_features
from linking_engine.pipeline.scoring import score_pairs

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TOPS = [f"top{i}_{part}" for i in (1, 2, 3) for part in ("feature", "contribution", "value")]
OWN = ScorerWeights(
    version="acme-1",
    features=(FeatureWeight(column="content_cosine", weight=1.0, normalisation="percentile"),),
)


@pytest.mark.integration
async def test_the_scores_of_every_candidate_pair_are_written_beside_the_matrix(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)

    with capture_logs() as logs:
        report, path = await score_pairs(graph, mongo, tenant, cache_dir=tmp_path)

    features, _ = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    defaults = default_weights()
    assert path == (
        tmp_path / tenant / f"{features.cache_key}.{weights_hash(defaults)[:12]}.scores.parquet"
    )
    assert (report.feature_cache_key, report.weights.version) == (features.cache_key, "baseline-1")
    assert report.weights_hash == weights_hash(defaults)
    scores = pq.read_table(path).to_pandas()
    assert list(scores.columns) == [*KEY_COLUMNS, "score", "tier", "raw_score", *TOPS]
    assert len(scores) == report.pairs == features.pairs
    assert (scores["score"].min(), scores["score"].max()) == (0.0, 100.0)
    assert set(scores["tier"]) <= {1, 2, 3}
    assert sum(report.tiers.values()) == report.pairs
    assert [name for name in entries(path.parent) if name.endswith(".tmp")] == []
    schema = pq.read_schema(path)
    assert (schema.field("tier").type, schema.field("score").type) == (pa.int8(), pa.float64())
    assert schema.field("top1_feature").type == pa.string()
    metadata = schema.metadata
    assert ScorerWeights.model_validate_json(metadata[b"weights"]) == defaults
    assert (metadata[b"feature_cache_key"].decode(), metadata[b"tenant_id"].decode()) == (
        features.cache_key,
        tenant,
    )
    [line] = [entry for entry in logs if entry["event"] == "scores.computed"]
    logged = " ".join(str(value) for value in line.values())
    assert [u for u in map(url, NAMES) if u in logged] == [], "urls in the log line"


@pytest.mark.integration
async def test_a_tenants_own_weights_override_the_default_for_that_tenant_only(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed(graph, mongo, tenant)
    await seed(graph, mongo, other)
    await mongo.set_scorer_weights(tenant, OWN)

    assert await mongo.get_scorer_weights(tenant) == OWN
    assert await mongo.get_scorer_weights(other) is None
    mine, my_path = await score_pairs(graph, mongo, tenant, cache_dir=tmp_path)
    theirs, _ = await score_pairs(graph, mongo, other, cache_dir=tmp_path)

    assert (mine.weights.version, mine.weights_hash) == ("acme-1", weights_hash(OWN))
    assert my_path.name.endswith(f".{weights_hash(OWN)[:12]}.scores.parquet")
    assert theirs.weights.version == "baseline-1"
    scores = pq.read_table(my_path).to_pandas()
    # One weighted feature: every pair's top contributor is it.
    assert set(scores["top1_feature"]) == {"content_cosine"}
    assert mine.top_contributors == {"content_cosine": mine.pairs}


@pytest.mark.integration
async def test_removing_a_tenants_weights_restores_the_default(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)
    await mongo.set_scorer_weights(tenant, OWN)
    await mongo.set_scorer_weights(tenant, None)

    report, _ = await score_pairs(graph, mongo, tenant, cache_dir=tmp_path)

    assert await mongo.get_scorer_weights(tenant) is None
    assert report.weights.version == "baseline-1"


@pytest.mark.integration
async def test_weights_naming_an_unknown_column_are_refused_before_any_work(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    bogus = ScorerWeights(
        version="acme-bad", features=(FeatureWeight(column="page_age_days", weight=1.0),)
    )
    await mongo.set_scorer_weights(tenant, bogus)

    with pytest.raises(ValueError, match="page_age_days"):
        await score_pairs(graph, mongo, tenant, cache_dir=tmp_path)
    assert entries(tmp_path) == []


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param("baseline-1", id="not-a-document"),
        pytest.param({"version": "x", "features": []}, id="no-features"),
        pytest.param(
            {
                "version": "x",
                "features": [{"column": "content_cosine", "weight": 1.0}],
                "tier_shares": [0.2, 0.3],
            },
            id="snake-case-key",
        ),
    ],
)
@pytest.mark.integration
async def test_malformed_stored_weights_fail_the_read(
    mongo: MongoRepo, tenant: str, stored: object
) -> None:
    await mongo._db["tenant_config"].update_one(
        {"tenantId": tenant}, {"$set": {"scorerWeights": stored}}, upsert=True
    )

    with pytest.raises(DatabaseReadError, match="scorer weights"):
        await mongo.get_scorer_weights(tenant)
