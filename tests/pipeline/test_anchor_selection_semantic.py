"""Anchor selection without Voyage (#22): no key, or an outage after retries, skips the
semantic rung with its reason, and the lexical rungs still choose anchors; nothing fails."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from quality_seed import seed_quality
from quality_seed import voyage as keyword_voyage
from voyage_fakes import client, settings
from voyageai.error import ServiceUnavailableError

from linking_engine.models import AnchorRung, ExtractionSettings, UnanchoredReason
from linking_engine.pipeline.anchor_selection import UNANCHORED_FILE, select_anchors
from linking_engine.pipeline.semantic_anchors import NO_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

LEXICAL = (AnchorRung.EXACT, AnchorRung.STEMMED, AnchorRung.STEM_SET)


def down() -> VoyageClient:
    fake = keyword_voyage()
    fake.failures = [ServiceUnavailableError("down", http_status=503)]
    return client(fake, settings(max_attempts=1))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("voyage", "reason"),
    [
        pytest.param(lambda: None, NO_KEY, id="no-key"),
        pytest.param(
            down, "Voyage unavailable after retries (ServiceUnavailableError)", id="outage"
        ),
    ],
)
async def test_without_voyage_the_lexical_rungs_still_choose_anchors(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    voyage: object,
    reason: str,
) -> None:
    await seed_quality(graph, mongo, tenant)

    report, path = await select_anchors(
        graph,
        mongo,
        tenant,
        cache_dir=tmp_path,
        voyage=voyage(),  # type: ignore[operator]
    )

    assert report.semantic_skipped_reason == reason
    # The planted copies hold 48 lexical matches over the 480 candidate pairs (see the quality
    # eval's extractability): the rung skip takes none of them away.
    assert report.lexical_pairs == 48
    assert (report.semantic_invocations, report.semantic_matched) == (0, 0)
    assert report.zero_overlap_matches == 0
    assert sum(report.semantic_histogram) == 0
    assert report.threshold.negatives == 0
    assert report.chosen > 0
    rows = pq.read_table(path).to_pylist()
    assert rows, "no anchor was chosen"
    assert {row["rung"] for row in rows} <= {rung.value for rung in LEXICAL}
    assert report.chosen + sum(report.unanchored.values()) == report.pairs
    assert not (tmp_path / tenant / "text_vectors").exists(), "sentences or phrases were cached"
    # Every keyworded pair the lexical rungs missed was never searched by meaning: that is no
    # finding about the source page, so none is reported as not mentioning the topic.
    not_run = report.pairs - report.lexical_pairs
    assert report.unanchored[UnanchoredReason.MEANING_SEARCH_NOT_RUN] == not_run == 432
    assert report.unanchored.get(UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC, 0) == 0
    unanchored = pq.read_table(tmp_path / tenant / UNANCHORED_FILE).to_pylist()
    unsearched = {
        (row["source_url"], row["target_url"])
        for row in unanchored
        if row["reason"] == UnanchoredReason.MEANING_SEARCH_NOT_RUN.value
    }
    assert len(unsearched) == not_run
    assert not unsearched & {(row["source_url"], row["target_url"]) for row in rows}


@pytest.mark.integration
async def test_the_tenants_semantic_threshold_override_reaches_the_rung(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    await mongo.set_extraction_settings(tenant, ExtractionSettings(semantic_threshold=0.45))
    fake = keyword_voyage()

    report, _ = await select_anchors(graph, mongo, tenant, cache_dir=tmp_path, voyage=client(fake))

    assert report.semantic_skipped_reason is None
    assert (report.threshold.value, report.threshold.negatives) == (0.45, 0)
    assert (report.threshold.bounded, report.threshold.fallback) == (False, False)
    assert fake.call_count > 0


@pytest.mark.integration
async def test_page_vectors_of_another_model_leave_the_placement_features_empty(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.embeddingModel = 'voyage-3-large'", t=tenant
    )

    report, path = await select_anchors(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(keyword_voyage())
    )

    assert report.embedding_skipped_reason == (
        "page vectors are from voyage-3-large, not voyage-4-large"
    )
    assert report.chosen > 0
    assert report.features_filled == 0
    rows = [row for row in pq.read_table(path).to_pylist() if row["rank"] == 1]
    assert rows
    assert {(row["context_relevance"], row["anchor_target_fit"]) for row in rows} == {(None, None)}
