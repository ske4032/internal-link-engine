"""select_anchors end to end on real Neo4j and Mongo: the planted gate, the output files, the
running per-target state, read-only stores and another tenant untouched."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import pytest
from selection_seed import (
    BARE,
    ECHO,
    EXISTING,
    MISSING,
    OFFSET,
    PAGES,
    keyword,
    page_vectors,
    path,
    resolved,
    seed_selection,
    voyage,
)
from store_state import graph_state, mongo_state
from structlog.testing import capture_logs
from test_keyword_stage import url
from voyage_fakes import client

from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.models.enums import UnanchoredReason
from linking_engine.pipeline.anchor_selection import (
    CHOICES_SCHEMA,
    UNANCHORED_FILE,
    UNANCHORED_SCHEMA,
    select_anchors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

CHOICE_COLUMNS = [
    "source_url",
    "target_url",
    "rank",
    "anchor_type",
    "keyword",
    "keyword_rank",
    "keyword_source",
    "rung",
    "phrase",
    "start",
    "end",
    "sentence",
    "sentence_index",
    "sentence_start",
    "stem_jaccard",
    "semantic_similarity",
    "score_semantic",
    "score_keyword",
    "score_diversity",
    "score_length",
    "score_rank_weight",
    "score_profile_bonus",
    "score_total",
    "context_relevance",
    "anchor_target_fit",
]
UNANCHORED_COLUMNS = ["source_url", "target_url", "reason", "advice", "best_score"]
GATE, TOLERANCE = 0.78, 0.05
NO_MENTION = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
NO_KEYWORD = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
NO_TEXT = UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE
# Its copy only writes a keyword that is missing anyway, so no planted anchor is lost.
TEXTLESS = 10


def cosine(a: str, b: str) -> float:
    vectors = page_vectors()
    left, right = np.asarray(vectors[a]), np.asarray(vectors[b])
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def names(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


def test_the_output_file_columns_are_the_contracts() -> None:
    assert CHOICES_SCHEMA.names == CHOICE_COLUMNS
    assert UNANCHORED_SCHEMA.names == UNANCHORED_COLUMNS


@pytest.mark.integration
async def test_the_planted_gate_resolves_exactly_the_targets_whose_keyword_is_written(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed_selection(graph, mongo, tenant)
    await seed_selection(graph, mongo, other)
    stored = (await graph_state(graph, tenant), await graph_state(graph, other))
    documents = await mongo_state(mongo)

    with capture_logs() as logs:
        report, path_ = await select_anchors(
            graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage())
        )

    assert (await graph_state(graph, tenant), await graph_state(graph, other)) == stored
    assert await mongo_state(mongo) == documents, "the stage is read-only against Mongo"
    assert names(tmp_path) == [tenant], "nothing is written for the other tenant"

    share = report.targets_with_anchor / report.targets
    assert report.targets == PAGES
    assert abs(share - GATE) <= TOLERANCE, (
        f"planted gate: {GATE:.0%} ± {TOLERANCE:.0%} of target pages resolve, got {share:.0%}"
    )
    choices = pq.read_table(path_).to_pandas()
    assert list(choices.columns) == CHOICE_COLUMNS
    chosen = choices[choices["rank"] == 1]
    assert set(chosen["target_url"]) == resolved(), "exactly the planted targets resolve"
    # One pair per written keyword, two for the echoed one; every other pair is a gap.
    assert len(chosen) == PAGES - len(MISSING) + 1 == report.chosen
    assert report.pairs == PAGES * (PAGES - 1) - 1
    assert report.chosen + sum(report.unanchored.values()) == report.pairs
    assert Counter(chosen["rung"]) == {"EXACT": 12, "STEMMED": 12, "STEM_SET": 16}
    assert report.semantic_matched == 0, "no two hashed texts are close"
    assert report.features_filled == report.chosen
    assert chosen["context_relevance"].notna().all()
    assert chosen["anchor_target_fit"].notna().all()

    unanchored = pq.read_table(path_.parent / UNANCHORED_FILE).to_pandas()
    assert list(unanchored.columns) == UNANCHORED_COLUMNS
    assert len(unanchored) == report.pairs - report.chosen
    missing = {url(path(i)) for i in MISSING}
    into_missing = unanchored[unanchored["target_url"].isin(missing)]
    assert len(into_missing) == len(MISSING) * (PAGES - 1)
    assert set(into_missing["reason"]) == {NO_MENTION.value}
    assert set(into_missing["advice"]) == {UNANCHORED_ADVICE[NO_MENTION]}
    assert into_missing["best_score"].isna().all()
    assert not [name for name in names(path_.parent) if name.endswith(".tmp")]

    [line] = [entry for entry in logs if entry["event"] == "anchors.selected"]
    logged = " ".join(str(value) for value in line.values())
    texts = [url(path(i)) for i in range(PAGES)] + [keyword(i) for i in range(PAGES)]
    assert [text for text in texts if text in logged] == []


@pytest.mark.integration
async def test_a_targets_earlier_pairs_and_existing_anchors_shape_its_later_choices(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_selection(graph, mongo, tenant, bare=True, textless=TEXTLESS)

    report, path_ = await select_anchors(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage())
    )

    # A target without keywords is its own gap, advised as such, never a content gap.
    unanchored = pq.read_table(path_.parent / UNANCHORED_FILE).to_pandas()
    into_bare = unanchored[unanchored["target_url"] == url(BARE)]
    assert len(into_bare) == PAGES
    assert set(into_bare["reason"]) == {NO_KEYWORD.value}
    assert set(into_bare["advice"]) == {UNANCHORED_ADVICE[NO_KEYWORD]}
    assert report.unanchored[NO_KEYWORD] == PAGES
    from_bare = unanchored[unanchored["source_url"] == url(BARE)]
    assert set(from_bare["reason"]) == {NO_MENTION.value}
    # A source without stored text was never searched: no finding about its content.
    unread = unanchored[
        (unanchored["source_url"] == url(path(TEXTLESS))) & (unanchored["target_url"] != url(BARE))
    ]
    assert len(unread) == PAGES - 1
    assert set(unread["reason"]) == {NO_TEXT.value}
    assert set(unread["advice"]) == {UNANCHORED_ADVICE[NO_TEXT]}
    assert report.unanchored[NO_TEXT] == PAGES - 1

    choices = pq.read_table(path_).to_pandas()
    chosen = choices[choices["rank"] == 1].set_index(["source_url", "target_url"])
    # The echoed keyword: the more similar source's pair comes first and takes it whole.
    target = url(path(ECHO[1]))
    sources = sorted(
        (url(path((ECHO[1] + OFFSET) % PAGES)), url(path(ECHO[0]))),
        key=lambda source: -cosine(source, target),
    )
    first, second = (chosen.loc[(source, target)] for source in sources)
    assert (first["phrase"], second["phrase"]) == (keyword(ECHO[1]), keyword(ECHO[1]))
    assert (first["score_diversity"], second["score_diversity"]) == (1.0, 0.0)
    assert first["score_profile_bonus"] > second["score_profile_bonus"] == 0.0
    # The existing link's anchor is page 1's keyword: its plural shares one word of three.
    source, linked = EXISTING
    assert (url(path(source)), url(path(linked))) not in chosen.index
    into = chosen.loc[(url(path((linked + OFFSET) % PAGES)), url(path(linked)))]
    assert (into["rung"], into["anchor_type"]) == ("STEMMED", "PARTIAL")
    assert into["score_diversity"] == pytest.approx(2 / 3)
