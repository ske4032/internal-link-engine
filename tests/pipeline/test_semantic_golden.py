"""The semantic rung and anchor selection give the outputs captured before #91's memory rework:
matches, thresholds, guard counts and zero overlap identical, cosines within float32 rounding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from golden_outputs import (
    BIG_TARGETS,
    DIM,
    RIVAL_INBOUND,
    RIVAL_TARGETS,
    RUNG_FILE,
    SELECTION_FILE,
    TOLERANCE,
    big_outputs,
    big_vectors,
    differences,
    load,
    planted_gate_outputs,
    quality_outputs,
    rung_voyage,
    semantic_rung_outputs,
)
from voyage_fakes import client

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo


def report(scenario: str, found: list[str]) -> str:
    shown = "\n".join(found[:25])
    more = f"\n... and {len(found) - 25} more" if len(found) > 25 else ""
    return f"{scenario} departs from the golden outputs (floats within {TOLERANCE}):\n{shown}{more}"


def test_the_golden_rung_outputs_exercise_every_output() -> None:
    golden = load(RUNG_FILE)

    assert set(golden) == {"planted", "guarded", "big", "rivals", "rivals-two-languages"}
    assert len(golden["planted"]["matches"]) == 1
    guarded = golden["guarded"]
    assert (guarded["rejected_identifier"], guarded["rejected_other_target"]) == (1, 1)
    big = golden["big"]["threshold"]
    assert (big["negatives"] >= 300, big["bounded"], big["fallback"]) == (True, True, False)
    rivals, apart = golden["rivals"], golden["rivals-two-languages"]
    assert rivals["rejected_other_target"] > 0, "no pair lost its phrase to a rival"
    # The tent page's keyword is no rival to the trail pages once the topics' languages differ.
    assert apart["rejected_other_target"] < rivals["rejected_other_target"]
    assert len(apart["matches"]) > len(rivals["matches"])
    assert {"1", "2"} <= set(apart["log"]["by_keyword_rank"])
    assert all(golden[name]["zero_overlap"] > 0 for name in ("planted", "big", "rivals"))


async def test_semantic_rung_outputs_match_golden(tmp_path: Path) -> None:
    golden = load(RUNG_FILE)

    found = await semantic_rung_outputs(tmp_path)

    for scenario, expected in golden.items():
        missed = differences(expected, found[scenario], scenario)
        assert not missed, report(scenario, missed)


async def test_semantic_rung_outputs_from_a_warm_cache_match_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = load(RUNG_FILE)
    fake = rung_voyage()
    await big_outputs(
        big_vectors(client(rung_voyage()), tmp_path), targets=BIG_TARGETS, languages=False
    )
    await big_outputs(
        big_vectors(client(rung_voyage()), tmp_path),
        targets=RIVAL_TARGETS,
        languages=False,
        inbound=RIVAL_INBOUND,
        override=0.5,
    )
    # Offline, the cache is read at the tenant's configured model and dimension.
    monkeypatch.setenv("TENANT_EMBEDDING_DIMENSIONS", str(DIM))

    found: dict[str, Any] = {
        "big": await big_outputs(big_vectors(None, tmp_path), targets=BIG_TARGETS, languages=False),
        "rivals": await big_outputs(
            big_vectors(client(fake), tmp_path),
            targets=RIVAL_TARGETS,
            languages=False,
            inbound=RIVAL_INBOUND,
            override=0.5,
        ),
    }

    assert fake.call_count == 0, "a cached vector was embedded again"
    for scenario, outputs in found.items():
        missed = differences(golden[scenario], outputs, scenario)
        assert not missed, report(scenario, missed)


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["planted_gate", "quality"])
async def test_anchor_selection_outputs_match_golden(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, scenario: str
) -> None:
    expected = load(SELECTION_FILE)[scenario]
    run = planted_gate_outputs if scenario == "planted_gate" else quality_outputs

    found = await run(graph, mongo, tenant, tmp_path)

    report_ = expected["report"]
    assert report_["chosen"] > 0
    assert report_["features_filled"] == report_["chosen"], "every chosen anchor has features"
    if scenario == "quality":
        assert report_["semantic_matched"] > 0, "the golden quality tenant has semantic anchors"
        assert report_["semantic_rejected_other_target"] > 0, "and phrases closer to a rival"
    missed = differences(expected, found, scenario)
    assert not missed, report(scenario, missed)
