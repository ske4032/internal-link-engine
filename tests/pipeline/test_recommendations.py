"""The recommendations stage on synthetic inputs: the per-source walk, score and tier, stable
ids, the audit's verdicts, bridge marks, target fixes, profiles, hubs, duplicates and every
listing's order; both scorers' signals on a matrix in tmp_path; the stage's guards; and one run
written through real Neo4j and Mongo that replaces the previous one."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import lightgbm
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs

from linking_engine.anchor.scoring import brand_tokens
from linking_engine.discovery.features import KEY_COLUMNS
from linking_engine.discovery.scoring import default_weights, rank_tiers, score_frame
from linking_engine.errors import DatabaseReadError
from linking_engine.ml.ranker_tracking import Holder
from linking_engine.ml.tracking import recommendation_metrics
from linking_engine.models import (
    ActionType,
    AnchorMix,
    AnchorType,
    BridgeLink,
    BridgeReason,
    ContentGapFinding,
    ExcludedPage,
    ExclusionReason,
    HubPair,
    IssueFlag,
    KeywordRung,
    LinkAuditResult,
    OrphanLabel,
    PageProfile,
    Recommendation,
    RecommendationReport,
    RunInfo,
    ScorerName,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.models.page import HubNode, InboundAnchorText, PageFacts
from linking_engine.output.collections import (
    HUBS,
    PAGES,
    RECOMMENDATIONS,
    RUN_SCOPED,
    RUNS,
    TARGET_FIXES,
    from_document,
    recommendation_id,
)
from linking_engine.output.writer import OutputWriter
from linking_engine.pipeline import recommendations
from linking_engine.pipeline.anchor_selection import CHOICES_SCHEMA, UNANCHORED_SCHEMA
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE, RANKED_SCHEMA
from linking_engine.pipeline.recommendations import (
    ADD_LINK_LABEL,
    ANCHOR_MAX_CHARS,
    CONTENT_GAP_LABEL,
    MODEL_CHANGED,
    Assembly,
    Inputs,
    Walk,
    assemble,
    baseline_signals,
    learned_signals,
    percentiles,
    publish_recommendations,
    read_ranked,
    summarise_recommendations,
    top_contributions,
    walk,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TENANT = "test-acme"
AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
CHOICE_COLUMNS = (
    "source_url",
    "target_url",
    "rank",
    "anchor_type",
    "keyword",
    "phrase",
    "start",
    "end",
    "sentence",
    "sentence_index",
    "score_total",
)
UNANCHORED_COLUMNS = ("source_url", "target_url", "reason", "advice", "best_score")
BRAND = brand_tokens(["Dome tents | Acme", "Camp stoves | Acme", "Trail shoes | Acme"])


def url(path: str) -> str:
    return f"example.com/{path}"


def ranked(scores: Mapping[tuple[str, str], float]) -> pandas.DataFrame:
    """Ranked pairs as rank-pairs writes them: best first within each source, ties by target."""
    frame = pandas.DataFrame(
        [{"source_url": s, "target_url": t, "score": v} for (s, t), v in scores.items()],
        columns=["source_url", "target_url", "score"],
    )
    frame = frame.sort_values(
        ["source_url", "score", "target_url"], ascending=[True, False, True], kind="stable"
    )
    frame["rank_in_source"] = frame.groupby("source_url", sort=False).cumcount() + 1
    return frame.reset_index(drop=True)


def choice(
    source: str,
    target: str,
    phrase: str,
    *,
    rank: int = 1,
    kind: AnchorType = AnchorType.PARTIAL,
    total: float = 0.8,
) -> dict[str, object]:
    sentence = f"Read about {phrase} before you go."
    return {
        "source_url": source,
        "target_url": target,
        "rank": rank,
        "anchor_type": kind.value,
        "keyword": "dome tent",
        "phrase": phrase,
        "start": 211,
        "end": 211 + len(phrase),
        "sentence": sentence,
        "sentence_index": 4,
        "score_total": total,
    }


def unanchored(source: str, target: str, reason: UnanchoredReason) -> dict[str, object]:
    return {
        "source_url": source,
        "target_url": target,
        "reason": reason.value,
        "advice": UNANCHORED_ADVICE[reason],
        "best_score": 0.3
        if reason is UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
        else None,
    }


def excluded(*urls: str) -> tuple[ExcludedPage, ...]:
    return tuple(
        ExcludedPage(
            url=page,
            reason=ExclusionReason.SITEMAP,
            label="sitemap",
            words=10,
            link_words=9,
            links=5,
        )
        for page in urls
    )


def page(path: str, **fields: Any) -> PageFacts:
    return PageFacts(**{"url": url(path), "inbound": 1, "outbound": 1, **fields})


def make_inputs(
    scores: Mapping[tuple[str, str], float] | None = None,
    choices: Iterable[Mapping[str, object]] = (),
    missing: Iterable[Mapping[str, object]] = (),
    **fields: Any,
) -> Inputs:
    base = Inputs(
        tenant_id=TENANT,
        run_id="run-1",
        started_at=AT,
        limit=10,
        gap_limit=3,
        tier_shares=(0.1, 0.3),
        scorer=ScorerName.BASELINE,
        ranked=ranked(scores or {}),
        choices=pandas.DataFrame(list(choices), columns=list(CHOICE_COLUMNS)),
        unanchored=pandas.DataFrame(list(missing), columns=list(UNANCHORED_COLUMNS)),
        hub_pairs=(),
        bridge_links=(),
        audit=(),
        anchors={},
        excluded=(),
        pages=(),
        hubs=(),
        titles={},
        keywords={},
        ranked_keywords={},
        inbound=(),
        brand=BRAND,
    )
    return dataclasses.replace(base, **fields)


def run(
    inputs: Inputs, signals: Mapping[tuple[str, str], Sequence[tuple[str, float]]] | None = None
) -> tuple[Walk, Assembly]:
    walked = walk(inputs)
    return walked, assemble(inputs, walked, signals or {})


def new_links(records: Sequence[Recommendation]) -> list[tuple[str, str, ActionType, int | None]]:
    return [
        (r.source_url, r.target_url, r.action_type, r.rank_in_source)
        for r in records
        if r.position is None
    ]


# ── the walk ─────────────────────────────────────────────────────────────────


def test_the_walk_puts_links_first_lists_gaps_apart_and_counts_what_it_passes() -> None:
    a, b, c, d = url("a"), url("b"), url("c"), url("d")
    t = [url(f"t{i}") for i in range(10)]
    gap, awkward = (
        UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC,
        UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE,
    )
    scores = {(a, t[i]): 10.0 - i for i in range(1, 10)} | {(c, t[1]): 1.0}
    scores |= {(b, t[1]): 5.0, (b, t[3]): 4.0, (b, t[5]): 3.0, (b, t[4]): 2.0, (b, t[6]): 1.0}
    scores |= {(d, t[i]): 10.0 - i for i in range(5, 10)}
    inputs = make_inputs(
        scores,
        choices=[
            *(choice(a, t[i], "dome tent") for i in (3, 6, 8)),
            choice(b, t[4], "camp stove"),
            choice(c, t[1], "x"),
        ],
        missing=[
            unanchored(a, t[1], gap),
            unanchored(a, t[2], gap),
            unanchored(a, t[5], UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD),
            unanchored(a, t[7], awkward),
            unanchored(b, t[1], UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE),
            unanchored(b, t[3], UnanchoredReason.MEANING_SEARCH_NOT_RUN),
            unanchored(b, t[6], gap),
            *(unanchored(d, t[i], awkward if i == 6 else gap) for i in range(5, 9)),
            # A hub bridge that was no candidate: served nowhere, counted apart.
            unanchored(b, t[7], gap),
        ],
        excluded=excluded(t[2], c),
        limit=2,
    )
    walked, out = run(inputs)

    # a: its first two anchored pairs, t8 past the limit; the gap above its last link only.
    # b: its gap ranks below its one link. d: no link, so its first three gaps.
    assert new_links(out.recommendations) == [
        (a, t[3], ActionType.ADD_LINK, 1),
        (a, t[6], ActionType.ADD_LINK, 2),
        (a, t[1], ActionType.CONTENT_GAP, 1),
        (b, t[4], ActionType.ADD_LINK, 1),
        (d, t[5], ActionType.CONTENT_GAP, 1),
        (d, t[6], ActionType.CONTENT_GAP, 2),
        (d, t[7], ActionType.CONTENT_GAP, 3),
    ]
    # a's t4 above its last link, b's t5 and d's t9 on pages below the limit; a's t9 is
    # below a's stop.
    assert walked.not_assessed == 3
    assert walked.unanchored_not_ranked == 1
    gaps = {
        (r.source_url, r.target_url): r
        for r in out.recommendations
        if r.action_type is ActionType.CONTENT_GAP
    }
    assert gaps[(a, t[1])].finding is ContentGapFinding.NO_TOPICAL_MENTION
    assert gaps[(d, t[6])].finding is ContentGapFinding.AWKWARD_PHRASING
    assert {r.label for r in gaps.values()} == {CONTENT_GAP_LABEL}
    assert gaps[(a, t[1])].advice == UNANCHORED_ADVICE[gap]
    assert all(r.proposed_anchors is None for r in gaps.values())

    listed = {(u.source_url, u.target_url): u for u in out.unanchored}
    assert set(listed) == {
        (a, t[1]), (a, t[5]), (a, t[7]), (b, t[1]), (b, t[3]), (b, t[6]),
        (d, t[5]), (d, t[6]), (d, t[7]), (d, t[8]),
    }  # fmt: skip
    assert {key for key, u in listed.items() if u.recommended} == set(gaps)
    assert listed[(a, t[7])].rank_in_source == 7
    assert listed[(a, t[7])].best_score == pytest.approx(0.3)
    assert out.summary.recommendations == {ActionType.ADD_LINK: 3, ActionType.CONTENT_GAP: 4}
    # Links only: b and d have fewer than the limit, a has it.
    assert out.summary.sources_below_limit == 2
    assert out.summary.sources_with_recommendations == 3
    assert out.summary.unanchored[UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE] == 1
    everything = {u for r in out.recommendations for u in (r.source_url, r.target_url)}
    everything |= {u for row in out.unanchored for u in (row.source_url, row.target_url)}
    assert not everything & {t[2], c}


def test_a_gap_ranked_below_the_last_link_is_not_listed() -> None:
    a, b = url("a"), url("b")
    t1, t2 = url("t1"), url("t2")
    gap = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
    inputs = make_inputs(
        {(a, t1): 2.0, (a, t2): 1.0, (b, t1): 1.0, (b, t2): 2.0},
        choices=[choice(a, t1, "dome tent"), choice(b, t1, "dome tent")],
        missing=[unanchored(a, t2, gap), unanchored(b, t2, gap)],
    )
    out = run(inputs)[1]

    assert new_links(out.recommendations) == [
        (a, t1, ActionType.ADD_LINK, 1),
        (b, t1, ActionType.ADD_LINK, 1),
        (b, t2, ActionType.CONTENT_GAP, 1),
    ]
    assert [(u.source_url, u.recommended) for u in out.unanchored] == [(a, False), (b, True)]


def test_a_page_without_links_lists_its_best_gaps_up_to_the_gap_limit() -> None:
    a = url("a")
    t = [url(f"t{i}") for i in range(5)]
    scores = {(a, t[i]): float(i) for i in range(5)}
    gap = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
    inputs = make_inputs(scores, missing=[unanchored(a, x, gap) for x in t])
    records = run(inputs)[1].recommendations

    assert inputs.gap_limit == 3
    assert new_links(records) == [
        (a, t[4], ActionType.CONTENT_GAP, 1),
        (a, t[3], ActionType.CONTENT_GAP, 2),
        (a, t[2], ActionType.CONTENT_GAP, 3),
    ]
    # Score and tier stay the pair's place among all ranked pairs.
    assert [r.score for r in records] == [100.0, 75.0, 50.0]
    assert [r.tier for r in records] == [1, 2, 3]


def test_a_gap_limit_of_zero_lists_no_gap() -> None:
    a, t1, t2 = url("a"), url("t1"), url("t2")
    gap = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
    inputs = make_inputs(
        {(a, t1): 2.0, (a, t2): 1.0},
        choices=[choice(a, t2, "dome tent")],
        missing=[unanchored(a, t1, gap)],
        gap_limit=0,
    )
    out = run(inputs)[1]

    assert new_links(out.recommendations) == [(a, t2, ActionType.ADD_LINK, 1)]
    assert [u.recommended for u in out.unanchored] == [False]


def test_an_add_link_serves_its_anchors_chosen_first_with_placements() -> None:
    a, t = url("a"), url("t")
    inputs = make_inputs(
        {(a, t): 1.0},
        choices=[
            choice(a, t, "backpacking tent", rank=2, total=0.5),
            choice(a, t, "dome tent", kind=AnchorType.EXACT, total=1.07),
            choice(a, t, "tent", rank=3, total=0.2),
        ],
    )
    [record] = run(inputs)[1].recommendations

    assert record.label == ADD_LINK_LABEL
    anchors = record.proposed_anchors or ()
    assert [(x.text, x.anchor_type, x.score) for x in anchors] == [
        ("dome tent", AnchorType.EXACT, 1.0),
        ("backpacking tent", AnchorType.PARTIAL, 0.5),
        ("tent", AnchorType.PARTIAL, 0.2),
    ]
    assert {x.source for x in anchors} == {"EXTRACTED"}
    placement = anchors[0].placement
    assert placement is not None
    assert (placement.start, placement.end, placement.sentence_index) == (211, 220, 4)
    assert anchors[0].keyword == "dome tent"


def test_a_phrase_too_long_to_serve_is_left_out_and_never_counted_unassessed() -> None:
    a, t1, t2 = url("a"), url("t1"), url("t2")
    long = "tent " * (ANCHOR_MAX_CHARS // 5 + 1)
    inputs = make_inputs(
        {(a, t1): 2.0, (a, t2): 1.0},
        choices=[
            choice(a, t1, long.strip()),
            choice(a, t1, "dome tent", rank=2),
            choice(a, t2, long.strip()),
        ],
    )
    walked, out = run(inputs)

    [record] = out.recommendations
    assert [x.text for x in record.proposed_anchors or ()] == ["dome tent"]
    assert (walked.anchors_too_long, walked.not_assessed) == (1, 0)


def test_a_tenant_without_pairs_or_verdicts_gets_an_empty_run() -> None:
    walked, out = run(make_inputs())

    assert (out.recommendations, out.unanchored, out.target_fixes) == ((), (), ())
    assert walked.not_assessed == 0
    assert out.summary.sources_below_limit == 0


# ── score and tier ───────────────────────────────────────────────────────────


def test_score_is_the_percentile_among_all_pairs_ties_averaged() -> None:
    assert percentiles(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == pytest.approx(
        [5 / 6, 0.0, 5 / 6, 1 / 3]
    )
    assert percentiles(np.array([7.0])).tolist() == [0.5]
    assert percentiles(np.array([2.0, 2.0])).tolist() == [0.5, 0.5]

    a, b = url("a"), url("b")
    t = [url(f"t{i}") for i in range(5)]
    scores = {(a, t[i]): float(i) for i in range(5)} | {(b, t[i]): float(i) for i in range(5)}
    out = run(make_inputs(scores, choices=[choice(s, x, "dome tent") for s, x in scores]))[1]

    by_pair = {(r.source_url, r.target_url): r for r in out.recommendations}
    assert by_pair[(a, t[4])].score == by_pair[(b, t[4])].score == 94.4
    assert by_pair[(a, t[0])].score == 5.6
    # Ten pairs: one in tier 1, three in tier 2; ties go to the lower source url.
    assert by_pair[(a, t[4])].tier == 1
    assert [by_pair[k].tier for k in ((b, t[4]), (a, t[3]), (b, t[3]))] == [2, 2, 2]
    assert by_pair[(a, t[2])].tier == 3
    assert out.summary.tiers == {1: 1, 2: 3, 3: 6}


def test_tiers_follow_the_scorers_rule_on_any_value() -> None:
    frame = pandas.DataFrame(
        {
            "source_url": [url("b"), url("a"), url("a"), url("c")],
            "target_url": [url("x"), url("y"), url("x"), url("x")],
        }
    )
    values = np.array([1.0, 1.0, 5.0, -np.inf])

    # One pair to tier 1, one to tier 2: of the tied pairs, the lower source url.
    assert rank_tiers(values, frame["source_url"], frame["target_url"], (0.25, 0.25)).tolist() == [
        3,
        2,
        1,
        3,
    ]


# ── ids ──────────────────────────────────────────────────────────────────────


def test_ids_are_stable_across_runs_and_differ_across_tenants() -> None:
    a, t = url("a"), url("t")
    audit = (verdict(a, 3, t, ActionType.REMOVE, reasons=("off topic",)),)
    first = run(make_inputs({(a, t): 1.0}, choices=[choice(a, t, "dome tent")], audit=audit))[1]
    again = run(
        make_inputs({(a, t): 1.0}, choices=[choice(a, t, "dome tent")], audit=audit, run_id="run-2")
    )[1]

    assert [r.id for r in first.recommendations] == [r.id for r in again.recommendations]
    assert {r.run_id for r in again.recommendations} == {"run-2"}
    assert [r.id for r in first.recommendations] == [
        recommendation_id(TENANT, ActionType.ADD_LINK, a, t, None),
        recommendation_id(TENANT, ActionType.REMOVE, a, t, 3),
    ]
    assert recommendation_id("test-other", ActionType.ADD_LINK, a, t, None) != (
        first.recommendations[0].id
    )


# ── audit verdicts ───────────────────────────────────────────────────────────


def verdict(
    source: str,
    position: int,
    target: str,
    action: ActionType | None,
    *,
    flags: Iterable[IssueFlag] = (),
    reasons: tuple[str, ...] = ("a reason",),
    **fields: Any,
) -> LinkAuditResult:
    return LinkAuditResult(
        source_url=source,
        position=position,
        target_url=target,
        run_id="audit-1",
        issue_flags=frozenset(flags),
        verdict=action,
        reasons=reasons,
        audited_at=AT,
        **fields,
    )


def test_every_audit_verdict_becomes_a_record_uncapped_with_its_evidence() -> None:
    a, t, canon, gone = url("a"), url("t"), url("canon"), url("gone")
    audit = (
        verdict(
            a, 0, t, ActionType.FIX, flags=(IssueFlag.NOFOLLOW, IssueFlag.BROKEN),
            reasons=("the target is broken", "nofollow"), fix_target=canon,
            anchor_quality_score=40.0, context_relevance=0.7,
        ),
        verdict(
            a, 1, t, ActionType.REANCHOR, flags=(IssueFlag.GENERIC,),
            reasons=("the anchor text is generic", 'a better phrase: "dome tent"'),
            proposed_anchor="dome tent", anchor_quality_score=12.5, keyword_alignment=0.0,
            equity_efficiency=0.4,
        ),
        verdict(a, 2, t, ActionType.REMOVE, flags=(IssueFlag.OFF_TOPIC,), reasons=("off topic",)),
        verdict(a, 3, t, None, reasons=()),
        verdict(a, 4, gone, None, reasons=("not crawled",), unverified=True),
        verdict(a, 5, url("sitemap"), ActionType.REMOVE, reasons=("off topic",)),
        *(verdict(url("b"), i, t, ActionType.REMOVE, reasons=("off topic",)) for i in range(12)),
    )  # fmt: skip
    inputs = make_inputs(
        audit=audit,
        anchors={(a, 0): "Dome Tents", (a, 1): "click here", (a, 2): "   "},
        excluded=excluded(url("sitemap")),
        ranked_keywords={t: ["dome tent", "backpacking tent"]},
        pages=(page("a", language="en"),),
        limit=3,
    )
    records = run(inputs)[1].recommendations
    fix, reanchor, remove = records[:3]

    assert len(records) == 15
    assert (fix.action_type, fix.label, fix.position) == (ActionType.FIX, "fix this link", 0)
    assert fix.issue_flags == (IssueFlag.BROKEN, IssueFlag.NOFOLLOW)
    assert fix.fix_target == canon
    assert fix.current_anchor == "Dome Tents"
    assert fix.rationale == "the target is broken; nofollow"
    assert fix.signals == (("anchor_quality_score", 40.0), ("context_relevance", 0.7))
    assert (fix.score, fix.tier, fix.rank_in_source, fix.proposed_anchors) == (None,) * 4

    assert reanchor.label == "change the anchor text"
    [proposed] = reanchor.proposed_anchors or ()
    assert (proposed.text, proposed.anchor_type, proposed.source) == (
        "dome tent",
        AnchorType.EXACT,
        "EXTRACTED",
    )
    assert (proposed.score, proposed.keyword, proposed.placement) == (None, None, None)
    assert [name for name, _ in reanchor.signals] == [
        "anchor_quality_score",
        "keyword_alignment",
        "equity_efficiency",
    ]

    assert (remove.label, remove.current_anchor, remove.fix_target) == (
        "review this link",
        None,
        None,
    )
    assert {r.created_at for r in records} == {AT}
    assert {r.status.value for r in records} == {"PENDING"}


# ── bridges ──────────────────────────────────────────────────────────────────


def hub_pair(a: int, b: int, *reasons: BridgeReason) -> HubPair:
    return HubPair(
        language="en", hub_a=a, hub_b=b, size_a=3, size_b=3, pages_ab=0, pages_ba=1,
        link_density=0.1, centroid_cosine=0.6, bridge_gap=0.5, reasons=reasons,
    )  # fmt: skip


def bridge(
    source: str, target: str, hub_from: int, hub_to: int, *, rank: int = 1, slot: int = 1
) -> BridgeLink:
    return BridgeLink(
        language="en", hub_from=hub_from, hub_to=hub_to, slot=slot, rank=rank, source_url=source,
        target_url=target, similarity=0.7, reasons=(BridgeReason.SPANNING_TREE,),
    )  # fmt: skip


def test_a_bridge_link_marks_the_new_link_it_became_and_takes_no_slot() -> None:
    a, b, t1, t2, t3 = url("a"), url("b"), url("t1"), url("t2"), url("t3")
    inputs = make_inputs(
        {(a, t1): 2.0, (a, t2): 1.0},
        choices=[choice(a, t1, "dome tent"), choice(a, t2, "camp stove")],
        hub_pairs=(
            hub_pair(0, 1, BridgeReason.SPANNING_TREE),
            hub_pair(0, 2),
            hub_pair(1, 2),
            hub_pair(2, 3, BridgeReason.NEAREST_HUB),
        ),
        bridge_links=(
            bridge(a, t1, 0, 1),
            bridge(b, t3, 2, 1),
            bridge(a, url("sitemap"), 1, 2, rank=2),
        ),
        excluded=excluded(url("sitemap")),
        limit=1,
    )
    out = run(inputs)[1]

    [record] = out.recommendations
    assert record.bridge is not None
    assert (record.bridge.hub_from, record.bridge.hub_to, record.bridge.reasons) == (
        0,
        1,
        (BridgeReason.SPANNING_TREE,),
    )
    assert [(p.hub_a, p.hub_b) for p in out.bridges] == [(0, 1), (1, 2), (2, 3)]
    links = {(x.source_url, x.target_url): x for p in out.bridges for x in p.links}
    assert set(links) == {(a, t1), (b, t3)}
    assert links[(a, t1)].recommendation_id == record.id
    assert links[(b, t3)].recommendation_id is None
    assert (out.summary.bridge_pairs, out.summary.bridge_links) == (3, 2)


def test_bridge_links_count_one_proposal_per_slot_and_serve_the_alternatives() -> None:
    a, b = url("a"), url("b")
    t = [url(f"t{i}") for i in range(3)]
    inputs = make_inputs(
        hub_pairs=(hub_pair(0, 1, BridgeReason.SPANNING_TREE),),
        bridge_links=(
            *(bridge(a, t[i], 0, 1, rank=i + 1) for i in range(3)),
            *(bridge(b, t[i], 0, 1, rank=i + 1, slot=2) for i in range(2)),
        ),
    )
    out = run(inputs)[1]

    [pair] = out.bridges
    assert [(x.slot, x.rank) for x in pair.links] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    assert (out.summary.bridge_pairs, out.summary.bridge_links) == (1, 2)


def test_bridge_links_of_an_unknown_hub_pair_fail_the_run() -> None:
    inputs = make_inputs(
        hub_pairs=(hub_pair(0, 1),), bridge_links=(bridge(url("a"), url("b"), 4, 5),)
    )
    with pytest.raises(ValueError, match="rerun hub-bridges"):
        run(inputs)


# ── target fixes ─────────────────────────────────────────────────────────────


def test_target_fixes_count_the_waiting_sources_and_keep_the_best_five() -> None:
    x, y = url("x"), url("y")
    sources = [url(f"s{i}") for i in range(8)]
    scores = {(s, x): float(i) for i, s in enumerate(sources)} | {
        (sources[0], y): 1.0,
        (sources[1], y): 1.0,
    }
    reason = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
    inputs = make_inputs(
        scores,
        missing=[unanchored(s, t, reason) for s, t in scores],
        titles={x: "Trail shoes | Acme"},
        excluded=excluded(sources[7]),
    )
    out = run(inputs)[1]

    assert out.recommendations == ()
    first, second = out.target_fixes
    assert (first.target_url, first.waiting_sources, first.title) == (x, 7, "Trail shoes | Acme")
    assert first.best_sources == tuple(sources[6:1:-1])
    assert first.fix == UNANCHORED_ADVICE[reason]
    assert (second.target_url, second.waiting_sources, second.best_sources) == (
        y,
        2,
        (sources[0], sources[1]),
    )
    assert out.summary.target_fixes == 2


# ── profiles, hubs, duplicates ───────────────────────────────────────────────


def test_profiles_hubs_and_duplicates_read_the_graph_facts() -> None:
    a, t = url("a"), url("t")
    pages = (
        page("a", language="en", hub_id=0, is_hub_pillar=True, word_count=900),
        page("t", language="en", hub_id=0, is_orphan=True, orphan_label=OrphanLabel.MENUS_ONLY,
             duplicate_group=4, is_canonical=True),
        page("t-copy", language="de", hub_id=0, is_dead_end=True, duplicate_group=4,
             is_canonical=False),
        page("noise", hub_id=-1, duplicate_group=5, is_canonical=True),
        page("two", duplicate_group=5, is_canonical=True),
        page("lone", hub_id=1, duplicate_group=6, is_canonical=True),
        page("sitemap", hub_id=1, duplicate_group=6, is_canonical=False),
        page("other", hub_id=2, language="en"),
    )  # fmt: skip
    inbound = [
        InboundAnchorText(target_url=t, source_language="en", anchor_text=text, links=n)
        for text, n in (
            ("Dome tent", 2),
            ("tent pegs", 1),
            ("our favourite shelter", 1),
            ("Acme tents", 3),
            ("click here", 5),
            ("  ", 1),
        )
    ]
    inputs = make_inputs(
        {(a, t): 1.0},
        choices=[choice(a, t, "dome tent")],
        pages=pages,
        hubs=(
            HubNode(hub_id=0, size=3, pillar_url=a, active=True),
            HubNode(hub_id=1, size=2, pillar_url=url("sitemap"), active=True),
            HubNode(hub_id=2, size=1, pillar_url=url("other"), active=True),
            HubNode(hub_id=3, size=0, active=False),
        ),
        hub_pairs=(
            hub_pair(0, 2, BridgeReason.NEAREST_HUB),
            hub_pair(0, 1, BridgeReason.BRIDGE_GAP),
        ),
        titles={a: "Dome tents | Acme"},
        keywords={t: ("dome tent", KeywordRung.H1)},
        ranked_keywords={t: ["dome tent", "backpacking tent"]},
        inbound=inbound,
        audit=(verdict(a, 0, t, ActionType.REMOVE, reasons=("off topic",)),),
        excluded=excluded(url("sitemap")),
    )
    with capture_logs() as logs:
        out = run(inputs)[1]

    profiles = {p.url: p for p in out.pages}
    assert url("sitemap") not in profiles
    assert [p.url for p in out.pages] == sorted(profiles)
    target = profiles[t]
    assert target.anchor_mix == AnchorMix(exact=2, partial=1, natural=1, branded=3)
    assert (target.target_keyword, target.keyword_rung) == ("dome tent", KeywordRung.H1)
    assert (target.recommendations_in, target.recommendations_out) == (1, 0)
    assert (profiles[a].recommendations_out, profiles[a].audit_verdicts_out) == (1, 1)
    assert profiles[a].title == "Dome tents | Acme"
    assert profiles[url("noise")].hub_id is None

    hubs = {h.hub_id: h for h in out.hubs}
    assert list(hubs) == [0, 1, 2]
    assert (hubs[0].language, hubs[0].orphan_pages, hubs[0].dead_end_pages) == (None, 1, 1)
    assert (hubs[0].pillar_url, hubs[0].pillar_title) == (a, "Dome tents | Acme")
    assert (hubs[0].recommendations_in, hubs[0].bridge_hubs) == (1, (1, 2))
    assert (hubs[1].pillar_url, hubs[1].bridge_hubs) == (None, (0,))
    assert hubs[2].language == "en"

    [group] = out.duplicates
    assert (group.group_id, group.canonical, group.copies) == (4, t, (url("t-copy"),))
    skipped = [e for e in logs if e["event"] == "recommendations.duplicate_group_skipped"]
    assert [(e["group"], e["canonicals"], e["copies"]) for e in skipped] == [(5, 2, 0), (6, 1, 0)]
    assert out.summary.pages == 7
    assert out.summary.orphan_pages == {OrphanLabel.MENUS_ONLY: 1}
    assert (out.summary.duplicate_groups, out.summary.duplicate_copies, out.summary.hubs) == (
        1,
        1,
        3,
    )
    assert out.summary.excluded_pages == {ExclusionReason.SITEMAP: 1}


# ── stable orders ────────────────────────────────────────────────────────────


def test_every_listing_is_in_its_stable_order() -> None:
    a, b = url("a"), url("b")
    t = [url(f"t{i}") for i in range(4)]
    reason = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
    scores = {(b, t[1]): 4.0, (b, t[0]): 3.0, (a, t[2]): 2.0, (a, t[3]): 1.0, (a, t[0]): 0.5}
    scores |= {(b, t[3]): 0.1}
    inputs = make_inputs(
        scores,
        choices=[choice(b, t[0], "dome tent"), choice(a, t[3], "camp stove")],
        missing=[
            unanchored(b, t[1], UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC),
            unanchored(a, t[2], reason),
            unanchored(a, t[0], reason),
            unanchored(b, t[3], reason),
        ],
        audit=(
            verdict(a, 7, t[1], ActionType.REMOVE, reasons=("off topic",)),
            verdict(a, 2, t[0], ActionType.FIX, reasons=("broken",)),
            verdict(a, 2, t[2], ActionType.REMOVE, reasons=("off topic",)),
            verdict(b, 0, t[2], ActionType.REMOVE, reasons=("off topic",)),
        ),
        hub_pairs=(
            hub_pair(1, 2, BridgeReason.NEAREST_HUB),
            hub_pair(0, 3, BridgeReason.NEAREST_HUB),
        ),
        pages=(page("t1"), page("a"), page("b")),
        hubs=(HubNode(hub_id=2, size=1, active=True), HubNode(hub_id=0, size=1, active=True)),
    )
    out = run(inputs)[1]

    assert [(r.source_url, r.action_type, r.rank_in_source or r.position, r.target_url)
            for r in out.recommendations] == [
        (a, ActionType.ADD_LINK, 1, t[3]),
        (a, ActionType.FIX, 2, t[0]),
        (a, ActionType.REMOVE, 2, t[2]),
        (a, ActionType.REMOVE, 7, t[1]),
        (b, ActionType.ADD_LINK, 1, t[0]),
        (b, ActionType.CONTENT_GAP, 1, t[1]),
        (b, ActionType.REMOVE, 0, t[2]),
    ]  # fmt: skip
    assert [p.url for p in out.pages] == [a, b, t[1]]
    assert [h.hub_id for h in out.hubs] == [0, 2]
    assert [(p.hub_a, p.hub_b) for p in out.bridges] == [(0, 3), (1, 2)]
    assert [(u.source_url, u.rank_in_source) for u in out.unanchored] == [
        (a, 1),
        (a, 3),
        (b, 1),
        (b, 3),
    ]
    assert [(f.target_url, f.waiting_sources) for f in out.target_fixes] == [
        (t[0], 1),
        (t[2], 1),
        (t[3], 1),
    ]


# ── signals ──────────────────────────────────────────────────────────────────


def test_a_rationale_names_the_scorer_the_rank_and_the_signals_in_plain_words() -> None:
    a, t = url("a"), url("t")
    signals = {
        (a, t): (("content_cosine", 0.4), ("same_hub", 0.2), ("target_saturation_ratio", -0.1))
    }
    [record] = run(make_inputs({(a, t): 1.0}, choices=[choice(a, t, "dome tent")]), signals)[
        1
    ].recommendations

    assert record.signals == signals[(a, t)]
    assert record.rationale == (
        "Ranked 1 of the 1 candidate targets of this page by the baseline scorer; strongest "
        "signals: content cosine, same hub and target saturation ratio (against)."
    )


def test_top_contributions_keep_the_sign_largest_first_and_skip_zeros() -> None:
    names = ["a", "b", "c", "d", "e"]
    found = top_contributions(names, np.array([0.1, -0.5, 0.0, 0.3, -0.3]))
    assert found == (("b", -0.5), ("d", 0.3), ("e", -0.3))
    assert top_contributions(names[:2], np.array([0.0, 0.2])) == (("b", 0.2),)


def matrix_frame(pairs: int, columns: Sequence[str], seed: int = 3) -> pandas.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "source_url": [url(f"s{i % 4}") for i in range(pairs)],
            "target_url": [url(f"t{i}") for i in range(pairs)],
        }
    )
    for column in columns:
        values = rng.random(pairs)
        values[rng.random(pairs) < 0.1] = np.nan
        frame[column] = values
    return frame


def test_baseline_signals_rescore_the_whole_matrix_as_rank_pairs_did(tmp_path: Path) -> None:
    weights = default_weights()
    columns = [f.column for f in weights.features]
    frame = matrix_frame(40, columns)
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    scored = score_frame(frame, weights)
    emitted = scored.loc[[3, 17, 25], ["source_url", "target_url", "score"]]

    found = baseline_signals(path, weights, emitted)

    for row in (3, 17, 25):
        key = (scored.at[row, "source_url"], scored.at[row, "target_url"])
        assert found[key] == tuple(
            (scored.at[row, f"top{k}_feature"], round(scored.at[row, f"top{k}_contribution"], 6))
            for k in (1, 2, 3)
        )
    moved = emitted.assign(score=emitted["score"] + 1.0)
    with pytest.raises(ValueError, match="rerun rank-pairs"):
        baseline_signals(path, weights, moved)
    stranger = pandas.DataFrame([{"source_url": url("x"), "target_url": url("y"), "score": 1.0}])
    with pytest.raises(ValueError, match="rerun rank-pairs"):
        baseline_signals(path, weights, stranger)


def trained_holder(frame: pandas.DataFrame, columns: Sequence[str], version: str = "3") -> Holder:
    rng = np.random.default_rng(5)
    booster = lightgbm.train(
        {"objective": "regression", "verbose": -1, "num_leaves": 4, "min_data_in_leaf": 2},
        lightgbm.Dataset(
            frame.loc[:, list(columns)].to_numpy(dtype=np.float32, na_value=np.nan),
            label=rng.random(len(frame)),
            feature_name=list(columns),
        ),
        num_boost_round=5,
    )
    return Holder(
        version=version,
        run_id="mlflow-run",
        booster=booster,
        columns=tuple(columns),
        feature_set_version=None,
        split_seed=None,
        test_share=None,
        valid_share=None,
    )


def test_learned_signals_are_the_models_contributions_of_the_emitted_rows(tmp_path: Path) -> None:
    columns = ["content_cosine", "same_hub", "target_is_orphan", "pair_kw_overlap"]
    frame = matrix_frame(60, columns)
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    holder = trained_holder(frame, columns)
    values = frame.loc[:, columns].to_numpy(dtype=np.float32, na_value=np.nan)
    scores = holder.booster.predict(values)
    emitted = frame.loc[[2, 30], list(KEY_COLUMNS)].assign(score=scores[[2, 30]])

    found = learned_signals(path, holder, emitted)

    contributions = holder.booster.predict(values[[2, 30]], pred_contrib=True)
    for row, contribution in zip((2, 30), contributions, strict=True):
        key = (frame.at[row, "source_url"], frame.at[row, "target_url"])
        assert found[key] == top_contributions(columns, contribution[:-1])
    with pytest.raises(ValueError, match="rerun rank-pairs"):
        learned_signals(path, holder, emitted.assign(score=emitted["score"] + 1.0))
    stranger = pandas.DataFrame([{"source_url": url("x"), "target_url": url("y"), "score": 1.0}])
    with pytest.raises(ValueError, match="rerun rank-pairs"):
        learned_signals(path, holder, pandas.concat([emitted, stranger]))
    shuffled = dataclasses.replace(holder, columns=tuple(reversed(columns)))
    with pytest.raises(ValueError, match="columns differ"):
        learned_signals(path, shuffled, emitted)


async def test_learned_signals_need_the_model_that_ranked_the_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    columns = ["content_cosine", "same_hub"]
    frame = matrix_frame(20, columns)
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    holder = trained_holder(frame, columns, version="7")
    scores = holder.booster.predict(frame.loc[:, columns].to_numpy(np.float32, na_value=np.nan))
    keys = zip(frame["source_url"], frame["target_url"], strict=True)
    pairs = dict(zip(keys, scores.tolist(), strict=True))
    inputs = make_inputs(
        pairs,
        choices=[choice(s, t, "dome tent") for s, t in pairs],
        scorer=ScorerName.LEARNED,
        limit=2,
    )
    walked = walk(inputs)

    async def features(*_: object, **__: object) -> tuple[None, Path]:
        return None, path

    monkeypatch.setattr(recommendations, "assemble_features", features)
    stores = cast("Any", None)
    for found in (None, dataclasses.replace(holder, version="6")):
        monkeypatch.setattr(recommendations, "load_production", lambda *_, found=found: found)
        with pytest.raises(ValueError, match=MODEL_CHANGED):
            await recommendations._signals(
                stores, stores, inputs, walked, "7", default_weights(), cache_dir=tmp_path
            )
    monkeypatch.setattr(recommendations, "load_production", lambda *_: holder)
    signals, matrix = await recommendations._signals(
        stores, stores, inputs, walked, "7", default_weights(), cache_dir=tmp_path
    )

    assert matrix == path
    assert set(signals) == set(
        zip(walked.emitted["source_url"], walked.emitted["target_url"], strict=True)
    )
    assert len(signals) == 8


# ── guards ───────────────────────────────────────────────────────────────────


def write_ranked(path: Path, rows: list[dict[str, object]], tenant: str = TENANT) -> None:
    table = pa.Table.from_pylist(rows, schema=RANKED_SCHEMA.with_metadata({"tenant_id": tenant}))
    pq.write_table(table, path)


def ranked_row(
    target: str, scorer: str = "baseline", model: str | None = None
) -> dict[str, object]:
    return {
        "source_url": url("a"),
        "target_url": target,
        "score": 1.0,
        "rank_in_source": 1,
        "scorer": scorer,
        "model_version": model,
    }


def test_the_ranked_file_names_its_scorer_and_belongs_to_the_tenant(tmp_path: Path) -> None:
    path = tmp_path / RANKED_PAIRS_FILE
    write_ranked(path, [ranked_row(url("t"), "learned", "4")])
    frame, scorer, model = read_ranked(path, TENANT)
    assert (len(frame), scorer, model) == (1, ScorerName.LEARNED, "4")

    write_ranked(path, [])
    assert read_ranked(path, TENANT)[1:] == (ScorerName.BASELINE, None)

    write_ranked(path, [ranked_row(url("t"))], tenant="test-other")
    with pytest.raises(ValueError, match="another tenant"):
        read_ranked(path, TENANT)
    write_ranked(path, [ranked_row(url("t")), ranked_row(url("u"), "learned", "4")])
    with pytest.raises(ValueError, match="mixes scorers"):
        read_ranked(path, TENANT)
    write_ranked(path, [ranked_row(url("t")), ranked_row(url("t"))])
    with pytest.raises(ValueError, match="repeats a pair"):
        read_ranked(path, TENANT)


async def test_a_missing_stage_file_names_the_flow_to_run_before_any_read(tmp_path: Path) -> None:
    stores = cast("Any", None)
    folder = tmp_path / TENANT
    folder.mkdir()
    with pytest.raises(ValueError, match="run rank-pairs first"):
        await publish_recommendations(stores, stores, stores, TENANT, cache_dir=tmp_path)
    write_ranked(folder / RANKED_PAIRS_FILE, [])
    with pytest.raises(ValueError, match="run anchor-selection first"):
        await publish_recommendations(stores, stores, stores, TENANT, cache_dir=tmp_path)


def test_the_logged_summary_and_metrics_hold_no_url() -> None:
    a, t = url("a"), url("t")
    out = run(
        make_inputs(
            {(a, t): 1.0},
            choices=[choice(a, t, "dome tent")],
            pages=(page("a"), page("t")),
            excluded=excluded(url("sitemap")),
        )
    )[1]
    report = RecommendationReport(
        tenant_id=TENANT,
        run_id="run-1",
        scorer=ScorerName.BASELINE,
        limit_per_source=10,
        content_gap_limit=3,
        summary=out.summary,
        pairs_not_assessed=3,
        seconds=1.5,
        finished_at=AT,
    )
    text = summarise_recommendations(report)
    metrics = recommendation_metrics(report)
    fixture = {a, t, url("sitemap")}

    words = {word.strip(".,;:()") for word in text.split()}
    assert not words & fixture
    assert not set(metrics) & fixture
    assert metrics["action_add_link"] == 1.0
    assert metrics["excluded_sitemap"] == 1.0
    assert metrics["pairs_not_assessed"] == 3.0
    assert "ADD_LINK 1" in text


# ── one run through the stores ───────────────────────────────────────────────

A, T, U, S = url("guide"), url("dome-tents"), url("camp-stoves"), url("sitemap")
STORED_PAGES: tuple[dict[str, object], ...] = (
    {"url": A, "language": "en", "pageType": "ARTICLE", "wordCount": 900, "crawlDepth": 1,
     "pageRankPercentile": 0.9, "hubId": 0, "isHubPillar": True},
    {"url": T, "language": "en", "pageType": "CATEGORY", "wordCount": 400, "crawlDepth": 2,
     "hubId": 0, "isOrphan": True, "orphanLabel": "MENUS_ONLY"},
    {"url": U, "language": "en", "wordCount": 300, "hubId": -1, "isDeadEnd": True},
)  # fmt: skip


async def seed_tenant(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, folder: Path, *, audited: bool = True
) -> None:
    await graph._auto(
        "UNWIND $pages AS page CREATE (p:Page) SET p = page, p.tenantId = $tenant",
        pages=list(STORED_PAGES),
        tenant=tenant,
    )
    await graph._auto(
        "UNWIND $links AS link "
        "MATCH (s:Page {tenantId: $tenant, url: link[0]}), (t:Page {tenantId: $tenant, url: link[1]}) "
        "CREATE (s)-[:LINKS_TO {position: link[2], anchorText: link[3]}]->(t)",
        links=[[A, T, 0, "Dome tent"], [U, T, 0, "click here"]],
        tenant=tenant,
    )
    await graph._auto(
        "CREATE (:Hub {tenantId: $tenant, hubId: 0, size: 2, pillarUrl: $pillar, active: true}), "
        "(:Hub {tenantId: $tenant, hubId: 1, size: 0, active: false}) "
        "WITH 1 AS one MATCH (p:Page {tenantId: $tenant, url: $target}) "
        "CREATE (p)-[:TARGETS_KEYWORD {rank: 1, rung: 'H1', source: 'INFERRED'}]->"
        "(:Keyword {tenantId: $tenant, text: 'dome tent', language: 'en'})",
        tenant=tenant,
        pillar=A,
        target=T,
    )
    await mongo._db["pages"].insert_many(
        [{"tenantId": tenant, "url": A, "metaTitle": "Kit guide | Acme"}]
    )
    await mongo._db["links"].insert_one(
        {"tenantId": tenant, "sourceUrl": A, "position": 0, "targetUrl": T,
         "anchorText": "Dome tent", "surroundingText": "Pick a dome tent.", "isInternal": True}
    )  # fmt: skip
    await mongo.replace_excluded_pages(tenant, excluded(S))
    if audited:
        result = verdict(A, 0, T, ActionType.REMOVE, reasons=("off topic",))
        await mongo.insert_link_audit(tenant, "audit-1", [result])
        await mongo.complete_link_audit(tenant, "audit-1", audited_at=AT, documents=1, edges=1)

    write_stage_files(folder, tenant)


def write_stage_files(folder: Path, tenant: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    weights = default_weights()
    frame = matrix_frame(5, [f.column for f in weights.features])
    frame["source_url"] = [A, A, A, U, U]
    frame["target_url"] = [T, U, S, T, A]
    # (A, U) scores best on every feature, so its content gap ranks above A's link.
    for feature in weights.features:
        frame.loc[1, feature.column] = 1.0 if feature.direction == "higher" else 0.0
    frame.to_parquet(folder / "matrix.parquet", index=False)
    scores = score_frame(frame, weights)["score"].tolist()
    order = ranked(
        dict(zip(zip(frame["source_url"], frame["target_url"], strict=True), scores, strict=True))
    )
    write_ranked(
        folder / RANKED_PAIRS_FILE,
        [{**row, "scorer": "baseline", "model_version": None} for row in order.to_dict("records")],
        tenant=tenant,
    )
    rows = [
        choice(A, T, "dome tent", kind=AnchorType.EXACT),
        choice(U, T, "tent"),
        choice(A, S, "map"),
    ]
    stored = {
        "keyword_rank": 1,
        "keyword_source": "INFERRED",
        "rung": "EXACT",
        "sentence_start": 200,
    }
    stored |= {f"score_{part}": 0.5 for part in ("keyword", "diversity", "length", "rank_weight")}
    pq.write_table(
        pa.Table.from_pylist(
            [{**row, **stored, "score_profile_bonus": 0.0} for row in rows], schema=CHOICES_SCHEMA
        ),
        folder / "anchor_choices.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                unanchored(A, U, UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC),
                unanchored(U, A, UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD),
            ],
            schema=UNANCHORED_SCHEMA,
        ),
        folder / "unanchored_pairs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([hub_pair(0, 1, BridgeReason.SPANNING_TREE).model_dump(mode="json")]),
        folder / "hub_pairs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([bridge(A, T, 0, 1).model_dump(mode="json")]),
        folder / "bridges.parquet",
    )


async def output_of(writer: OutputWriter, tenant: str) -> dict[str, list[dict[str, object]]]:
    found: dict[str, list[dict[str, object]]] = {}
    for name in (*RUN_SCOPED, RUNS):
        cursor = writer._db[name].find({"tenantId": tenant}, {"_id": 0}).sort("ordinal", 1)
        found[name] = await cursor.to_list()
    return found


@pytest.mark.integration
async def test_a_run_is_published_through_the_stores_and_replaces_the_previous(
    graph: GraphRepo,
    mongo: MongoRepo,
    mongo_uri: str,
    tenant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = f"{tenant}-other"
    for name in (tenant, other):
        await seed_tenant(graph, mongo, name, tmp_path / name)

    async def features(_: object, __: object, name: str, **___: object) -> tuple[None, Path]:
        return None, tmp_path / name / "matrix.parquet"

    monkeypatch.setattr(recommendations, "assemble_features", features)
    async with await OutputWriter.connect(mongo_uri, "linking_engine_test") as writer:
        await publish_recommendations(graph, mongo, writer, other, cache_dir=tmp_path)
        before = await output_of(writer, other)
        first = await publish_recommendations(graph, mongo, writer, tenant, cache_dir=tmp_path)
        ids = [d["id"] for d in (await output_of(writer, tenant))[RECOMMENDATIONS]]
        second = await publish_recommendations(graph, mongo, writer, tenant, cache_dir=tmp_path)
        stored = await output_of(writer, tenant)

        assert first.run_id != second.run_id
        assert {d["runId"] for docs in stored.values() for d in docs} == {second.run_id}
        records = [from_document(Recommendation, d) for d in stored[RECOMMENDATIONS]]
        assert [d["id"] for d in stored[RECOMMENDATIONS]] == ids
        assert [d["ordinal"] for d in stored[RECOMMENDATIONS]] == list(range(len(records)))
        assert [(r.source_url, r.action_type) for r in records] == [
            (U, ActionType.ADD_LINK),
            (A, ActionType.ADD_LINK),
            (A, ActionType.CONTENT_GAP),
            (A, ActionType.REMOVE),
        ]
        by_pair = {(r.source_url, r.target_url, r.action_type): r for r in records}
        assert set(by_pair) == {
            (A, T, ActionType.ADD_LINK),
            (A, U, ActionType.CONTENT_GAP),
            (A, T, ActionType.REMOVE),
            (U, T, ActionType.ADD_LINK),
        }
        assert by_pair[(A, T, ActionType.ADD_LINK)].bridge is not None
        assert by_pair[(U, T, ActionType.ADD_LINK)].bridge is None
        assert by_pair[(A, T, ActionType.REMOVE)].current_anchor == "Dome tent"
        profiles = {p.url: p for p in (from_document(PageProfile, d) for d in stored[PAGES])}
        assert set(profiles) == {A, T, U}
        assert profiles[T].anchor_mix == AnchorMix(exact=1)
        assert (profiles[T].target_keyword, profiles[T].orphan_label) == (
            "dome tent",
            OrphanLabel.MENUS_ONLY,
        )
        assert profiles[A].title == "Kit guide | Acme"
        assert profiles[U].hub_id is None
        assert [d["hub_id"] for d in stored[HUBS]] == [0]
        [fix] = stored[TARGET_FIXES]
        assert fix["target_url"] == A

        [run_doc] = stored[RUNS]
        info = from_document(RunInfo, run_doc)
        assert (info.status, info.scorer, info.link_audit_run_id) == (
            "complete",
            ScorerName.BASELINE,
            "audit-1",
        )
        assert set(info.inputs) == {
            RANKED_PAIRS_FILE,
            "anchor_choices.parquet",
            "unanchored_pairs.parquet",
            "hub_pairs.parquet",
            "bridges.parquet",
            "matrix.parquet",
            "link_audit",
        }
        assert info.summary == second.summary
        assert second.summary.excluded_pages == {ExclusionReason.SITEMAP: 1}
        # The other tenant's identical urls were neither read nor rewritten.
        assert await output_of(writer, other) == before

        unaudited = f"{tenant}-unaudited"
        await seed_tenant(graph, mongo, unaudited, tmp_path / unaudited, audited=False)
        with pytest.raises(ValueError, match="run link-audit first"):
            await publish_recommendations(graph, mongo, writer, unaudited, cache_dir=tmp_path)
        assert await output_of(writer, unaudited) == {name: [] for name in (*RUN_SCOPED, RUNS)}


async def test_signals_need_a_new_link_and_a_scorer_that_ranks(tmp_path: Path) -> None:
    stores = cast("Any", None)
    a, t = url("a"), url("t")
    empty = make_inputs()
    assert await recommendations._signals(
        stores, stores, empty, walk(empty), None, default_weights(), cache_dir=tmp_path
    ) == ({}, None)

    held = make_inputs({(a, t): 1.0}, choices=[choice(a, t, "dome tent")], scorer=ScorerName.HOLDER)
    with pytest.raises(ValueError, match="ranked by holder"):
        await recommendations._signals(
            stores, stores, held, walk(held), None, default_weights(), cache_dir=tmp_path
        )


def test_anchor_files_that_repeat_a_pair_are_refused(tmp_path: Path) -> None:
    a, t = url("a"), url("t")
    choices, missing = tmp_path / "choices.parquet", tmp_path / "missing.parquet"
    pandas.DataFrame([choice(a, t, "dome tent"), choice(a, t, "tent")]).to_parquet(choices)
    pandas.DataFrame(
        [unanchored(a, url("u"), UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC)],
        columns=list(UNANCHORED_COLUMNS),
    ).to_parquet(missing)

    with pytest.raises(ValueError, match="rerun anchor-selection"):
        recommendations._read_anchor_files(choices, missing)


@pytest.mark.integration
async def test_the_stage_readers_are_tenant_scoped_and_refuse_data_that_does_not_fit(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed_tenant(graph, mongo, tenant, tmp_path / tenant)
    await mongo._db["pages"].insert_one({"tenantId": other, "url": A, "metaTitle": "Other | Co"})

    assert await mongo.page_titles_by_url(tenant) == {A: "Kit guide | Acme"}
    assert [hub.hub_id for hub in await graph.hub_nodes(tenant)] == [0, 1]
    assert [page.url for page in await graph.page_facts(tenant)] == sorted([A, T, U])
    assert await graph.page_facts(other) == []
    # Generic anchors are only left out once the tenant's own list is applied.
    assert [
        (found.target_url, found.anchor_text, found.links)
        for found in await graph.inbound_anchor_texts(tenant)
    ] == [(T, "Dome tent", 1), (T, "click here", 1)]

    await mongo._db["pages"].insert_one({"tenantId": tenant, "url": U, "metaTitle": 7})
    with pytest.raises(DatabaseReadError, match="title 7"):
        await mongo.page_titles_by_url(tenant)
    await graph._auto(
        "MATCH (p:Page {tenantId: $tenant, url: $url}) SET p.orphanLabel = 'NOT_LINKED'",
        tenant=tenant,
        url=A,
    )
    with pytest.raises(DatabaseReadError, match="page facts"):
        await graph.page_facts(tenant)
    await graph._auto("CREATE (:Hub {tenantId: $tenant, hubId: -3})", tenant=tenant)
    with pytest.raises(DatabaseReadError, match="hubs"):
        await graph.hub_nodes(tenant)
    await graph._auto(
        "MATCH (s:Page {tenantId: $tenant, url: $a}), (t:Page {tenantId: $tenant, url: $u}) "
        "CREATE (s)-[:LINKS_TO {position: 1}]->(t)",
        tenant=tenant,
        a=A,
        u=U,
    )
    with pytest.raises(DatabaseReadError, match="inbound anchors"):
        await graph.inbound_anchor_texts(tenant)


def test_a_reanchor_without_a_servable_phrase_is_still_a_record() -> None:
    a, t = url("a"), url("t")
    reasons = ("the anchor text is generic", "no better phrase for the target in the copy")
    audit = (
        verdict(a, 0, t, ActionType.REANCHOR, flags=(IssueFlag.GENERIC,), reasons=reasons),
        verdict(
            a, 1, t, ActionType.REANCHOR, flags=(IssueFlag.OVER_OPTIMISED,),
            reasons=("the anchor is over-optimised", "a better phrase"),
            proposed_anchor="tent " * (ANCHOR_MAX_CHARS // 5 + 1),
        ),
    )  # fmt: skip
    none, too_long = run(make_inputs(audit=audit, anchors={(a, 0): "click here"}))[
        1
    ].recommendations

    assert (none.action_type, none.label, none.proposed_anchors) == (
        ActionType.REANCHOR,
        "change the anchor text",
        None,
    )
    assert (
        none.rationale == "the anchor text is generic; no better phrase for the target in the copy"
    )
    assert (none.current_anchor, none.issue_flags) == ("click here", (IssueFlag.GENERIC,))
    assert too_long.proposed_anchors is None
