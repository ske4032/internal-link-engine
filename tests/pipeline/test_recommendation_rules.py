"""The walk's rules where a filter or a tie-break decides: the hub rule and the eligible sources
of an orphan slot, why a page is left short, the order of the cap, the guarantees and
best_rank on tied scores, and the summary's edge cases."""

from __future__ import annotations

import dataclasses
from typing import Any

from test_recommendations import (
    choice,
    excluded,
    make_inputs,
    page,
    run,
    slots,
    unanchored,
    url,
)

from linking_engine.models import ActionType, OrphanSlotReason, UnanchoredReason
from linking_engine.models.page import HubNode
from linking_engine.pipeline import recommendations
from linking_engine.pipeline.recommendations import walk

ORPHAN: dict[str, Any] = {"is_orphan": True, "inbound": 0}

# ── the eligible sources of a slot ───────────────────────────────────────────


def test_a_slot_into_a_page_in_a_hub_skips_a_better_source_from_another_hub() -> None:
    other, same, x, o = url("s-other"), url("s-same"), url("x"), url("o")
    scores = {(other, x): 10.0, (same, x): 9.0, (other, o): 8.0, (same, o): 7.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            page("s-other", hub_id=1, outbound=0),
            page("s-same", hub_id=0, outbound=0),
            page("x"),
            page("o", hub_id=0, inbound=0),
        ),
        tier_shares=(0.5, 0.5),
        guaranteed_inbound_links=1,
    )
    out = run(inputs)[1]

    # s-other scores best into o but is in another hub, so the slot is s-same's.
    assert slots(out.recommendations) == [
        (other, x, 1, True, False),
        (other, o, 2, False, False),
        (same, x, 1, False, False),
        (same, o, 2, True, True),
    ]
    [rescue] = out.orphans
    assert ([r.source_url for r in rescue.sources], rescue.unmet_reason) == ([same], None)


def test_a_page_in_noise_or_in_no_hub_takes_slots_from_sources_in_any_hub() -> None:
    h0, h1, h2, noise = url("s-h0"), url("s-h1"), url("s-h2"), url("s-noise")
    x, in_noise, free = url("x"), url("o-noise"), url("o-free")
    scores = {(h1, x): 10.0, (h0, x): 9.5, (h2, x): 9.0, (noise, x): 8.5}
    scores |= {(h1, in_noise): 8.0, (h0, in_noise): 7.0, (h2, free): 6.0, (noise, free): 5.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            page("s-h0", hub_id=0, outbound=0),
            page("s-h1", hub_id=1, outbound=0),
            page("s-h2", hub_id=2, outbound=0),
            page("s-noise", hub_id=-1, outbound=0),
            page("x"),
            page("o-noise", hub_id=-1, inbound=0),
            page("o-free", inbound=0),
        ),
        tier_shares=(0.5, 0.5),
    )
    out = run(inputs)[1]

    # Noise is no hub: o-noise takes its slots from hubs 0 and 1, o-free from hub 2 and noise.
    placed = {(r.source_url, r.target_url) for r in out.recommendations if r.orphan_slot}
    assert placed == {(h1, in_noise), (h0, in_noise), (h2, free), (noise, free)}
    assert {r.profile.url: (r.suggested_in, r.unmet_reason) for r in out.orphans} == {
        in_noise: (2, None),
        free: (2, None),
    }


def test_a_source_whose_suggestions_all_go_into_guaranteed_pages_gives_no_slot() -> None:
    s1, s2, s3 = url("s1"), url("s2"), url("s3")
    x, o1, o2, o3 = url("x"), url("o1"), url("o2"), url("o3")
    scores = {(s1, o1): 9.0, (s1, o2): 8.0, (s2, x): 7.0, (s2, o2): 6.0}
    scores |= {(s3, o1): 5.0, (s3, o3): 4.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            *(page(s, outbound=0) for s in ("s1", "s2", "s3")),
            page("x"),
            *(page(o, inbound=0) for o in ("o1", "o2", "o3")),
        ),
        tier_shares=(0.5, 0.5),
        guaranteed_inbound_links=1,
    )
    out = run(inputs)[1]

    # s1 is o2's best source and s3 o3's only one, but each suggests only o1, and a link into
    # a guaranteed page is never given up: o2 takes s2 and o3 is left short.
    assert slots(out.recommendations) == [
        (s1, o1, 1, True, False),
        (s1, o2, 2, False, False),
        (s2, x, 1, False, False),
        (s2, o2, 2, True, True),
        (s3, o1, 1, True, False),
        (s3, o3, 2, False, False),
    ]
    assert {r.profile.url: r.unmet_reason for r in out.orphans} == {
        o1: None,
        o2: None,
        o3: OrphanSlotReason.SOURCES_FULL,
    }


# ── why a page is left short ─────────────────────────────────────────────────


def test_pairs_from_another_hub_or_an_excluded_page_are_no_relevant_source() -> None:
    other, low, gone, x = url("s-other"), url("s-low"), url("e"), url("x")
    in_hub, free = url("o-hub"), url("o-free")
    scores = {(other, x): 10.0, (gone, free): 9.0, (other, in_hub): 8.0, (low, in_hub): 1.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            page("s-other", hub_id=1, outbound=0),
            page("s-low", hub_id=0),
            page("x"),
            page("o-hub", hub_id=0, inbound=0),
            page("o-free", inbound=0),
        ),
        excluded=excluded(gone),
        # Two pairs in tier 1, one in tier 2: s-low's pair into o-hub is tier 3.
        tier_shares=(0.5, 0.25),
        guaranteed_inbound_links=1,
    )
    out = run(inputs)[1]

    # o-hub's tier-2 pair is from another hub and its same-hub pair is tier 3; o-free's only
    # pair is from an excluded page.
    rescue = {r.profile.url: r for r in out.orphans}
    assert {page_url: r.unmet_reason for page_url, r in rescue.items()} == dict.fromkeys(
        (in_hub, free), OrphanSlotReason.NO_RELEVANT_SOURCE
    )
    assert [(r.source_url, r.tier) for r in rescue[in_hub].sources] == [(low, 3)]
    assert rescue[free].sources == ()
    assert out.summary.orphan_slots == 0


def test_a_page_has_no_anchor_when_its_relevant_pairs_lack_one_whatever_its_other_pairs() -> None:
    other, same, same2, low = url("s-other"), url("s-same"), url("s-same2"), url("s-low")
    x, weak, cross = url("x"), url("o-weak"), url("o-cross")
    scores = {(other, x): 10.0, (other, cross): 9.0, (same, weak): 8.0, (same2, cross): 7.0}
    scores |= {(low, weak): 1.0}
    no_keyword = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in ((other, x), (other, cross), (low, weak))],
        missing=[unanchored(same, weak, no_keyword), unanchored(same2, cross, no_keyword)],
        pages=(
            page("s-other", hub_id=1, outbound=0),
            *(page(s, hub_id=0) for s in ("s-same", "s-same2", "s-low")),
            page("x"),
            *(page(o, hub_id=0, inbound=0) for o in ("o-weak", "o-cross")),
        ),
        # Two pairs in tier 1, two in tier 2: s-low's pair is tier 3.
        tier_shares=(0.4, 0.4),
        guaranteed_inbound_links=1,
    )
    out = run(inputs)[1]

    # Each page's tier-2 same-hub pair has no anchor. The anchored pair into o-weak is tier 3 and
    # the one into o-cross is from another hub, so neither makes the page's sources full.
    rescue = {r.profile.url: r for r in out.orphans}
    assert {page_url: r.unmet_reason for page_url, r in rescue.items()} == dict.fromkeys(
        (weak, cross), OrphanSlotReason.NO_ANCHOR
    )
    assert [(r.source_url, r.tier, r.anchor) for r in rescue[weak].sources] == [
        (low, 3, "dome tent"),
        (same, 2, None),
    ]


# ── tie-breaks ───────────────────────────────────────────────────────────────


def test_the_cap_keeps_the_better_suggestion_where_the_served_scores_tie() -> None:
    z, b, w, spare = url("z"), url("b"), url("w"), url("spare")
    # Three thousand pairs put neighbouring ranks within one 0.1 step of the served score.
    scores = {(url("f"), url(f"t{i}")): float(i) for i in range(3000)}
    scores |= {(z, w): 1501.6, (b, w): 1501.4, (z, spare): -1.0, (b, spare): -2.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in ((z, w), (b, w), (z, spare), (b, spare))],
        max_suggested_inbound=1,
    )
    out = run(inputs)[1]

    by_pair = {(r.source_url, r.target_url): r for r in out.recommendations}
    assert by_pair[(z, w)].score == by_pair[(b, w)].score
    # w takes one: z's suggestion is the better one unrounded, though b comes first by url.
    assert slots(out.recommendations) == [
        (b, w, 1, False, False),
        (b, spare, 2, True, False),
        (z, w, 1, True, False),
        (z, spare, 2, False, False),
    ]
    assert out.summary.links_moved_by_cap == 1


def test_guaranteed_pages_tied_on_eligible_sources_go_in_url_order() -> None:
    s, x, o1, o2 = url("s"), url("x"), url("o1"), url("o2")
    scores = {(s, x): 3.0, (s, o2): 2.0, (s, o1): 1.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for _, t in scores],
        pages=(page("s", outbound=0), page("o2", inbound=0), page("o1", inbound=0)),
        tier_shares=(0.5, 0.5),
        guaranteed_inbound_links=1,
    )
    out = run(inputs)[1]

    # s is the only source of both; o1 goes first by url, though s ranks o2 higher.
    assert {(r.source_url, r.target_url) for r in out.recommendations if r.orphan_slot} == {(s, o1)}
    assert [(r.profile.url, r.unmet_reason) for r in out.orphans] == [
        (o1, None),
        (o2, OrphanSlotReason.SOURCES_FULL),
    ]
    # The url decides whatever order the guaranteed pages come in.
    before = walk(dataclasses.replace(inputs, guaranteed_inbound_links=0))
    links = before.emitted.loc[before.emitted["action"] == ActionType.ADD_LINK.value]
    placed, unmet = recommendations._guarantee(inputs, before.pairs, links, (o2, o1))
    assert placed.loc[placed["orphan_slot"].astype(bool), "target_url"].tolist() == [o1]
    assert unmet == {o2: OrphanSlotReason.SOURCES_FULL}


def test_sources_tied_on_score_give_their_slots_by_pagerank() -> None:
    a, m, z, x, o = url("a"), url("m"), url("z"), url("x"), url("o")
    scores = {(s, x): 9.0 for s in (a, m, z)} | {(s, o): 5.0 for s in (a, m, z)}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            page("a", outbound=0),
            page("m", outbound=0, page_rank_percentile=0.5),
            page("z", outbound=0, page_rank_percentile=0.9),
            page("o", inbound=0),
        ),
        tier_shares=(0.5, 0.5),
    )
    out = run(inputs)[1]

    # Tied on score, the two strongest sources give the slots; a, first by url, has no PageRank.
    assert slots(out.recommendations) == [
        (a, x, 1, True, False),
        (a, o, 2, False, False),
        (m, x, 1, False, False),
        (m, o, 2, True, True),
        (z, x, 1, False, False),
        (z, o, 2, True, True),
    ]


def test_best_rank_breaks_score_ties_by_source_then_place_then_action() -> None:
    a, b = url("a"), url("b")
    t1, t2, t3, g = url("t1"), url("t2"), url("t3"), url("g")
    scores = {(a, t1): 5.0, (a, g): 3.0, (a, t2): 3.0, (b, t3): 3.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores if t != g],
        missing=[unanchored(a, g, UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC)],
        pages=(page("a", word_count=400, outbound=0),),
    )
    records = run(inputs)[1].recommendations

    ranked = sorted(records, key=lambda r: r.best_rank or 0)
    # Three records tie on the score: a's before b's though b's is first in its list, and a's
    # gap, first in the gap list, before a's second link.
    assert [(r.source_url, r.target_url, r.action_type, r.rank_in_source) for r in ranked] == [
        (a, t1, ActionType.ADD_LINK, 1),
        (a, g, ActionType.CONTENT_GAP, 1),
        (a, t2, ActionType.ADD_LINK, 2),
        (b, t3, ActionType.ADD_LINK, 1),
    ]
    assert [r.best_rank for r in ranked] == [1, 2, 3, 4]


# ── the summary ──────────────────────────────────────────────────────────────


def test_an_orphan_linked_only_by_a_reserve_is_not_reached() -> None:
    s, s2, x, o1, o2 = url("s"), url("s2"), url("x"), url("o1"), url("o2")
    scores = {(s, x): 3.0, (s, o1): 2.0, (s2, o2): 1.0}
    inputs = make_inputs(
        scores,
        choices=[choice(source, t, "dome tent") for source, t in scores],
        pages=(
            page("s", outbound=0),
            page("s2", outbound=0),
            page("o1", **ORPHAN),
            page("o2", **ORPHAN),
        ),
        guaranteed_inbound_links=0,
    )
    out = run(inputs)[1]

    assert slots(out.recommendations) == [
        (s, x, 1, True, False),
        (s, o1, 2, False, False),
        (s2, o2, 1, True, False),
    ]
    assert out.summary.orphans_reached == 1


def test_orphans_to_pillar_counts_only_orphans_suggesting_their_own_hubs_main_page() -> None:
    p0, p1 = url("p0"), url("p1")
    mine, member, stray = url("o-mine"), url("member"), url("o-stray")
    scores = {(mine, p0): 1.0, (member, p0): 1.0, (stray, p1): 1.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        pages=(
            page("p0", hub_id=0, is_hub_pillar=True),
            page("p1", hub_id=1, is_hub_pillar=True),
            page("o-mine", hub_id=0, outbound=0, **ORPHAN),
            page("member", hub_id=0, outbound=0),
            page("o-stray", hub_id=0, outbound=0, **ORPHAN),
        ),
        hubs=(
            HubNode(hub_id=0, size=4, pillar_url=p0, active=True),
            HubNode(hub_id=1, size=1, pillar_url=p1, active=True),
        ),
        guaranteed_inbound_links=0,
    )
    summary = run(inputs)[1].summary

    # All three links are suggested: member is no orphan and o-stray links to hub 1's main page.
    assert summary.suggested_links == 3
    assert summary.orphans_to_pillar == 1


def test_the_inbound_gini_is_none_when_ranked_pairs_give_no_suggested_link() -> None:
    s1, s2, t1, t2 = url("s1"), url("s2"), url("t1"), url("t2")
    scores = {(s1, t1): 2.0, (s2, t2): 1.0}
    inputs = make_inputs(
        scores,
        choices=[choice(s, t, "dome tent") for s, t in scores],
        # Each already has the one link its words allow, so both new links are reserves.
        pages=(page("s1"), page("s2")),
    )
    summary = run(inputs)[1].summary

    assert (summary.suggested_links, summary.reserve_links) == (0, 2)
    assert summary.inbound_gini is None
