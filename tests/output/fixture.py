"""Two tenants for the #89 output acceptance tests: the same urls, different content, verdicts
and hubs, so a response that mixes the tenants shows it.

`served_output` builds one tenant's complete served output from the output models alone, and
`write_served` stores it the way the output stage leaves it; `open_api` serves the stored
output through the real app. `plan_inputs` and `plant_inputs` plant instead what the
recommendations stage reads: pages, links, keywords, hubs, the link audit, excluded pages and
the stage files, ranked by the real ranker. The planted data separate cleanly by construction:
recovering them proves the plumbing, not the method.
"""

from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import httpx
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, SecretStr

from linking_engine.api.app import ApiSettings, create_app
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import (
    EXCLUSION_LABELS,
    NEW_LINK_ACTIONS,
    ActionType,
    AnchorCandidate,
    AnchorMatch,
    AnchorPlacement,
    AnchorRung,
    AnchorType,
    BridgeLink,
    BridgeLinkOut,
    BridgeMark,
    BridgePair,
    BridgeReason,
    ContentGapFinding,
    DuplicateGroup,
    ExcludedPage,
    ExclusionReason,
    HubPair,
    HubSummary,
    IssueFlag,
    KeywordRung,
    KeywordSource,
    LanguageRules,
    Link,
    LinkAuditResult,
    LinkRecord,
    OrphanLabel,
    Page,
    PageDetail,
    PageProfile,
    PageRecord,
    PageType,
    Recommendation,
    RecommendationStatus,
    RunInfo,
    ScorerName,
    SiteSummary,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.output.collections import (
    BRIDGES,
    DUPLICATES,
    HUBS,
    PAGES,
    RECOMMENDATIONS,
    TARGET_FIXES,
    UNANCHORED,
    recommendation_id,
)
from linking_engine.output.writer import OutputWriter
from linking_engine.pipeline.anchor_selection import (
    CHOICES_SCHEMA,
    UNANCHORED_FILE,
    UNANCHORED_SCHEMA,
)
from linking_engine.pipeline.bridges import (
    _LINK_SCHEMA,
    _PAIR_SCHEMA,
    BRIDGES_FILE,
    HUB_PAIRS_FILE,
)
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE
from linking_engine.pipeline.keywords import resolve_tenant_keywords
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE, rank_pairs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from fastapi import FastAPI

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

DATABASE: Final = "linking_engine_test"
PREFIX: Final = "/v1/tenants/{tenant}"
# Every route of the API contract under the tenant prefix.
ROUTES: Final = (
    "/recommendations",
    "/recommendations/{recommendation_id}",
    "/summary",
    "/runs/latest",
    "/pages",
    "/page",
    "/orphans",
    "/hubs",
    "/bridges",
    "/duplicates",
    "/unanchored",
    "/target-fixes",
    "/excluded-pages",
)
NOT_FOUND: Final = {"detail": "not found"}
Params = dict[str, str | int | list[str]]
UNAUTHORISED: Final = {"detail": "invalid or missing API key"}

WORDS: Final = ("alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel")
URLS: Final = tuple(f"example.com/gear/{word}" for word in WORDS)
# New links go to the first seven pages; the last has no keyword and waits on a target fix.
LINKED: Final = 7
KEYWORDLESS: Final = URLS[7]
SOURCES: Final = URLS[:6]
DEAD_END: Final = URLS[6]
ORPHAN: Final = URLS[7]
CANONICAL, COPY = URLS[4], URLS[5]
EXCLUDED: Final = ("example.com/misc/offcuts", "example.com/sitemap")
STARTED: Final = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
# New links and content gaps listed per source page, the tenant defaults.
LIMIT: Final = 10
GAP_LIMIT: Final = 3


@dataclass(frozen=True, slots=True)
class Variant:
    """What differs between the two tenants over the same urls."""

    name: str
    hubs: tuple[int, int]
    gap_finding: ContentGapFinding
    orphan_label: OrphanLabel
    excluded_reason: ExclusionReason
    # (source, target, position, action, flags, current anchor) of each audited link.
    verdicts: tuple[tuple[int, int, int, ActionType, tuple[IssueFlag, ...], str | None], ...]
    # The stage inputs: the word every keyword ends in, the retired hub, and how far the
    # audited page's verdicts are rotated.
    suffix: str
    retired_hub: int
    rotation: int


ALPHA: Final = Variant(
    name="Alpha",
    hubs=(0, 1),
    gap_finding=ContentGapFinding.NO_TOPICAL_MENTION,
    orphan_label=OrphanLabel.MENUS_ONLY,
    excluded_reason=ExclusionReason.INSUFFICIENT_CONTENT,
    verdicts=(
        (0, 5, 4, ActionType.FIX, (IssueFlag.REDIRECTED,), "elm kit"),
        (1, 2, 0, ActionType.REANCHOR, (IssueFlag.GENERIC,), "click here"),
        (3, 4, 2, ActionType.REMOVE, (IssueFlag.OFF_TOPIC, IssueFlag.WASTED_EQUITY), None),
    ),
    suffix="Kit",
    retired_hub=2,
    rotation=0,
)
BETA: Final = Variant(
    name="Beta",
    hubs=(3, 5),
    gap_finding=ContentGapFinding.AWKWARD_PHRASING,
    orphan_label=OrphanLabel.NOT_LINKED,
    excluded_reason=ExclusionReason.TENANT_EXCLUDED,
    verdicts=(
        (0, 5, 4, ActionType.REMOVE, (IssueFlag.OFF_TOPIC,), "fir"),
        (2, 3, 1, ActionType.FIX, (IssueFlag.BROKEN,), "dogwood"),
        # Over-optimised, and the copy holds no better phrase.
        (4, 6, 3, ActionType.REANCHOR, (IssueFlag.OVER_OPTIMISED,), "ginkgo"),
    ),
    suffix="Pack",
    retired_hub=4,
    rotation=4,
)


def word(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def hub_of(variant: Variant, url: str) -> int:
    return variant.hubs[URLS.index(url) % 2]


@dataclass(frozen=True, slots=True)
class ServedOutput:
    """One tenant's run as the API serves it, every listing in its stable order."""

    tenant_id: str
    run: RunInfo
    recommendations: tuple[Recommendation, ...]
    pages: tuple[PageProfile, ...]
    hubs: tuple[HubSummary, ...]
    bridges: tuple[BridgePair, ...]
    duplicates: tuple[DuplicateGroup, ...]
    unanchored: tuple[UnanchoredOut, ...]
    target_fixes: tuple[TargetFix, ...]
    excluded: tuple[ExcludedPage, ...]

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def collections(self) -> dict[str, Sequence[BaseModel]]:
        """Each run-scoped collection and its documents' models, in ordinal order."""
        return {
            RECOMMENDATIONS: self.recommendations,
            PAGES: self.pages,
            HUBS: self.hubs,
            BRIDGES: self.bridges,
            DUPLICATES: self.duplicates,
            UNANCHORED: self.unanchored,
            TARGET_FIXES: self.target_fixes,
        }


def _anchor(target: str, rank: int, *, alternative: bool) -> AnchorCandidate:
    keyword = word(target)
    text = f"{keyword} kit" if alternative else keyword
    sentence = f"Pack the {text} before the weekend."
    start = sentence.index(text)
    return AnchorCandidate(
        text=text,
        anchor_type=AnchorType.PARTIAL if alternative else AnchorType.EXACT,
        source="EXTRACTED",
        score=0.6 if alternative else 0.9,
        keyword=keyword,
        placement=AnchorPlacement(
            sentence=sentence,
            sentence_index=rank,
            start=40 * rank + start,
            end=40 * rank + start + len(text),
        ),
    )


def _new_links(tenant: str, run_id: str, variant: Variant, revision: int) -> list[Recommendation]:
    records = []
    for index, source in enumerate(SOURCES):
        for rank in (1, 2, 3):
            target = URLS[(index + rank) % LINKED]
            gap = rank == 3
            action = ActionType.CONTENT_GAP if gap else ActionType.ADD_LINK
            crossing = index == 0 and rank == 1
            records.append(
                Recommendation(
                    id=recommendation_id(tenant, action, source, target, None),
                    run_id=run_id,
                    source_url=source,
                    target_url=target,
                    action_type=action,
                    label="content gap: add copy first" if gap else "add a link",
                    finding=variant.gap_finding if gap else None,
                    advice=f"Write a sentence about {word(target)} first." if gap else None,
                    score=round(90.0 - 10 * index - rank, 1),
                    tier=1 if index < 3 else 2,
                    rank_in_source=1 if gap else rank,
                    status=RecommendationStatus.PENDING,
                    proposed_anchors=None
                    if gap
                    else (
                        _anchor(target, rank, alternative=False),
                        _anchor(target, rank, alternative=True),
                    ),
                    bridge=BridgeMark(
                        hub_from=hub_of(variant, source),
                        hub_to=hub_of(variant, target),
                        reasons=(BridgeReason.BRIDGE_GAP,),
                    )
                    if crossing
                    else None,
                    rationale=f"{variant.name} run {revision}: ranked {rank} by the baseline "
                    "scorer; content cosine and same hub",
                    signals=(("content_cosine", 0.4), ("same_hub", -0.1)),
                    created_at=STARTED + timedelta(days=revision),
                )
            )
    return records


def _verdicts(tenant: str, run_id: str, variant: Variant, revision: int) -> list[Recommendation]:
    records = []
    for source, target, position, action, flags, anchor in variant.verdicts:
        source_url, target_url = URLS[source], URLS[target]
        records.append(
            Recommendation(
                id=recommendation_id(tenant, action, source_url, target_url, position),
                run_id=run_id,
                source_url=source_url,
                target_url=target_url,
                action_type=action,
                label={
                    ActionType.FIX: "fix this link",
                    ActionType.REANCHOR: "change the anchor text",
                    ActionType.REMOVE: "review this link",
                }[action],
                position=position,
                status=RecommendationStatus.PENDING,
                current_anchor=anchor,
                proposed_anchors=(
                    AnchorCandidate(
                        text=word(target_url), anchor_type=AnchorType.EXACT, source="EXTRACTED"
                    ),
                )
                if action is ActionType.REANCHOR and IssueFlag.GENERIC in flags
                else None,
                issue_flags=flags,
                fix_target=CANONICAL if action is ActionType.FIX and target_url == COPY else None,
                rationale=f"{variant.name} run {revision}: the anchor does not fit the target",
                signals=(("anchor_quality_score", 0.2), ("context_relevance", 0.3)),
                created_at=STARTED + timedelta(days=revision),
            )
        )
    return records


def _recommendation_order(record: Recommendation) -> tuple[object, ...]:
    """Links, then content gaps, each by rank, then verdicts by position."""
    group = (
        (ActionType.ADD_LINK, ActionType.CONTENT_GAP).index(record.action_type)
        if (record.action_type in NEW_LINK_ACTIONS)
        else 2
    )
    place = record.position if group == 2 else record.rank_in_source
    return (record.source_url, group, place, record.action_type, record.target_url)


def _pages(variant: Variant, records: Sequence[Recommendation]) -> list[PageProfile]:
    new = [r for r in records if r.action_type in NEW_LINK_ACTIONS]
    out = Counter(r.source_url for r in new)
    into = Counter(r.target_url for r in new)
    audited = Counter(r.source_url for r in records if r.action_type not in NEW_LINK_ACTIONS)
    pages = []
    for index, url in enumerate(URLS):
        grouped = url in {CANONICAL, COPY}
        keyword = None if url == KEYWORDLESS else word(url)
        pages.append(
            PageProfile(
                url=url,
                title=f"{word(url).title()} | Acme {variant.name}",
                language="en",
                word_count=300 + 10 * index,
                inbound=0 if url == ORPHAN else 2,
                outbound=0 if url == DEAD_END else 3,
                crawl_depth=None if url == ORPHAN else 1 + index % 3,
                page_rank_percentile=index / len(URLS),
                hub_id=hub_of(variant, url),
                is_hub_pillar=index < 2,
                is_orphan=url == ORPHAN,
                orphan_label=variant.orphan_label if url == ORPHAN else None,
                is_dead_end=url == DEAD_END,
                duplicate_group=0 if grouped else None,
                is_canonical=url == CANONICAL if grouped else None,
                target_keyword=keyword,
                keyword_rung=None if keyword is None else KeywordRung.TITLE,
                recommendations_out=out[url],
                recommendations_in=into[url],
                audit_verdicts_out=audited[url],
            )
        )
    return pages


def _hubs(variant: Variant, pages: Sequence[PageProfile]) -> list[HubSummary]:
    first, second = variant.hubs
    return [
        HubSummary(
            hub_id=hub,
            language="en",
            size=sum(1 for page in pages if page.hub_id == hub),
            pillar_url=URLS[side],
            pillar_title=f"{WORDS[side].title()} | Acme {variant.name}",
            orphan_pages=sum(1 for page in pages if page.hub_id == hub and page.is_orphan),
            dead_end_pages=sum(1 for page in pages if page.hub_id == hub and page.is_dead_end),
            recommendations_in=sum(page.recommendations_in for page in pages if page.hub_id == hub),
            bridge_hubs=(other,),
        )
        for side, (hub, other) in enumerate(((first, second), (second, first)))
    ]


def _bridges(variant: Variant, records: Sequence[Recommendation]) -> list[BridgePair]:
    marked = next(r for r in records if r.bridge is not None)
    assert marked.bridge is not None
    first, second = variant.hubs
    return [
        BridgePair(
            language="en",
            hub_a=first,
            hub_b=second,
            size_a=4,
            size_b=4,
            pages_ab=0,
            pages_ba=1,
            link_density=0.0625,
            centroid_cosine=0.6,
            query_jaccard=None,
            bridge_gap=0.54,
            reasons=(BridgeReason.BRIDGE_GAP,),
            links=(
                BridgeLinkOut(
                    hub_from=marked.bridge.hub_from,
                    hub_to=marked.bridge.hub_to,
                    slot=0,
                    rank=0,
                    source_url=marked.source_url,
                    target_url=marked.target_url,
                    similarity=0.71,
                    source_page_rank_percentile=0.0,
                    anchor_keyword=word(marked.target_url),
                    anchor_rung=KeywordRung.TITLE,
                    reasons=(BridgeReason.BRIDGE_GAP,),
                    recommendation_id=marked.id,
                ),
            ),
        )
    ]


def _unanchored(records: Sequence[Recommendation]) -> list[UnanchoredOut]:
    gaps = [
        UnanchoredOut(
            source_url=r.source_url,
            target_url=r.target_url,
            reason=UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
            if r.finding is ContentGapFinding.NO_TOPICAL_MENTION
            else UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE,
            advice=r.advice or "",
            best_score=0.3,
            rank_in_source=3,
            recommended=True,
        )
        for r in records
        if r.action_type is ActionType.CONTENT_GAP
    ]
    waiting = [
        UnanchoredOut(
            source_url=source,
            target_url=KEYWORDLESS,
            reason=UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD,
            advice="Give the target page a keyword.",
            rank_in_source=4,
        )
        for source in SOURCES[2:4]
    ]
    return sorted(gaps + waiting, key=lambda u: (u.source_url, u.rank_in_source, u.target_url))


def _summary(output: ServedOutput) -> SiteSummary:
    new = [r for r in output.recommendations if r.action_type in NEW_LINK_ACTIONS]
    per_source = Counter(r.source_url for r in new)
    links = Counter(r.source_url for r in new if r.action_type is ActionType.ADD_LINK)
    audits = [r for r in output.recommendations if r.action_type not in NEW_LINK_ACTIONS]
    return SiteSummary(
        pages=len(output.pages),
        excluded_pages=dict(Counter(page.reason for page in output.excluded)),
        orphan_pages=dict(Counter(p.orphan_label for p in output.pages if p.orphan_label)),
        dead_end_pages=sum(1 for page in output.pages if page.is_dead_end),
        duplicate_groups=len(output.duplicates),
        duplicate_copies=sum(len(group.copies) for group in output.duplicates),
        hubs=len(output.hubs),
        bridge_pairs=len(output.bridges),
        bridge_links=sum(1 for pair in output.bridges for link in pair.links if link.rank == 1),
        recommendations=dict(Counter(r.action_type for r in output.recommendations)),
        tiers=dict(Counter(r.tier for r in new if r.tier is not None)),
        sources_with_recommendations=len(per_source),
        sources_below_limit=sum(1 for source in per_source if links[source] < LIMIT),
        links_audited=len(audits),
        unverified_links=0,
        audit_flags=dict(Counter(flag for r in audits for flag in r.issue_flags)),
        unanchored=dict(Counter(u.reason for u in output.unanchored)),
        target_fixes=len(output.target_fixes),
    )


def served_output(
    tenant: str, run_id: str, variant: Variant, *, revision: int = 1, sources: int = 6
) -> ServedOutput:
    """The tenant's complete output of one run. A later ``revision`` words every record
    differently; fewer ``sources`` drop the last source pages' new links."""
    new = [
        r for r in _new_links(tenant, run_id, variant, revision) if r.source_url in URLS[:sources]
    ]
    records = sorted(
        [*new, *_verdicts(tenant, run_id, variant, revision)], key=_recommendation_order
    )
    pages = _pages(variant, records)
    started = STARTED + timedelta(days=revision)
    draft = ServedOutput(
        tenant_id=tenant,
        run=RunInfo(
            tenant_id=tenant,
            run_id=run_id,
            status="writing",
            started_at=started,
            scorer=ScorerName.BASELINE,
            weights_version="default",
            feature_code="0" * 16,
            package_version="0.1.0",
            link_audit_run_id=f"audit-{run_id}",
            inputs={"ranked_pairs.parquet": started - timedelta(hours=1)},
            limit_per_source=LIMIT,
            content_gap_limit=GAP_LIMIT,
        ),
        recommendations=tuple(records),
        pages=tuple(pages),
        hubs=tuple(_hubs(variant, pages)),
        bridges=tuple(_bridges(variant, records)),
        duplicates=(DuplicateGroup(group_id=0, canonical=CANONICAL, copies=(COPY,)),),
        unanchored=tuple(_unanchored(records)),
        target_fixes=(
            TargetFix(
                target_url=KEYWORDLESS,
                title=f"{word(KEYWORDLESS).title()} | Acme {variant.name}",
                fix="Give the target page a keyword.",
                waiting_sources=2,
                best_sources=SOURCES[2:4],
            ),
        ),
        excluded=(
            ExcludedPage(
                url=EXCLUDED[0],
                reason=variant.excluded_reason,
                label=f"{variant.name} {variant.excluded_reason.value.lower()}",
                words=12,
                link_words=0,
                links=0,
            ),
            ExcludedPage(
                url=EXCLUDED[1],
                reason=ExclusionReason.SITEMAP,
                label="sitemap",
                words=80,
                link_words=80,
                links=40,
            ),
        ),
    )
    run = RunInfo.model_validate(
        {
            **draft.run.model_dump(),
            "status": "complete",
            "completed_at": started + timedelta(minutes=5),
            "summary": _summary(draft),
        }
    )
    return replace(draft, run=run)


async def stage_run(writer: OutputWriter, output: ServedOutput, *, only: str | None = None) -> None:
    """The run begun and written, not completed: what a crash mid-run leaves. ``only`` stops
    after that collection."""
    writing = output.run.model_copy(
        update={"status": "writing", "completed_at": None, "summary": None}
    )
    await writer.begin(writing)
    for name, models in output.collections().items():
        await writer.write(name, output.tenant_id, output.run_id, models)
        if name == only:
            return


async def finish_run(writer: OutputWriter, output: ServedOutput) -> None:
    """The staged run marked complete, the tenant's other runs pruned."""
    run = output.run
    assert run.summary is not None
    assert run.completed_at is not None
    await writer.complete(output.tenant_id, output.run_id, run.summary, run.completed_at)
    await writer.prune(output.tenant_id, keep=output.run_id)


async def publish_run(writer: OutputWriter, output: ServedOutput) -> None:
    """One complete run of the tenant through the writer, the previous runs pruned."""
    await stage_run(writer, output)
    await finish_run(writer, output)


async def write_served(mongo: MongoRepo, output: ServedOutput) -> None:
    """Store a complete run of the tenant through the real writer, and its excluded pages as
    prepare-corpus does."""
    # On the repo's own client, which the repo closes.
    writer = OutputWriter(mongo._client, DATABASE)
    await writer.ensure_indexes()
    await publish_run(writer, output)
    await mongo.replace_excluded_pages(output.tenant_id, output.excluded)


def api_app(mongo_uri: str) -> FastAPI:
    return create_app(ApiSettings(mongo_uri=SecretStr(mongo_uri), mongo_db=DATABASE))


@asynccontextmanager
async def open_api(mongo_uri: str) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """The real app over the test database, its lifespan run as a server runs it."""
    app = api_app(mongo_uri)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client,
    ):
        yield app, client


async def walk(
    client: httpx.AsyncClient,
    path: str,
    key: str,
    params: Params | None = None,
    *,
    limit: int = 50,
) -> tuple[list[dict[str, object]], int]:
    """Every item of a listing, page by page, and its reported total; each page's total must
    agree."""
    items: list[dict[str, object]] = []
    totals: set[int] = set()
    after: str | None = None
    while True:
        query: Params = {**(params or {}), "limit": limit, **({"after": after} if after else {})}
        response = await client.get(path, params=query, headers={"X-API-Key": key})
        assert response.status_code == 200, (path, response.status_code, response.text)
        body = response.json()
        items.extend(body["items"])
        totals.add(body["total"])
        after = body["next_cursor"]
        if after is None:
            break
        assert body["items"], f"{path}: an empty page with a next cursor"
    assert len(totals) == 1, f"{path}: totals disagree across pages: {sorted(totals)}"
    return items, totals.pop()


LISTINGS: Final[dict[str, type[BaseModel]]] = {
    "/recommendations": Recommendation,
    "/pages": PageProfile,
    "/orphans": PageProfile,
    "/hubs": HubSummary,
    "/bridges": BridgePair,
    "/duplicates": DuplicateGroup,
    "/unanchored": UnanchoredOut,
    "/target-fixes": TargetFix,
    "/excluded-pages": ExcludedPage,
}
State = dict[str, object]


async def fetch(client: httpx.AsyncClient, path: str, key: str, **params: str) -> object:
    response = await client.get(path, params=params, headers={"X-API-Key": key})
    assert response.status_code == 200, (path, params, response.status_code, response.text)
    return response.json()


async def served_state(client: httpx.AsyncClient, tenant: str, key: str) -> State:
    """What every route serves the tenant: each listing walked in small pages, the summary and
    run, and every page's detail and recommendation by id."""
    base = PREFIX.format(tenant=tenant)
    state: State = {}
    for route, model in LISTINGS.items():
        items, total = await walk(client, base + route, key, limit=7)
        assert total == len(items), f"{route}: total {total} for {len(items)} items"
        state[route] = tuple(model.model_validate(item) for item in items)
    state["/runs/latest"] = RunInfo.model_validate(await fetch(client, f"{base}/runs/latest", key))
    state["/summary"] = SiteSummary.model_validate(await fetch(client, f"{base}/summary", key))
    pages = state["/pages"]
    records = state["/recommendations"]
    assert isinstance(pages, tuple)
    assert isinstance(records, tuple)
    state["/page"] = tuple(
        [
            PageDetail.model_validate(await fetch(client, f"{base}/page", key, url=page.url))
            for page in pages
        ]
    )
    state["/recommendations/{recommendation_id}"] = tuple(
        [
            Recommendation.model_validate(
                await fetch(client, f"{base}/recommendations/{r.id}", key)
            )
            for r in records
        ]
    )
    return state


def expected_state(output: ServedOutput) -> State:
    """What every route should serve for this output, keyed as `served_state` keys it."""
    records = output.recommendations
    return {
        "/recommendations": records,
        "/pages": output.pages,
        "/orphans": tuple(page for page in output.pages if page.is_orphan),
        "/hubs": output.hubs,
        "/bridges": output.bridges,
        "/duplicates": output.duplicates,
        "/unanchored": output.unanchored,
        "/target-fixes": output.target_fixes,
        "/excluded-pages": tuple(sorted(output.excluded, key=lambda page: page.url)),
        "/runs/latest": output.run,
        "/summary": output.run.summary,
        "/page": tuple(
            PageDetail(
                profile=page,
                outgoing=tuple(r for r in records if r.source_url == page.url),
                incoming_total=sum(
                    1
                    for r in records
                    if r.target_url == page.url and r.action_type in NEW_LINK_ACTIONS
                ),
            )
            for page in output.pages
        ),
        "/recommendations/{recommendation_id}": records,
    }


# ── Stage inputs: what the recommendations stage reads ──────────────────────────────────────
#
# Fourteen pages in two hubs, even and odd. FULL has an anchor for every ranked target, more
# than the limit; GAPS_ONLY has only content gaps, more than the gap limit; SPLIT has two links
# among gaps; AUDITED links to twelve pages and the audit has a verdict on eleven of those links;
# MIXED has one pair of every unanchored reason and four pairs no stage assessed. Every pair
# into NO_KEYWORD waits on a target fix. The two excluded pages come back
# through files and an audit from before they were excluded: ranked pairs with an anchor or a
# content gap, and a verdict each.

KIT: Final = (
    "alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo",
    "hazel", "ivy", "juniper", "larch", "maple", "oak", "pine",
)  # fmt: skip
KIT_URLS: Final = tuple(f"example.com/kit/{name}" for name in KIT)
FULL, AUDITED, MIXED = KIT_URLS[0], KIT_URLS[1], KIT_URLS[2]
KIT_CANONICAL, KIT_COPY = KIT_URLS[10], KIT_URLS[11]
NO_KEYWORD: Final = KIT_URLS[12]
KIT_ORPHAN: Final = KIT_URLS[13]
KIT_DEAD_END: Final = FULL
# Content gaps only; and two same-hub links among gaps to pages of the other hub, which rank
# lower.
GAPS_ONLY, SPLIT = KIT_URLS[4], KIT_URLS[6]
SPLIT_ANCHORED: Final = frozenset({KIT_URLS[8], KIT_URLS[10]})
# The other two pages with a new link each.
MORE_ANCHORED: Final = frozenset({(AUDITED, KIT_ORPHAN), (KIT_URLS[5], KIT_URLS[7])})
UNCRAWLED: Final = "example.com/kit/retired"
OFFCUTS, SITEMAP_PAGE = EXCLUDED
DIMENSION: Final = 2048
EMBEDDING_MODEL: Final = "voyage-4-large"
AUDITED_AT: Final = datetime(2026, 9, 20, 6, 0, tzinfo=UTC)
# MIXED's planted pairs: None is an anchor; its other ranked targets no stage assessed.
MIXED_PLAN: Final[dict[str, UnanchoredReason | None]] = {
    KIT_URLS[3]: None,
    KIT_URLS[4]: UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC,
    KIT_URLS[5]: UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE,
    KIT_URLS[6]: UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE,
    KIT_URLS[7]: UnanchoredReason.MEANING_SEARCH_NOT_RUN,
}
# Bridge links as (slot, rank, source, target): the first is also MIXED's new link, the second
# its alternative, the third a content gap at most.
BRIDGED: Final = (
    (1, 1, MIXED, KIT_URLS[3]),
    (1, 2, MIXED, KIT_URLS[5]),
    (2, 1, GAPS_ONLY, KIT_URLS[9]),
)
# The audited page's verdicts by link, the link to the duplicate copy left out: (verdict, flags).
VERDICT_CYCLE: Final[tuple[tuple[ActionType | None, tuple[IssueFlag, ...]], ...]] = (
    (ActionType.REMOVE, (IssueFlag.OFF_TOPIC,)),
    (ActionType.REANCHOR, (IssueFlag.GENERIC,)),
    (ActionType.REANCHOR, (IssueFlag.OVER_OPTIMISED,)),
    (ActionType.FIX, (IssueFlag.REDIRECTED,)),
    (ActionType.REMOVE, (IssueFlag.WASTED_EQUITY,)),
    (ActionType.REANCHOR, (IssueFlag.MISALIGNED,)),
    (ActionType.REMOVE, (IssueFlag.OFF_TOPIC, IssueFlag.WASTED_EQUITY)),
    (ActionType.FIX, (IssueFlag.BROKEN,)),
    (None, ()),
    (ActionType.REMOVE, (IssueFlag.GENERIC,)),
    (ActionType.REANCHOR, (IssueFlag.GENERIC,)),
)
AUDITED_TARGETS: Final = (KIT_URLS[0], *KIT_URLS[2:13])


def keyword(variant: Variant, url: str) -> str:
    return f"{word(url).title()} {variant.suffix}"


def phrase(variant: Variant, url: str) -> str:
    """The keyword as the copy writes it."""
    return keyword(variant, url).lower()


def kit_hub(variant: Variant, url: str) -> int:
    return variant.hubs[KIT_URLS.index(url) % 2]


def pillars(variant: Variant) -> tuple[str, str]:
    """The main page of the even hub and of the odd hub."""
    return (KIT_URLS[0], KIT_URLS[1]) if variant.rotation == 0 else (KIT_URLS[2], KIT_URLS[3])


def anchor_sentences(text: str) -> tuple[str, str]:
    """The sentences holding a pair's chosen anchor and its alternative, ``text`` + " range"."""
    return f"Pack the {text} before the weekend.", f"The {text} range grew again."


@dataclass(frozen=True, slots=True)
class PlantedLink:
    source: str
    position: int
    target: str
    anchor: str


@dataclass(frozen=True, slots=True)
class Planted:
    """One tenant's planted stage inputs, and the truths its output is checked against."""

    tenant_id: str
    variant: Variant
    links: tuple[PlantedLink, ...]
    # Pairs with an anchor, by the chosen phrase; the alternative is the phrase + " range".
    anchored: dict[tuple[str, str], str]
    unanchored: dict[tuple[str, str], UnanchoredReason]
    audit: tuple[LinkAuditResult, ...]
    audit_run_id: str
    bridge_links: tuple[BridgeLink, ...]
    hub_pair: HubPair
    excluded: tuple[ExcludedPage, ...]
    # Each page's body text, sentence by sentence.
    copy: dict[str, tuple[str, ...]]

    @property
    def urls(self) -> frozenset[str]:
        """Every url the tenant's stores and stage files hold."""
        return frozenset({*KIT_URLS, *EXCLUDED, UNCRAWLED})

    @property
    def excluded_urls(self) -> frozenset[str]:
        return frozenset(page.url for page in self.excluded)

    def body(self, url: str) -> str:
        return " ".join(self.copy[url])

    def keyword_of(self, url: str) -> str | None:
        return None if url not in KIT_URLS or url == NO_KEYWORD else keyword(self.variant, url)

    def title(self, url: str) -> str | None:
        found = self.keyword_of(url)
        return None if found is None else f"{found} | Acme"

    def candidates(self) -> frozenset[tuple[str, str]]:
        """The pairs candidate retrieval finds: any two pages not linked yet, the duplicate copy
        on neither side."""
        linked = {(link.source, link.target) for link in self.links}
        pages = [url for url in KIT_URLS if url != KIT_COPY]
        return frozenset((s, t) for s in pages for t in pages if s != t and (s, t) not in linked)


def _audited_verdicts(variant: Variant) -> list[tuple[ActionType | None, tuple[IssueFlag, ...]]]:
    """The audited page's verdicts in link order; the link to the duplicate copy is a FIX."""
    turn = variant.rotation % len(VERDICT_CYCLE)
    cycle = [*VERDICT_CYCLE[turn:], *VERDICT_CYCLE[:turn]]
    cycle.insert(AUDITED_TARGETS.index(KIT_COPY), (ActionType.FIX, ()))
    return cycle


def _links(variant: Variant) -> list[PlantedLink]:
    links = []
    for position, (target, (verdict, flags)) in enumerate(
        zip(AUDITED_TARGETS, _audited_verdicts(variant), strict=True)
    ):
        if IssueFlag.GENERIC in flags:
            anchor = "click here"
        elif verdict is ActionType.REANCHOR:
            anchor = phrase(variant, target)
        else:
            anchor = f"best {phrase(variant, target)}"
        links.append(PlantedLink(AUDITED, position, target, anchor))
    return [
        *links,
        PlantedLink(MIXED, 0, FULL, phrase(variant, FULL)),
        PlantedLink(MIXED, 1, AUDITED, "read more"),
        PlantedLink(KIT_URLS[3], 0, AUDITED, f"best {phrase(variant, AUDITED)}"),
        PlantedLink(KIT_URLS[4], 0, AUDITED, "this guide"),
    ]


def _audit_result(
    source: str,
    position: int,
    target: str,
    run_id: str,
    verdict: ActionType | None,
    flags: tuple[IssueFlag, ...],
    **extra: object,
) -> LinkAuditResult:
    reasons = tuple(f"the anchor is {flag.value.lower().replace('_', ' ')}" for flag in flags)
    if verdict is not None and not reasons:
        reasons = ("the link points at a duplicate copy",)
    return LinkAuditResult.model_validate(
        {
            "source_url": source,
            "position": position,
            "target_url": target,
            "run_id": run_id,
            "anchor_quality_score": 35.0 + position,
            "keyword_alignment": 0.3,
            "context_relevance": 0.5,
            "anchor_target_fit": 0.4,
            # Absent on odd positions: a verdict's signals are the dimensions present.
            "equity_efficiency": None if position % 2 else 0.6,
            "issue_flags": frozenset(flags),
            "verdict": verdict,
            "reasons": reasons,
            "audited_at": AUDITED_AT,
            **extra,
        }
    )


def _audit(variant: Variant, run_id: str, links: Sequence[PlantedLink]) -> list[LinkAuditResult]:
    verdicts = _audited_verdicts(variant)
    results = []
    for link in links:
        verdict, flags = verdicts[link.position] if link.source == AUDITED else (None, ())
        if (link.source, link.target) == (MIXED, AUDITED):
            verdict = ActionType.REANCHOR if variant.rotation == 0 else ActionType.REMOVE
            flags = (IssueFlag.GENERIC,)
        extra: dict[str, object] = {}
        if verdict is ActionType.REANCHOR and IssueFlag.OVER_OPTIMISED not in flags:
            extra["proposed_anchor"] = phrase(variant, link.target)
        if verdict is ActionType.FIX and link.target == KIT_COPY:
            extra["fix_target"] = KIT_CANONICAL
        results.append(
            _audit_result(link.source, link.position, link.target, run_id, verdict, flags, **extra)
        )
    results.append(
        LinkAuditResult(
            source_url=KIT_URLS[3],
            position=1,
            target_url=UNCRAWLED,
            run_id=run_id,
            issue_flags=frozenset(),
            verdict=None,
            unverified=True,
            audited_at=AUDITED_AT,
        )
    )
    # From an audit of the pages before they were excluded.
    results.append(_audit_result(OFFCUTS, 0, MIXED, run_id, ActionType.FIX, (IssueFlag.BROKEN,)))
    results.append(
        _audit_result(AUDITED, 12, SITEMAP_PAGE, run_id, ActionType.REMOVE, (IssueFlag.OFF_TOPIC,))
    )
    return results


def plan_inputs(tenant: str, variant: Variant) -> Planted:
    """The tenant's stage inputs, before anything is stored."""
    links = _links(variant)
    linked = {(link.source, link.target) for link in links}
    pages = [url for url in KIT_URLS if url != KIT_COPY]
    anchored: dict[tuple[str, str], str] = {}
    unanchored: dict[tuple[str, str], UnanchoredReason] = {}
    for source, target in ((s, t) for s in pages for t in pages if s != t):
        pair = (source, target)
        if pair in linked:
            continue
        if target == NO_KEYWORD:
            unanchored[pair] = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
        elif (
            source == FULL
            or pair in MORE_ANCHORED
            or (source == SPLIT and target in SPLIT_ANCHORED)
        ):
            anchored[pair] = phrase(variant, target)
        elif source in {GAPS_ONLY, SPLIT}:
            unanchored[pair] = (
                UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
                if KIT_URLS.index(target) % 2
                else UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
            )
        elif source == MIXED:
            if target not in MIXED_PLAN:
                continue
            reason = MIXED_PLAN[target]
            if reason is None:
                anchored[pair] = phrase(variant, target)
            else:
                unanchored[pair] = reason
        else:
            unanchored[pair] = UnanchoredReason.MEANING_SEARCH_NOT_RUN
    anchored[(FULL, OFFCUTS)] = "offcuts bin"
    anchored[(SITEMAP_PAGE, MIXED)] = phrase(variant, MIXED)
    unanchored[(SITEMAP_PAGE, FULL)] = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC

    copy = {}
    for url in KIT_URLS:
        opening = "Odds and ends guide." if url == NO_KEYWORD else f"{keyword(variant, url)} guide."
        sentences = [opening]
        for (source, _), text in sorted(anchored.items()):
            if source == url:
                sentences += anchor_sentences(text)
        sentences.append(f"Everything about {word(url)} for the {variant.name} season.")
        copy[url] = tuple(sentences)

    first, second = sorted(variant.hubs)
    audit_run_id = f"audit-{variant.name.lower()}"
    return Planted(
        tenant_id=tenant,
        variant=variant,
        links=tuple(links),
        anchored=anchored,
        unanchored=unanchored,
        audit=tuple(_audit(variant, audit_run_id, links)),
        audit_run_id=audit_run_id,
        bridge_links=tuple(
            BridgeLink(
                language="en",
                hub_from=kit_hub(variant, source),
                hub_to=kit_hub(variant, target),
                slot=slot,
                rank=rank,
                source_url=source,
                target_url=target,
                similarity=0.7,
                reasons=(BridgeReason.BRIDGE_GAP,),
            )
            for slot, rank, source, target in BRIDGED
        ),
        hub_pair=HubPair(
            language="en",
            hub_a=first,
            hub_b=second,
            size_a=7,
            size_b=7,
            pages_ab=1,
            pages_ba=2,
            link_density=0.08,
            centroid_cosine=0.6,
            bridge_gap=0.42,
            reasons=(BridgeReason.BRIDGE_GAP, BridgeReason.SPANNING_TREE),
        ),
        excluded=tuple(
            ExcludedPage(
                url=url,
                reason=reason,
                label=EXCLUSION_LABELS[reason],
                words=40,
                link_words=30,
                links=12,
            )
            for url, reason in zip(
                EXCLUDED, (variant.excluded_reason, ExclusionReason.SITEMAP), strict=True
            )
        ),
        copy=copy,
    )


def _choice_rows(planted: Planted) -> list[dict[str, object]]:
    rows = []
    for (source, target), text in sorted(planted.anchored.items()):
        # An excluded source's copy is gone; its stale anchor keeps only its sentences.
        sentences = planted.copy.get(source, anchor_sentences(text))
        for rank, (sentence, chosen) in enumerate(
            zip(anchor_sentences(text), (text, f"{text} range"), strict=True), 1
        ):
            index = sentences.index(sentence)
            sentence_start = sum(len(s) + 1 for s in sentences[:index])
            start = sentence_start + sentence.index(chosen)
            match = AnchorMatch(
                source_url=source,
                target_url=target,
                keyword=planted.keyword_of(target) or text.title(),
                keyword_rank=1,
                keyword_source=KeywordSource.INFERRED,
                rung=AnchorRung.EXACT,
                phrase=chosen,
                start=start,
                end=start + len(chosen),
                sentence=sentence,
                sentence_index=index,
                sentence_start=sentence_start,
            )
            rows.append(
                {
                    **match.model_dump(mode="json"),
                    "rank": rank,
                    "anchor_type": (AnchorType.EXACT if rank == 1 else AnchorType.PARTIAL).value,
                    "score_semantic": None,
                    "score_keyword": 1.0,
                    "score_diversity": 0.8,
                    "score_length": 0.9,
                    "score_rank_weight": 1.0,
                    "score_profile_bonus": 0.0,
                    "score_total": 0.82 if rank == 1 else 0.64,
                    "context_relevance": 0.7,
                    "anchor_target_fit": 0.6,
                }
            )
    return rows


async def plant_inputs(
    graph: GraphRepo, mongo: MongoRepo, planted: Planted, cache_dir: Path
) -> Path:
    """Store the tenant's pages, links, keywords, hubs, excluded pages and audit, write its
    anchor, unanchored and bridge files, rank its pairs with the real ranker (the baseline:
    no model is promoted), and slip the stale excluded pairs into the ranked file. Returns
    the tenant's stage folder."""
    tenant, variant = planted.tenant_id, planted.variant
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    by_source = Counter(link.source for link in planted.links)
    await mongo.write_pages(
        tenant,
        [
            PageRecord.model_validate(
                {
                    "url": url,
                    "status_code": 200,
                    "usable": True,
                    "meta_title": planted.title(url),
                    "meta_description": None,
                    "h1": planted.keyword_of(url),
                    "headings": (),
                    "body_text": planted.body(url),
                    "word_count": len(planted.body(url).split()),
                    "link_count": by_source[url],
                    "content_hash": None,
                    "body_hash": body_hash(planted.body(url)),
                    "scraped_at": None,
                    "source": "test",
                    "crawl_url": f"https://{url}",
                    "language": "en",
                }
            )
            for url in KIT_URLS
        ],
        [
            LinkRecord(
                source_url=link.source,
                position=link.position,
                target_url=link.target,
                anchor_text=link.anchor,
                surrounding_text=f"See the {link.anchor} here.",
                is_internal=True,
            )
            for link in planted.links
        ],
    )
    main = pillars(variant)
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url,
                status_code=200,
                is_indexable=True,
                word_count=len(planted.body(url).split()),
                language="en",
                crawl_depth=None if url == KIT_ORPHAN else 1 + index % 3,
                page_type=PageType.PILLAR if url in main else PageType.ARTICLE,
            )
            for index, url in enumerate(KIT_URLS)
        ],
    )
    await graph.replace_links(
        tenant,
        sorted(by_source),
        [
            Link(
                source_url=link.source,
                target_url=link.target,
                position=link.position,
                anchor_text=link.anchor,
                surrounding_text=f"See the {link.anchor} here.",
            )
            for link in planted.links
        ],
    )
    rng = np.random.default_rng(89 + variant.rotation)
    centres = np.eye(2, DIMENSION)
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model, p.hubId = row.hub, "
        "p.isHubPillar = row.pillar, p.pageRankPercentile = row.pr, p.isOrphan = row.orphan, "
        "p.orphanLabel = row.label, p.isDeadEnd = row.dead_end, p.duplicateGroup = row.group, "
        "p.isCanonical = row.canonical",
        t=tenant,
        model=EMBEDDING_MODEL,
        rows=[
            {
                "url": url,
                "vec": (centres[index % 2] + 0.05 * rng.normal(size=DIMENSION)).tolist(),
                "hub": kit_hub(variant, url),
                "pillar": url in main,
                "pr": index / len(KIT_URLS),
                "orphan": url == KIT_ORPHAN,
                "label": variant.orphan_label.value if url == KIT_ORPHAN else None,
                "dead_end": url == KIT_DEAD_END,
                "group": 0 if url in {KIT_CANONICAL, KIT_COPY} else None,
                "canonical": (url == KIT_CANONICAL) if url in {KIT_CANONICAL, KIT_COPY} else None,
            }
            for index, url in enumerate(KIT_URLS)
        ],
    )
    await graph._auto(
        "UNWIND $hubs AS hub MERGE (h:Hub {tenantId: $t, hubId: hub.id}) "
        "SET h.size = hub.size, h.pillarUrl = hub.pillar, h.active = hub.active",
        t=tenant,
        hubs=[
            *(
                {"id": variant.hubs[side], "size": 7, "pillar": main[side], "active": True}
                for side in (0, 1)
            ),
            {"id": variant.retired_hub, "size": 0, "pillar": None, "active": False},
        ],
    )
    await resolve_tenant_keywords(graph, mongo, tenant)
    await mongo.replace_excluded_pages(tenant, planted.excluded)
    await mongo.insert_link_audit(tenant, planted.audit_run_id, planted.audit)
    await mongo.complete_link_audit(
        tenant,
        planted.audit_run_id,
        audited_at=AUDITED_AT,
        documents=len(planted.audit),
        edges=0,
    )

    folder = cache_dir / tenant
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(_choice_rows(planted), schema=CHOICES_SCHEMA),
        folder / ANCHOR_CHOICES_FILE,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_url": source,
                    "target_url": target,
                    "reason": reason.value,
                    "advice": UNANCHORED_ADVICE[reason],
                    "best_score": None,
                }
                for (source, target), reason in sorted(planted.unanchored.items())
            ],
            schema=UNANCHORED_SCHEMA,
        ),
        folder / UNANCHORED_FILE,
    )
    pq.write_table(
        pa.Table.from_pylist([planted.hub_pair.model_dump(mode="json")], schema=_PAIR_SCHEMA),
        folder / HUB_PAIRS_FILE,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [link.model_dump(mode="json") for link in planted.bridge_links], schema=_LINK_SCHEMA
        ),
        folder / BRIDGES_FILE,
    )

    await rank_pairs(graph, mongo, tenant, cache_dir=cache_dir)
    path = folder / RANKED_PAIRS_FILE
    table = pq.read_table(path)
    frame = table.to_pandas()
    ranked = set(zip(frame["source_url"], frame["target_url"], strict=True))
    assert ranked == planted.candidates(), "the ranker ranked other pairs than planted"
    # A ranked file from before the two pages were excluded.
    full = frame["source_url"] == FULL
    frame.loc[full, "rank_in_source"] += 1
    stale = pandas.DataFrame(
        [
            (FULL, OFFCUTS, frame.loc[full, "score"].max(), 1),
            (SITEMAP_PAGE, MIXED, frame["score"].median(), 1),
            (SITEMAP_PAGE, FULL, frame["score"].min(), 2),
        ],
        columns=["source_url", "target_url", "score", "rank_in_source"],
    ).assign(scorer=frame["scorer"].iloc[0], model_version=None)
    pq.write_table(
        pa.Table.from_pandas(
            pandas.concat([frame, stale], ignore_index=True),
            schema=table.schema,
            preserve_index=False,
        ),
        path,
    )
    return folder
