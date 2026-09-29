"""Stage 1 output: the scored result of auditing one existing link.

One instance per `LINKS_TO` edge. Written back onto the relationship and kept in
the `link_audit` collection until the next completed run replaces it.
"""

from datetime import datetime
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import ActionType, IssueFlag
from linking_engine.models.relevance import ScoreDistribution

AUDIT_VERDICTS: Final = frozenset({ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE})
TECHNICAL_FLAGS: Final = frozenset(
    {IssueFlag.BROKEN, IssueFlag.REDIRECTED, IssueFlag.NOINDEX_TARGET, IssueFlag.NOFOLLOW}
)


class AuditReason(StrEnum):
    """Why an edge carries a flag or its verdict; each result states it in plain words."""

    BROKEN_TARGET = "BROKEN_TARGET"
    REDIRECTED_TARGET = "REDIRECTED_TARGET"
    NOINDEX_TARGET = "NOINDEX_TARGET"
    NOFOLLOW = "NOFOLLOW"
    NON_CANONICAL_TARGET = "NON_CANONICAL_TARGET"
    GENERIC_ANCHOR = "GENERIC_ANCHOR"
    MISALIGNED_ANCHOR = "MISALIGNED_ANCHOR"
    OVER_OPTIMISED_ANCHOR = "OVER_OPTIMISED_ANCHOR"
    OFF_TOPIC = "OFF_TOPIC"
    WASTED_EQUITY = "WASTED_EQUITY"
    WEAK_FIT = "WEAK_FIT"
    REMOVE_OFF_TOPIC = "REMOVE_OFF_TOPIC"
    INDEX_LIKE_SOURCE_KEPT = "INDEX_LIKE_SOURCE_KEPT"
    LISTING_SOURCE = "LISTING_SOURCE"
    BETTER_PHRASE = "BETTER_PHRASE"
    NO_BETTER_PHRASE = "NO_BETTER_PHRASE"
    UNVERIFIED_TARGET = "UNVERIFIED_TARGET"


class AuditEdge(BaseModel):
    """One body link from a crawled page, with what the audit reads of it and of both pages.
    Urls are verbatim as stored, so results match their edges again on write-back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1)
    position: int = Field(ge=0)
    target_url: str = Field(min_length=1)
    anchor_text: str
    is_follow: bool = True
    # #16's stored scores; None before score-links, for a target without a vector, and the fit
    # always for a generic anchor.
    context_relevance: float | None = Field(default=None, ge=0, le=1)
    anchor_target_fit: float | None = Field(default=None, ge=0, le=1)
    source_language: str | None = None
    source_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    # For the source's link density; None when ingestion stored none.
    source_word_count: int | None = Field(default=None, ge=0)
    # A target never crawled: nothing about it can be verified.
    target_placeholder: bool = False
    target_status_code: int | None = Field(default=None, ge=100, le=599)
    target_indexable: bool | None = None
    # HDBSCAN hub; -1 is noise.
    target_hub_id: int | None = Field(default=None, ge=-1)
    target_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    # The canonical page of the target's duplicate group, set only when the target is a
    # non-canonical copy.
    target_canonical_url: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.target_canonical_url == self.target_url:
            raise ValueError("a non-canonical copy cannot be its own canonical page")
        return self


class AuditCutoff(BaseModel):
    """One per-tenant threshold the audit derived from the tenant's own links, and how; None
    when the data gave none, and the reason says why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float | None = None
    reason: str = Field(min_length=1)


class LinkAuditResult(BaseModel):
    """Scored dimensions, issue flags and a verdict for one existing link.

    `anchor_quality_score` is the composite on a 0-100 scale and the other
    dimensions are 0-1, which is why the bounds differ. A2 (audit with
    embeddings) is what fills `context_relevance` and `anchor_target_fit`; A1
    scores the rest without any vector. A dimension that cannot be computed is
    None rather than invented.

    `verdict` is ``None`` when the edge is healthy and needs no action. Every
    flag and verdict carries a reason in `reasons`, in plain words.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1)
    # Ordinal in the source body; (source_url, position) identifies the edge.
    position: int = Field(ge=0)
    target_url: str = Field(min_length=1)
    run_id: str = Field(min_length=1)

    anchor_quality_score: float | None = Field(default=None, ge=0, le=100)
    keyword_alignment: float | None = Field(default=None, ge=0, le=1)
    context_relevance: float | None = Field(default=None, ge=0, le=1)
    anchor_target_fit: float | None = Field(default=None, ge=0, le=1)
    equity_efficiency: float | None = Field(default=None, ge=0, le=1)

    # frozenset, not set: the model is frozen and therefore hashable, and one
    # edge can carry several defects at once.
    issue_flags: frozenset[IssueFlag]
    verdict: ActionType | None
    reasons: tuple[str, ...] = ()
    # REANCHOR only: a phrase of the source copy, extracted by the anchor ladder, never generated.
    proposed_anchor: str | None = Field(default=None, min_length=1)
    # FIX only: the canonical page to link to instead of its duplicate copy.
    fix_target: str | None = Field(default=None, min_length=1)
    # The target was never crawled: no score, flag or verdict.
    unverified: bool = False
    audited_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.verdict is not None and self.verdict not in AUDIT_VERDICTS:
            raise ValueError("an audit verdict is FIX, REANCHOR or REMOVE")
        if self.proposed_anchor is not None and self.verdict is not ActionType.REANCHOR:
            raise ValueError("only a REANCHOR verdict proposes an anchor")
        if self.fix_target is not None and self.verdict is not ActionType.FIX:
            raise ValueError("only a FIX verdict names a fix target")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("reasons cannot be blank")
        if len(self.reasons) < len(self.issue_flags) or (self.verdict and not self.reasons):
            raise ValueError("every flag and verdict carries a reason")
        scores = (
            self.anchor_quality_score,
            self.keyword_alignment,
            self.context_relevance,
            self.anchor_target_fit,
            self.equity_efficiency,
        )
        if self.unverified and (
            self.issue_flags or self.verdict or any(score is not None for score in scores)
        ):
            raise ValueError("an unverified link has no score, flag or verdict")
        return self


class LinkAuditReport(BaseModel):
    """One existing-link audit run over a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    # Body links from crawled pages, and those into a placeholder, which are unverified.
    links: int = Field(ge=0)
    unverified: int = Field(ge=0)
    source_pages: int = Field(ge=0)
    # Sources whose outbound links lie above the tenant's upper fence; never REMOVE.
    index_like_pages: int = Field(ge=0)
    # Listing and archive sources, whose link density lies above the tenant's fence: never
    # REANCHOR or REMOVE.
    listing_pages: int = Field(default=0, ge=0)
    # Natural link hubs, not evaluated: sitemap pages, then paginated pages, and the links from
    # or into them, none counted in links.
    sitemap_pages: int = Field(default=0, ge=0)
    sitemap_links: int = Field(default=0, ge=0)
    paginated_pages: int = Field(default=0, ge=0)
    paginated_links: int = Field(default=0, ge=0)
    by_flag: dict[IssueFlag, int]
    by_verdict: dict[ActionType, int]
    by_reason: dict[AuditReason, int]
    # Verified links with no verdict.
    healthy: int = Field(ge=0)
    # (source, target) pairs sent to the anchor ladder, and REANCHOR verdicts with a phrase.
    ladder_pairs: int = Field(ge=0)
    proposals: int = Field(ge=0)
    cutoffs: tuple[AuditCutoff, ...]
    # A2 ran on #16's stored scores; otherwise A1 alone, and why.
    embeddings: bool
    embeddings_skipped_reason: str | None = None
    # Links whose keyword alignment includes the phrase-keyword cosine, and why Voyage served
    # none of the missing vectors when it did not.
    keyword_cosines: int = Field(ge=0)
    vectors_skipped_reason: str | None = None
    keyword_alignment: ScoreDistribution | None = None
    context_relevance: ScoreDistribution | None = None
    anchor_target_fit: ScoreDistribution | None = None
    equity_efficiency: ScoreDistribution | None = None
    # The 0-100 composite divided by 100.
    anchor_quality: ScoreDistribution | None = None
    seconds: float = Field(ge=0)
    finished_at: datetime

    @property
    def fixable_rate(self) -> float | None:
        """Links with a verdict over every audited link; None without links."""
        return sum(self.by_verdict.values()) / self.links if self.links else None

    @property
    def verified_fixable_rate(self) -> float | None:
        """Links with a verdict over the links into a crawled page, the only ones the audit
        can judge; None without such links."""
        verified = self.links - self.unverified
        return sum(self.by_verdict.values()) / verified if verified else None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.by_verdict) - AUDIT_VERDICTS:
            raise ValueError("an audit verdict is FIX, REANCHOR or REMOVE")
        counts = [*self.by_flag.values(), *self.by_verdict.values(), *self.by_reason.values()]
        if any(count < 0 for count in counts):
            raise ValueError("counts cannot be negative")
        if sum(self.by_verdict.values()) + self.healthy + self.unverified != self.links:
            raise ValueError("every link is unverified, healthy or has one verdict")
        if self.proposals > self.by_verdict.get(ActionType.REANCHOR, 0):
            raise ValueError("more proposals than REANCHOR verdicts")
        if self.index_like_pages > self.source_pages or self.listing_pages > self.source_pages:
            raise ValueError("more index-like or listing pages than source pages")
        if self.embeddings == (self.embeddings_skipped_reason is not None):
            raise ValueError("A1 alone is reported with its reason, A2 without one")
        if self.keyword_cosines > self.links - self.unverified:
            raise ValueError("more keyword cosines than verified links")
        return self
