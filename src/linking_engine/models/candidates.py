"""Candidate retrieval: the pages that could link to each target, before any ranking.

Eligibility is four hard constraints and nothing else: the target is indexable, the
source is not the target, the source does not already link to the target, and neither
is a non-canonical duplicate copy. Every other consideration is a ranker feature.

Urls are kept verbatim, as stored, because some tenants store bare paths that are not
url keys.
"""

from datetime import datetime
from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

VectorIndex = Literal["page_content", "page_gnn"]


class CandidateTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    # No isIndexable flag is stored, so the page counts as indexable from its 2xx status.
    indexable_assumed: bool


class TargetSelection(BaseModel):
    """The tenant's crawled pages split into queryable targets and why the rest are not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crawled_pages: int = Field(ge=0)
    not_indexable: int = Field(ge=0)
    # Indexable pages with no vector in the chosen index, so they cannot be scored.
    without_vector: int = Field(ge=0)
    targets: tuple[CandidateTarget, ...]

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if len({t.url for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate target urls")
        if self.not_indexable + self.without_vector + len(self.targets) != self.crawled_pages:
            raise ValueError("targets and exclusions must add up to the crawled pages")
        return self


class TargetCandidates(BaseModel):
    """The nearest eligible sources of one target, capped per target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: str = Field(min_length=1)
    # Best first by cosine, url ascending on ties.
    sources: tuple[str, ...]
    similarities: tuple[float, ...]
    # Sources in the tenant that pass the hard constraints, before the cap.
    eligible: int = Field(ge=0)
    # Pool pages that already link to the target, all excluded, and how many of them would
    # have ranked above the last kept candidate had they been eligible (all of them when the
    # target is short).
    linked: int = Field(ge=0)
    linked_nearer: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(self.sources) != len(self.similarities):
            raise ValueError("one similarity per source")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("duplicate source urls")
        if self.target_url in self.sources:
            raise ValueError("a target cannot be its own source")
        if any(not -1.0 <= similarity <= 1.0 for similarity in self.similarities):
            raise ValueError("similarity must be a cosine in [-1, 1]")
        ranked = zip(self.sources, self.similarities, strict=True)
        if any((-a[1], a[0]) > (-b[1], b[0]) for a, b in pairwise(ranked)):
            raise ValueError("sources must be ordered best first, url ascending on ties")
        if len(self.sources) > self.eligible:
            raise ValueError("more sources than eligible pages")
        if self.linked_nearer > self.linked:
            raise ValueError("linked_nearer cannot exceed linked")
        return self


class CandidateReport(BaseModel):
    """One candidate retrieval run over a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    index: VectorIndex
    per_target: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    crawled_pages: int = Field(ge=0)
    not_indexable: int = Field(ge=0)
    without_vector: int = Field(ge=0)
    targets: int = Field(ge=0)
    indexable_assumed: int = Field(ge=0)
    # Crawled pages with a vector: every one is a possible source, except non-canonical
    # duplicate copies, which are neither sources nor targets.
    source_pages: int = Field(ge=0)
    non_canonical_excluded: int = Field(default=0, ge=0)
    candidates: int = Field(ge=0)
    # Targets that reached the cap, got fewer, and got none.
    full_targets: int = Field(ge=0)
    short_targets: int = Field(ge=0)
    empty_targets: int = Field(ge=0)
    min_per_target: int | None = Field(default=None, ge=0)
    median_per_target: float | None = Field(default=None, ge=0)
    max_per_target: int | None = Field(default=None, ge=0)
    # Sums of the targets' `linked` and `linked_nearer`: existing links from pool pages only.
    linked_pairs: int = Field(ge=0)
    linked_nearer: int = Field(ge=0)
    # linked_nearer / (candidates + linked_nearer), pooled over all targets: the share of the
    # neighbourhood down to each last kept candidate that already links to its target. None
    # when that neighbourhood is empty.
    drop_rate: float | None = Field(default=None, ge=0, le=1)
    load_seconds: float = Field(ge=0)
    search_seconds: float = Field(ge=0)
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.full_targets + self.short_targets + self.empty_targets != self.targets:
            raise ValueError("full + short + empty targets must equal targets")
        if self.indexable_assumed > self.targets:
            raise ValueError("more assumed-indexable targets than targets")
        if self.targets > self.source_pages:
            raise ValueError("every target has a vector, so it is also a source page")
        if self.linked_nearer > self.linked_pairs:
            raise ValueError("linked_nearer cannot exceed linked_pairs")
        stats = (self.min_per_target, self.median_per_target, self.max_per_target)
        if any((stat is None) != (self.targets == 0) for stat in stats):
            raise ValueError("per-target stats are set exactly when there are targets")
        if self.max_per_target is not None and self.max_per_target > self.per_target:
            raise ValueError("no target can keep more than per_target sources")
        if (self.drop_rate is None) != (self.candidates + self.linked_nearer == 0):
            raise ValueError("drop_rate is set exactly when the neighbourhood is not empty")
        return self


class CandidateSet(BaseModel):
    """Every target's candidates and the report describing them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: CandidateReport
    targets: tuple[TargetCandidates, ...]

    @model_validator(mode="after")
    def _matches_report(self) -> Self:
        if len(self.targets) != self.report.targets:
            raise ValueError("one entry per reported target")
        if len({t.target_url for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate target urls")
        report = self.report
        counts = [len(t.sources) for t in self.targets]
        if sum(counts) != report.candidates:
            raise ValueError("candidate count does not match the report")
        if any(count > report.per_target for count in counts):
            raise ValueError("no target can keep more than per_target sources")
        if (
            sum(t.linked for t in self.targets) != report.linked_pairs
            or sum(t.linked_nearer for t in self.targets) != report.linked_nearer
        ):
            raise ValueError("linked counts do not match the report")
        split = (
            sum(count == report.per_target for count in counts),
            sum(0 < count < report.per_target for count in counts),
            counts.count(0),
        )
        if split != (report.full_targets, report.short_targets, report.empty_targets):
            raise ValueError("full, short and empty targets do not match the report")
        return self
