"""Keyword resolution: each page's target keyword, and the keyword edges written to the graph."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import KeywordRung, KeywordSource


class ResolvedKeyword(BaseModel):
    """The keyword a page's inbound anchors should be about, and the rung that chose it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(min_length=2)
    rung: KeywordRung
    # Estimated click upside, impressions x (CTR@1 - CTR@position); GSC rung only.
    opportunity_value: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _opportunity_for_gsc_only(self) -> Self:
        if (self.opportunity_value is not None) != (self.rung is KeywordRung.GSC):
            raise ValueError("opportunity_value is set exactly for the GSC rung")
        return self


class KeywordTarget(BaseModel):
    """One `TARGETS_KEYWORD` edge to write: a page, a keyword and where the keyword came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(min_length=2)
    source: KeywordSource
    priority: int | None = Field(default=None, ge=1, le=5)
    is_primary: bool = False
    # The page's resolved keyword, which the anchor ladder reads; one per page.
    rung: KeywordRung | None = None
    # Position in the page's ranked keyword set: 1 is the resolved keyword, then the other
    # strategic keywords, then further usable GSC queries. A page's inbound anchors are spread
    # across the set.
    rank: int | None = Field(default=None, ge=1)

    @property
    def resolved(self) -> bool:
        return self.rung is not None

    @model_validator(mode="before")
    @classmethod
    def _resolved_defaults_to_rank_one(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("rung") is not None and data.get("rank") is None:
            return {**data, "rank": 1}
        return data

    @model_validator(mode="after")
    def _resolved_ranks_first(self) -> Self:
        if self.resolved and self.rank != 1:
            raise ValueError("the resolved keyword has rank 1")
        if not self.resolved and self.rank == 1:
            raise ValueError("rank 1 is reserved for the resolved keyword")
        return self


class KeywordReport(BaseModel):
    """One keyword resolution run over a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pages: int = Field(ge=0)
    resolved: int = Field(ge=0)
    by_rung: dict[KeywordRung, int]
    # False when the tenant's GSC data is too thin for a CTR curve: the GSC rung is skipped.
    gsc_enabled: bool
    gsc_rows: int = Field(ge=0)
    # GSC candidates rejected by the quality bar, over all pages.
    gsc_rejected: int = Field(ge=0)
    # The tenant's brand as detected from its titles, at the end and at the start.
    brand_suffix: str | None = None
    brand_prefix: str | None = None
    # H1 and title fallbacks the quality checks rejected, by reason ("h1_repeated",
    # "title_generic", ...), and kept fallbacks longer than MAX_KEYWORD_TOKENS words.
    fallbacks_rejected: dict[str, int] = Field(default_factory=dict)
    long_fallbacks: int = Field(default=0, ge=0)
    # Keywords ranked after the resolved one, and the pages that have any.
    secondary_keywords: int = Field(default=0, ge=0)
    pages_with_secondaries: int = Field(default=0, ge=0)
    edges_written: dict[KeywordSource, int]
    stale_edges_deleted: dict[KeywordSource, int]
    # Strategic keyword rows whose url is not a crawled page.
    skipped_rows: int = Field(ge=0)
    by_language: dict[str, int]
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.pages_with_secondaries > self.resolved:
            raise ValueError("only resolved pages have secondary keywords")
        if self.resolved > self.pages:
            raise ValueError("more resolved pages than pages")
        if (
            any(count < 0 for count in self.by_rung.values())
            or sum(self.by_rung.values()) != self.resolved
        ):
            raise ValueError("by_rung must add up to the resolved pages")
        if any(count < 0 for count in self.fallbacks_rejected.values()):
            raise ValueError("rejection counts cannot be negative")
        if sum(self.by_language.values()) != self.pages:
            raise ValueError("by_language must add up to the pages")
        return self
