"""Stage 1 output: the scored result of auditing one existing link.

One instance per `LINKS_TO` edge. Written back onto the relationship and kept in
the `link_audit` collection so issue history over time is queryable.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from linking_engine.models.enums import ActionType, IssueFlag
from linking_engine.urls import UrlKey


class LinkAuditResult(BaseModel):
    """Six scored dimensions and a verdict for one existing link.

    `anchor_quality_score` is the composite on a 0-100 scale and the other five
    dimensions are the 0-1 components that feed it, which is why the bounds
    differ. A2 (audit with embeddings) is what fills `context_relevance` and
    `anchor_target_fit`; A1 can score the rest without any vector.

    `verdict` is ``None`` when the edge is healthy and needs no action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: UrlKey
    target_url: UrlKey

    anchor_quality_score: float = Field(ge=0, le=100)
    keyword_alignment: float = Field(ge=0, le=1)
    context_relevance: float = Field(ge=0, le=1)
    anchor_target_fit: float = Field(ge=0, le=1)
    equity_efficiency: float = Field(ge=0, le=1)

    # frozenset, not set: the model is frozen and therefore hashable, and one
    # edge can carry several defects at once.
    issue_flags: frozenset[IssueFlag]
    verdict: ActionType | None
    audited_at: datetime
