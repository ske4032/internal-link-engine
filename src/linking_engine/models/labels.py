"""Hand labels of candidate links: the blind export a reviewer labels, and the label events
imported into anchor_feedback."""

from datetime import datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import (
    ActionType,
    AnchorRung,
    AnchorType,
    RecommendationStatus,
    ScorerName,
)

# The label words of the export file.
LABEL_WORDS: Final = {
    "accept": RecommendationStatus.ACCEPTED,
    "modify": RecommendationStatus.MODIFIED,
    "dismiss": RecommendationStatus.DISMISSED,
}
# Relevance grades the ranker trains on; a pair without a label is 0.
GRADES: Final = {
    RecommendationStatus.ACCEPTED: 3,
    RecommendationStatus.MODIFIED: 2,
    RecommendationStatus.DISMISSED: 1,
}
UNLABELLED_GRADE: Final = 0


class LabelSettings(BaseModel):
    """How many source pages an export samples, and how many candidate pairs of each."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: int = Field(default=20, ge=1)
    # Each sampled page's anchored candidates, split into this many score bands, one pair each.
    pairs_per_page: int = Field(default=10, ge=2)
    seed: int = Field(ge=0)


class ExportedPair(BaseModel):
    """One exported candidate pair as it was proposed; kept on the server, never in the file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    target_url: str = Field(min_length=1)
    anchor: str = Field(min_length=1)
    anchor_type: AnchorType
    rung: AnchorRung
    keyword: str = Field(min_length=1)
    sentence: str = Field(min_length=1)
    # The anchor's character offset in its sentence.
    anchor_start: int = Field(ge=0)
    score: float
    rank_in_source: int = Field(ge=1)
    # The pair's score band within its source page, 1 the best.
    band: int = Field(ge=1)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.source_url == self.target_url:
            raise ValueError("a page cannot link to itself")
        if self.sentence[self.anchor_start : self.anchor_start + len(self.anchor)] != self.anchor:
            raise ValueError("the anchor is not at its offset in the sentence")
        return self


class LabelExport(BaseModel):
    """One export of a tenant's candidate pairs for hand labelling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_id: str = Field(min_length=1)
    created_at: datetime
    seed: int = Field(ge=0)
    pages: int = Field(ge=1)
    pairs_per_page: int = Field(ge=2)
    pairs: int = Field(ge=2)
    scorer: ScorerName
    model_version: str | None = None
    weights_version: str = Field(min_length=1)
    # Source pages with at least pairs_per_page anchored candidates, and the pairs they hold.
    eligible_pages: int = Field(ge=1)
    eligible_pairs: int = Field(ge=2)
    # Source pages with anchored candidates, but fewer than pairs_per_page.
    small_pages: int = Field(ge=0)
    file_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if self.pairs != self.pages * self.pairs_per_page:
            raise ValueError("an export holds pairs_per_page pairs of each page")
        if self.pages > self.eligible_pages or self.pairs > self.eligible_pairs:
            raise ValueError("an export samples eligible pages and pairs only")
        if (self.scorer is ScorerName.LEARNED) != (self.model_version is not None):
            raise ValueError("model_version is set when the learned ranker scored the pairs only")
        return self


class LabelEvent(ExportedPair):
    """One pair's hand label, as stored in anchor_feedback with the export's snapshot."""

    import_id: str = Field(min_length=1)
    export_id: str = Field(min_length=1)
    scorer: ScorerName
    model_version: str | None = None
    weights_version: str = Field(min_length=1)
    action_type: Literal[ActionType.ADD_LINK] = ActionType.ADD_LINK
    feedback_source: Literal["hand_label"] = "hand_label"
    status: RecommendationStatus
    # The link is wanted: accepted as proposed, or with another anchor.
    accepted: bool
    grade: int = Field(ge=1, le=3)
    # The anchor the reviewer would use: the proposed one when accepted, none when dismissed.
    anchor_used: str | None = Field(default=None, min_length=1)
    reason: str | None = Field(default=None, min_length=1)
    reviewer: str = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def _label(self) -> Self:
        if self.status is RecommendationStatus.PENDING:
            raise ValueError("a label is accepted, modified or dismissed")
        if self.grade != GRADES[self.status]:
            raise ValueError(f"a {self.status.value} label has grade {GRADES[self.status]}")
        if self.accepted != (self.status is not RecommendationStatus.DISMISSED):
            raise ValueError("accepted and modified links are wanted, dismissed ones are not")
        used = {
            RecommendationStatus.ACCEPTED: self.anchor_used == self.anchor,
            RecommendationStatus.MODIFIED: self.anchor_used not in {None, self.anchor},
            RecommendationStatus.DISMISSED: self.anchor_used is None,
        }
        if not used[self.status]:
            raise ValueError(
                "anchor_used is the proposed anchor when accepted, another when modified, "
                "none when dismissed"
            )
        return self


class LabelProblem(BaseModel):
    """Why a label file, or one of its rows, cannot be imported."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The row's line in the file, the header line 1; none for the whole file.
    line: int | None = Field(default=None, ge=2)
    pair_id: str | None = None
    problem: str = Field(min_length=1)


class LabelImportReport(BaseModel):
    """One label file checked, and imported unless only checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    export_id: str = Field(min_length=1)
    # None when the file was only checked.
    import_id: str | None = None
    rows: int = Field(ge=1)
    labelled: int = Field(ge=1)
    by_status: dict[RecommendationStatus, int]

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if sum(self.by_status.values()) != self.labelled or self.labelled > self.rows:
            raise ValueError("labelled rows split by status and never exceed the rows")
        return self
