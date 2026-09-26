"""Embedding input and output."""

from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class PageText(BaseModel):
    """A page's text to embed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    text: str


class PageEmbedding(BaseModel):
    """A unit-norm page vector; token counts use the model's tokenizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)
    tokens: int = Field(ge=0)
    original_tokens: int = Field(ge=0)
    truncated: bool


class EmbeddingBatch(BaseModel):
    """Vectors from one Voyage request, in input order, plus the tokens Voyage billed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embeddings: tuple[PageEmbedding, ...] = Field(min_length=1)
    api_tokens: int = Field(ge=0)


class EmbeddingTarget(BaseModel):
    """A page the resume query selected, with the graph's bodyHash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    body_hash: str | None


class EmbeddingSelection(BaseModel):
    """Result of the single up-front resume query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[EmbeddingTarget, ...]
    up_to_date: int = Field(ge=0)
    placeholders: int = Field(ge=0)
    non_2xx: int = Field(ge=0)


class EmbeddingModelCount(BaseModel):
    """Stored vectors per embeddingModel value; None means vectors without a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str | None
    vectors: int = Field(ge=1)


class EmbedRunReport(BaseModel):
    """Outcome of one embedding run; every selected url is embedded or skipped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    selected: int = Field(ge=0)
    embedded: int = Field(ge=0)
    skipped_not_usable: int = Field(ge=0)
    skipped_empty_body: int = Field(ge=0)
    # In the graph but not in Mongo.
    skipped_missing: int = Field(ge=0)
    # Body text does not hash to the graph's bodyHash.
    skipped_hash_mismatch: int = Field(ge=0)
    up_to_date: int = Field(ge=0)
    placeholders: int = Field(ge=0)
    non_2xx: int = Field(ge=0)
    flushes: int = Field(ge=0)
    api_tokens: int = Field(ge=0)
    tokens: int = Field(ge=0)
    truncated: int = Field(ge=0)
    elapsed_s: float = Field(ge=0)
    finished_at: AwareDatetime

    @model_validator(mode="after")
    def _selected_accounted_for(self) -> Self:
        handled = (
            self.embedded
            + self.skipped_not_usable
            + self.skipped_empty_body
            + self.skipped_missing
            + self.skipped_hash_mismatch
        )
        if self.selected != handled:
            raise ValueError("selected must equal embedded plus every skipped count")
        return self
