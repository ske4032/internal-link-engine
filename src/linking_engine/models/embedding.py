"""Embedding input and output."""

from pydantic import BaseModel, ConfigDict, Field


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
