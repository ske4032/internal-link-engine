"""Output of corpus preparation: clean body text plus the links found in it.

Markdown link markup is the only place a scraped page records a link's anchor
text and the sentence around it, and the audit and anchor stages read both. So
preparation extracts every link while it strips the markup, and the clean body
keeps the anchor words in place.
"""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ExtractedLink(BaseModel):
    """One link found in a page body, before any anchor classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: HttpUrl
    anchor_text: str = Field(min_length=1)
    surrounding_text: str
    is_internal: bool


class CleanedPage(BaseModel):
    """A page's text, ready for storage and for embedding.

    ``headings`` is the page outline as ``(level, text)`` pairs in document
    order, levels 1 to 6. ``removed`` counts what cleaning took out, by kind, as
    sorted ``(kind, count)`` pairs rather than a dict so the model stays hashable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    title: str | None = None
    h1: str | None = None
    headings: tuple[tuple[int, str], ...] = ()
    body_text: str
    links: tuple[ExtractedLink, ...] = ()
    removed: tuple[tuple[str, int], ...] = ()
