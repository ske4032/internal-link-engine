"""Output of corpus preparation: clean body text plus the links found in it.

Markdown link markup is the only place a scraped page records a link's anchor
text and the sentence around it, and the audit and anchor stages read both. So
preparation extracts every link while it strips the markup, and the clean body
keeps the anchor words in place.
"""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl

from linking_engine.urls import UrlKey


class ExtractedLink(BaseModel):
    """One link found in a page body, before any anchor classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: HttpUrl
    anchor_text: str = Field(min_length=1)
    surrounding_text: str
    is_internal: bool


class TemplateLink(BaseModel):
    """An internal link on a dropped template or breadcrumb line; never a body link (ADR-004)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: HttpUrl
    # footer: after the last line of body text; menu: above or within it.
    zone: Literal["menu", "footer"]


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
    template_links: tuple[TemplateLink, ...] = ()
    removed: tuple[tuple[str, int], ...] = ()


class TemplateInlinks(BaseModel):
    """Distinct pages linking to ``url`` from menu and from footer template lines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: UrlKey
    menu_inlinks: int = Field(ge=0)
    footer_inlinks: int = Field(ge=0)


# MongoDB documents; stored keys are the camelCase form of these field names.
# body_hash is required so a document written without one fails on read.

_SHA256_HEX = r"^[0-9a-f]{64}$"


class Heading(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(ge=1, le=6)
    text: str = Field(min_length=1)


class PageRecord(BaseModel):
    """``pages`` document, keyed by (tenantId, url)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: UrlKey
    # The url as crawled, before normalisation.
    crawl_url: str = Field(min_length=1)
    status_code: int | None = Field(ge=100, le=599)
    usable: bool | None
    meta_title: str | None
    meta_description: str | None
    h1: str | None
    headings: tuple[Heading, ...]
    body_text: str
    word_count: int = Field(ge=0)
    link_count: int = Field(ge=0)
    content_hash: str | None
    body_hash: str = Field(pattern=_SHA256_HEX)
    scraped_at: AwareDatetime | None
    source: str = Field(min_length=1)
    # Distinct crawled pages linking here from template lines, by zone; never edges.
    menu_inlinks: int = Field(default=0, ge=0)
    footer_inlinks: int = Field(default=0, ge=0)


class CrawlPage(BaseModel):
    """A page as the crawler stored it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    title: str | None = None
    description: str | None = None
    content: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    usable: bool | None = None
    content_hash: str | None = None
    scraped_at: AwareDatetime | None = None


class PageSummary(BaseModel):
    """``pages`` document without text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: UrlKey
    status_code: int | None = Field(ge=100, le=599)
    word_count: int = Field(ge=0)
    content_hash: str | None
    body_hash: str = Field(pattern=_SHA256_HEX)
    menu_inlinks: int = Field(default=0, ge=0)
    footer_inlinks: int = Field(default=0, ge=0)


class LinkRecord(BaseModel):
    """``links`` document, keyed by (tenantId, sourceUrl, position)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: UrlKey
    position: int = Field(ge=0)
    target_url: UrlKey
    anchor_text: str = Field(min_length=1)
    surrounding_text: str
    is_internal: bool


class GraphLoadReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: int = Field(ge=0)
    placeholders: int = Field(ge=0)
    links: int = Field(ge=0)
    external_links_skipped: int = Field(ge=0)
    self_links_skipped: int = Field(ge=0)
    stale_links_deleted: int = Field(ge=0)
    finished_at: datetime


class QueryParamEvidence(BaseModel):
    """How changing one query parameter alone changed crawled content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    urls: int = Field(ge=0)
    content_changed: int = Field(ge=0)
    content_same: int = Field(ge=0)
    kept: bool
