"""Exact duplicate pages: one body served at several urls of one language.

Each group has one canonical copy, the only one of its urls that retrieval keeps as a link
target or source. Near duplicates (different bodies) are never grouped. Urls are kept
verbatim, as stored.
"""

from datetime import datetime
from itertools import pairwise
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Url = Annotated[str, Field(min_length=1)]


class DuplicateInput(BaseModel):
    """What the grouping reads about one crawled page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: _Url
    body_hash: str = Field(min_length=1)
    language: str | None = None
    # The candidate-target rule: the isIndexable flag, else the page's 2xx status.
    indexable: bool
    # Body links from other crawled pages of the tenant, distinct pages.
    inbound: int = Field(ge=0)


class DuplicateGroup(BaseModel):
    """The canonical copy of one body and its other urls, ascending."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: int = Field(ge=0)
    canonical: _Url
    copies: tuple[_Url, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if any(a >= b for a, b in pairwise(self.copies)):
            raise ValueError("copies must be unique and ascending")
        if self.canonical in self.copies:
            raise ValueError("the canonical url cannot also be a copy")
        return self


class DuplicateReport(BaseModel):
    """One duplicate grouping run over a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    # Ordered by canonical url; a group's id is its position, so ids are stable on unchanged data.
    groups: tuple[DuplicateGroup, ...]
    pages_in_groups: int = Field(ge=0)
    non_canonical: int = Field(ge=0)
    largest_group: int = Field(ge=0)
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if [group.group_id for group in self.groups] != list(range(len(self.groups))):
            raise ValueError("group ids must count up from 0 in order")
        if any(a.canonical >= b.canonical for a, b in pairwise(self.groups)):
            raise ValueError("groups must be ordered by canonical url")
        urls = [url for group in self.groups for url in (group.canonical, *group.copies)]
        if len(set(urls)) != len(urls):
            raise ValueError("a page can belong to one group only")
        if self.pages_in_groups != len(urls):
            raise ValueError("pages_in_groups must count every url of every group")
        if self.non_canonical != len(urls) - len(self.groups):
            raise ValueError("non_canonical must count every copy of every group")
        if self.largest_group != max((1 + len(g.copies) for g in self.groups), default=0):
            raise ValueError("largest_group must be the size of the largest group")
        return self
