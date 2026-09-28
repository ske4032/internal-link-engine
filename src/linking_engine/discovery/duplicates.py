"""Exact duplicate pages: crawled pages of one language with an identical body, grouped with
one canonical copy each."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from linking_engine.models import DuplicateGroup, DuplicateInput

if TYPE_CHECKING:
    from collections.abc import Iterable


def canonical_order(page: DuplicateInput) -> tuple[bool, int, int, str]:
    """Indexable first, so the canonical copy can be a link target whenever any copy can;
    then the most inbound body links, the shortest url and the url."""
    return (not page.indexable, -page.inbound, len(page.url), page.url)


def duplicate_groups(pages: Iterable[DuplicateInput]) -> list[DuplicateGroup]:
    """Every set of two or more pages sharing a body hash and a language (None is a language of
    its own), ordered by canonical url; a group's id is its position."""
    members: defaultdict[tuple[str, str | None], list[DuplicateInput]] = defaultdict(list)
    seen: set[str] = set()
    for page in pages:
        if page.url in seen:
            raise ValueError(f"page {page.url!r} is listed twice")
        seen.add(page.url)
        members[page.body_hash, page.language].append(page)
    ranked = sorted(
        (sorted(group, key=canonical_order) for group in members.values() if len(group) > 1),
        key=lambda group: group[0].url,
    )
    return [
        DuplicateGroup(
            group_id=group_id,
            canonical=canonical.url,
            copies=tuple(sorted(page.url for page in copies)),
        )
        for group_id, (canonical, *copies) in enumerate(ranked)
    ]
