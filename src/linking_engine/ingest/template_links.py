"""Menu and footer inlinks: which pages reach a page only through site template."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from linking_engine.models import TemplateInlinks
from linking_engine.urls import normalise_url

if TYPE_CHECKING:
    from collections.abc import Iterable

    from linking_engine.models import CleanedPage


def count_template_inlinks(pages: Iterable[tuple[str, CleanedPage]]) -> list[TemplateInlinks]:
    """Per target key, the distinct other pages linking to it from each zone, sorted by key."""
    sources: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for source_url, page in pages:
        source = normalise_url(source_url)
        for link in page.template_links:
            try:
                target = normalise_url(str(link.target_url))
            except ValueError:
                # No key, so it can never be a crawled page.
                continue
            if target != source:
                sources[target, link.zone].add(source)
    return [
        TemplateInlinks(
            url=target,
            menu_inlinks=len(sources.get((target, "menu"), ())),
            footer_inlinks=len(sources.get((target, "footer"), ())),
        )
        for target in sorted({target for target, _ in sources})
    ]
