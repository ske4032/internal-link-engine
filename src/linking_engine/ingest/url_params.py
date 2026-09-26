"""Which query parameters change a site's content, measured on its crawl."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from linking_engine.models import QueryParamEvidence
from linking_engine.urls import is_kept_param, normalise_url

if TYPE_CHECKING:
    from collections.abc import Iterable


def query_param_evidence(pages: Iterable[tuple[str, str | None]]) -> list[QueryParamEvidence]:
    """For crawled (url, content hash) pairs: per parameter, how many URL pairs that
    differ only in it had different content and how many had the same."""
    groups: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    for url, content_hash in pages:
        parts = urlsplit(url)
        params = {name.lower(): value for name, value in parse_qsl(parts.query)}
        for name in params:
            seen[name] += 1
        if content_hash:
            base = normalise_url(urlunsplit(parts._replace(query="", fragment="")))
            groups[base].append((params, content_hash))

    changed: dict[str, int] = defaultdict(int)
    same: dict[str, int] = defaultdict(int)
    for variants in groups.values():
        for (a, hash_a), (b, hash_b) in combinations(variants, 2):
            differing = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
            if len(differing) == 1:
                (name,) = differing
                if hash_a == hash_b:
                    same[name] += 1
                else:
                    changed[name] += 1
    return sorted(
        (
            QueryParamEvidence(
                name=name,
                urls=count,
                content_changed=changed[name],
                content_same=same[name],
                kept=is_kept_param(name),
            )
            for name, count in seen.items()
        ),
        key=lambda item: (-item.content_changed, -item.urls, item.name),
    )
