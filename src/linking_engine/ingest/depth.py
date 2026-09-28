"""Crawl depth: the fewest clicks from the site root, over every link a visitor can follow."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def crawl_depths(edges: Mapping[str, Iterable[str]], roots: Iterable[str]) -> dict[str, int]:
    """Breadth-first distance of every url reachable from ``roots`` (depth 0).

    ``edges`` maps a url to the urls it links to. Unreachable urls are absent.
    """
    depths = dict.fromkeys(roots, 0)
    queue = deque(depths)
    while queue:
        url = queue.popleft()
        depth = depths[url] + 1
        for target in edges.get(url, ()):
            if target not in depths:
                depths[target] = depth
                queue.append(target)
    return depths
