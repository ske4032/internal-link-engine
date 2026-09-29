"""URL identity shared by every tenant: one normalised key per page."""

from __future__ import annotations

import posixpath
import re
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Final
from urllib.parse import SplitResult, parse_qsl, quote, unquote, urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

_WEB_SCHEMES = frozenset({"http", "https"})
_WWW = re.compile(r"^www\d*\.(?=[^.]+\.)")  # keep a bare "www.com"
# "mailto:", "tel:", "javascript:"; a digit after the colon is a host:port instead.
_OTHER_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:(?!\d)")
_PATH_PARAMS = re.compile(r";[^/]*")  # ;jsessionid=... and other matrix parameters
_INDEX_DOCUMENT = re.compile(r"/(?:index|default)\.(?:html?|php|aspx?|jsp)$", re.IGNORECASE)
_SLASHES = re.compile(r"/{2,}")
_HOST = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")
_ID_VALUE = re.compile(r"^[\w.-]{1,64}$")
# Characters a path may carry unescaped (RFC 3986 pchar plus '/').
_PATH_SAFE = "/:@!$&'()*+,;=-._~"

# Query parameters that select a different page; every other parameter is stripped.
PAGE_NUMBER_PARAMS: Final = frozenset(
    {
        "page",
        "paged",
        "pg",
        "pagenum",
        "page_no",
        "pagenumber",
        "seite",
        "pagina",
        "sayfa",
        "strona",
    }
)
OFFSET_PARAMS: Final = frozenset({"start", "offset", "limitstart"})
DOCUMENT_ID_PARAMS: Final = frozenset({"page_id", "p", "post", "id", "article", "product_id"})

# Pagination parameters the url keys do not keep as page numbers or offsets.
EXTRA_PAGINATION_PARAMS: Final = frozenset({"p", "currentpage", "pageindex", "page_index"})
# Query parameters that make a url a page of a paginated listing, whatever their value.
PAGINATION_PARAMS: Final = PAGE_NUMBER_PARAMS | OFFSET_PARAMS | EXTRA_PAGINATION_PARAMS
# A path segment that numbers a page: page-2, or /page/2 and /p/2 as two segments.
_PAGE_SEGMENT: Final = re.compile(r"page-\d+")
_PAGE_PREFIXES: Final = frozenset({"page", "p"})
_NOT_ALNUM: Final = re.compile(r"[^0-9a-z]+")
_SCHEME: Final = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class UrlRules(BaseModel):
    """Per-tenant additions to the built-in query parameter rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keep_params: frozenset[str] = frozenset()
    drop_params: frozenset[str] = frozenset()

    @field_validator("keep_params", "drop_params", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        if isinstance(value, list | tuple | set | frozenset):
            return frozenset(str(item).strip().lower() for item in value if str(item).strip())
        return value


_NO_RULES: Final = UrlRules()
_RULES: ContextVar[UrlRules] = ContextVar("url_rules")


@contextmanager
def url_rules(rules: UrlRules) -> Iterator[None]:
    """Apply a tenant's rules to every absolute URL normalised in this context."""
    token = _RULES.set(rules)
    try:
        yield
    finally:
        _RULES.reset(token)


def normalise_url(url: str) -> str:
    """Key for a web URL: no scheme, www, port 80/443, credentials or fragment;
    lowercase host and path; dot segments, duplicate and trailing slashes, index
    documents and path parameters removed; percent-encoding canonical. Only
    pagination and document-id query parameters survive (see ``_query``).

    ``https://www.Example.com/Blog//a/../News/index.html?utm=x&page=2#top`` -> ``example.com/blog/news?page=2``.
    A key (no scheme) keeps its query as is, so re-normalising a key never changes it.
    Raises ValueError for anything that is not an http(s) URL with a host.
    """
    text = url.strip()
    if not text:
        raise ValueError("url is empty")
    is_key = "://" not in text and not text.startswith("//")
    if "://" not in text:
        if _OTHER_SCHEME.match(text):
            raise ValueError(f"not an http(s) url: {url!r}")
        text = f"http:{text}" if text.startswith("//") else f"http://{text}"
    parts = urlsplit(text)
    if parts.scheme.lower() not in _WEB_SCHEMES:
        raise ValueError(f"not an http(s) url: {url!r}")

    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise ValueError(f"url has no host: {url!r}")
    try:
        host = _WWW.sub("", host).encode("idna").decode("ascii")
        port = parts.port
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"invalid host or port in {url!r}") from error
    if not _HOST.match(host):
        raise ValueError(f"invalid host in {url!r}")
    if port is not None and port not in (80, 443):
        host = f"{host}:{port}"

    path = _SLASHES.sub("/", _PATH_PARAMS.sub("", unquote(parts.path)))
    path = posixpath.normpath(path) if path else "/"
    path = _INDEX_DOCUMENT.sub("/", path).lower().rstrip("/")
    query = _query(parts.query, keep_all=is_key)
    return host + quote(path, safe=_PATH_SAFE) + (f"?{query}" if query else "")


def _query(raw: str, *, keep_all: bool) -> str:
    """Pagination with a page past the first, and document ids, sorted by name.
    ``page=1`` and ``offset=0`` are the base page, so they are dropped."""
    rules = _RULES.get(_NO_RULES)
    kept: dict[str, str] = {}
    for name, value in parse_qsl(raw):
        name, value = name.strip().lower(), value.strip().lower()
        if name in kept or (name in rules.drop_params and not keep_all):
            continue
        if keep_all:
            kept[name] = value
        elif name in PAGE_NUMBER_PARAMS or name in OFFSET_PARAMS:
            first = 1 if name in PAGE_NUMBER_PARAMS else 0
            if value.isdigit() and int(value) > first:
                kept[name] = str(int(value))
        elif (name in DOCUMENT_ID_PARAMS or name in rules.keep_params) and _ID_VALUE.match(value):
            kept[name] = str(int(value)) if value.isdigit() else value
    return "&".join(f"{name}={quote(value, safe='')}" for name, value in sorted(kept.items()))


def is_kept_param(name: str) -> bool:
    """Whether the active rules keep this query parameter in keys."""
    rules, name = _RULES.get(_NO_RULES), name.strip().lower()
    if name in rules.drop_params:
        return False
    return name in PAGE_NUMBER_PARAMS | OFFSET_PARAMS | DOCUMENT_ID_PARAMS | rules.keep_params


def host_of(key: str) -> str:
    """The host part of a normalised key."""
    return re.split(r"[/?]", key, maxsplit=1)[0]


def _url_parts(url: str) -> SplitResult | None:
    """The url's parts. A stored key has no scheme ("host/path"), so it is split as
    network-path reference, which keeps its host out of the path; a bare path stays a path.
    None for a url that cannot be split."""
    text = url.strip()
    try:
        return urlsplit(text if _SCHEME.match(text) or text.startswith("/") else f"//{text}")
    except ValueError:
        return None


def is_sitemap(url: str) -> bool:
    """A path segment naming a sitemap, case-insensitive: /sitemap, /sitemap.html,
    /html-sitemap, /site-map; the host never counts."""
    parts = _url_parts(url)
    if parts is None:
        return False
    for segment in parts.path.lower().split("/"):
        words = [word for word in _NOT_ALNUM.split(posixpath.splitext(segment)[0]) if word]
        if "sitemap" in words or ("site", "map") in pairwise(words):
            return True
    return False


def is_pagination(url: str) -> bool:
    """A page of a paginated listing: a PAGINATION_PARAMS query parameter with any value, or a
    /page/<n>, /page-<n> or /p/<n> path segment; the host never counts."""
    parts = _url_parts(url)
    if parts is None:
        return False
    if any(
        name.strip().lower() in PAGINATION_PARAMS
        for name, _ in parse_qsl(parts.query, keep_blank_values=True)
    ):
        return True
    segments = [segment for segment in parts.path.lower().split("/") if segment]
    return any(_PAGE_SEGMENT.fullmatch(segment) for segment in segments) or any(
        first in _PAGE_PREFIXES and second.isdigit() for first, second in pairwise(segments)
    )


def normalise_path(path: str) -> str:
    """A configured path as the exclusion rules compare it: lowercase, one leading slash, no
    trailing slash except the root."""
    return "/" + path.strip().strip("/").lower()


def under_paths(url: str, paths: Collection[str]) -> bool:
    """Whether the url's path is one of ``paths`` or lies below one, case-insensitive; the
    host never counts. ``paths`` are normalised with ``normalise_path``."""
    parts = _url_parts(url)
    if parts is None or not paths:
        return False
    path = normalise_path(parts.path)
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in paths)


# A URL field that is normalised on validation, so stored keys never diverge.
UrlKey = Annotated[str, AfterValidator(normalise_url)]
