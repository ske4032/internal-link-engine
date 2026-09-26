"""URL identity shared by every tenant: one normalised key per page."""

from __future__ import annotations

import posixpath
import re
from typing import Annotated
from urllib.parse import quote, unquote, urlsplit

from pydantic import AfterValidator

_WEB_SCHEMES = frozenset({"http", "https"})
_WWW = re.compile(r"^www\d*\.(?=[^.]+\.)")  # keep a bare "www.com"
# "mailto:", "tel:", "javascript:"; a digit after the colon is a host:port instead.
_OTHER_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:(?!\d)")
_PATH_PARAMS = re.compile(r";[^/]*")  # ;jsessionid=... and other matrix parameters
_INDEX_DOCUMENT = re.compile(r"/(?:index|default)\.(?:html?|php|aspx?|jsp)$", re.IGNORECASE)
_SLASHES = re.compile(r"/{2,}")
_HOST = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")
# Characters a path may carry unescaped (RFC 3986 pchar plus '/').
_PATH_SAFE = "/:@!$&'()*+,;=-._~"


def normalise_url(url: str) -> str:
    """Key for a web URL: no scheme, www, port 80/443, credentials, query or fragment;
    lowercase host and path; dot segments, duplicate and trailing slashes, index
    documents and path parameters removed; percent-encoding canonical.

    ``https://www.Example.com:443/Blog//a/../Post/index.html?utm=x#top`` -> ``example.com/blog/post``.
    Raises ValueError for anything that is not an http(s) URL with a host.
    """
    text = url.strip()
    if not text:
        raise ValueError("url is empty")
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
    return host + quote(path, safe=_PATH_SAFE)


def host_of(key: str) -> str:
    """The host part of a normalised key."""
    return key.split("/", 1)[0]


# A URL field that is normalised on validation, so stored keys never diverge.
UrlKey = Annotated[str, AfterValidator(normalise_url)]
