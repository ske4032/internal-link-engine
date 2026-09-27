"""Startup checks: Prefect and MLflow clients must not be newer than their servers, and MLflow
artifact transfers must go through the tracking server."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Final

import httpx

from linking_engine.errors import ServiceError

if TYPE_CHECKING:
    from collections.abc import Mapping

SERVER_INFO_PATH: Final = "/api/3.0/mlflow/server-info"
# (server capability, transfer, client setting). With the setting false the client streams
# through the tracking server instead of presigned storage urls, which can point inside the
# server's cluster where the client cannot reach them.
PROXY_SETTINGS: Final = (
    ("multipart_downloads_enabled", "downloads", "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"),
    ("multipart_uploads_enabled", "uploads", "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"),
)


def major_minor(text: str) -> tuple[int, int]:
    parts = text.strip().strip('"').lstrip("v").split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as error:
        raise ServiceError(f"unparseable version {text!r}") from error


def check_client_not_newer(name: str, client: str, server: str) -> None:
    if major_minor(client) > major_minor(server):
        raise ServiceError(f"{name} client {client} is newer than server {server}")


async def _get(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ServiceError(f"{url} returned {error.response.status_code}") from error
    except httpx.HTTPError as error:
        raise ServiceError(f"cannot reach {url}: {error}") from error
    return response.text


async def check_service_versions(
    prefect_api_url: str,
    mlflow_uri: str,
    mlflow_auth: tuple[str, str] | None,
    *,
    http_timeout_s: float = 10.0,
) -> dict[str, tuple[str, str]]:
    """Returns {service: (client, server)}; raises ServiceError on mismatch or failure."""
    async with httpx.AsyncClient(timeout=http_timeout_s) as client:
        prefect = await _get(client, f"{prefect_api_url.rstrip('/')}/admin/version")
    async with httpx.AsyncClient(timeout=http_timeout_s, auth=mlflow_auth) as client:
        mlflow = await _get(client, f"{mlflow_uri.rstrip('/')}/version")
    found = {
        "prefect": (version("prefect"), prefect.strip().strip('"')),
        "mlflow": (version("mlflow"), mlflow.strip()),
    }
    for name, (client_version, server_version) in found.items():
        check_client_not_newer(name, client_version, server_version)
    return found


async def mlflow_server_info(client: httpx.AsyncClient, mlflow_uri: str) -> dict[str, object]:
    """What the server advertises; empty for a server too old to say, which never presigns."""
    url = f"{mlflow_uri.rstrip('/')}{SERVER_INFO_PATH}"
    try:
        response = await client.get(url)
    except httpx.HTTPError as error:
        raise ServiceError(f"cannot reach {url}: {error}") from error
    if response.status_code == httpx.codes.NOT_FOUND:
        return {}
    if response.status_code != httpx.codes.OK:
        raise ServiceError(f"{url} returned {response.status_code}")
    try:
        info = response.json()
    except ValueError as error:
        raise ServiceError(f"{url} did not return JSON") from error
    if not isinstance(info, dict):
        raise ServiceError(f"{url} did not return an object")
    return info


def presigned_transfer_warnings(
    server_info: Mapping[str, object], environ: Mapping[str, str]
) -> list[str]:
    """One warning per presigned transfer the server offers that this client would take."""
    return [
        f"MLflow offers presigned {transfer}: set {setting}=false, or the client transfers "
        "artifacts directly with storage that may be unreachable outside the server's cluster"
        for capability, transfer, setting in PROXY_SETTINGS
        if server_info.get(capability) is True
        and environ.get(setting, "").strip().lower() not in ("false", "0")
    ]
