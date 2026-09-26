"""Startup check: installed Prefect and MLflow clients must not be newer than their servers."""

from __future__ import annotations

from importlib.metadata import version

import httpx

from linking_engine.errors import ServiceError


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
