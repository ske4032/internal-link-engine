from __future__ import annotations

import httpx
import pytest

from linking_engine.errors import ServiceError
from linking_engine.pipeline.health import (
    check_client_not_newer,
    check_service_versions,
    major_minor,
    mlflow_server_info,
    presigned_transfer_warnings,
)


@pytest.mark.parametrize(
    ("text", "expected"), [("3.8.6", (3, 8)), ('"3.8.6"', (3, 8)), ("v3.16.1\n", (3, 16))]
)
def test_major_minor(text: str, expected: tuple[int, int]) -> None:
    assert major_minor(text) == expected


def test_unparseable_version_raises() -> None:
    with pytest.raises(ServiceError, match="unparseable"):
        major_minor("latest")


def test_client_may_equal_or_trail_the_server() -> None:
    check_client_not_newer("prefect", "3.8.6", "3.8.2")
    check_client_not_newer("prefect", "3.7.0", "3.8.2")


def test_newer_client_minor_raises() -> None:
    with pytest.raises(ServiceError, match="newer than server"):
        check_client_not_newer("mlflow", "3.17.0", "3.16.1")


async def test_unreachable_service_raises() -> None:
    with pytest.raises(ServiceError, match="cannot reach"):
        await check_service_versions(
            "http://127.0.0.1:1/api", "http://127.0.0.1:1", None, http_timeout_s=1
        )


@pytest.mark.parametrize(
    ("setting", "warned"),
    [(None, True), ("false", False), ("FALSE", False), (" 0 ", False), ("true", True), ("", True)],
    ids=["unset", "false", "upper-case", "zero", "true", "empty"],
)
def test_presigned_transfers_are_flagged_until_the_client_turns_them_off(
    setting: str | None, warned: bool
) -> None:
    info = {"multipart_downloads_enabled": True, "multipart_uploads_enabled": False}
    environ = {} if setting is None else {"MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD": setting}
    warnings = presigned_transfer_warnings(info, environ)
    assert bool(warnings) is warned
    if warned:
        assert "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false" in warnings[0]


def test_both_transfers_warn_and_a_silent_server_warns_about_none() -> None:
    both = {"multipart_downloads_enabled": True, "multipart_uploads_enabled": True}
    assert len(presigned_transfer_warnings(both, {})) == 2
    assert presigned_transfer_warnings({}, {}) == []


def server(status: int, body: bytes = b"") -> httpx.AsyncClient:
    def reply(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mlflow/api/3.0/mlflow/server-info"
        return httpx.Response(status, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(reply))


async def test_server_info_is_read_below_any_path_prefix() -> None:
    async with server(200, b'{"multipart_downloads_enabled": true}') as client:
        assert await mlflow_server_info(client, "https://tracking.example/mlflow/") == {
            "multipart_downloads_enabled": True
        }


async def test_a_server_too_old_to_say_advertises_nothing() -> None:
    async with server(404) as client:
        assert await mlflow_server_info(client, "https://tracking.example/mlflow") == {}


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (500, b"", "returned 500"),
        (200, b"<html>", "did not return JSON"),
        (200, b"[]", "an object"),
    ],
    ids=["server-error", "not-json", "not-an-object"],
)
async def test_a_bad_server_info_answer_raises(status: int, body: bytes, message: str) -> None:
    async with server(status, body) as client:
        with pytest.raises(ServiceError, match=message):
            await mlflow_server_info(client, "https://tracking.example/mlflow")


async def test_an_unreachable_server_info_raises() -> None:
    async with httpx.AsyncClient(timeout=1) as client:
        with pytest.raises(ServiceError, match="cannot reach"):
            await mlflow_server_info(client, "http://127.0.0.1:1")
