from __future__ import annotations

import pytest

from linking_engine.errors import ServiceError
from linking_engine.pipeline.health import (
    check_client_not_newer,
    check_service_versions,
    major_minor,
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
            "http://127.0.0.1:1/api", "http://127.0.0.1:1", None, timeout=1
        )
