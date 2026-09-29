"""The output API against its own OpenAPI schema (#89): every route fuzzed with valid and
malformed input, a valid key and the fixture tenant's stored output, each response checked for
its documented status, media type and schema, and none a server error. Needs Docker."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Final, cast

import pytest
import schemathesis
from fixture import ALPHA, PREFIX, ROUTES, URLS, Params, api_app, served_output, write_served
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from schemathesis import GenerationMode
from schemathesis.checks import CheckFunction, not_a_server_error
from schemathesis.core.result import Err, Ok
from schemathesis.python.asgi import shutdown_lifespans
from schemathesis.specs.openapi.checks import (
    content_type_conformance,
    response_schema_conformance,
    status_code_conformance,
)

from linking_engine.output.keys import KeyStore

if TYPE_CHECKING:
    from schemathesis import Case

    from linking_engine.ingest.mongo_repo import MongoRepo

pytestmark = pytest.mark.integration

# Typed as a check class or function upstream; all four are functions.
CHECKS: Final = cast(
    "list[CheckFunction]",
    [
        not_a_server_error,
        status_code_conformance,
        content_type_conformance,
        response_schema_conformance,
    ],
)
EXAMPLES: Final = 25


@pytest.fixture
async def served(mongo: MongoRepo, tenant: str) -> tuple[str, str, str]:
    """The tenant's stored output and a key for it: (tenant, key, a recommendation id)."""
    output = served_output(tenant, uuid.uuid4().hex, ALPHA)
    await write_served(mongo, output)
    key, _ = await KeyStore(mongo._db).issue(tenant, "contract")
    return tenant, key, output.recommendations[0].id


def test_the_contract_holds(served: tuple[str, str, str], mongo_uri: str) -> None:
    tenant, key, recommendation = served
    headers = {"X-API-Key": key}
    schema = schemathesis.openapi.from_asgi("/openapi.json", api_app(mongo_uri))
    try:
        results = list(schema.get_all_operations())
        assert not [result.err() for result in results if isinstance(result, Err)]
        operations = [result.ok() for result in results if isinstance(result, Ok)]
        paths = {operation.path for operation in operations}
        assert {PREFIX + route for route in ROUTES} <= paths, "a route is missing from OpenAPI"
        assert {operation.method.upper() for operation in operations} == {"GET"}

        # Requests that reach stored data, which random values rarely do.
        examples: dict[str, tuple[dict[str, str], Params]] = {
            f"{PREFIX}/recommendations/{{recommendation_id}}": (
                {"recommendation_id": recommendation},
                {},
            ),
            f"{PREFIX}/page": ({}, {"url": URLS[0]}),
            f"{PREFIX}/recommendations": ({}, {"source": URLS[0], "limit": 2}),
        }
        statuses: dict[str, Counter[int]] = defaultdict(Counter)
        for operation in operations:
            fixed = {"tenant": tenant} if "{tenant}" in operation.path else {}

            @settings(
                max_examples=EXAMPLES,
                deadline=None,
                derandomize=True,
                database=None,
                suppress_health_check=list(HealthCheck),
            )
            @given(
                case=st.one_of(
                    [
                        operation.as_strategy(mode, path_parameters=fixed, headers=headers)
                        for mode in GenerationMode
                    ]
                )
            )
            def fuzz(case: Case[Any]) -> None:
                response = case.call_and_validate(checks=CHECKS)
                statuses[case.operation.path][response.status_code] += 1

            fuzz()

            known = examples.get(operation.path)
            if known is not None:
                path_parameters, query = known
                case = operation.Case(
                    path_parameters={**fixed, **path_parameters}, query=query, headers=headers
                )
                response = case.call_and_validate(checks=CHECKS)
                assert response.status_code == 200, (operation.path, response.text)
                statuses[operation.path][response.status_code] += 1

        # Each route served the fixture tenant at least once, so the checks saw real bodies.
        unserved = sorted(path for path in paths if statuses[path][200] == 0)
        assert not unserved, f"never answered 200: {unserved}; statuses {dict(statuses)}"
    finally:
        shutdown_lifespans()
