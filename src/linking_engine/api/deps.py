"""Request dependencies: the stores the app opened, the tenant an API key belongs to, and that
tenant's latest complete run.

A key used on another tenant's path gets the same 404 as a missing resource, so a caller never
learns whether that tenant exists.
"""

from typing import Annotated, Final

from fastapi import Depends, HTTPException, Path, Request, Security, status
from fastapi.security import APIKeyHeader

from linking_engine.models import RunInfo
from linking_engine.output.keys import KeyStore
from linking_engine.output.reader import OutputReader

NOT_FOUND: Final = "not found"
INVALID_KEY: Final = "invalid or missing API key"

api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="An API key issued for the tenant."
)


def not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)


def output_reader(request: Request) -> OutputReader:
    reader = getattr(request.app.state, "reader", None)
    if not isinstance(reader, OutputReader):
        raise RuntimeError("the output reader is opened by the app's lifespan")
    return reader


def key_store(request: Request) -> KeyStore:
    keys = getattr(request.app.state, "keys", None)
    if not isinstance(keys, KeyStore):
        raise RuntimeError("the key store is opened by the app's lifespan")
    return keys


async def authorised_tenant(
    tenant: Annotated[str, Path(description="The tenant's id.")],
    key: Annotated[str | None, Security(api_key_header)],
    keys: Annotated[KeyStore, Depends(key_store)],
) -> str:
    """The path's tenant, when the key is live and belongs to it."""
    owner = None if key is None else await keys.tenant_for(key)
    if owner is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail=INVALID_KEY, headers={"WWW-Authenticate": "APIKey"}
        )
    if owner != tenant:
        raise not_found()
    return tenant


async def latest_run(
    tenant: Annotated[str, Depends(authorised_tenant)],
    reader: Annotated[OutputReader, Depends(output_reader)],
) -> RunInfo:
    """The tenant's latest complete run; 404 when it has none."""
    run = await reader.latest_run(tenant)
    if run is None:
        raise not_found()
    return run


AuthorisedTenant = Annotated[str, Depends(authorised_tenant)]
LatestRun = Annotated[RunInfo, Depends(latest_run)]
Reader = Annotated[OutputReader, Depends(output_reader)]
