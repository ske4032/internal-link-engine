"""Per-tenant keys of the output API. A key is shown once, when it is issued: only its sha256
hash is stored, and neither the key nor its hash is ever logged."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import structlog
from pydantic import ValidationError
from pydantic.alias_generators import to_camel
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from linking_engine.errors import DatabaseReadError, DatabaseWriteError
from linking_engine.models import ApiKeyInfo
from linking_engine.output.collections import API_KEYS, INDEXES, Document
from linking_engine.output.reader import store_errors

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymongo.asynchronous.database import AsyncDatabase

log = structlog.get_logger(__name__)

PREFIX: Final = "lek_"
# token_urlsafe(32) is 43 characters; anything much longer is not a key and is not hashed.
_MAX_KEY_LENGTH: Final = 128
_ID_ATTEMPTS: Final = 3
_INFO_KEYS: Final = {to_camel(field): field for field in ApiKeyInfo.model_fields}


class KeyStore:
    """Issues, resolves and revokes API keys; each key belongs to one tenant."""

    def __init__(self, db: AsyncDatabase[Document]) -> None:
        self._keys = db[API_KEYS]

    async def ensure_indexes(self) -> None:
        with store_errors(f"indexes on {API_KEYS}", write=True):
            await self._keys.create_indexes(list(INDEXES[API_KEYS]))

    async def issue(self, tenant_id: str, label: str | None = None) -> tuple[str, ApiKeyInfo]:
        """A new key for the tenant, and how it is listed. The key cannot be read back."""
        _require_tenant(tenant_id)
        # BSON keeps milliseconds: the info returned equals the info listed later.
        now = datetime.now(UTC)
        created = now.replace(microsecond=now.microsecond // 1000 * 1000)
        for _ in range(_ID_ATTEMPTS):
            key = PREFIX + secrets.token_urlsafe(32)
            info = ApiKeyInfo(
                key_id=secrets.token_hex(4),
                tenant_id=tenant_id,
                label=(label or "").strip() or None,
                created_at=created,
            )
            document: Document = {
                "keyHash": _hash(key),
                "keyId": info.key_id,
                "tenantId": tenant_id,
                "label": info.label,
                "createdAt": created,
                "revokedAt": None,
            }
            with store_errors("issue an API key", write=True):
                try:
                    await self._keys.insert_one(document)
                except DuplicateKeyError:
                    continue
            log.info("api_keys.issued", tenant_id=tenant_id, key_id=info.key_id)
            return key, info
        raise DatabaseWriteError(
            "mongodb", f"issue an API key: no free key id after {_ID_ATTEMPTS} attempts"
        )

    async def revoke(self, tenant_id: str, key_id: str) -> bool:
        """Revoke one of the tenant's keys; False when it has no such live key."""
        _require_tenant(tenant_id)
        with store_errors("revoke an API key", write=True):
            result = await self._keys.update_one(
                {"tenantId": tenant_id, "keyId": key_id, "revokedAt": None},
                {"$set": {"revokedAt": datetime.now(UTC)}},
            )
        revoked = result.modified_count == 1
        if revoked:
            log.info("api_keys.revoked", tenant_id=tenant_id, key_id=key_id)
        return revoked

    async def tenant_for(self, key: str) -> str | None:
        """The tenant a live key belongs to; None for an unknown or revoked key."""
        if not key.startswith(PREFIX) or len(key) > _MAX_KEY_LENGTH:
            return None
        with store_errors("look up an API key"):
            document = await self._keys.find_one(
                {"keyHash": _hash(key), "revokedAt": None}, {"tenantId": 1}
            )
        tenant = None if document is None else document.get("tenantId")
        return tenant if isinstance(tenant, str) else None

    async def keys(self, tenant_id: str) -> tuple[ApiKeyInfo, ...]:
        """The tenant's keys, revoked ones included, oldest first."""
        _require_tenant(tenant_id)
        with store_errors("list API keys"):
            documents = (
                await self._keys.find({"tenantId": tenant_id}, dict.fromkeys(_INFO_KEYS, 1))
                # Keys issued within one millisecond tie on createdAt; _id keeps insertion order.
                .sort([("createdAt", ASCENDING), ("_id", ASCENDING)])
                .to_list()
            )
        return tuple(_info(document) for document in documents)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _info(document: Mapping[str, object]) -> ApiKeyInfo:
    data = {field: document[key] for key, field in _INFO_KEYS.items() if key in document}
    try:
        return ApiKeyInfo.model_validate(data)
    except ValidationError as error:
        raise DatabaseReadError(
            "mongodb", f"{API_KEYS} document does not fit ApiKeyInfo: {error}"
        ) from error


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
