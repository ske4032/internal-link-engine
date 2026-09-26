"""Voyage embedding client: token-aware batching, backoff, unit-norm vectors."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, Self

import numpy as np
import structlog
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from tokenizers import Tokenizer
from voyageai.client_async import AsyncClient
from voyageai.error import (
    APIConnectionError,
    AuthenticationError,
    InvalidRequestError,
    MalformedRequestError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    VoyageError,
)

from linking_engine.errors import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from linking_engine.models import PageEmbedding, PageText

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tenacity import RetryCallState
    from tokenizers import Encoding

log = structlog.get_logger(__name__)

INPUT_TYPE: Final = "document"
NORM_TOLERANCE: Final = 1e-6
_RETRYABLE: Final = (
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    APIConnectionError,
    TimeoutError,
)
_REJECTED: Final = (InvalidRequestError, MalformedRequestError, AuthenticationError)
_AUTH_STATUSES: Final = frozenset({401, 403})


class VoyageSettings(BaseSettings):
    """Voyage credentials, limits and backoff, from ``VOYAGE_*`` process env."""

    # Process env only: a dotenv source would reject unrelated keys under extra="forbid".
    model_config = SettingsConfigDict(frozen=True, extra="forbid", env_prefix="VOYAGE_")

    api_key: SecretStr
    context_tokens: int = Field(default=32_000, ge=1)
    max_request_tokens: int = Field(default=120_000, ge=1)
    # ~8% under the 120k cap: headroom for the server-side input_type prompt and tokenizer drift.
    request_token_budget: int = Field(default=110_000, ge=1)
    max_batch_items: int = Field(default=1_000, ge=1)
    # Jittered wait caps 1+2+4+8+16+32+60 = 123s over 7 retries: outlasts a per-minute limit.
    max_attempts: int = Field(default=8, ge=1)
    backoff_initial_s: float = Field(default=1.0, ge=0)
    backoff_max_s: float = Field(default=60.0, ge=0)
    timeout_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _budget_fits(self) -> Self:
        if self.request_token_budget > self.max_request_tokens:
            raise ValueError("request_token_budget must not exceed max_request_tokens")
        if self.request_token_budget < self.context_tokens:
            raise ValueError("request_token_budget must be at least context_tokens")
        return self


class VoyageEmbedResult(Protocol):
    @property
    def embeddings(self) -> Sequence[Sequence[float]]: ...

    @property
    def total_tokens(self) -> int: ...


class VoyageSdk(Protocol):
    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int,
        truncation: bool,
    ) -> VoyageEmbedResult: ...


class _Prepared(NamedTuple):
    url: str
    text: str
    tokens: int
    original_tokens: int

    @property
    def truncated(self) -> bool:
        return self.tokens < self.original_tokens


def token_batches(counts: Sequence[int], *, budget: int, max_items: int) -> Iterator[range]:
    """Yield contiguous index ranges whose token sum <= budget and length <= max_items."""
    if budget < 1 or max_items < 1:
        raise ValueError("budget and max_items must be positive")
    start, total = 0, 0
    for index, count in enumerate(counts):
        if count < 0 or count > budget:
            raise ValueError(f"token count {count} at index {index} is outside 0..{budget}")
        if index > start and (total + count > budget or index - start >= max_items):
            yield range(start, index)
            start, total = index, 0
        total += count
    if len(counts) > start:
        yield range(start, len(counts))


async def backoff_sleep(seconds: float) -> None:
    """Wait between retries; patch this module attribute in tests to skip real sleeps."""
    await asyncio.sleep(seconds)


class VoyageClient:
    """Embeds page texts as documents with Voyage, in token-budgeted batches."""

    def __init__(
        self,
        settings: VoyageSettings,
        *,
        model: str,
        dimension: int,
        sdk: VoyageSdk | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._settings = settings
        self._model = model
        self._dimension = dimension
        if sdk is None:
            # SDK timeout sits above ours so the asyncio.timeout deadline always fires first.
            sdk = AsyncClient(
                api_key=settings.api_key.get_secret_value(),
                max_retries=0,
                timeout=settings.timeout_s + 5,
            )
        self._sdk = sdk
        self._tokenizer = tokenizer

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token count per text with the model's own tokenizer."""
        return [len(encoding) for encoding in self._encode(texts)]

    async def embed(self, pages: Sequence[PageText]) -> list[PageEmbedding]:
        """One unit-norm vector per page, in input order."""
        if not pages:
            return []
        encodings = await asyncio.to_thread(self._encode, [page.text for page in pages])
        prepared = [
            self._fit(page, encoding) for page, encoding in zip(pages, encodings, strict=True)
        ]
        # Release token/offset arrays before the network-bound batches.
        del encodings
        results: list[PageEmbedding] = []
        for span in token_batches(
            [item.tokens for item in prepared],
            budget=self._settings.request_token_budget,
            max_items=self._settings.max_batch_items,
        ):
            batch = prepared[span.start : span.stop]
            vectors = await self._embed_batch(batch)
            results.extend(
                PageEmbedding(
                    url=item.url,
                    vector=vector,
                    tokens=item.tokens,
                    original_tokens=item.original_tokens,
                    truncated=item.truncated,
                )
                for item, vector in zip(batch, vectors, strict=True)
            )
        return results

    def _encode(self, texts: Sequence[str]) -> list[Encoding]:
        return self._load_tokenizer().encode_batch(list(texts))

    def _load_tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            try:
                tokenizer = Tokenizer.from_pretrained(f"voyageai/{self._model}")
            except Exception as error:
                raise EmbeddingError(f"cannot load tokenizer for {self._model}") from error
            tokenizer.no_truncation()
            self._tokenizer = tokenizer
        return self._tokenizer

    def _fit(self, page: PageText, encoding: Encoding) -> _Prepared:
        original = len(encoding)
        limit = self._settings.context_tokens
        if original <= limit:
            return _Prepared(page.url, page.text, original, original)
        # Later evaluation: chunked embeddings with voyage-context-4 vs this whole-page truncation.
        spans = [(start, end) for start, end in encoding.offsets if end > start]
        keep = max(limit - (original - len(spans)), 0)
        cut = spans[keep - 1][1] if keep else 0
        if keep < len(spans):
            # A multi-byte character split across tokens: drop it whole.
            cut = min(cut, spans[keep][0])
        sent = original - len(spans) + sum(1 for _, end in spans[:keep] if end <= cut)
        log.warning("embedding.truncated", url=page.url, original_tokens=original, limit=limit)
        return _Prepared(page.url, page.text[:cut], sent, original)

    async def _embed_batch(self, batch: Sequence[_Prepared]) -> list[tuple[float, ...]]:
        texts = [item.text for item in batch]
        started = time.perf_counter()
        result, retries = await self._call(texts)
        latency_ms = round((time.perf_counter() - started) * 1000)
        vectors = self._validate(result, expected=len(batch))
        log.info(
            "embedding.batch",
            items=len(batch),
            tokens=sum(item.tokens for item in batch),
            api_tokens=result.total_tokens,
            latency_ms=latency_ms,
            retries=retries,
            truncated=sum(item.truncated for item in batch),
        )
        return vectors

    async def _call(self, texts: list[str]) -> tuple[VoyageEmbedResult, int]:
        what = f"embed {len(texts)} texts"
        attempts = 0
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(self._settings.max_attempts),
                wait=wait_random_exponential(
                    multiplier=self._settings.backoff_initial_s,
                    max=self._settings.backoff_max_s,
                ),
                sleep=backoff_sleep,
                before_sleep=_log_retry,
                reraise=True,
            ):
                attempts = attempt.retry_state.attempt_number
                with attempt:
                    # Client-side deadline too: the SDK timeout does not bound every await.
                    async with asyncio.timeout(self._settings.timeout_s):
                        result = await self._sdk.embed(
                            texts,
                            model=self._model,
                            input_type=INPUT_TYPE,
                            output_dimension=self._dimension,
                            truncation=True,
                        )
                    return result, attempts - 1
        except VoyageError as error:
            raise _translate(error, what=what, attempts=attempts) from error
        except TimeoutError as error:
            error_type = type(error).__name__
            limit = self._settings.timeout_s
            raise EmbeddingUnavailableError(
                f"{what}: retries exhausted after {attempts} attempts: "
                f"{error_type}: no response within {limit}s",
                error_type=error_type,
            ) from error
        raise AssertionError("unreachable")

    def _validate(self, result: VoyageEmbedResult, *, expected: int) -> list[tuple[float, ...]]:
        embeddings = result.embeddings
        if len(embeddings) != expected:
            raise EmbeddingResponseError(f"expected {expected} vectors, got {len(embeddings)}")
        for index, vector in enumerate(embeddings):
            if len(vector) != self._dimension:
                raise EmbeddingResponseError(
                    f"vector {index} has dimension {len(vector)}, expected {self._dimension}"
                )
        matrix = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        unusable = ~np.isfinite(norms) | (norms == 0)
        if unusable.any():
            index = int(np.flatnonzero(unusable)[0])
            raise EmbeddingResponseError(f"vector {index} is zero or not finite")
        unit = matrix / norms[:, np.newaxis]
        off = np.abs(np.linalg.norm(unit, axis=1) - 1.0) > NORM_TOLERANCE
        if off.any():
            index = int(np.flatnonzero(off)[0])
            raise EmbeddingResponseError(f"vector {index} is not unit norm after normalising")
        return [tuple(row) for row in unit.tolist()]


def _is_rejected(error: BaseException) -> bool:
    """Non-429 4xx or a known bad-request/auth class: retrying cannot help."""
    if not isinstance(error, VoyageError):
        return False
    status = error.http_status
    return isinstance(error, _REJECTED) or (
        isinstance(status, int) and 400 <= status < 500 and status != 429
    )


def _is_retryable(error: BaseException) -> bool:
    """429, 5xx, SDK timeout, connection failure or the client-side deadline."""
    if _is_rejected(error):
        return False
    if isinstance(error, _RETRYABLE):
        return True
    if not isinstance(error, VoyageError):
        return False
    status = error.http_status
    return isinstance(status, int) and (status == 429 or status >= 500)


def _log_retry(state: RetryCallState) -> None:
    error = state.outcome.exception() if state.outcome else None
    log.warning(
        "embedding.retry",
        attempt=state.attempt_number,
        wait_s=round(state.upcoming_sleep, 3),
        error=type(error).__name__,
        http_status=getattr(error, "http_status", None),
    )


def _detail(error: VoyageError) -> str:
    """Provider message without the request-id prefix or the body/headers APIError appends."""
    body = error.json_body
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    message = error.user_message
    return str(message) if message else "<empty message>"


def _translate(error: VoyageError, *, what: str, attempts: int) -> EmbeddingError:
    raw_status = error.http_status
    status = raw_status if isinstance(raw_status, int) else None
    error_type = type(error).__name__
    http = f"HTTP {status} " if status is not None else ""
    cause = f"{http}{error_type}: {_detail(error)}"
    if isinstance(error, AuthenticationError) or status in _AUTH_STATUSES:
        return EmbeddingAuthError(f"{what}: {cause}", status_code=status, error_type=error_type)
    if _is_rejected(error):
        return EmbeddingRequestError(f"{what}: {cause}", status_code=status, error_type=error_type)
    if _is_retryable(error):
        return EmbeddingUnavailableError(
            f"{what}: retries exhausted after {attempts} attempts: {cause}",
            status_code=status,
            error_type=error_type,
        )
    return EmbeddingUnavailableError(f"{what}: {cause}", status_code=status, error_type=error_type)
