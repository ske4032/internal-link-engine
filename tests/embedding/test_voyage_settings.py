from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from linking_engine.embedding.voyage_client import VoyageSettings

REQUIRED = {
    "context_tokens": 50,
    "max_request_tokens": 130,
    "request_token_budget": 120,
    "max_batch_items": 4,
}


def test_defaults() -> None:
    config = VoyageSettings(api_key="test", **REQUIRED)  # type: ignore[arg-type]
    assert (
        config.max_attempts,
        config.backoff_initial_s,
        config.backoff_max_s,
        config.timeout_s,
    ) == (8, 1.0, 60.0, 60.0)


def test_limit_defaults() -> None:
    config = VoyageSettings(api_key="test")  # type: ignore[call-arg]
    assert (
        config.context_tokens,
        config.max_request_tokens,
        config.request_token_budget,
        config.max_batch_items,
    ) == (32_000, 120_000, 110_000, 1_000)


def test_budget_above_request_cap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        VoyageSettings(api_key="test", **{**REQUIRED, "request_token_budget": 131})  # type: ignore[arg-type]


def test_budget_below_context_is_rejected() -> None:
    # A single page cut to the context limit would not fit any batch.
    with pytest.raises(ValidationError):
        VoyageSettings(api_key="test", **{**REQUIRED, "request_token_budget": 49})  # type: ignore[arg-type]


def test_budget_equal_to_both_bounds_is_accepted() -> None:
    config = VoyageSettings(
        api_key="test",
        context_tokens=120,
        max_request_tokens=120,
        request_token_budget=120,
        max_batch_items=4,
    )
    assert config.request_token_budget == 120


def test_everything_reads_from_voyage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-secret-value")
    monkeypatch.setenv("VOYAGE_CONTEXT_TOKENS", "16000")
    monkeypatch.setenv("VOYAGE_MAX_REQUEST_TOKENS", "100000")
    monkeypatch.setenv("VOYAGE_REQUEST_TOKEN_BUDGET", "90000")
    monkeypatch.setenv("VOYAGE_MAX_BATCH_ITEMS", "500")
    monkeypatch.setenv("VOYAGE_MAX_ATTEMPTS", "5")
    config = VoyageSettings()  # type: ignore[call-arg]
    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "pa-secret-value"
    assert "pa-secret-value" not in repr(config)
    assert "pa-secret-value" not in str(config)
    assert (
        config.context_tokens,
        config.max_request_tokens,
        config.request_token_budget,
        config.max_batch_items,
        config.max_attempts,
    ) == (16_000, 100_000, 90_000, 500, 5)


def test_missing_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        VoyageSettings(**REQUIRED)  # type: ignore[arg-type]


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        VoyageSettings(api_key="test", batch_size=36, **REQUIRED)  # type: ignore[arg-type,call-arg]


def test_settings_are_frozen() -> None:
    config = VoyageSettings(api_key="test", **REQUIRED)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        config.max_attempts = 1  # type: ignore[misc]
