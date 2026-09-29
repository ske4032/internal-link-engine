"""`TenantConfig` defaults, against the tenant config block in wiki/Data-Model.md.

Every value here is copied from that JSON. The two toggles are asserted separately and
in all four combinations because they are independent by design — Stage 1, the audit,
ships and runs with discovery off, and a validator that quietly couples them would break
the only configuration that currently exists.

The autouse `isolated_settings_env` fixture in tests/conftest.py strips anything from
the environment that could feed these fields and runs the test from an empty directory,
so a default asserted here is the model's own and not the developer's `.env`.
"""

from __future__ import annotations

import pytest
from pydantic_settings import BaseSettings

from linking_engine.models.tenant import TenantConfig

EXPECTED_DEFAULTS = {
    "embedding_provider": "VOYAGE_API",
    "embedding_model": "voyage-4-large",
    "embedding_dimensions": 2048,
    "embedding_local_fallback_model": "voyage-4-nano",
    "content_gap_model": "qwen3.5:4b",
    "content_gap_min_priority": 4,
    "content_gap_min_opportunity": 500,
    "audit_enabled": True,
    "discovery_enabled": False,
    "embedding_strategy": "MULTILINGUAL_EMBED",
    "client_tier": "STARTER",
    "diversity_cap_pct": 0.15,
    "max_recommendations_per_source": 10,
    "max_content_gaps_per_source": 3,
    "words_per_link": 200,
    "guaranteed_inbound_links": 2,
    "guaranteed_inbound_below": 1,
    "max_suggested_inbound": 5,
    "pillar_floor_quantile": 0.10,
    "pillar_floor_min_links": 50,
}

EXPECTED_ANCHOR_PROFILE = {"exact": 0.15, "partial": 0.20, "natural": 0.50, "branded": 0.15}


def _env_var_for(field_name: str) -> str:
    """The environment variable pydantic-settings would read for a field."""
    prefix = TenantConfig.model_config.get("env_prefix", "")
    alias = TenantConfig.model_fields[field_name].validation_alias
    name = alias if isinstance(alias, str) else field_name
    return f"{prefix}{name}".upper()


@pytest.fixture
def config() -> TenantConfig:
    """A tenant carrying nothing but its identifier: everything else is a default."""
    return TenantConfig(tenant_id="client_abc")


def test_tenant_config_is_a_settings_model() -> None:
    assert issubclass(TenantConfig, BaseSettings), (
        "TenantConfig is built on pydantic-settings so deployment can override "
        "defaults from the environment"
    )


def test_the_identifier_is_carried(config: TenantConfig) -> None:
    assert config.tenant_id == "client_abc"


@pytest.mark.parametrize(("field", "expected"), sorted(EXPECTED_DEFAULTS.items()))
def test_default_matches_the_data_model_page(config: TenantConfig, field, expected) -> None:
    assert getattr(config, field) == expected, (
        f"{field} default drifted from the tenant config in wiki/Data-Model.md"
    )


def test_site_languages_default(config: TenantConfig) -> None:
    assert tuple(config.site_languages) == ("en", "de", "fr")


def test_anchor_type_profile_default(config: TenantConfig) -> None:
    profile = config.anchor_type_profile
    as_dict = profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)

    assert as_dict == EXPECTED_ANCHOR_PROFILE
    assert sum(as_dict.values()) == pytest.approx(1.0), (
        "the anchor type profile is a distribution and has to sum to 1"
    )


def test_stage_one_is_the_shipping_default(config: TenantConfig) -> None:
    assert config.audit_enabled is True
    assert config.discovery_enabled is False


@pytest.mark.parametrize("audit", [True, False])
@pytest.mark.parametrize("discovery", [True, False])
def test_audit_and_discovery_are_independent(audit, discovery) -> None:
    configured = TenantConfig(
        tenant_id="client_abc", audit_enabled=audit, discovery_enabled=discovery
    )
    assert configured.audit_enabled is audit
    assert configured.discovery_enabled is discovery


def test_values_can_be_overridden_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(_env_var_for("audit_enabled"), "false")
    monkeypatch.setenv(_env_var_for("max_recommendations_per_source"), "3")

    configured = TenantConfig(tenant_id="client_abc")

    assert configured.audit_enabled is False, "pydantic-settings is not reading the env"
    assert configured.max_recommendations_per_source == 3
