"""Per-tenant configuration, with the shipped defaults.

Loaded from the `tenant_config` collection for a real tenant. The
``pydantic-settings`` base means a local single-tenant run can override any of
these from the environment with a ``TENANT_`` prefix without a Mongo document
existing yet, and every field still has the shipped default from the Data Model.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnchorTypeProfile(BaseModel):
    """Desired distribution of anchor types across a tenant's recommendations.

    A preference, not a constraint. Extraction runs first and the profile only
    chooses among phrases that already exist in the source copy: if the one
    usable phrase is a partial match it is used even when the profile wants
    exact. The four shares are therefore bounded individually and not required
    to sum to exactly one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact: float = Field(default=0.15, ge=0, le=1)
    partial: float = Field(default=0.20, ge=0, le=1)
    natural: float = Field(default=0.50, ge=0, le=1)
    branded: float = Field(default=0.15, ge=0, le=1)


class LanguageRules(BaseModel):
    """How a tenant's pages get their language: links are only ever made within one language.

    A page takes the language of the longest url path prefix that matches it, else the
    tenant's default. A single-language site needs no prefixes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_language: str = Field(default="en", min_length=2)
    # (url path prefix, language), e.g. ("/de/", "de").
    prefixes: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _prefixes(self) -> Self:
        paths = [path for path, _ in self.prefixes]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate language prefixes")
        if any(not path.startswith("/") or len(language) < 2 for path, language in self.prefixes):
            raise ValueError("a language prefix is a path starting with / and a language code")
        return self


class AnchorRules(BaseModel):
    """A tenant's additions to and exemptions from the generic-anchor dictionary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generic_add: frozenset[str] = frozenset()
    generic_remove: frozenset[str] = frozenset()


class TenantConfig(BaseSettings):
    """Everything that varies per tenant.

    Every field except `tenant_id` carries the shipped default from the Data
    Model, so a config document only has to state what it changes.
    """

    # Process environment only, never a dotenv file: `extra="forbid"` is
    # non-negotiable, and pydantic-settings' dotenv source rejects every
    # unrelated key in the file rather than ignoring it. The env prefix filters
    # the process environment properly, so `TENANT_DISCOVERY_ENABLED=true` works
    # and `NEO4J_URI` alongside it does not raise. Load a dotenv with
    # `uv run --env-file .env` when developing locally.
    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="TENANT_",
        env_nested_delimiter="__",
    )

    # ── identity ────────────────────────────────────────────────────────────
    # The key of the Mongo config document, and the value bound to every log
    # line for the run. No default: a config that does not say whose it is has
    # no meaning.
    tenant_id: str = Field(min_length=1)

    # ── embedding ───────────────────────────────────────────────────────────
    embedding_provider: str = "VOYAGE_API"
    embedding_model: str = "voyage-4-large"
    # Must match the Neo4j vector index, whose dimension is fixed at creation.
    # Changing it means dropping both indexes, re-embedding every page and
    # retraining the GNN, so it is an ADR rather than a config edit.
    embedding_dimensions: int = Field(default=2048, ge=1)
    embedding_local_fallback_model: str = "voyage-4-nano"
    embedding_strategy: str = "MULTILINGUAL_EMBED"
    site_languages: tuple[str, ...] = ("en", "de", "fr")

    # ── content gap ─────────────────────────────────────────────────────────
    # The local model is used only to phrase a CONTENT_GAP finding, never to
    # write anchor text: anchors are extracted from the copy, never generated.
    content_gap_model: str = "qwen3.5:4b"
    content_gap_min_priority: int = Field(default=4, ge=1, le=5)
    content_gap_min_opportunity: int = Field(default=500, ge=0)

    # ── anchors ─────────────────────────────────────────────────────────────
    anchor_type_profile: AnchorTypeProfile = AnchorTypeProfile()

    # ── stages ──────────────────────────────────────────────────────────────
    # Independent flags. Stage 1 ships and runs alone; discovery is opt-in.
    audit_enabled: bool = True
    discovery_enabled: bool = False

    # ── output shaping ──────────────────────────────────────────────────────
    client_tier: str = "STARTER"
    diversity_cap_pct: float = Field(default=0.15, ge=0, le=1)
    max_recommendations_per_source: int = Field(default=10, ge=1)
    # Content gaps listed per source page beside its new links; 0 lists none.
    max_content_gaps_per_source: int = Field(default=3, ge=0)
