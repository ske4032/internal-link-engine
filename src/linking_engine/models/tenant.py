"""Per-tenant configuration, with the shipped defaults.

Loaded from the `tenant_config` collection for a real tenant. The
``pydantic-settings`` base means a local single-tenant run can override any of
these from the environment with a ``TENANT_`` prefix without a Mongo document
existing yet, and every field still has the shipped default from the Data Model.
"""

from pydantic import BaseModel, ConfigDict, Field
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
    lifecycle_boost_enabled: bool = True
    reserved_new_page_slot_pct: float = Field(default=0.15, ge=0, le=1)
    diversity_cap_pct: float = Field(default=0.15, ge=0, le=1)
    max_recommendations_per_source: int = Field(default=10, ge=1)
