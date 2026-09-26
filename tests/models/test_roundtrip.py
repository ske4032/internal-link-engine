"""Issue #2 Acceptance: round-trip every model type, and prove `extra="forbid"` bites.

Both criteria are registry-driven rather than written out eight times, so a model added
to `factories.MODEL_SPECS` is covered by all of it and a model dropped from the registry
fails `test_registry_covers_every_documented_model` instead of quietly reducing the
suite to nothing.
"""

from __future__ import annotations

import pytest
from factories import MODEL_IDS, MODEL_SPECS
from pydantic import BaseModel, ValidationError

DOCUMENTED_MODELS = {
    "Page",
    "Keyword",
    "Link",
    "LinkAuditResult",
    "PairFeatures",
    "AnchorCandidate",
    "Recommendation",
    "TenantConfig",
}


def test_registry_covers_every_documented_model() -> None:
    """Guards the parametrised tests below from going vacuous."""
    assert {spec.name for spec in MODEL_SPECS} == DOCUMENTED_MODELS
    assert len({spec.model for spec in MODEL_SPECS}) == len(MODEL_SPECS)
    for spec in MODEL_SPECS:
        assert issubclass(spec.model, BaseModel), f"{spec.name} is not a pydantic model"
        assert spec.model.__name__ == spec.name


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_python_mode_round_trip(spec) -> None:
    """model -> model_dump() -> model.

    Plain `model_dump()` leaves `HttpUrl` objects and `datetime`s as objects rather than
    strings; both re-validate unchanged, so equality is the right assertion.
    """
    instance = spec.build()
    dumped = instance.model_dump()

    assert isinstance(dumped, dict)
    assert set(dumped) == set(spec.model.model_fields), (
        f"{spec.name}.model_dump() keys do not match the declared fields: "
        f"{sorted(set(dumped) ^ set(spec.model.model_fields))}"
    )

    rebuilt = spec.model.model_validate(dumped)
    assert rebuilt is not instance
    assert rebuilt == instance, f"{spec.name} did not survive model_dump() -> validate"


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_json_mode_round_trip(spec) -> None:
    """model -> model_dump(mode="json") -> model.

    The stricter of the two: urls become strings, datetimes become ISO-8601, frozensets
    and tuples become lists. All of it has to re-validate back to an equal instance, and
    a field whose type cannot be serialised raises here rather than at an API boundary.
    """
    instance = spec.build()
    dumped = instance.model_dump(mode="json")

    assert set(dumped) == set(spec.model.model_fields)
    rebuilt = spec.model.model_validate(dumped)
    assert rebuilt == instance, f"{spec.name} did not survive a json-mode round trip"


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_round_trip_carries_values_not_just_shape(spec) -> None:
    """A model that dropped every field would pass the two tests above."""
    instance = spec.build()
    mutated = spec.model.model_validate({**instance.model_dump(), **spec.mutation})

    assert mutated != instance, (
        f"changing {sorted(spec.mutation)} on {spec.name} produced an equal instance: "
        "either the field is being dropped or __eq__ ignores it"
    )
    for field, value in spec.mutation.items():
        assert getattr(mutated, field) == value


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_unknown_field_is_rejected(spec) -> None:
    """extra="forbid" catches a renamed field at the boundary instead of dropping it."""
    with pytest.raises(ValidationError) as exc_info:
        spec.model(**spec.kwargs_with(definitely_not_a_field="x"))

    errors = exc_info.value.errors()
    assert {error["type"] for error in errors} == {"extra_forbidden"}, (
        f"{spec.name} rejected the payload for the wrong reason: {errors}"
    )
    assert {error["loc"] for error in errors} == {("definitely_not_a_field",)}
