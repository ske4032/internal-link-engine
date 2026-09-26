"""Issue #2 Gotcha: `frozenset` for `issue_flags`, not `set` — the models are frozen.

`frozen=True` gives a model `__hash__`, and `__hash__` walks the field values. One
`set[IssueFlag]` or one `list[float]` anywhere in a model and `hash(instance)` raises
`TypeError` at the first point something puts a result in a set or uses it as a dict
key. The type annotation alone does not say this; `hash()` does.
"""

from __future__ import annotations

import pytest
from factories import (
    ANCHOR_SPEC,
    AUDIT_SPEC,
    LINK_SPEC,
    MODEL_IDS,
    MODEL_SPECS,
    PAGE_SPEC,
    RECOMMENDATION_SPEC,
    VECTOR,
)
from pydantic import ValidationError

from linking_engine.models.enums import IssueFlag


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_model_config_is_frozen_and_forbids_extras(spec) -> None:
    config = spec.model.model_config
    assert config.get("frozen") is True, f"{spec.name} is not frozen"
    assert config.get("extra") == "forbid", f"{spec.name} does not forbid extra fields"


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=MODEL_IDS)
def test_assigning_to_a_field_raises(spec) -> None:
    instance = spec.build()
    field = next(iter(spec.model.model_fields))
    with pytest.raises(ValidationError) as exc_info:
        setattr(instance, field, getattr(instance, field))
    assert exc_info.value.errors()[0]["type"] == "frozen_instance", (
        f"{spec.name}.{field} accepted an assignment for a reason other than frozen"
    )


def test_link_audit_result_is_hashable() -> None:
    """The test that catches `set[IssueFlag]`."""
    result = AUDIT_SPEC.build()
    twin = AUDIT_SPEC.build()

    assert isinstance(hash(result), int)
    assert twin == result
    assert hash(twin) == hash(result)
    assert len({result, twin}) == 1


def test_issue_flags_is_a_frozenset() -> None:
    result = AUDIT_SPEC.build()
    assert isinstance(result.issue_flags, frozenset)
    assert not isinstance(result.issue_flags, set), (
        "frozenset is not a subclass of set; a plain set here is unhashable"
    )
    assert result.issue_flags == frozenset({IssueFlag.GENERIC, IssueFlag.MISALIGNED})


def test_issue_flags_coerces_a_list_and_stays_hashable() -> None:
    duplicated = [IssueFlag.BROKEN, IssueFlag.BROKEN, IssueFlag.NOFOLLOW]
    result = AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(issue_flags=duplicated))
    assert result.issue_flags == frozenset({IssueFlag.BROKEN, IssueFlag.NOFOLLOW})
    assert isinstance(hash(result), int)


def test_a_clean_edge_has_no_flags_and_no_verdict() -> None:
    result = AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(issue_flags=frozenset(), verdict=None))
    assert result.issue_flags == frozenset()
    assert result.verdict is None


def test_page_vectors_are_tuples_and_optional() -> None:
    page = PAGE_SPEC.model(
        **PAGE_SPEC.kwargs_with(content_embedding=list(VECTOR), gnn_embedding=None)
    )
    assert isinstance(page.content_embedding, tuple), "a list here makes Page unhashable"
    assert page.content_embedding == VECTOR
    assert page.gnn_embedding is None, "a page that has not been through the GNN yet"
    assert isinstance(hash(page), int)


def test_link_surrounding_embedding_is_a_tuple() -> None:
    link = LINK_SPEC.model(**LINK_SPEC.kwargs_with(surrounding_embedding=list(VECTOR)))
    assert isinstance(link.surrounding_embedding, tuple)
    assert link.surrounding_embedding == VECTOR
    assert isinstance(hash(link), int)


def test_proposed_anchors_is_a_tuple_of_hashable_candidates() -> None:
    candidate = ANCHOR_SPEC.build()
    kwargs = RECOMMENDATION_SPEC.kwargs_with(proposed_anchors=[candidate])
    recommendation = RECOMMENDATION_SPEC.model(**kwargs)

    assert isinstance(recommendation.proposed_anchors, tuple)
    assert recommendation.proposed_anchors == (candidate,)
    assert isinstance(hash(candidate), int)
