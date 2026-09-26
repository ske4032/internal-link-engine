"""`has_gsc_data` exists so the features can tell missing apart from zero.

A new page has no Search Console history at all; a mature page can genuinely have zero
impressions. Typing the GSC block as plain `float` with a 0.0 default collapses those
two into the same row, and the ranker learns that new pages are bad rather than unknown.
"""

from __future__ import annotations

from factories import GSC_FIELDS, PAIR_SPEC


def _no_gsc_history():
    """A NEW page: never crawled by Search Console, every GSC field null."""
    overrides = dict.fromkeys(GSC_FIELDS, None)
    return PAIR_SPEC.model(**PAIR_SPEC.kwargs_with(has_gsc_data=False, **overrides))


def _measured_zero():
    """A page with GSC history whose numbers happen to be zero.

    The literal "0" is used rather than 0.0 because the spec does not pin whether
    `target_position_band` is numeric or a bucket label; "0" validates as either.
    """
    overrides = dict.fromkeys(GSC_FIELDS, "0")
    return PAIR_SPEC.model(**PAIR_SPEC.kwargs_with(has_gsc_data=True, **overrides))


def test_a_page_with_no_gsc_history_is_valid() -> None:
    features = _no_gsc_history()
    assert features.has_gsc_data is False
    for field in GSC_FIELDS:
        assert getattr(features, field) is None, f"{field} must accept None"


def test_a_page_with_measured_zeroes_is_valid() -> None:
    features = _measured_zero()
    assert features.has_gsc_data is True
    for field in GSC_FIELDS:
        value = getattr(features, field)
        assert value is not None
        assert float(value) == 0.0


def test_missing_and_zero_are_different_rows() -> None:
    missing = _no_gsc_history()
    zero = _measured_zero()

    assert missing != zero, "missing GSC data and zero GSC data must not compare equal"
    assert missing.has_gsc_data is not zero.has_gsc_data
    for field in GSC_FIELDS:
        assert getattr(missing, field) != getattr(zero, field), (
            f"{field} lost the distinction between no data and a measured zero"
        )


def test_null_gsc_fields_survive_serialisation_as_null() -> None:
    missing = _no_gsc_history()
    dumped = missing.model_dump(mode="json")

    for field in GSC_FIELDS:
        assert dumped[field] is None, f"{field} serialised as {dumped[field]!r}, not null"
    assert PAIR_SPEC.model.model_validate(dumped) == missing
