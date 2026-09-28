"""The anchor selection run in MLflow (#23): counts and the threshold as metrics, the two
histograms step-indexed, tables of types, keyword ranks and content gaps; never text."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text

from linking_engine.ml.tracking import (
    analytics_experiment,
    log_anchor_selection,
    selection_metrics,
    selection_params,
    selection_tables,
)
from linking_engine.models import (
    AnchorSelectionReport,
    AnchorType,
    AnchorTypeProfile,
    SemanticThreshold,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE

if TYPE_CHECKING:
    from pathlib import Path

NO_MENTION = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
NO_GOOD_PHRASE = UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
NO_KEYWORD = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
SCORES = (*([0] * 12), 3, 4, *([0] * 6))
SIMILARITIES = (*([0] * 13), 2, *([0] * 6))


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs go to a throwaway local store, never the remote server. Only the environment is
    set: an explicit set_tracking_uri would outlive the test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def report(**fields: object) -> AnchorSelectionReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pairs": 10,
        "lexical_pairs": 6,
        "semantic_invocations": 3,
        "semantic_matched": 2,
        "semantic_rejected_identifier": 4,
        "semantic_rejected_other_target": 5,
        "zero_overlap_matches": 1,
        "threshold": SemanticThreshold(
            value=0.62, quantile=0.99, negatives=400, positives=20, positive_recall=0.85
        ),
        "sentences_embedded": 30,
        "sentences_cached": 10,
        "phrases_embedded": 90,
        "phrases_cached": 0,
        "chosen": 7,
        "alternatives": 9,
        "unanchored": {NO_MENTION: 1, NO_GOOD_PHRASE: 1, NO_KEYWORD: 1},
        "chosen_types": {
            AnchorType.EXACT: 2,
            AnchorType.PARTIAL: 2,
            AnchorType.NATURAL: 3,
            AnchorType.BRANDED: 0,
        },
        "chosen_ranks": {1: 5, 2: 2},
        "profile": AnchorTypeProfile(),
        "targets": 4,
        "targets_with_anchor": 3,
        "features_filled": 6,
        "score_histogram": SCORES,
        "semantic_histogram": SIMILARITIES,
        "seconds": 1.5,
        "finished_at": datetime(2026, 9, 28, tzinfo=UTC),
        **fields,
    }
    return AnchorSelectionReport.model_validate(values)


def test_selection_metrics_are_counts_threshold_shares_and_breakdowns() -> None:
    assert selection_metrics(report()) == {
        "pairs": 10,
        "lexical_pairs": 6,
        "semantic_invocations": 3,
        "semantic_matched": 2,
        "semantic_rejected_identifier": 4,
        "semantic_rejected_other_target": 5,
        "zero_overlap_matches": 1,
        "sentences_embedded": 30,
        "sentences_cached": 10,
        "phrases_embedded": 90,
        "phrases_cached": 0,
        "chosen": 7,
        "alternatives": 9,
        "targets": 4,
        "targets_with_anchor": 3,
        "features_filled": 6,
        "seconds": 1.5,
        "threshold_value": 0.62,
        "threshold_overridden": 0.0,
        "threshold_negatives": 400,
        "threshold_positives": 20,
        "threshold_bounded": 0.0,
        "threshold_fallback": 0.0,
        "threshold_positive_recall": 0.85,
        "semantic_skipped": 0.0,
        "embedding_skipped": 0.0,
        "targets_with_anchor_share": 0.75,
        "unanchored_source_does_not_mention_topic": 1,
        "unanchored_topic_mentioned_but_no_good_phrase": 1,
        "unanchored_target_page_has_no_keyword": 1,
        "type_exact": 2,
        "type_partial": 2,
        "type_natural": 3,
        "type_branded": 0,
        "keyword_rank_1": 5,
        "keyword_rank_2": 2,
    }


def test_a_skipped_rung_without_positives_or_targets_logs_no_shares() -> None:
    skipped = report(
        pairs=0,
        lexical_pairs=0,
        semantic_invocations=0,
        semantic_matched=0,
        zero_overlap_matches=0,
        threshold=SemanticThreshold(
            value=0.6, quantile=0.99, negatives=0, positives=0, fallback=True
        ),
        semantic_skipped_reason="no Voyage API key",
        embedding_skipped_reason="no Voyage API key",
        chosen=0,
        alternatives=0,
        unanchored={},
        chosen_types={},
        chosen_ranks={},
        targets=0,
        targets_with_anchor=0,
        features_filled=0,
        score_histogram=(0,) * 20,
        semantic_histogram=(0,) * 20,
    )

    metrics = selection_metrics(skipped)

    assert (metrics["semantic_skipped"], metrics["embedding_skipped"]) == (1.0, 1.0)
    # No threshold was used, so none is reported.
    assert [name for name in metrics if name.startswith("threshold_")] == []
    absent = ("targets_with_anchor_share", "keyword_rank_1")
    assert [name for name in absent if name in metrics] == []
    assert selection_tables(skipped)["types.json"]["share"] == []


def test_selection_tables_set_types_against_the_profile() -> None:
    assert selection_tables(report()) == {
        "types.json": {
            "type": ["EXACT", "PARTIAL", "NATURAL", "BRANDED"],
            "chosen": [2, 2, 3, 0],
            "share": [2 / 7, 2 / 7, 3 / 7, 0.0],
            "profile": [0.15, 0.2, 0.5, 0.15],
        },
        "keyword_ranks.json": {"keyword_rank": [1, 2], "chosen": [5, 2]},
        "unanchored.json": {
            "reason": [
                "SOURCE_DOES_NOT_MENTION_TOPIC",
                "TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE",
                "TARGET_PAGE_HAS_NO_KEYWORD",
            ],
            "pairs": [1, 1, 1],
            "advice": [
                UNANCHORED_ADVICE[reason] for reason in (NO_MENTION, NO_GOOD_PHRASE, NO_KEYWORD)
            ],
        },
    }


OVERRIDDEN = SemanticThreshold(
    value=0.7, quantile=0.99, negatives=0, positives=4, positive_recall=0.5, overridden=True
)


def test_an_overridden_threshold_reports_its_value_but_no_derivation() -> None:
    metrics = selection_metrics(report(threshold=OVERRIDDEN))

    threshold = {name: value for name, value in metrics.items() if name.startswith("threshold_")}
    assert threshold == {
        "threshold_value": 0.7,
        "threshold_overridden": 1.0,
        "threshold_positives": 4.0,
        "threshold_positive_recall": 0.5,
    }


SCORING_PARAMS = {
    "semantic_weight": 0.4,
    "keyword_weight": 0.4,
    "diversity_weight": 0.1,
    "length_weight": 0.1,
    "stem_share": 0.7,
    "cosine_share": 0.3,
    "secondary_weight": 0.7,
    "profile_bonus": 0.1,
    "awkward_floor": 0.4,
    "score_bins": 20,
}
DERIVATION_PARAMS = {
    "threshold_quantile": 0.99,
    "threshold_bounds": "0.35,0.9",
    "min_negatives": 300,
    "negative_sample": 5000,
    "default_semantic_threshold": 0.6,
}
RUNG_PARAMS = {"top_sentences": 3, "max_phrases_per_sentence": 10}


def test_params_hold_the_rungs_and_the_derivations_constants_only_when_used() -> None:
    derived = selection_params(report())
    overridden = selection_params(report(threshold=OVERRIDDEN))
    skipped = selection_params(report(semantic_skipped_reason="no Voyage API key"))

    assert derived == {
        **SCORING_PARAMS,
        **RUNG_PARAMS,
        "threshold_overridden": False,
        **DERIVATION_PARAMS,
    }
    assert overridden == {**SCORING_PARAMS, **RUNG_PARAMS, "threshold_overridden": True}
    assert skipped == SCORING_PARAMS


def table(run_id: str, name: str) -> dict[str, list[object]]:
    stored = load_dict(f"runs:/{run_id}/{name}")
    return {
        column: [row[i] for row in stored["data"]] for i, column in enumerate(stored["columns"])
    }


def test_a_selection_run_logs_params_metrics_histograms_and_tables(local_mlflow: str) -> None:
    logged = report()

    run_id = log_anchor_selection(logged, "Anchor selection for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"], run.data.tags["tenant_id"]) == (
        "anchor selection",
        "anchor-selection",
        "acme",
    )
    assert run.data.tags["mlflow.note.content"] == "Anchor selection for tenant acme."
    assert run.data.tags["embedding_skipped"] == "none"
    assert run.data.params == {name: str(value) for name, value in selection_params(logged).items()}
    assert selection_metrics(logged).items() <= run.data.metrics.items()
    for name, bins in (("anchor_score_hist", SCORES), ("semantic_similarity_hist", SIMILARITIES)):
        history = sorted(client.get_metric_history(run_id, name), key=lambda m: m.step)
        assert [(m.step, m.value) for m in history] == list(enumerate(map(float, bins))), name
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "types.json",
        "keyword_ranks.json",
        "unanchored.json",
        "report.json",
        "summary.md",
    }
    for name, expected in selection_tables(logged).items():
        stored = table(run_id, name)
        assert list(stored) == list(expected), name
        for column, values in expected.items():
            assert stored[column] == pytest.approx(values, abs=1e-9), f"{name} {column}"
    assert (
        AnchorSelectionReport.model_validate_json(load_text(f"runs:/{run_id}/report.json"))
        == logged
    )
    assert load_text(f"runs:/{run_id}/summary.md") == "Anchor selection for tenant acme."
