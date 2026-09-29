"""The learned ranker in MLflow: one training run per tenant with its metrics, tables and model,
the model registered per tenant, and the production alias that ranking loads.

The alias moves only where RANKER_PROMOTION is "allowed" (CI and production), so a local run
can register a model but never promote it. Nothing logged holds a page url.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final

import lightgbm
import mlflow
import mlflow.lightgbm
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.types import ColSpec, DataType, Schema

from linking_engine.ml.quality import LINK_DERIVED_COLUMNS
from linking_engine.ml.ranking import (
    BOOTSTRAP_SAMPLES,
    DOMINANT_SHARE,
    MIN_TEST_GROUPS,
    MIN_TRAIN_GROUPS,
    PLACEMENT_COLUMNS,
    PREDICT_CHUNK,
    PRODUCT_K,
    TRAIN_THREADS,
    Trained,
    placement_gain_share,
)
from linking_engine.ml.tracking import EXPERIMENT_KIND, EXPERIMENT_KIND_TAG, start_stage_run
from linking_engine.models.ranking import NDCG_HISTOGRAM_BINS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from mlflow.entities.model_registry import ModelVersion

    from linking_engine.models import RankerReport, SeedResult

PROMOTION_ENV: Final = "RANKER_PROMOTION"
PROMOTION_ALLOWED: Final = "allowed"
ALIAS: Final = "production"
STAGE: Final = "ranker-training"
ISSUE: Final = "24-27"
MODEL_ARTIFACT: Final = "model"
# The server advertises multipart transfers through storage urls inside its own cluster;
# false streams them through the tracking server instead.
PROXY_ENV: Final = (
    "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD",
    "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD",
)
_MISSING: Final = ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


class RegistryUnavailableError(Exception):
    """The model registry or its artifact store failed: unreachable, refusing or erroring.

    Raised without its cause, whose text can hold the server's address: ``cause_type`` names
    the failure's type and ``error_code`` MLflow's code, None for a transport error.
    """

    def __init__(self, cause_type: str, error_code: str | None = None) -> None:
        detail = f"{cause_type} {error_code}" if error_code else cause_type
        super().__init__(f"model registry failed: {detail}")
        self.cause_type = cause_type
        self.error_code = error_code

    def __reduce__(self) -> tuple[type[RegistryUnavailableError], tuple[str, str | None]]:
        return type(self), (self.cause_type, self.error_code)


@contextmanager
def _registry() -> Iterator[None]:
    try:
        yield
    except MlflowException as error:
        raise RegistryUnavailableError(type(error).__name__, str(error.error_code)) from None
    except OSError as error:
        raise RegistryUnavailableError(type(error).__name__) from None


@dataclass(frozen=True)
class Holder:
    """A registered version of a tenant's ranker, the columns it reads in order, and how it was
    trained, from the version's tags: None where a tag is missing or malformed."""

    version: str
    run_id: str | None
    booster: lightgbm.Booster
    columns: tuple[str, ...]
    feature_set_version: str | None
    split_seed: int | None
    test_share: float | None
    valid_share: float | None


def trained_holder(
    report: RankerReport, trained: Trained, run_id: str, model_version: str
) -> Holder:
    """The Holder of a model this process trained and registered, for its local copy."""
    return Holder(
        version=model_version,
        run_id=run_id,
        booster=trained.booster,
        columns=trained.columns,
        feature_set_version=report.feature_set_version,
        split_seed=report.settings.split_seed,
        test_share=report.settings.test_share,
        valid_share=report.settings.valid_share,
    )


def ranker_experiment(tenant_id: str) -> str:
    return f"ranker-{tenant_id}"


def registered_model(tenant_id: str) -> str:
    return f"link-ranker-{tenant_id}"


def promotion_allowed() -> bool:
    return os.environ.get(PROMOTION_ENV) == PROMOTION_ALLOWED


def _proxy_transfers() -> None:
    for name in PROXY_ENV:
        os.environ.setdefault(name, "false")


def _use_ranker_experiment(tenant_id: str) -> None:
    experiment = mlflow.set_experiment(ranker_experiment(tenant_id))
    if experiment.tags.get(EXPERIMENT_KIND_TAG) != EXPERIMENT_KIND:
        mlflow.set_experiment_tag(EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)


def ranker_params(report: RankerReport) -> dict[str, object]:
    settings, params = report.settings, report.params
    return {
        "rounds": settings.rounds,
        "share": settings.share,
        "hide_seed": settings.seed,
        "test_share": settings.test_share,
        "valid_share": settings.valid_share,
        "split_seed": settings.split_seed,
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "min_data_in_leaf": params.min_data_in_leaf,
        "feature_fraction": params.feature_fraction,
        "max_rounds": params.max_rounds,
        "early_stopping_rounds": params.early_stopping_rounds,
        "eval_at": params.eval_at,
        "seed": params.seed,
        "monotone_increasing": ",".join(params.monotone_increasing) or "none",
        "evaluation_seeds": ",".join(map(str, settings.evaluation_seeds)),
        "product_k": PRODUCT_K,
        "columns": len(report.columns),
        "excluded_columns": ",".join(report.excluded_columns) or "none",
        "link_derived_columns": ",".join(LINK_DERIVED_COLUMNS),
        "placement_columns": ",".join(PLACEMENT_COLUMNS),
        "train_threads": TRAIN_THREADS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "min_train_groups": MIN_TRAIN_GROUPS,
        "min_test_groups": MIN_TEST_GROUPS,
        "dominant_share": DOMINANT_SHARE,
        "predict_chunk": PREDICT_CHUNK,
        "lightgbm": version("lightgbm"),
    }


def ranker_metrics(report: RankerReport) -> dict[str, float]:
    """Every numeric result of the run, flat, under names stable across runs and tenants."""
    metrics: dict[str, float] = {
        "corpus_pages": float(report.corpus_pages),
        "body_links": float(report.body_links),
        "train_groups": float(report.train_groups),
        "valid_groups": float(report.valid_groups),
        "test_groups": float(report.test_groups),
        "positives": float(report.positives),
        "seconds": report.seconds,
        "skipped": float(report.skipped_reason is not None),
        "unlabelable_targets": float(report.unlabelable_targets),
        "unlabelable_rows": float(report.unlabelable_rows),
        "seeds_skipped": float(len(report.skipped_seeds)),
    }
    if results := report.seed_results:
        metrics.update(
            {
                "seeds_evaluated": float(len(results)),
                "seeds_worse": float(report.seeds_worse),
                "seeds_better": float(report.seeds_better),
                "seed_mean_learned_ndcg_at_10": _mean(r.learned[0] for r in results),
                "seed_mean_plain_ndcg_at_10": _mean(r.plain[0] for r in results),
                "seed_mean_baseline_ndcg_at_10": _mean(r.baseline[0] for r in results),
                "seed_mean_delta": _mean(r.delta for r in results),
            }
        )
    for measured in report.product_measures:
        prefix = f"product_{measured.scorer.value}"
        found = {
            "top_relevance": measured.top_relevance,
            "same_hub_share": measured.same_hub_share,
            "orphan_slot_share": measured.orphan_slot_share,
            "orphan_page_share": measured.orphan_page_share,
            "orphans_reached": measured.orphans_reached,
            "orphans_to_pillar": measured.orphans_to_pillar,
            "inbound_gini": measured.inbound_gini,
        }
        metrics.update({f"{prefix}_{k}": v for k, v in found.items() if v is not None})
    if report.best_iteration is not None:
        metrics["best_iteration"] = float(report.best_iteration)
    for entry in report.metrics:
        name = entry.scorer.value
        metrics.update(
            {
                f"{name}_ndcg_at_10": entry.ndcg_at_10,
                f"{name}_ndcg_at_10_ci_low": entry.ci_low,
                f"{name}_ndcg_at_10_ci_high": entry.ci_high,
                f"{name}_precision_at_5": entry.precision_at_5,
                f"{name}_groups": float(entry.groups),
                f"{name}_groups_with_positive_share": entry.groups_with_positive_share,
                f"{name}_labelled_pairs": float(entry.n_labelled_pairs),
            }
        )
    if report.importance:
        metrics["top_gain_share"] = report.importance[0].gain_share
        metrics["placement_gain_share"] = placement_gain_share(report.importance)
    if (decision := report.promotion) is not None:
        metrics.update(
            {
                "promotion_delta": decision.delta,
                "promotion_delta_ci_low": decision.delta_ci_low,
                "promotion_delta_ci_high": decision.delta_ci_high,
                "would_promote": float(decision.would_promote),
            }
        )
    return metrics


def _mean(values: Iterable[float]) -> float:
    found = list(values)
    return sum(found) / len(found)


def ranker_step_metrics(report: RankerReport) -> dict[str, dict[int, float]]:
    """Distributions as metrics whose step is the round, the histogram bin, or the seed's index
    among the evaluation seeds (seed_value names the seed)."""
    steps: dict[str, dict[int, float]] = {
        f"round_{field}": {
            r.round: float(value) for r in report.rounds if (value := getattr(r, field)) is not None
        }
        for field in (
            "hidden",
            "recoverable",
            "pairs",
            "positives",
            "groups_with_positive",
            "positive_placement_share",
            "negative_placement_share",
        )
    }
    for entry in report.metrics:
        name = entry.scorer.value
        steps[f"{name}_ndcg_at_10_by_round"] = dict(entry.per_round)
        steps[f"{name}_ndcg_hist"] = {b: float(n) for b, n in enumerate(entry.histogram)}
    index = {seed: i for i, seed in enumerate(report.settings.evaluation_seeds)}
    for field, value_of in _SEED_STEPS:
        steps[f"seed_{field}"] = {index[r.seed]: value_of(r) for r in report.seed_results}
    return {name: values for name, values in steps.items() if values}


_SEED_STEPS: Final[tuple[tuple[str, Callable[[SeedResult], float]], ...]] = (
    ("value", lambda r: float(r.seed)),
    ("test_pages", lambda r: float(r.test_pages)),
    ("learned_ndcg_at_10", lambda r: r.learned[0]),
    ("plain_ndcg_at_10", lambda r: r.plain[0]),
    ("baseline_ndcg_at_10", lambda r: r.baseline[0]),
    ("delta", lambda r: r.delta),
    ("delta_ci_low", lambda r: r.delta_ci_low),
    ("delta_ci_high", lambda r: r.delta_ci_high),
)


def ranker_tables(report: RankerReport) -> dict[str, dict[str, list[object]]]:
    """Per-item results as MLflow tables, keyed by artifact file; no page urls."""
    tables: dict[str, dict[str, list[object]]] = {}
    if report.rounds:
        tables["rounds.json"] = {
            "round": [r.round for r in report.rounds],
            "hidden": [r.hidden for r in report.rounds],
            "recoverable": [r.recoverable for r in report.rounds],
            "pairs": [r.pairs for r in report.rounds],
            "positives": [r.positives for r in report.rounds],
            "negatives": [r.pairs - r.positives for r in report.rounds],
            "positive_share": [r.positives / r.pairs if r.pairs else 0.0 for r in report.rounds],
            "groups_with_positive": [r.groups_with_positive for r in report.rounds],
            "positive_placement_share": [r.positive_placement_share for r in report.rounds],
            "negative_placement_share": [r.negative_placement_share for r in report.rounds],
        }
    if report.metrics:
        entries = report.metrics
        tables["ranking_metrics.json"] = {
            "scorer": [e.scorer.value for e in entries],
            "ndcg_at_10": [e.ndcg_at_10 for e in entries],
            "ci_low": [e.ci_low for e in entries],
            "ci_high": [e.ci_high for e in entries],
            "precision_at_5": [e.precision_at_5 for e in entries],
            "groups": [e.groups for e in entries],
            "groups_with_positive_share": [e.groups_with_positive_share for e in entries],
            "labelled_pairs": [e.n_labelled_pairs for e in entries],
        }
        by_round = [(e.scorer.value, r, v) for e in entries for r, v in sorted(e.per_round.items())]
        tables["ndcg_by_round.json"] = {
            "scorer": [scorer for scorer, _, _ in by_round],
            "round": [r for _, r, _ in by_round],
            "ndcg_at_10": [v for _, _, v in by_round],
        }
        width = 1 / NDCG_HISTOGRAM_BINS
        tables["ndcg_histogram.json"] = {
            "scorer": [e.scorer.value for e in entries for _ in range(NDCG_HISTOGRAM_BINS)],
            "bin": [b for _ in entries for b in range(NDCG_HISTOGRAM_BINS)],
            "low": [round(b * width, 6) for _ in entries for b in range(NDCG_HISTOGRAM_BINS)],
            "high": [
                round((b + 1) * width, 6) for _ in entries for b in range(NDCG_HISTOGRAM_BINS)
            ],
            "groups": [n for e in entries for n in e.histogram],
        }
    if report.importance:
        tables["importance.json"] = {
            "column": [e.column for e in report.importance],
            "gain": [e.gain for e in report.importance],
            "gain_share": [e.gain_share for e in report.importance],
            "link_derived": [e.column in LINK_DERIVED_COLUMNS for e in report.importance],
            "placement": [e.column in PLACEMENT_COLUMNS for e in report.importance],
        }
    if results := report.seed_results:
        table: dict[str, list[object]] = {
            "seed": [r.seed for r in results],
            "test_pages": [r.test_pages for r in results],
        }
        for name in ("learned", "plain", "baseline"):
            triples = [getattr(r, name) for r in results]
            table[f"{name}_ndcg_at_10"] = [t[0] for t in triples]
            table[f"{name}_ci_low"] = [t[1] for t in triples]
            table[f"{name}_ci_high"] = [t[2] for t in triples]
        table["delta"] = [r.delta for r in results]
        table["delta_ci_low"] = [r.delta_ci_low for r in results]
        table["delta_ci_high"] = [r.delta_ci_high for r in results]
        table["against_plain"] = [
            "worse" if r.significantly_worse else "better" if r.significantly_better else "same"
            for r in results
        ]
        tables["seed_results.json"] = table
    if skipped := report.skipped_seeds:
        tables["skipped_seeds.json"] = {"seed": list(skipped), "reason": list(skipped.values())}
    if measures := report.product_measures:
        tables["product_measures.json"] = {
            "scorer": [m.scorer.value for m in measures],
            "k": [m.k for m in measures],
            "top_relevance": [m.top_relevance for m in measures],
            "same_hub_share": [m.same_hub_share for m in measures],
            "orphan_slot_share": [m.orphan_slot_share for m in measures],
            "orphan_page_share": [m.orphan_page_share for m in measures],
            "orphans_reached": [m.orphans_reached for m in measures],
            "orphans_to_pillar": [m.orphans_to_pillar for m in measures],
            "inbound_gini": [m.inbound_gini for m in measures],
        }
    return tables


def _signature(columns: tuple[str, ...]) -> ModelSignature:
    return ModelSignature(
        inputs=Schema([ColSpec(DataType.float, column) for column in columns]),
        outputs=Schema([ColSpec(DataType.double)]),
    )


def log_ranker(
    report: RankerReport, trained: Trained | None, summary: str
) -> tuple[str, str | None]:
    """Log one training run and register its model under the tenant's name; returns the run id
    and the registered version, None for a skipped run. The run is logged as not promoted: the
    alias is not touched here, and tag_promoted records a move on the run once it happened.
    Registry failures raise RegistryUnavailableError."""
    if (trained is None) != (report.skipped_reason is not None):
        raise ValueError("a model is logged exactly when the run was not skipped")
    if trained is not None and trained.columns != report.columns:
        raise ValueError("the model's columns differ from the report's")
    _proxy_transfers()
    with _registry():
        return _log_ranker(report, trained, summary)


def _log_ranker(
    report: RankerReport, trained: Trained | None, summary: str
) -> tuple[str, str | None]:
    tenant_id = report.tenant_id
    _use_ranker_experiment(tenant_id)
    with start_stage_run(
        run_name="ranker training",
        tags={
            "tenant_id": tenant_id,
            "kind": "train",
            "stage": STAGE,
            "issue": ISSUE,
            "corpus_pages": str(report.corpus_pages),
            "body_links": str(report.body_links),
            "seed": str(report.params.seed),
            "hide_seed": str(report.settings.seed),
            "split_seed": str(report.settings.split_seed),
            "feature_set_version": report.feature_set_version,
            "git_sha": report.git_sha,
            "skipped_reason": report.skipped_reason or "none",
            "dominant_feature": report.dominant_feature or "none",
            "product_skipped_reason": report.product_skipped_reason or "none",
            "skipped_seeds": ",".join(map(str, report.skipped_seeds)) or "none",
            "promoted": "false",
            "mlflow.note.content": summary,
        },
    ) as run:
        run_id = str(run.info.run_id)
        client = mlflow.MlflowClient()
        mlflow.log_params(ranker_params(report))
        metrics = ranker_metrics(report)
        mlflow.log_metrics({**metrics, "promoted": 0.0})
        steps = ranker_step_metrics(report)
        if steps:
            now = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[
                    Metric(name, value, now, step)
                    for name, values in steps.items()
                    for step, value in sorted(values.items())
                ],
            )
        for artifact, table in ranker_tables(report).items():
            mlflow.log_table(table, artifact)
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(
            {"columns": list(report.columns), "excluded": dict(report.excluded_columns)},
            "columns.json",
        )
        mlflow.log_text(summary, "summary.md")
        if trained is None:
            return run_id, None
        info = mlflow.lightgbm.log_model(
            trained.booster,
            name=MODEL_ARTIFACT,
            registered_model_name=registered_model(tenant_id),
            signature=_signature(trained.columns),
            pip_requirements=[f"lightgbm=={version('lightgbm')}"],
        )
        registered = info.registered_model_version
        if registered is None:
            raise RuntimeError("the model was logged but not registered")
        model_version = str(registered)
        name = registered_model(tenant_id)
        client.set_registered_model_tag(name, "tenant_id", tenant_id)
        settings = report.settings
        for key, value in (
            ("tenant_id", tenant_id),
            ("feature_set_version", report.feature_set_version),
            ("git_sha", report.git_sha),
            ("split_seed", str(settings.split_seed)),
            ("test_share", str(settings.test_share)),
            ("valid_share", str(settings.valid_share)),
        ):
            client.set_model_version_tag(name, model_version, key, value)
        mlflow.set_tag("registered_version", model_version)
        return run_id, model_version


def tag_promoted(run_id: str, model_version: str) -> None:
    """Record on a training run that its model now holds the production alias; call only once
    move_alias succeeded. Registry failures raise RegistryUnavailableError."""
    client = mlflow.MlflowClient()
    with _registry():
        client.set_tag(run_id, "promoted", "true")
        client.set_tag(run_id, "model_version", model_version)
        client.log_metric(run_id, "promoted", 1.0)


def _production(tenant_id: str) -> ModelVersion | None:
    """The version holding the tenant's production alias; None before the first promotion.
    A version tagged with another tenant raises."""
    client = mlflow.MlflowClient()
    name = registered_model(tenant_id)
    try:
        aliases = client.get_registered_model(name).aliases
    except MlflowException as error:
        if error.error_code == _MISSING:
            return None
        raise
    if ALIAS not in aliases:
        return None
    found = client.get_model_version(name, str(aliases[ALIAS]))
    if found.tags.get("tenant_id") != tenant_id:
        raise ValueError("the production version is not tagged with this tenant")
    return found


def _int_tag(tags: Mapping[str, str], key: str) -> int | None:
    try:
        return int(tags[key])
    except (KeyError, ValueError):
        return None


def _float_tag(tags: Mapping[str, str], key: str) -> float | None:
    try:
        return float(tags[key])
    except (KeyError, ValueError):
        return None


def _holder(found: ModelVersion, booster: lightgbm.Booster) -> Holder:
    tags = found.tags
    return Holder(
        version=str(found.version),
        run_id=found.run_id or None,
        booster=booster,
        columns=tuple(booster.feature_name()),
        feature_set_version=tags.get("feature_set_version") or None,
        split_seed=_int_tag(tags, "split_seed"),
        test_share=_float_tag(tags, "test_share"),
        valid_share=_float_tag(tags, "valid_share"),
    )


def _download(tenant_id: str, found: ModelVersion) -> Holder:
    _proxy_transfers()
    booster = mlflow.lightgbm.load_model(f"models:/{registered_model(tenant_id)}/{found.version}")
    if not isinstance(booster, lightgbm.Booster):
        raise TypeError("the registered model is not a LightGBM booster")
    return _holder(found, booster)


def holder(tenant_id: str) -> Holder | None:
    """The version holding the tenant's production alias, downloaded; None when there is none.
    Registry failures raise RegistryUnavailableError."""
    with _registry():
        found = _production(tenant_id)
        return None if found is None else _download(tenant_id, found)


def move_alias(tenant_id: str, model_version: str) -> None:
    """Point the tenant's production alias at ``model_version``; only where promotion is
    allowed, and only to a version registered for this tenant."""
    if not promotion_allowed():
        raise RuntimeError(
            f"promotion is not allowed here: {PROMOTION_ENV} is not {PROMOTION_ALLOWED!r}"
        )
    client = mlflow.MlflowClient()
    name = registered_model(tenant_id)
    with _registry():
        tagged = client.get_model_version(name, model_version).tags.get("tenant_id")
        if tagged != tenant_id:
            raise ValueError("the version is not tagged with this tenant")
        client.set_registered_model_alias(name, ALIAS, model_version)


def _ranker_folder(cache_dir: Path, tenant_id: str) -> Path:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    if tenant_id in {".", ".."} or Path(tenant_id).name != tenant_id:
        raise ValueError("tenant_id must be usable as a directory name")
    return cache_dir / tenant_id / "ranker"


def _local_paths(cache_dir: Path, tenant_id: str, model_version: str) -> tuple[Path, Path]:
    folder = _ranker_folder(cache_dir, tenant_id)
    return folder / f"model-{model_version}.txt", folder / f"model-{model_version}.columns.json"


def _write(path: Path, text: str) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def save_local(tenant_id: str, cache_dir: Path, found: Holder) -> Path:
    """Keep a copy of a registered version beside the tenant's other caches; returns the model
    file. It is trusted only while the registry still names the same training run for that
    version."""
    model_path, columns_path = _local_paths(cache_dir, tenant_id, found.version)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _write(model_path, found.booster.model_to_string())
    _write(
        columns_path,
        json.dumps(
            {"version": found.version, "run_id": found.run_id, "columns": list(found.columns)}
        ),
    )
    return model_path


def _read_local(tenant_id: str, cache_dir: Path, found: ModelVersion) -> Holder | None:
    model_version = str(found.version)
    model_path, columns_path = _local_paths(cache_dir, tenant_id, model_version)
    if not found.run_id or not (model_path.is_file() and columns_path.is_file()):
        return None
    try:
        stored = json.loads(columns_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(stored, dict)
        or stored.get("version") != model_version
        or stored.get("run_id") != found.run_id
    ):
        return None
    try:
        booster = lightgbm.Booster(model_file=str(model_path))
    except (OSError, ValueError, lightgbm.basic.LightGBMError):
        return None
    if stored.get("columns") != booster.feature_name():
        return None
    return _holder(found, booster)


def load_production(tenant_id: str, cache_dir: Path) -> Holder | None:
    """The tenant's production model: the alias and the version's tags from the registry, the
    model from the local copy when it holds that version from the same training run, else
    downloaded and copied. None when nothing is promoted; registry failures raise
    RegistryUnavailableError."""
    _ranker_folder(cache_dir, tenant_id)
    with _registry():
        found = _production(tenant_id)
    if found is None:
        return None
    local = _read_local(tenant_id, cache_dir, found)
    if local is not None:
        return local
    with _registry():
        downloaded = _download(tenant_id, found)
    save_local(tenant_id, cache_dir, downloaded)
    return downloaded
