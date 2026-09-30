"""Hand labels for the ranker: a blind export of a tenant's candidate pairs for a reviewer to
label, and the import of the labelled file into anchor_feedback.

The export samples source pages, not scattered pairs: a page's ranking can only be judged
against other labelled candidates of the same page. Each sampled page contributes one pair
from each band of its scores, best to worst, so the labels cover what the ranker would reject
as well as what it would serve. The file holds no score, rank or other model output, and its
rows are shuffled; the snapshot the pairs were proposed from stays on the server.
"""

from __future__ import annotations

import asyncio
import csv
import secrets
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pyarrow.parquet as pq
import structlog

from linking_engine.discovery.features import KEY_COLUMNS
from linking_engine.discovery.scoring import default_weights
from linking_engine.models import (
    GRADES,
    LABEL_WORDS,
    ExportedPair,
    LabelEvent,
    LabelExport,
    LabelImportReport,
    LabelProblem,
    LabelSettings,
    RecommendationStatus,
)
from linking_engine.pipeline.anchors import cache_folder
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE
from linking_engine.pipeline.recommendations import ANCHOR_MAX_CHARS, read_ranked

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    import pandas

    from linking_engine.ingest.mongo_repo import MongoRepo

log = structlog.get_logger(__name__)

STAGE: Final = "labels"
# The file's columns: the pair as the reviewer sees it, then what the reviewer fills in.
PAIR_COLUMNS: Final = (
    "pair_id",
    "source_url",
    "target_url",
    "target_title",
    "sentence",
    "proposed_anchor",
)
LABEL_COLUMNS: Final = ("label", "anchor_used", "reason", "reviewer")
COLUMNS: Final = (*PAIR_COLUMNS, *LABEL_COLUMNS)
# The proposed anchor is marked in its sentence.
MARK_OPEN: Final = "[["
MARK_CLOSE: Final = "]]"
MISSING_SHOWN: Final = 10
# A cell opening with one of these is read as a formula by spreadsheets.
FORMULA_STARTS: Final = ("=", "+", "-", "@", "\t", "\r")
_CHOICE_COLUMNS: Final = (
    "source_url",
    "target_url",
    "rank",
    "anchor_type",
    "keyword",
    "rung",
    "phrase",
    "start",
    "sentence",
    "sentence_start",
)
_SAVE_AS: Final = "save it as comma-separated CSV, UTF-8"

Pair = tuple[str, str]


class LabelFileError(ValueError):
    """A label file that cannot be imported, with every problem found in it."""

    def __init__(self, name: str, problems: Sequence[LabelProblem]) -> None:
        self.problems = tuple(problems)
        count = len(self.problems)
        lines = [f"{name}: {count} problem{'' if count == 1 else 's'}; nothing imported"]
        lines.extend(_describe(problem) for problem in self.problems)
        super().__init__("\n".join(lines))


@dataclass(frozen=True, slots=True)
class Sample:
    """The pairs drawn for an export, and what they were drawn from."""

    pairs: tuple[ExportedPair, ...]
    pages: int
    eligible_pages: int
    eligible_pairs: int
    small_pages: int


async def export_labels(
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    out_dir: Path,
    settings: LabelSettings,
) -> tuple[LabelExport, Path]:
    """Sample the tenant's ranked, anchored candidate pairs, store them with the snapshot they
    were proposed from, and write the blind label file into ``out_dir``. The export is stored
    complete before the file is written, so every written file can be imported."""
    started = time.perf_counter()
    folder = cache_folder(cache_dir, tenant_id)
    ranked_path, choices_path = folder / RANKED_PAIRS_FILE, folder / ANCHOR_CHOICES_FILE
    for path, flow in ((ranked_path, "rank-pairs"), (choices_path, "anchor-selection")):
        if not path.is_file():
            raise ValueError(f"no {path.name} for tenant {tenant_id!r}; run {flow} first")
    if choices_path.stat().st_mtime > ranked_path.stat().st_mtime:
        raise ValueError("the anchor choices changed since rank-pairs; rerun rank-pairs")
    ranked, scorer, model_version = await asyncio.to_thread(read_ranked, ranked_path, tenant_id)
    choices = await asyncio.to_thread(_read_choices, choices_path)
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    excluded = frozenset(page.url for page in await mongo.excluded_pages(tenant_id))
    titles = await mongo.page_titles_by_url(tenant_id)
    sample = sample_pairs(candidate_pool(ranked, choices, excluded), settings)
    export_id = uuid.uuid4().hex
    export = LabelExport(
        export_id=export_id,
        created_at=datetime.now(UTC),
        seed=settings.seed,
        pages=sample.pages,
        pairs_per_page=settings.pairs_per_page,
        pairs=len(sample.pairs),
        scorer=scorer,
        model_version=model_version,
        weights_version=weights.version,
        eligible_pages=sample.eligible_pages,
        eligible_pairs=sample.eligible_pairs,
        small_pages=sample.small_pages,
        file_name=f"{tenant_id}-labels-{export_id[:12]}.csv",
    )
    await mongo.insert_label_pairs(tenant_id, export_id, sample.pairs)
    await mongo.complete_label_export(tenant_id, export)
    path = await asyncio.to_thread(
        write_label_file, out_dir / export.file_name, sample.pairs, titles, settings.seed
    )
    log.info(
        "labels.exported",
        stage=STAGE,
        tenant_id=tenant_id,
        export_id=export_id,
        pages=export.pages,
        pairs=export.pairs,
        eligible_pages=export.eligible_pages,
        small_pages=export.small_pages,
        scorer=scorer.value,
        seconds=round(time.perf_counter() - started, 3),
    )
    return export, path


def _read_choices(path: Path) -> pandas.DataFrame:
    frame = pq.read_table(path, columns=list(_CHOICE_COLUMNS)).to_pandas()
    if frame.duplicated([*KEY_COLUMNS, "rank"]).any():
        raise ValueError("the anchor choices repeat a pair; rerun anchor-selection")
    return frame


def candidate_pool(
    ranked: pandas.DataFrame, choices: pandas.DataFrame, excluded: frozenset[str]
) -> pandas.DataFrame:
    """The ranked pairs a reviewer can judge as served: with a first-choice anchor short enough
    to serve, between pages that are not excluded."""
    first = choices.loc[choices["rank"] == 1].drop(columns="rank")
    pool = ranked.merge(first, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    keep = (
        ~pool["source_url"].isin(excluded)
        & ~pool["target_url"].isin(excluded)
        & (pool["phrase"].str.len() <= ANCHOR_MAX_CHARS)
    )
    return pool.loc[keep].reset_index(drop=True)


def sample_pairs(pool: pandas.DataFrame, settings: LabelSettings) -> Sample:
    """``settings.pages`` source pages drawn at random from those with at least
    ``pairs_per_page`` candidates, all of them when fewer. Each drawn page's candidates, best
    score first, are split into ``pairs_per_page`` bands of near-equal size, and one pair is
    drawn from each band. The same pool and seed draw the same pairs."""
    per_page = settings.pairs_per_page
    sizes = pool.groupby("source_url").size()
    eligible = sorted(str(url) for url in sizes.index[sizes >= per_page])
    if not eligible:
        raise ValueError(f"no source page has {per_page} anchored candidate pairs to label")
    rng = np.random.default_rng(settings.seed)
    picks = rng.choice(len(eligible), size=min(settings.pages, len(eligible)), replace=False)
    drawn = sorted(eligible[int(pick)] for pick in picks)
    pairs: list[ExportedPair] = []
    grouped = pool.loc[pool["source_url"].isin(drawn)].groupby("source_url", sort=True)
    for _, group in grouped:
        rows = group.sort_values(["score", "target_url"], ascending=[False, True])
        records = rows.to_dict("records")
        for band, positions in enumerate(np.array_split(np.arange(len(rows)), per_page), start=1):
            pick = int(positions[int(rng.integers(len(positions)))])
            pairs.append(_exported(records[pick], band))
    return Sample(
        pairs=tuple(pairs),
        pages=len(drawn),
        eligible_pages=len(eligible),
        eligible_pairs=int(sizes[sizes >= per_page].sum()),
        small_pages=int((sizes < per_page).sum()),
    )


def _exported(row: Mapping[Hashable, Any], band: int) -> ExportedPair:
    return ExportedPair(
        pair_id=secrets.token_hex(8),
        source_url=row["source_url"],
        target_url=row["target_url"],
        anchor=row["phrase"],
        anchor_type=row["anchor_type"],
        rung=row["rung"],
        keyword=row["keyword"],
        sentence=row["sentence"],
        anchor_start=int(row["start"]) - int(row["sentence_start"]),
        score=float(row["score"]),
        rank_in_source=int(row["rank_in_source"]),
        band=band,
    )


def _cell(text: str) -> str:
    """Page text a spreadsheet would read as a formula, shown as text instead."""
    return f"'{text}" if text.startswith(FORMULA_STARTS) else text


def marked_sentence(pair: ExportedPair) -> str:
    """The pair's sentence with its proposed anchor marked."""
    end = pair.anchor_start + len(pair.anchor)
    return (
        f"{pair.sentence[: pair.anchor_start]}{MARK_OPEN}{pair.anchor}{MARK_CLOSE}"
        f"{pair.sentence[end:]}"
    )


def write_label_file(
    path: Path, pairs: Sequence[ExportedPair], titles: Mapping[str, str | None], seed: int
) -> Path:
    """The blind label file at ``path``, rows shuffled, label columns empty. UTF-8 with a byte
    order mark, so spreadsheets read accented text as written. An existing file is never
    overwritten."""
    if path.exists():
        raise FileExistsError(f"{path} exists; nothing written")
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng((seed, 1)).permutation(len(pairs))
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - renamed into place once complete
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for index in order:
                pair = pairs[int(index)]
                writer.writerow(
                    (
                        pair.pair_id,
                        pair.source_url,
                        pair.target_url,
                        _cell(titles.get(pair.target_url) or ""),
                        _cell(marked_sentence(pair)),
                        _cell(pair.anchor),
                        *("" for _ in LABEL_COLUMNS),
                    )
                )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


async def import_labels(
    mongo: MongoRepo, tenant_id: str, path: Path, *, check: bool = False
) -> LabelImportReport:
    """Check a labelled file against the tenant's export it came from and, unless ``check``,
    import its labels into anchor_feedback. A file with any problem is refused whole, with every
    problem reported; an import is marked complete only once all its events are written, so a
    failed one is never read. Rows without a label are left unlabelled."""
    started = time.perf_counter()
    rows = await asyncio.to_thread(read_label_file, path)
    ids = [row["pair_id"] for _, row in rows if row["pair_id"]]
    exports = await mongo.label_export_ids(tenant_id, ids)
    if len(exports) != 1:
        problem = (
            f"no pair_id belongs to an export of tenant {tenant_id!r}"
            if not exports
            else f"the rows come from {len(exports)} exports: {', '.join(sorted(exports))}"
        )
        raise LabelFileError(path.name, [LabelProblem(problem=problem)])
    (export_id,) = exports
    found = await mongo.label_export(tenant_id, export_id)
    if found is None:
        raise ValueError(f"export {export_id!r} of {tenant_id!r} is not complete")
    export, pairs = found
    import_id = uuid.uuid4().hex
    events, problems = check_rows(rows, export, pairs, import_id, datetime.now(UTC))
    if problems:
        raise LabelFileError(path.name, problems)
    report = LabelImportReport(
        tenant_id=tenant_id,
        export_id=export_id,
        import_id=None if check else import_id,
        rows=len(rows),
        labelled=len(events),
        by_status=dict(Counter(event.status for event in events)),
    )
    if not check:
        await mongo.insert_label_events(tenant_id, import_id, events)
        await mongo.complete_label_import(
            tenant_id, import_id, export_id=export_id, events=len(events)
        )
    log.info(
        "labels.checked" if check else "labels.imported",
        stage=STAGE,
        tenant_id=tenant_id,
        export_id=export_id,
        import_id=report.import_id,
        rows=report.rows,
        labelled=report.labelled,
        seconds=round(time.perf_counter() - started, 3),
    )
    return report


def read_label_file(path: Path) -> list[tuple[int, dict[str, str]]]:
    """The file's rows with the line each ends on, every column stripped; rows with every column
    empty are skipped, and columns other than the file's own are ignored."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = [name.strip() for name in reader.fieldnames or ()]
            missing = [name for name in COLUMNS if name not in header]
            repeated = sorted({name for name in header if header.count(name) > 1} & set(COLUMNS))
            if missing or repeated:
                problem = (
                    f"missing columns {', '.join(missing)}; {_SAVE_AS}"
                    if missing
                    else f"repeated columns {', '.join(repeated)}"
                )
                raise LabelFileError(path.name, [LabelProblem(problem=problem)])
            reader.fieldnames = header
            rows: list[tuple[int, dict[str, str]]] = []
            for row in reader:
                cells = {name: (row.get(name) or "").strip() for name in COLUMNS}
                if any(cells.values()):
                    rows.append((reader.line_num, cells))
    except UnicodeDecodeError as error:
        raise LabelFileError(path.name, [LabelProblem(problem=f"not UTF-8; {_SAVE_AS}")]) from error
    if not rows:
        raise LabelFileError(path.name, [LabelProblem(problem="no rows")])
    return rows


def check_rows(
    rows: Sequence[tuple[int, Mapping[str, str]]],
    export: LabelExport,
    pairs: Sequence[ExportedPair],
    import_id: str,
    now: datetime,
) -> tuple[list[LabelEvent], list[LabelProblem]]:
    """The label events of the rows, and every problem that keeps the file from importing."""
    known = {pair.pair_id: pair for pair in pairs}
    seen: dict[str, int] = {}
    events: list[LabelEvent] = []
    problems: list[LabelProblem] = []
    for line, row in rows:
        pair_id = row["pair_id"]
        if pair_id in seen:
            found: str | None = f"pair_id repeats line {seen[pair_id]}"
        else:
            if pair_id:
                seen[pair_id] = line
            found = _row_problem(row, known, export.export_id)
        if found is not None:
            problems.append(LabelProblem(line=line, pair_id=pair_id or None, problem=found))
            continue
        word = row["label"].lower()
        if not word:
            continue
        pair, status = known[pair_id], LABEL_WORDS[word]
        events.append(
            LabelEvent(
                **pair.model_dump(),
                import_id=import_id,
                export_id=export.export_id,
                scorer=export.scorer,
                model_version=export.model_version,
                weights_version=export.weights_version,
                status=status,
                accepted=status is not RecommendationStatus.DISMISSED,
                grade=GRADES[status],
                anchor_used=(
                    pair.anchor
                    if status is RecommendationStatus.ACCEPTED
                    else row["anchor_used"] or None
                ),
                reason=row["reason"] or None,
                reviewer=row["reviewer"],
                created_at=now,
            )
        )
    missing = sorted(set(known) - set(seen))
    if missing:
        shown = ", ".join(missing[:MISSING_SHOWN])
        more = f" and {len(missing) - MISSING_SHOWN} more" if len(missing) > MISSING_SHOWN else ""
        problems.append(
            LabelProblem(
                problem=f"{len(missing)} exported pairs are missing from the file: {shown}{more}"
            )
        )
    if not problems and not events:
        problems.append(LabelProblem(problem="no row is labelled"))
    return events, problems


def _row_problem(
    row: Mapping[str, str], known: Mapping[str, ExportedPair], export_id: str
) -> str | None:
    pair_id = row["pair_id"]
    if not pair_id:
        return "no pair_id"
    pair = known.get(pair_id)
    if pair is None:
        return f"pair_id is not in export {export_id}"
    if row["source_url"] != pair.source_url or row["target_url"] != pair.target_url:
        return "source_url or target_url differs from the exported pair"
    word = row["label"].lower()
    anchor_used = row["anchor_used"]
    if not word:
        return (
            "anchor_used or reason given without a label" if anchor_used or row["reason"] else None
        )
    status = LABEL_WORDS.get(word)
    if status is None:
        return f"label {row['label']!r} is not accept, modify or dismiss"
    if status is RecommendationStatus.MODIFIED:
        if not anchor_used:
            return "modify needs the anchor you would use in anchor_used"
        if anchor_used == pair.anchor:
            return "anchor_used is the proposed anchor; label it accept"
        if len(anchor_used) > ANCHOR_MAX_CHARS:
            return f"anchor_used is longer than {ANCHOR_MAX_CHARS} characters"
    elif anchor_used:
        return f"anchor_used goes with modify only, not {word}"
    if not row["reviewer"]:
        return "a labelled row needs the reviewer"
    return None


def label_grades(labels: Iterable[LabelEvent]) -> dict[Pair, int]:
    """The ranker's relevance grade of each labelled pair; unlabelled pairs are not listed."""
    return {(label.source_url, label.target_url): label.grade for label in labels}


def _describe(problem: LabelProblem) -> str:
    where = f"line {problem.line}" if problem.line is not None else "file"
    pair = f" ({problem.pair_id})" if problem.pair_id else ""
    return f"  {where}{pair}: {problem.problem}"
