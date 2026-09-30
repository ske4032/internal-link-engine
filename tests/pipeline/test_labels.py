"""Hand labels (#29, #82): the blind export a reviewer labels, and the import of the labelled file
into anchor_feedback that the ranker reads its grades from."""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas
import pytest
from test_anchor_placements import choice, write_choices
from test_recommendations import write_ranked

from linking_engine.errors import DatabaseWriteError
from linking_engine.models import (
    ExportedPair,
    LabelEvent,
    LabelExport,
    LabelProblem,
    LabelSettings,
    RecommendationStatus,
    ScorerName,
)
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE
from linking_engine.pipeline.labels import (
    COLUMNS,
    LabelFileError,
    candidate_pool,
    check_rows,
    export_labels,
    import_labels,
    label_grades,
    marked_sentence,
    read_label_file,
    sample_pairs,
    write_label_file,
)
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE
from linking_engine.pipeline.recommendations import ANCHOR_MAX_CHARS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from linking_engine.ingest.mongo_repo import MongoRepo

AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
ACCEPTED, MODIFIED, DISMISSED = (
    RecommendationStatus.ACCEPTED,
    RecommendationStatus.MODIFIED,
    RecommendationStatus.DISMISSED,
)
SENTENCE = "Our dome tent guide covers wet nights."
PHRASE = "dome tent"
SETTINGS = LabelSettings(pages=2, pairs_per_page=3, seed=7)

Row = dict[str, str]


def url(path: str) -> str:
    return f"https://example.com/{path}"


# ── the pool and the sample ─────────────────────────────────────────────────


def ranked(*rows: tuple[str, str, float]) -> pandas.DataFrame:
    frame = pandas.DataFrame(rows, columns=["source_url", "target_url", "score"])
    order = frame.groupby("source_url")["score"].rank(ascending=False, method="first")
    return frame.assign(rank_in_source=order.astype("int32"))


def chosen(source: str, target: str, *, rank: int = 1, phrase: str = PHRASE) -> dict[str, Any]:
    """A choice row whose sentence starts at character 100 of the page."""
    return {
        "source_url": source,
        "target_url": target,
        "rank": rank,
        "anchor_type": "EXACT",
        "keyword": PHRASE,
        "rung": "EXACT",
        "phrase": phrase,
        "start": 104,
        "sentence": SENTENCE,
        "sentence_start": 100,
    }


def target(page: str, position: int) -> str:
    return url(f"{page}-t{position:02d}")


def pool(sizes: Mapping[str, int]) -> pandas.DataFrame:
    """Source page -> its number of candidates; each page's targets best score first."""
    keys = [
        (url(page), target(page, n), 1.0 - n / 100)
        for page, size in sizes.items()
        for n in range(size)
    ]
    choices = pandas.DataFrame([chosen(source, dest) for source, dest, _ in keys])
    return candidate_pool(ranked(*keys), choices, frozenset())


def drawn(pairs: tuple[ExportedPair, ...]) -> list[tuple[str, str, int]]:
    return [(pair.source_url, pair.target_url, pair.band) for pair in pairs]


def test_the_pool_keeps_rank_one_choices_between_included_pages_with_servable_anchors() -> None:
    a, x = url("a"), url("excluded")
    t = [url(f"t{n}") for n in range(6)]
    at_limit, too_long = "t" * ANCHOR_MAX_CHARS, "t" * (ANCHOR_MAX_CHARS + 1)
    frame = candidate_pool(
        ranked(
            (a, t[0], 0.9),
            (a, t[1], 0.8),
            (a, t[2], 0.7),
            (a, t[3], 0.6),
            (a, t[5], 0.5),
            (x, t[0], 0.4),
            (a, x, 0.3),
        ),
        pandas.DataFrame(
            [
                chosen(a, t[0]),
                chosen(a, t[0], rank=2, phrase="tent"),
                chosen(a, t[2], phrase=at_limit),
                chosen(a, t[3], phrase=too_long),
                chosen(a, t[5], rank=2),
                chosen(x, t[0]),
                chosen(a, x),
                chosen(a, t[4]),
            ]
        ),
        frozenset({x}),
    )

    # t1 has no choice, t3's anchor is too long to serve, t5 has an alternative only, t4 was
    # never ranked; the excluded page is neither a source nor a target.
    assert sorted(zip(frame["source_url"], frame["target_url"], frame["phrase"], strict=True)) == [
        (a, t[0], PHRASE),
        (a, t[2], at_limit),
    ]


def test_the_same_pool_and_seed_draw_the_same_pairs() -> None:
    frame = pool({f"p{n}": 12 for n in range(6)})
    settings = LabelSettings(pages=3, pairs_per_page=4, seed=11)
    first = drawn(sample_pairs(frame, settings).pairs)

    assert drawn(sample_pairs(frame, settings).pairs) == first
    assert drawn(sample_pairs(frame, settings.model_copy(update={"seed": 12})).pairs) != first


def test_each_drawn_page_gives_one_pair_per_score_band_best_band_first() -> None:
    sample = sample_pairs(
        pool({"a": 20, "b": 20, "c": 20}), LabelSettings(pages=2, pairs_per_page=10, seed=3)
    )
    position = {target(page, n): n for page in "abc" for n in range(20)}

    assert (sample.pages, len(sample.pairs)) == (2, 20)
    per_page = Counter(pair.source_url for pair in sample.pairs)
    assert set(per_page.values()) == {10}
    for source in per_page:
        bands = [pair.band for pair in sample.pairs if pair.source_url == source]
        assert sorted(bands) == list(range(1, 11)), "one pair of each band per page"
    # Twenty candidates in ten bands: band 1 holds the best two scores, band 10 the worst two.
    assert all(position[pair.target_url] // 2 == pair.band - 1 for pair in sample.pairs)


def test_pages_with_too_few_candidates_are_counted_and_never_drawn() -> None:
    frame = pool({"a": 10, "b": 10, "e": 12, "c": 9, "d": 3})
    for seed in range(10):
        sample = sample_pairs(frame, LabelSettings(pages=2, pairs_per_page=10, seed=seed))

        assert {pair.source_url for pair in sample.pairs} <= {url("a"), url("b"), url("e")}
        counts = (sample.pages, sample.eligible_pages, sample.eligible_pairs, sample.small_pages)
        assert counts == (2, 3, 32, 2)


def test_fewer_eligible_pages_than_requested_draws_all_of_them() -> None:
    sample = sample_pairs(
        pool({"a": 4, "b": 5, "c": 1}), LabelSettings(pages=20, pairs_per_page=4, seed=0)
    )

    assert (sample.pages, len(sample.pairs)) == (2, 8)
    assert {pair.source_url for pair in sample.pairs} == {url("a"), url("b")}


def test_no_eligible_page_is_an_error() -> None:
    with pytest.raises(ValueError, match="no source page has 10 anchored candidate pairs"):
        sample_pairs(pool({"a": 9}), LabelSettings(pages=1, pairs_per_page=10, seed=0))


async def test_an_export_needs_ranked_pairs_newer_than_the_anchor_choices(tmp_path: Path) -> None:
    repo = cast("Any", None)
    tenant = "test-labels"
    kwargs: dict[str, Any] = {"cache_dir": tmp_path, "out_dir": tmp_path, "settings": SETTINGS}
    with pytest.raises(ValueError, match="run rank-pairs first"):
        await export_labels(repo, tenant, **kwargs)

    seed_cache(tmp_path, tenant)
    ranked_at = (tmp_path / tenant / RANKED_PAIRS_FILE).stat().st_mtime
    os.utime(tmp_path / tenant / ANCHOR_CHOICES_FILE, (ranked_at + 5, ranked_at + 5))
    with pytest.raises(ValueError, match="changed since rank-pairs"):
        await export_labels(repo, tenant, **kwargs)


# ── the label file ──────────────────────────────────────────────────────────


def exported(pair_id: str, score: float = 0.5) -> ExportedPair:
    return ExportedPair(
        pair_id=pair_id,
        source_url=url("a"),
        target_url=url(f"t-{pair_id}"),
        anchor=PHRASE,
        anchor_type="EXACT",
        rung="EXACT",
        keyword=PHRASE,
        sentence=SENTENCE,
        anchor_start=4,
        score=score,
        rank_in_source=1,
        band=1,
    )


def read_csv(path: Path) -> tuple[list[str], list[Row]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def test_the_file_is_blind_to_every_model_output_and_shuffled(tmp_path: Path) -> None:
    pairs = [exported(f"p{n}", score=1.0 - n / 10) for n in range(10)]
    titles = {pairs[0].target_url: "Zelte für Regen"}
    path = write_label_file(tmp_path / "labels.csv", pairs, titles, 7)
    header, rows = read_csv(path)

    assert header == [
        "pair_id",
        "source_url",
        "target_url",
        "target_title",
        "sentence",
        "proposed_anchor",
        "label",
        "anchor_used",
        "reason",
        "reviewer",
    ]
    assert list(COLUMNS) == header
    expected = {
        (p.pair_id, p.source_url, p.target_url, titles.get(p.target_url, ""), marked_sentence(p))
        for p in pairs
    }
    assert {tuple(row.values())[:5] for row in rows} == expected
    assert all(row["proposed_anchor"] == PHRASE for row in rows)
    assert all(row[name] == "" for row in rows for name in COLUMNS[6:])
    # The pairs came best score first; the file keeps no trace of that order.
    ids = [row["pair_id"] for row in rows]
    assert ids not in ([p.pair_id for p in pairs], [p.pair_id for p in reversed(pairs)])
    again = write_label_file(tmp_path / "again.csv", pairs, titles, 7)
    assert [row["pair_id"] for row in read_csv(again)[1]] == ids, "the seed fixes the order"


def test_the_anchor_is_marked_at_its_offset_when_the_phrase_repeats(tmp_path: Path) -> None:
    sentence = "A tent guide: pick the tent that stays dry."
    pair = exported("p1").model_copy(
        update={"anchor": "tent", "sentence": sentence, "anchor_start": sentence.rindex("tent")}
    )
    marked = "A tent guide: pick the [[tent]] that stays dry."

    assert marked_sentence(pair) == marked
    path = write_label_file(tmp_path / "labels.csv", [pair, exported("p2")], {}, 7)
    assert {row["pair_id"]: row["sentence"] for row in read_csv(path)[1]}["p1"] == marked


def test_the_file_starts_with_a_byte_order_mark_and_is_never_overwritten(tmp_path: Path) -> None:
    path = write_label_file(
        tmp_path / "out" / "labels.csv", [exported("p1"), exported("p2")], {}, 7
    )
    written = path.read_bytes()

    assert written.startswith(b"\xef\xbb\xbfpair_id,")
    with pytest.raises(FileExistsError):
        write_label_file(path, [exported("p3"), exported("p4")], {}, 7)
    assert path.read_bytes() == written
    assert [entry.name for entry in path.parent.iterdir()] == ["labels.csv"]


def test_page_text_a_spreadsheet_would_run_as_a_formula_is_written_as_text(
    tmp_path: Path,
) -> None:
    sentence = "=HYPERLINK(1) dome tent"
    pair = exported("p1").model_copy(
        update={"sentence": sentence, "anchor_start": sentence.index(PHRASE)}
    )
    titles = {pair.target_url: "@SUM(1)"}
    path = write_label_file(tmp_path / "labels.csv", [pair, exported("p2")], titles, 7)
    written = {row["pair_id"]: row for row in read_csv(path)[1]}["p1"]

    assert written["target_title"] == "'@SUM(1)"
    assert written["sentence"] == "'=HYPERLINK(1) [[dome tent]]"
    assert written["source_url"] == pair.source_url


# ── checking a labelled file ────────────────────────────────────────────────

PAIRS = (exported("p1", 0.9), exported("p2", 0.8), exported("p3", 0.7), exported("p4", 0.6))
EXPORT = LabelExport(
    export_id="e1",
    created_at=AT,
    seed=7,
    pages=2,
    pairs_per_page=2,
    pairs=4,
    scorer=ScorerName.BASELINE,
    weights_version="w1",
    eligible_pages=2,
    eligible_pairs=4,
    small_pages=0,
    file_name="labels.csv",
)
P1, P2, P3, P4 = PAIRS


def row(pair: ExportedPair, label: str = "", **cells: str) -> Row:
    return {
        "pair_id": pair.pair_id,
        "source_url": pair.source_url,
        "target_url": pair.target_url,
        "target_title": "",
        "sentence": marked_sentence(pair),
        "proposed_anchor": pair.anchor,
        "label": label,
        "anchor_used": "",
        "reason": "",
        "reviewer": "qa" if label else "",
        **cells,
    }


def base() -> list[Row]:
    return [row(P1, "accept"), row(P2), row(P3), row(P4)]


def check(rows: list[Row]) -> tuple[list[LabelEvent], list[LabelProblem]]:
    return check_rows(list(enumerate(rows, start=2)), EXPORT, PAIRS, "i1", AT)


def changed(index: int, **cells: str) -> Callable[[list[Row]], list[Row]]:
    def change(rows: list[Row]) -> list[Row]:
        rows[index] = {**rows[index], **cells}
        return rows

    return change


def test_a_valid_file_checks_clean() -> None:
    events, problems = check(base())

    assert problems == []
    assert [(event.pair_id, event.status, event.grade) for event in events] == [("p1", ACCEPTED, 3)]


@pytest.mark.parametrize(
    ("change", "line", "pair_id", "problem"),
    [
        (changed(1, pair_id=""), 3, None, "no pair_id"),
        (lambda rows: [*rows, row(P1, "dismiss")], 6, "p1", "pair_id repeats line 2"),
        (changed(1, pair_id="p9"), 3, "p9", "pair_id is not in export e1"),
        (changed(1, source_url=url("b")), 3, "p2", "source_url or target_url differs"),
        (changed(1, target_url=url("b")), 3, "p2", "source_url or target_url differs"),
        (changed(1, label="maybe", reviewer="qa"), 3, "p2", "'maybe' is not accept, modify or"),
        (changed(1, label="modify", reviewer="qa"), 3, "p2", "modify needs the anchor"),
        (
            changed(1, label="modify", anchor_used=PHRASE, reviewer="qa"),
            3,
            "p2",
            "anchor_used is the proposed anchor; label it accept",
        ),
        (
            changed(1, label="modify", anchor_used="t" * (ANCHOR_MAX_CHARS + 1), reviewer="qa"),
            3,
            "p2",
            f"anchor_used is longer than {ANCHOR_MAX_CHARS} characters",
        ),
        (changed(0, anchor_used="tent"), 2, "p1", "anchor_used goes with modify only, not accept"),
        (
            changed(1, label="dismiss", anchor_used="tent", reviewer="qa"),
            3,
            "p2",
            "anchor_used goes with modify only, not dismiss",
        ),
        (changed(1, anchor_used="tent"), 3, "p2", "anchor_used or reason given without a label"),
        (changed(1, reason="off topic"), 3, "p2", "anchor_used or reason given without a label"),
        (changed(0, reviewer=""), 2, "p1", "a labelled row needs the reviewer"),
        (lambda rows: rows[:3], None, None, "1 exported pairs are missing from the file: p4"),
        (changed(0, label="", reviewer=""), None, None, "no row is labelled"),
    ],
)
def test_each_problem_of_a_row_is_found(
    change: Callable[[list[Row]], list[Row]], line: int | None, pair_id: str | None, problem: str
) -> None:
    _, problems = check(change(base()))

    assert problems, "a broken file must not check clean"
    assert any(
        (found.line, found.pair_id) == (line, pair_id) and problem in found.problem
        for found in problems
    ), problems


def test_every_problem_of_a_file_is_reported_together() -> None:
    _, problems = check([row(P1, "maybe"), row(P2, "modify"), row(P3, "accept", reviewer="")])
    error = LabelFileError("labels.csv", problems)

    assert [(problem.line, problem.pair_id) for problem in problems] == [
        (2, "p1"),
        (3, "p2"),
        (4, "p3"),
        (None, None),
    ]
    assert str(error).splitlines()[0] == "labels.csv: 4 problems; nothing imported"
    assert str(error).splitlines()[4] == "  file: 1 exported pairs are missing from the file: p4"


HEADER = ",".join(COLUMNS)


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (
            ",".join(COLUMNS[:-1]) + "\r\np1,a,t,,s,x,accept,,,\r\n",
            "missing columns reviewer; save it as comma-separated CSV, UTF-8",
        ),
        (
            ";".join(COLUMNS) + "\r\n",
            f"missing columns {', '.join(COLUMNS)}; save it as comma-separated CSV, UTF-8",
        ),
        (HEADER + ",label\r\np1,a,t,,s,x,accept,,,qa,\r\n", "repeated columns label"),
        (HEADER + "\r\n", "no rows"),
        (HEADER + "\r\n,,,,,,,,,\r\n , , , , , , , , , \r\n", "no rows"),
    ],
    ids=["missing-column", "semicolons", "repeated-column", "header-only", "blank-rows-only"],
)
def test_a_file_that_cannot_be_read_is_refused_whole(
    tmp_path: Path, content: str, problem: str
) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(LabelFileError) as caught:
        read_label_file(path)
    assert [(found.line, found.problem) for found in caught.value.problems] == [(None, problem)]


def test_a_file_saved_in_another_encoding_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_bytes(f"{HEADER}\r\np1,a,t,Zelte für Regen,s,x,,,,\r\n".encode("cp1252"))

    with pytest.raises(LabelFileError) as caught:
        read_label_file(path)
    assert [problem.problem for problem in caught.value.problems] == [
        "not UTF-8; save it as comma-separated CSV, UTF-8"
    ]


def test_blank_rows_and_extra_columns_are_ignored_and_label_words_are_case_insensitive(
    tmp_path: Path,
) -> None:
    longest = "t" * ANCHOR_MAX_CHARS
    rows = [
        row(P1, " ACCEPT "),
        dict.fromkeys(COLUMNS, " "),
        row(P2, "Modify", anchor_used=longest),
        row(P3, "dismiss", reason="off topic"),
        row(P4),
    ]
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*COLUMNS, "notes"])
        writer.writerows([*cells.values(), "a note"] for cells in rows)

    found = read_label_file(path)
    events, problems = check_rows(found, EXPORT, PAIRS, "i1", AT)

    assert [line for line, _ in found] == [2, 4, 5, 6]
    assert problems == []
    assert [(e.pair_id, e.status, e.anchor_used, e.reason) for e in events] == [
        ("p1", ACCEPTED, PHRASE, None),
        ("p2", MODIFIED, longest, None),
        ("p3", DISMISSED, None, "off topic"),
    ]


def test_a_problem_is_reported_on_the_line_its_row_is_on_after_an_empty_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.csv"
    cells = row(P1, "maybe")
    path.write_text(f"{HEADER}\n\n{','.join(cells.values())}\n", encoding="utf-8")

    found = read_label_file(path)
    _, problems = check_rows(found, EXPORT, PAIRS[:1], "i1", AT)

    assert [(problem.line, problem.pair_id) for problem in problems] == [(3, "p1")]


# ── export and import against MongoDB ───────────────────────────────────────


def seed_cache(cache_dir: Path, tenant: str, pages: int = 2, per_page: int = 3) -> None:
    """The tenant's anchor choices, then the ranked pairs written after them."""
    keys = [(url(f"s{p}"), url(f"s{p}-t{n}"), n) for p in range(pages) for n in range(per_page)]
    write_choices(cache_dir, tenant, [choice(source, dest) for source, dest, _ in keys])
    rows: list[dict[str, object]] = [
        {
            "source_url": source,
            "target_url": dest,
            "score": 1.0 - n / 10,
            "rank_in_source": n + 1,
            "scorer": "baseline",
            "model_version": None,
        }
        for source, dest, n in keys
    ]
    write_ranked(cache_dir / tenant / RANKED_PAIRS_FILE, rows, tenant)


async def export_file(mongo: MongoRepo, tenant: str, tmp_path: Path) -> tuple[LabelExport, Path]:
    cache = tmp_path / "cache"
    if not (cache / tenant).exists():
        seed_cache(cache, tenant)
    return await export_labels(
        mongo, tenant, cache_dir=cache, out_dir=tmp_path / "out", settings=SETTINGS
    )


def fill(path: Path, labels: Mapping[int, Mapping[str, str]], out: Path | None = None) -> list[Row]:
    """The reviewer's edit: the file's rows at ``labels``' positions filled in, saved to ``out``
    (the file itself by default)."""
    header, rows = read_csv(path)
    for index, cells in labels.items():
        rows[index] = {**rows[index], "reviewer": "qa", **cells}
    with (out or path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return rows


async def stored(mongo: MongoRepo, tenant: str) -> tuple[int, int]:
    """The tenant's label events and import markers."""
    query = {"tenantId": tenant}
    return (
        await mongo._db["anchor_feedback"].count_documents(query),
        await mongo._db["label_imports"].count_documents(query),
    )


def key(cells: Row) -> tuple[str, str]:
    return cells["source_url"], cells["target_url"]


@pytest.mark.integration
async def test_a_labelled_export_imports_into_the_rankers_grades(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    export, path = await export_file(mongo, tenant, tmp_path)
    rows = fill(
        path,
        {
            0: {"label": "accept"},
            1: {"label": "modify", "anchor_used": "trail shoe guide"},
            2: {"label": "dismiss", "reason": "off topic"},
        },
    )

    report = await import_labels(mongo, tenant, path)
    labels = await mongo.hand_labels(tenant)

    counts = (export.pairs, export.pages, export.eligible_pages, export.small_pages)
    assert (counts, export.scorer, export.model_version) == (
        (6, 2, 2, 0),
        ScorerName.BASELINE,
        None,
    )
    assert (report.export_id, report.rows, report.labelled) == (export.export_id, 6, 3)
    assert report.by_status == {ACCEPTED: 1, MODIFIED: 1, DISMISSED: 1}
    # Unlabelled rows get no grade: the ranker reads them as 0.
    assert label_grades(labels) == {key(rows[0]): 3, key(rows[1]): 2, key(rows[2]): 1}
    by_pair = {label.pair_id: label for label in labels}
    accepted, modified = by_pair[rows[0]["pair_id"]], by_pair[rows[1]["pair_id"]]
    assert (accepted.anchor_used, modified.anchor_used) == ("trail shoes", "trail shoe guide")
    assert {(label.import_id, label.export_id) for label in labels} == {
        (report.import_id, export.export_id)
    }


@pytest.mark.integration
async def test_a_checked_file_writes_nothing(mongo: MongoRepo, tenant: str, tmp_path: Path) -> None:
    _, path = await export_file(mongo, tenant, tmp_path)
    fill(path, {0: {"label": "accept"}})

    report = await import_labels(mongo, tenant, path, check=True)

    assert (report.import_id, report.labelled) == (None, 1)
    assert await stored(mongo, tenant) == (0, 0)
    assert await mongo.hand_labels(tenant) == ()


@pytest.mark.integration
async def test_a_file_with_any_problem_writes_nothing(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    _, path = await export_file(mongo, tenant, tmp_path)
    rows = fill(
        path,
        {0: {"label": "accept"}, 1: {"label": "maybe"}, 2: {"label": "dismiss", "reviewer": ""}},
    )

    with pytest.raises(LabelFileError) as caught:
        await import_labels(mongo, tenant, path)

    assert [(problem.line, problem.pair_id) for problem in caught.value.problems] == [
        (3, rows[1]["pair_id"]),
        (4, rows[2]["pair_id"]),
    ]
    assert await stored(mongo, tenant) == (0, 0)


@pytest.mark.integration
async def test_a_relabelled_pair_reads_its_latest_label_and_keeps_the_earlier_one(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    _, path = await export_file(mongo, tenant, tmp_path)
    rows = fill(path, {0: {"label": "accept"}, 1: {"label": "accept"}})
    first = await import_labels(mongo, tenant, path)
    fill(path, {0: {"label": "dismiss"}, 1: {"label": "", "reviewer": ""}})
    second = await import_labels(mongo, tenant, path)

    labels = await mongo.hand_labels(tenant)

    assert {(label.pair_id, label.status, label.import_id) for label in labels} == {
        (rows[0]["pair_id"], DISMISSED, second.import_id),
        (rows[1]["pair_id"], ACCEPTED, first.import_id),
    }
    history = {"tenantId": tenant, "pairId": rows[0]["pair_id"]}
    assert await mongo._db["anchor_feedback"].count_documents(history) == 2


@pytest.mark.integration
async def test_the_events_of_an_import_never_marked_complete_are_not_read(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    export, path = await export_file(mongo, tenant, tmp_path)
    rows = fill(path, {0: {"label": "accept"}})
    done = await import_labels(mongo, tenant, path)
    found = await mongo.label_export(tenant, export.export_id)
    assert found is not None
    pair = next(pair for pair in found[1] if pair.pair_id == rows[0]["pair_id"])
    stray = LabelEvent(
        **pair.model_dump(),
        import_id="unfinished",
        export_id=export.export_id,
        scorer=export.scorer,
        weights_version=export.weights_version,
        status=DISMISSED,
        accepted=False,
        grade=1,
        reviewer="qa",
        created_at=AT,
    )

    assert await mongo.insert_label_events(tenant, "unfinished", [stray]) == 1
    with pytest.raises(DatabaseWriteError, match="has 1 of 2 label events"):
        await mongo.complete_label_import(
            tenant, "unfinished", export_id=export.export_id, events=2
        )
    labels = await mongo.hand_labels(tenant)
    assert [(label.pair_id, label.status, label.import_id) for label in labels] == [
        (pair.pair_id, ACCEPTED, done.import_id)
    ]
    assert await stored(mongo, tenant) == (2, 1)


@pytest.mark.integration
async def test_an_export_missing_pairs_is_never_marked_complete(
    mongo: MongoRepo, tenant: str
) -> None:
    short = EXPORT.model_copy(update={"export_id": "e-short"})

    assert await mongo.insert_label_pairs(tenant, "e-short", PAIRS[:2]) == 2
    with pytest.raises(DatabaseWriteError, match="has 2 of 4 pairs"):
        await mongo.complete_label_export(tenant, short)
    assert await mongo.label_export(tenant, "e-short") is None
    assert await mongo.label_export_ids(tenant, ["p1", "p2"]) == frozenset()


@pytest.mark.integration
async def test_label_writes_refuse_blank_ids_and_repeated_pairs(
    mongo: MongoRepo, tenant: str
) -> None:
    event = LabelEvent(
        **P1.model_dump(),
        import_id="i1",
        export_id="e1",
        scorer=ScorerName.BASELINE,
        weights_version="w1",
        status=DISMISSED,
        accepted=False,
        grade=1,
        reviewer="qa",
        created_at=AT,
    )

    with pytest.raises(ValueError, match="export_id must be"):
        await mongo.insert_label_pairs(tenant, " ", PAIRS)
    with pytest.raises(ValueError, match="duplicate pair_id"):
        await mongo.insert_label_pairs(tenant, "e1", [P1, P1])
    with pytest.raises(ValueError, match="import_id must be"):
        await mongo.insert_label_events(tenant, " ", [event])
    with pytest.raises(ValueError, match="every event must belong to import 'i2'"):
        await mongo.insert_label_events(tenant, "i2", [event])
    with pytest.raises(ValueError, match="duplicate pair_id"):
        await mongo.insert_label_events(tenant, "i1", [event, event])
    with pytest.raises(ValueError, match="import_id and export_id must be"):
        await mongo.complete_label_import(tenant, "i1", export_id=" ", events=0)
    assert await stored(mongo, tenant) == (0, 0)
    assert await mongo._db["label_pairs"].count_documents({"tenantId": tenant}) == 0


@pytest.mark.integration
async def test_a_tenants_file_is_refused_for_another_tenant(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-b"
    _, path = await export_file(mongo, tenant, tmp_path)
    fill(path, {0: {"label": "accept"}})
    await import_labels(mongo, tenant, path)
    before = await mongo.hand_labels(tenant)

    with pytest.raises(LabelFileError, match="no pair_id belongs to an export of tenant"):
        await import_labels(mongo, other, path)

    assert await stored(mongo, other) == (0, 0)
    assert await mongo.hand_labels(other) == ()
    assert await mongo.hand_labels(tenant) == before
    assert len(before) == 1


@pytest.mark.integration
async def test_a_file_mixing_the_rows_of_two_exports_is_refused(
    mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    _, path = await export_file(mongo, tenant, tmp_path)
    _, other_path = await export_file(mongo, tenant, tmp_path)
    header, rows = read_csv(path)
    mixed = tmp_path / "mixed.csv"
    with mixed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows([*rows, read_csv(other_path)[1][0]])
    fill(mixed, {0: {"label": "accept"}})

    with pytest.raises(LabelFileError, match="the rows come from 2 exports"):
        await import_labels(mongo, tenant, mixed)
    assert await stored(mongo, tenant) == (0, 0)
