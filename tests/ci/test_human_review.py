from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "human_review.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("human_review", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


hr = _load()

TICKED = f"## Human review\n- [x] {hr.ATTESTATION}\n"
UNTICKED = f"## Human review\n- [ ] {hr.ATTESTATION}\n"
PROVENANCE = """## AI provenance
- Agent: claude-code
- Model: claude-opus-5-5
- Swarm:
- Run IDs: run-1, run-2
- Task IDs: n/a
"""
HUMAN = {"login": "owner", "type": "User"}
REVIEWERS = {"owner"}


def pr(
    body: str = PROVENANCE + TICKED, labels: tuple[str, ...] = (hr.LABEL,), **extra: Any
) -> dict[str, Any]:
    return {
        "number": 7,
        "body": body,
        "labels": [{"name": name} for name in labels],
        "head": {"sha": "abc1234def", "ref": "feat/x"},
        "user": HUMAN,
        **extra,
    }


def labelled(actor: dict[str, str], label: str = hr.LABEL) -> dict[str, Any]:
    return {"event": "labeled", "label": {"name": label}, "actor": actor}


def commit(
    message: str = "Issue #1: change", author: dict[str, str] | None = HUMAN
) -> dict[str, Any]:
    return {
        "sha": "0123456789",
        "commit": {"message": message},
        "author": author,
        "committer": HUMAN,
    }


# ── review ──────────────────────────────────────────────────────────────────


def test_a_ticked_box_and_a_reviewer_label_approve() -> None:
    review = hr.evaluate_review(pr(), [labelled(HUMAN)], REVIEWERS)
    assert (review.approved, review.reviewer, review.missing) == (True, "owner", ())


@pytest.mark.parametrize(
    ("body", "labels", "events", "reason"),
    [
        (UNTICKED, (hr.LABEL,), [labelled(HUMAN)], "box"),
        (TICKED, (), [], "label is missing"),
        (TICKED, (hr.LABEL,), [labelled({"login": "someone", "type": "User"})], "not an allowed"),
        (TICKED, (hr.LABEL,), [], "no event records"),
    ],
    ids=["unticked", "no-label", "not-a-reviewer", "no-label-event"],
)
def test_missing_steps_keep_the_gate_pending(
    body: str, labels: tuple[str, ...], events: list[dict[str, Any]], reason: str
) -> None:
    review = hr.evaluate_review(pr(body, labels), events, REVIEWERS)
    assert not review.approved
    assert review.rejected_actor is None
    assert any(reason in item for item in review.missing)


@pytest.mark.parametrize(
    "actor",
    [
        {"login": "github-actions[bot]", "type": "Bot"},
        {"login": "Copilot", "type": "Bot"},
        {"login": "copilot-swe-agent[bot]", "type": "Bot"},
        {"login": "claude-helper", "type": "User"},
    ],
    ids=["actions-bot", "copilot", "copilot-agent", "ai-named-user"],
)
def test_a_label_from_a_bot_or_ai_account_is_rejected(actor: dict[str, str]) -> None:
    review = hr.evaluate_review(pr(), [labelled(actor)], REVIEWERS | {actor["login"].lower()})
    assert not review.approved
    assert review.rejected_actor == actor["login"]


def test_the_latest_label_event_decides() -> None:
    events = [labelled(HUMAN), labelled({"login": "Copilot", "type": "Bot"})]
    assert hr.evaluate_review(pr(), events, REVIEWERS).rejected_actor == "Copilot"
    assert hr.evaluate_review(pr(), list(reversed(events)), REVIEWERS).approved


def test_reviewer_logins_match_case_insensitively() -> None:
    review = hr.evaluate_review(pr(), [labelled({"login": "Owner", "type": "User"})], REVIEWERS)
    assert review.approved


def test_untick_clears_only_the_attestation_box() -> None:
    body = f"- [x] other item\n- [X] {hr.ATTESTATION}\n"
    assert hr.untick(body) == f"- [x] other item\n- [ ] {hr.ATTESTATION}\n"


def test_the_box_must_be_a_list_item_not_quoted_text() -> None:
    assert hr.evaluate_review(pr(f"> [x] {hr.ATTESTATION}"), [labelled(HUMAN)], REVIEWERS).missing


# ── provenance ──────────────────────────────────────────────────────────────


def test_a_filled_provenance_section_is_declared() -> None:
    found = hr.detect_provenance(pr(), [commit()])
    assert (found.ai_detected, found.classification, found.sources) == (
        True,
        "ai-declared",
        ("pr-description",),
    )
    assert (found.agents, found.models, found.run_ids, found.task_ids) == (
        ("claude-code",),
        ("claude-opus-5-5",),
        ("run-1", "run-2"),
        (),
    )
    assert not found.swarm_detected
    assert not found.copilot_detected


def test_nothing_declared_or_detected_is_undeclared_but_still_reviewed() -> None:
    found = hr.detect_provenance(pr(body=TICKED), [commit()])
    assert (found.ai_detected, found.classification, found.sources) == (False, "undeclared", ())


def test_ai_trailers_in_commits_are_declarations() -> None:
    found = hr.detect_provenance(
        pr(body=TICKED), [commit("Fix\n\nCo-authored-by: Claude <noreply@anthropic.com>")]
    )
    assert (found.classification, found.sources) == ("ai-declared", ("commit-trailer:0123456",))


def test_copilot_commits_and_ai_branches_are_detected() -> None:
    copilot = {"login": "copilot-swe-agent[bot]", "type": "Bot"}
    found = hr.detect_provenance(
        pr(body=TICKED, head={"sha": "abc", "ref": "copilot/fix-typo"}), [commit(author=copilot)]
    )
    assert found.classification == "ai-detected"
    assert found.copilot_detected
    assert found.bot_commits == 1
    assert set(found.sources) == {"commit-author:copilot-swe-agent[bot]", "branch:copilot/fix-typo"}


def test_a_swarm_is_several_declared_roles_or_several_ai_tools() -> None:
    roles = PROVENANCE.replace("- Swarm:", "- Swarm: python-engineer, qa-engineer")
    assert hr.detect_provenance(pr(body=roles), [commit()]).swarm_detected
    mixed = [commit("x\n\nCo-authored-by: Copilot <copilot@github.com>")]
    assert hr.detect_provenance(pr(), mixed).swarm_detected


def test_one_tool_named_two_ways_is_not_a_swarm() -> None:
    same = [commit("x\n\nCo-authored-by: Claude <noreply@anthropic.com>")]
    assert not hr.detect_provenance(pr(), same).swarm_detected


def test_fields_outside_the_provenance_section_are_ignored() -> None:
    body = "## Notes\n- Agent: copilot\n" + TICKED
    assert hr.declared_fields(body) == {}


def test_untrusted_values_cannot_inject_markdown_html_or_mentions() -> None:
    body = "## AI provenance\n- Run IDs: <!-- x -->@owner`[a](b)`\n"
    assert hr.declared_fields(body)["run ids"] == ("--x--owner[a]b",)


# ── the run: api calls and outputs ──────────────────────────────────────────


class FakeApi:
    def __init__(
        self, pull: dict[str, Any], events: list[dict[str, Any]], comments: list[dict[str, Any]]
    ) -> None:
        self.pull = pull
        self.events = events
        self.comments = comments
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def get(self, path: str) -> Any:
        assert path == "/pulls/7"
        return json.loads(json.dumps(self.pull))

    def list(self, path: str) -> list[dict[str, Any]]:
        return {
            "/issues/7/events": self.events,
            "/pulls/7/commits": [commit()],
            "/issues/7/comments": self.comments,
        }[path]

    def send(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, path, body))
        if method == "DELETE" and path.endswith(f"/labels/{hr.LABEL}"):
            self.pull["labels"] = []
        if method == "PATCH" and path == "/pulls/7":
            self.pull["body"] = body["body"] if body else ""
        return None

    def status(self) -> dict[str, Any]:
        [(_, path, body)] = [c for c in self.calls if c[1].startswith("/statuses/")]
        assert path == "/statuses/abc1234def"
        assert body is not None
        return body


def run(api: FakeApi, action: str, tmp_path: Path) -> Any:
    event = {"action": action, "pull_request": {"number": 7}}
    return hr.run(api, event, REVIEWERS, "https://run", tmp_path / "ai-provenance.json")


def test_an_approved_review_posts_success_on_the_head_commit(tmp_path: Path) -> None:
    api = FakeApi(pr(), [labelled(HUMAN)], [])

    run(api, "labeled", tmp_path)

    status = api.status()
    assert (status["state"], status["context"]) == ("success", hr.STATUS_CONTEXT)
    report = json.loads((tmp_path / "ai-provenance.json").read_text())
    assert report["review"]["reviewer"] == "owner"
    assert report["provenance"]["classification"] == "ai-declared"
    [(method, _, body)] = [c for c in api.calls if c[1] == "/issues/7/comments"]
    assert method == "POST"
    assert body is not None
    assert hr.MARKER in body["body"]
    assert "Approved by @owner" in body["body"]


def test_a_new_push_removes_the_label_and_unticks_the_box(tmp_path: Path) -> None:
    api = FakeApi(pr(), [labelled(HUMAN)], [])

    review = run(api, "synchronize", tmp_path)

    assert ("DELETE", f"/issues/7/labels/{hr.LABEL}", None) in api.calls
    assert f"- [ ] {hr.ATTESTATION}" in api.pull["body"]
    assert not review.approved
    assert api.status()["state"] == "pending"


def test_a_bot_label_is_removed_and_fails_the_gate(tmp_path: Path) -> None:
    api = FakeApi(pr(), [labelled({"login": "github-actions[bot]", "type": "Bot"})], [])

    run(api, "labeled", tmp_path)

    assert ("DELETE", f"/issues/7/labels/{hr.LABEL}", None) in api.calls
    assert api.status()["state"] == "failure"


def test_the_gate_comment_is_edited_in_place(tmp_path: Path) -> None:
    ours = {"id": 5, "user": {"login": hr.BOT_LOGIN}, "body": hr.MARKER + " old"}
    forged = {"id": 6, "user": HUMAN, "body": hr.MARKER + " forged"}
    api = FakeApi(pr(body=UNTICKED), [], [forged, ours])

    run(api, "edited", tmp_path)

    comment_calls = [c for c in api.calls if "comments" in c[1]]
    assert [(m, p) for m, p, _ in comment_calls] == [("PATCH", "/issues/comments/5")]
    assert api.status()["state"] == "pending"


def test_main_refuses_to_run_without_reviewers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEWERS", " , ")
    assert hr.main() == 1
