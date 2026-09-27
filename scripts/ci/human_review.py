"""Human review gate: every pull request needs an allowed reviewer's approval before merging.

Run by .github/workflows/human-review.yml on pull_request_target. The pull request is only
read through the API; its code is never checked out or run. Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

Json = dict[str, Any]

LABEL: Final = "human-reviewed"
STATUS_CONTEXT: Final = "human-review"
MARKER: Final = "<!-- human-review-gate -->"
BOT_LOGIN: Final = "github-actions[bot]"
ATTESTATION: Final = "I have personally read and understood every line of code in this PR"
TICKED: Final = re.compile(
    r"^([ \t]*[-*][ \t]*)\[[xX]\]([ \t]*" + re.escape(ATTESTATION) + ")", re.M
)
AI_NAMES: Final = re.compile(
    r"claude|anthropic|copilot|openai|chatgpt|codex|gemini|cursor|devin|aider", re.I
)
AI_TRAILER: Final = re.compile(r"^(?:co-authored-by|assisted-by|generated-by):(.*)$", re.I | re.M)
GENERATED_WITH: Final = re.compile(r"generated (?:with|by) .*", re.I)
AI_BRANCH: Final = re.compile(r"^(?:copilot|claude|codex|cursor|devin)[/-]", re.I)
SECTION: Final = re.compile(
    r"^##[ \t]+AI provenance[ \t]*$(.*?)(?=^##[ \t]|\Z)", re.M | re.S | re.I
)
FIELD: Final = re.compile(
    r"^[ \t]*[-*][ \t]*(Agent|Model|Swarm|Run IDs|Task IDs):[ \t]*(.*)$", re.M
)
EMPTY_VALUES: Final = {"", "-", "none", "n/a", "na"}
API: Final = "https://api.github.com"


@dataclass(frozen=True)
class Provenance:
    ai_detected: bool
    copilot_detected: bool
    swarm_detected: bool
    # ai-declared (PR description or commit trailers), ai-detected (bot authors or AI
    # branch names only) or undeclared. Review is required either way.
    classification: str
    sources: tuple[str, ...]
    agents: tuple[str, ...]
    models: tuple[str, ...]
    swarm: tuple[str, ...]
    run_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    commits: int
    bot_commits: int


@dataclass(frozen=True)
class Review:
    approved: bool
    missing: tuple[str, ...]
    reviewer: str | None
    # Login of a non-human that applied the label; the label is then removed.
    rejected_actor: str | None


class Api(Protocol):
    def get(self, path: str) -> Any: ...
    def list(self, path: str) -> list[Json]: ...
    def send(self, method: str, path: str, body: Json | None = None) -> Any: ...


def clean(value: str, limit: int = 80) -> str:
    """Untrusted text reduced to characters that cannot form html, mentions or break a code span.
    Brackets stay for bot logins such as ``name[bot]``; values are always rendered in backticks."""
    return re.sub(r"[^\w.:/+=\[\]-]", "", value)[:limit]


def declared_fields(body: str) -> dict[str, tuple[str, ...]]:
    """Fields of the PR description's `## AI provenance` section; blank fields are empty."""
    fields: dict[str, tuple[str, ...]] = {}
    section = SECTION.search(body)
    if section is None:
        return fields
    for name, raw in FIELD.findall(section.group(1)):
        values = tuple(
            clean(v.strip()) for v in raw.split(",") if v.strip().lower() not in EMPTY_VALUES
        )
        fields[name.lower()] = tuple(v for v in values if v)
    return fields


def tool_name(text: str) -> str:
    """The AI tool a login, trailer or agent name refers to, so one tool counts once."""
    name = AI_NAMES.search(text)
    return name.group(0).lower() if name else clean(text).lower()


def is_bot(user: Json | None) -> bool:
    if not user:
        return False
    login = str(user.get("login", ""))
    return user.get("type") != "User" or login.endswith("[bot]")


def detect_provenance(pr: Json, commits: list[Json]) -> Provenance:
    fields = declared_fields(pr.get("body") or "")
    declared: list[str] = []
    detected: list[str] = []
    ai_names: set[str] = set()
    if fields.get("agent") or fields.get("model"):
        declared.append("pr-description")
        ai_names.update(tool_name(agent) for agent in fields.get("agent", ()))
    bot_commits = 0
    for commit in commits:
        sha = str(commit.get("sha", ""))[:7]
        message = str((commit.get("commit") or {}).get("message", ""))
        for value in AI_TRAILER.findall(message) + GENERATED_WITH.findall(message):
            if AI_NAMES.search(value):
                declared.append(f"commit-trailer:{sha}")
                ai_names.add(tool_name(value))
        for role in ("author", "committer"):
            user = commit.get(role) or {}
            if not is_bot(user):
                continue
            if role == "author":
                bot_commits += 1
            login = str(user.get("login", ""))
            if AI_NAMES.search(login):
                detected.append(f"commit-{role}:{clean(login)}")
                ai_names.add(tool_name(login))
    branch = str((pr.get("head") or {}).get("ref", ""))
    if AI_BRANCH.search(branch):
        detected.append(f"branch:{clean(branch)}")
    author = pr.get("user") or {}
    if AI_NAMES.search(str(author.get("login", ""))):
        detected.append(f"pr-author:{clean(str(author.get('login')))}")
    sources = tuple(dict.fromkeys(declared + detected))
    swarm = fields.get("swarm", ())
    classification = "ai-declared" if declared else "ai-detected" if detected else "undeclared"
    return Provenance(
        ai_detected=bool(sources),
        copilot_detected=any("copilot" in s.lower() for s in sources)
        or any("copilot" in a.lower() for a in fields.get("agent", ())),
        swarm_detected=len(swarm) >= 2 or len(ai_names) >= 2,
        classification=classification,
        sources=sources,
        agents=fields.get("agent", ()),
        models=fields.get("model", ()),
        swarm=swarm,
        run_ids=fields.get("run ids", ()),
        task_ids=fields.get("task ids", ()),
        commits=len(commits),
        bot_commits=bot_commits,
    )


def evaluate_review(pr: Json, events: list[Json], reviewers: set[str]) -> Review:
    """Approved when the box is ticked and the label's latest application came from a reviewer."""
    missing: list[str] = []
    if not TICKED.search(pr.get("body") or ""):
        missing.append("the review box in the PR description is not ticked")
    labelled = [
        e
        for e in events
        if e.get("event") == "labeled" and (e.get("label") or {}).get("name") == LABEL
    ]
    has_label = any(label.get("name") == LABEL for label in pr.get("labels") or [])
    reviewer: str | None = None
    rejected: str | None = None
    if not has_label:
        missing.append(f"the `{LABEL}` label is missing")
    elif not labelled:
        missing.append(f"no event records who added `{LABEL}`")
    else:
        actor = labelled[-1].get("actor") or {}
        login = str(actor.get("login", ""))
        if is_bot(actor) or AI_NAMES.search(login):
            rejected = login
            missing.append(f"`{LABEL}` was added by `{clean(login)}`, which is not a human")
        elif login.lower() not in reviewers:
            missing.append(
                f"`{LABEL}` was added by `{clean(login)}`, who is not an allowed reviewer"
            )
        else:
            reviewer = login
    return Review(
        approved=not missing, missing=tuple(missing), reviewer=reviewer, rejected_actor=rejected
    )


def untick(body: str) -> str:
    return TICKED.sub(r"\1[ ]\2", body)


def render(pr: Json, provenance: Provenance, review: Review) -> str:
    def listed(values: tuple[str, ...]) -> str:
        return ", ".join(f"`{v}`" for v in values) or "none"

    head = str(pr["head"]["sha"])[:7]
    if review.approved:
        status = f"✅ Approved by @{review.reviewer} at `{head}`."
    else:
        status = f"⏳ Waiting for human review of `{head}`:\n" + "\n".join(
            f"- {reason}" for reason in review.missing
        )
    return f"""{MARKER}
## AI provenance

AI detected: {str(provenance.ai_detected).lower()}
Copilot detected: {str(provenance.copilot_detected).lower()}
Swarm detected: {str(provenance.swarm_detected).lower()}
Disclosure classification: {provenance.classification}
Detection sources: {listed(provenance.sources)}
Agents: {listed(provenance.agents)} · Models: {listed(provenance.models)} · Swarm: {listed(provenance.swarm)}
Run IDs: {listed(provenance.run_ids)}
Task IDs: {listed(provenance.task_ids)}
Commits: {provenance.commits} ({provenance.bot_commits} by bot accounts)

Artifact published: `ai-provenance.json`

## Human review

{status}

Every pull request needs a human review before merging. The reviewer must:

1. Read and understand every changed line.
2. Verify correctness, security, and test quality.
3. Confirm the AI provenance fields above are correct.
4. Tick the box in the PR description: `{ATTESTATION}`
5. Add the `{LABEL}` label from their own account. A label added by a bot or AI account is removed.

Any new push removes the label and unticks the box, so the new commits need a fresh review.
"""


def reset_review(gh: Api, pr: Json) -> None:
    """A new push invalidates the review of the previous head."""
    number = pr["number"]
    if any(label.get("name") == LABEL for label in pr.get("labels") or []):
        gh.send("DELETE", f"/issues/{number}/labels/{LABEL}")
    body = pr.get("body") or ""
    if TICKED.search(body):
        gh.send("PATCH", f"/pulls/{number}", {"body": untick(body)})


def upsert_comment(gh: Api, number: int, text: str) -> None:
    for comment in gh.list(f"/issues/{number}/comments"):
        user = comment.get("user") or {}
        if user.get("login") == BOT_LOGIN and MARKER in str(comment.get("body", "")):
            gh.send("PATCH", f"/issues/comments/{comment['id']}", {"body": text})
            return
    gh.send("POST", f"/issues/{number}/comments", {"body": text})


def run(gh: Api, event: Json, reviewers: set[str], run_url: str, out: Path) -> Review:
    number = int(event["pull_request"]["number"])
    pr = gh.get(f"/pulls/{number}")
    if event.get("action") == "synchronize":
        reset_review(gh, pr)
        pr = gh.get(f"/pulls/{number}")
    review = evaluate_review(pr, gh.list(f"/issues/{number}/events"), reviewers)
    if review.rejected_actor is not None:
        gh.send("DELETE", f"/issues/{number}/labels/{LABEL}")
    provenance = detect_provenance(pr, gh.list(f"/pulls/{number}/commits"))
    upsert_comment(gh, number, render(pr, provenance, review))
    out.write_text(
        json.dumps(
            {
                "pull_request": number,
                "head_sha": pr["head"]["sha"],
                "provenance": asdict(provenance),
                "review": asdict(review),
            },
            indent=2,
        )
    )
    if review.approved:
        state, description = "success", f"Approved by {review.reviewer}"
    elif review.rejected_actor is not None:
        state, description = "failure", "Label added by a non-human account was removed"
    else:
        state, description = "pending", "Waiting for human review"
    gh.send(
        "POST",
        f"/statuses/{pr['head']['sha']}",
        {
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description[:140],
            "target_url": run_url,
        },
    )
    return review


class GitHub:
    def __init__(self, token: str, repo: str) -> None:
        self._token = token
        self._base = f"{API}/repos/{repo}"

    def _request(self, method: str, url: str, body: Json | None) -> tuple[Any, str]:
        request = urllib.request.Request(  # noqa: S310 - fixed https API base
            url,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                raw = response.read()
                return (json.loads(raw) if raw else None), response.headers.get("Link", "")
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise RuntimeError(f"{method} {url} failed: {error.code} {detail}") from error

    def get(self, path: str) -> Any:
        return self._request("GET", self._base + path, None)[0]

    def list(self, path: str) -> list[Json]:
        items: list[Json] = []
        url: str | None = f"{self._base}{path}?per_page=100"
        while url:
            page, link = self._request("GET", url, None)
            items.extend(page)
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
        return items

    def send(self, method: str, path: str, body: Json | None = None) -> Any:
        return self._request(method, self._base + path, body)[0]


def main() -> int:
    reviewers = {r.strip().lower() for r in os.environ.get("REVIEWERS", "").split(",") if r.strip()}
    if not reviewers:
        print("REVIEWERS is empty: no one could approve", file=sys.stderr)
        return 1
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    gh = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    review = run(gh, event, reviewers, os.environ.get("RUN_URL", ""), Path("ai-provenance.json"))
    print("approved" if review.approved else "waiting: " + "; ".join(review.missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
