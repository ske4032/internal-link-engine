#!/usr/bin/env bash
# Creates all 33 core-engine issues with labels and milestone.
# Requires: gh CLI, authenticated, run from inside the repo.
#
#   ./create_issues.sh          # dry run, prints what it would do
#   ./create_issues.sh --apply  # actually create
set -euo pipefail

APPLY=${1:-}
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

echo "repo: $REPO"

LABELS=(
  "core:0E8A16:Core intelligence engine"
  "phase-0-foundation:C5DEF5:Scaffold, models, schema"
  "phase-1-embedding:C5DEF5:Voyage client and write path"
  "phase-2-graph:C5DEF5:igraph, Leiden, HDBSCAN"
  "phase-3-retrieval:C5DEF5:Candidates and signals"
  "phase-4-features:C5DEF5:Feature assembly and scoring"
  "phase-5-anchors:C5DEF5:Keyword resolution and extraction"
  "phase-6a-ranking:C5DEF5:LambdaMART on proxy labels"
  "phase-6b-real-labels:C5DEF5:LambdaMART on real labels"
  "phase-7-validation:C5DEF5:End-to-end validation"
  "gate:D93F0B:Decision point - may delete downstream work"
)

if [[ "$APPLY" == "--apply" ]]; then
  for l in "${LABELS[@]}"; do
    IFS=: read -r name color desc <<< "$l"
    gh label create "$name" --color "$color" --description "$desc" --force
  done
  gh api "repos/$REPO/milestones" -f title="Core Engine" \
    -f description="Voyage embedding through to a validated intelligence core" \
    >/dev/null 2>&1 || echo "milestone exists"
fi

python3 - "$APPLY" <<'PY'
import json, subprocess, sys
apply = len(sys.argv) > 1 and sys.argv[1] == "--apply"
for i in json.load(open("issues.json")):
    cmd = ["gh","issue","create","--title",i["title"],"--body",i["body"],
           "--milestone",i["milestone"]]
    for l in i["labels"]:
        cmd += ["--label", l]
    if apply:
        subprocess.run(cmd, check=True)
    else:
        print(f"[dry] {i['title']}  labels={','.join(i['labels'])}")
PY

echo
[[ "$APPLY" == "--apply" ]] || echo "dry run. re-run with --apply to create."
