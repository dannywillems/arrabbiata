#!/usr/bin/env bash
# Fail unless the main Lean theorems depend only on the standard axioms
# (propext, Classical.choice, Quot.sound). A green `lake build` is not a
# proof: `sorry` elaborates to the axiom `sorryAx`, which this catches.
set -euo pipefail

cd "$(dirname "$0")/../lean"
output=$(lake env lean Axioms.lean)
echo "$output"

if echo "$output" | grep -q "sorryAx"; then
    echo "error: a theorem depends on sorry" >&2
    exit 1
fi

allowed='^(propext|Classical\.choice|Quot\.sound)$'
axioms=$(echo "$output" | grep -o '\[.*\]' | tr -d '[]' | tr ',' '\n' \
    | sed 's/^ *//; s/ *$//' | grep -v '^$' || true)
while IFS= read -r axiom; do
    [ -z "$axiom" ] && continue
    if ! echo "$axiom" | grep -Eq "$allowed"; then
        echo "error: non-standard axiom: $axiom" >&2
        exit 1
    fi
done <<< "$axioms"

echo "axioms ok"
