#!/usr/bin/env bash
# Extract crates to Lean with Charon and Aeneas.
#
# Usage: scripts/extract.sh <crate>...
#
# For each crate, Charon compiles the Rust (through MIR) to LLBC, and Aeneas
# translates the LLBC into a pure Lean model under lean/extracted/<crate>.
# Both tools must be on PATH; see lean/README.md. The invocations follow the
# Aeneas README (`charon cargo --preset=aeneas` from inside the crate, then
# `aeneas -backend lean LLBC_FILE`); the output-directory flag is not
# documented there and is checked against `aeneas --help` on first use.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/lean/extracted"
mkdir -p "$out"

for crate in "$@"; do
    manifest=$(grep -rl --include=Cargo.toml "^name = \"$crate\"$" "$root/crates" | head -n 1)
    dir=$(dirname "$manifest")
    if [ -z "$manifest" ] || [ ! -d "$dir" ]; then
        echo "error: no crate directory for $crate" >&2
        exit 1
    fi
    llbc="$out/$crate.llbc"
    echo "== charon: $crate"
    (cd "$dir" && charon cargo --preset=aeneas --dest-file "$llbc")
    echo "== aeneas: $crate"
    aeneas -backend lean -dest "$out/$crate" "$llbc"
done
