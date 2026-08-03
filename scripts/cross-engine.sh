#!/usr/bin/env bash
# Prove the simulation is engine-independent.
#
# Runs the same match under Node (V8) and Bun (JavaScriptCore) and diffs the
# checksums. These are two independent implementations of JavaScript's `Math`,
# so a simulation that leaned on a non-portable function — `Math.sin`, `pow`,
# `exp` — or that lost bits to float drift would diverge here. Identical output
# means a Chrome player and a Safari player stay in sync.
set -euo pipefail

cd "$(dirname "$0")/.."
out_dir="$(mktemp -d)"
trap 'rm -rf "$out_dir"' EXIT

echo "running under Node (V8)..."
npx vite-node scripts/checksum-probe.ts > "$out_dir/node.txt"

if ! command -v bun >/dev/null 2>&1; then
  echo
  echo "bun not installed — ran V8 only, cross-engine check SKIPPED."
  cat "$out_dir/node.txt"
  exit 0
fi

echo "running under Bun (JavaScriptCore)..."
bun run scripts/checksum-probe.ts > "$out_dir/bun.txt"

echo
if diff -u "$out_dir/node.txt" "$out_dir/bun.txt"; then
  cat "$out_dir/node.txt"
  echo
  echo "PASS — V8 and JavaScriptCore agree on every checkpoint."
else
  echo
  echo "FAIL — engines disagree. The simulation is not portable;"
  echo "a desync between browsers is guaranteed. Check for non-portable"
  echo "Math usage or float arithmetic that escaped the fixed-point layer."
  exit 1
fi
