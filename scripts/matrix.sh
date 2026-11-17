#!/usr/bin/env bash
# Run one task across models x harness x tools x search (API models; no GPU needed).
# Each cell gets its own log dir, so `inspect view --log-dir logs/matrix` shows them side by side.
#
#   MODELS="anthropic/claude-opus-5-5,openai/<model>,google/<model>" \
#   HARNESSES="plain react" TOOLS="none sandbox" SEARCH="off controlled" \
#     scripts/matrix.sh tasks/demo.py [extra inspect args...]
#
# For inspect_evals tasks set SWAP_SOLVER=1 (and usually --sandbox apptainer:<config>):
#   SWAP_SOLVER=1 TOOLS=sandbox scripts/matrix.sh inspect_evals/gaia --sandbox apptainer
# Invalid cells (plain + tools/search) are skipped. eval-set retries failures and skips finished
# cells on re-run, so the script is safe to resume.
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
TASK=${1:?usage: matrix.sh <task> [inspect args...]}; shift
MODELS=${MODELS:?set MODELS (comma-separated)}
HARNESSES=${HARNESSES:-"plain react"}
TOOLS=${TOOLS:-"none"}
SEARCH=${SEARCH:-"off"}
OUT=${OUT:-$ROOT/logs/matrix/$(basename "${TASK%.py}")}

cd "$ROOT"
for h in $HARNESSES; do for t in $TOOLS; do for s in $SEARCH; do
  if [[ $h == plain && ( $t != none || $s != off ) ]]; then continue; fi
  cell="h-${h}_t-${t}_s-${s}"
  echo "=== $cell"
  if [[ -n ${SWAP_SOLVER:-} ]]; then   # foreign task (e.g. inspect_evals/...): replace its solver
    knobs=(--solver labkit/agent -S harness="$h" -S tools="$t" -S search="$s")
  else                                 # our tasks take the knobs as task args
    knobs=(-T harness="$h" -T tools="$t" -T search="$s")
  fi
  .venv/bin/inspect eval-set "$TASK" --model "$MODELS" --log-dir "$OUT/$cell" "${knobs[@]}" "$@"
done; done; done
