#!/usr/bin/env bash
# Run pytest + ruff (lint and format-check) across every workspace member, so a
# prod checkout can be validated in one command.
#
# The repo is a uv workspace: one lock and one shared .venv at the root, every
# member identically shaped (pyproject.toml, src/<name>/, tests/). One sync up
# front installs everything; each member's suite then runs from its own
# directory so its pyproject's pytest/ruff config applies. The script continues
# past failures and prints a summary at the end; it exits non-zero if any step
# failed. Run from anywhere -- it resolves the repo root from its own location.
#
#   scripts/run-suite-tests.sh
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PACKAGES=(
  bus
  airc-core
  airc-tools
  deepagent
  airc-room
)

overall=0
results=()

step() { # label, workdir, cmd...
  local label="$1" dir="$2"
  shift 2
  echo
  echo ">>> ${label}: $* (in ${dir})"
  if (cd "${ROOT}/${dir}" && "$@"); then
    results+=("PASS  ${label}")
  else
    results+=("FAIL  ${label}")
    overall=1
  fi
}

echo "repo:   ${ROOT}"
echo "commit: $(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "uv:     $(uv --version 2>/dev/null || echo 'NOT FOUND')"

# Core has no optional extras; a plain --all-packages sync is the whole core
# surface.
step "workspace sync" "." uv sync --all-packages --quiet

for name in "${PACKAGES[@]}"; do
  step "${name} pytest" "${name}" uv run pytest -q
  step "${name} ruff" "${name}" uv run ruff check src/ tests/
  step "${name} format" "${name}" uv run ruff format --check src/ tests/
done

echo
echo "================ summary ================"
printf '%s\n' "${results[@]}"
if [ "${overall}" -ne 0 ]; then
  echo "RESULT: FAIL"
else
  echo "RESULT: PASS"
fi
exit "${overall}"
