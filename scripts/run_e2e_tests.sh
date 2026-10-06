#!/usr/bin/env bash
# Unit tests, used as the build-time quality gate in the Dockerfile's
# e2e-test stage. Model-backed cases skip when no artifact is staged, so this
# stays hermetic and needs no GCS access.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
exec "${PYTHON}" -m pytest tests/unit -q
