#!/usr/bin/env bash
# Build the serving image and push it to the internal registry.
#   ./scripts/nexus_build.sh [tag] [env]
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
ENVIRONMENT="${2:-dev}"
REGISTRY="new-nexus.gcp.bri.co.id"
PROJECT="${NEXUS_PROJECT:-ocr-nilam}"
NAME="ms-bribrain-ocr-nilam-guardrail-statement"
IMAGE="${REGISTRY}/${PROJECT}/${ENVIRONMENT}/${NAME}"

echo "Building ${IMAGE}:${TAG}"
# --target production also runs the e2e-test stage: the production stage copies
# a marker out of it, so the unit tests cannot be skipped.
DOCKER_BUILDKIT=1 docker build --target production -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .

if [[ "${PUSH:-1}" == "1" ]]; then
  docker push "${IMAGE}:${TAG}"
  docker push "${IMAGE}:latest"
  echo "Pushed ${IMAGE}:${TAG}"
else
  echo "PUSH=0 — built locally, not pushed."
fi
