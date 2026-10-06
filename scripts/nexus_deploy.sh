#!/usr/bin/env bash
# Run the serving container on this host.
#   ./scripts/nexus_deploy.sh [tag] [env]
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-latest}"
ENVIRONMENT="${2:-dev}"
REGISTRY="new-nexus.gcp.bri.co.id"
PROJECT="${NEXUS_PROJECT:-ocr-nilam}"
NAME="ms-bribrain-ocr-nilam-guardrail-statement"
IMAGE="${REGISTRY}/${PROJECT}/${ENVIRONMENT}/${NAME}"
PORT="${PORT:-8090}"

# .env is optional now: the service has no secrets and /v1/predict is open.
ENV_FILE_ARG=""
[[ -f .env ]] && ENV_FILE_ARG="--env-file .env"

# No GPU flag on purpose: this service is CPU-only by design (see pyproject.toml).
docker rm -f "${NAME}" 2>/dev/null || true
docker run -d \
  --name "${NAME}" \
  -p "${PORT}:${PORT}" \
  -e "PORT=${PORT}" \
  ${ENV_FILE_ARG} \
  --restart always \
  "${IMAGE}:${TAG}"

echo "Started ${NAME} on port ${PORT}"
echo "Waiting for readiness (the model is fetched from GCS on first boot)..."
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${PORT}/health/ready" >/dev/null 2>&1; then
    echo "Ready."
    curl -s "http://localhost:${PORT}/health" | head -c 800
    echo
    exit 0
  fi
  sleep 5
done
echo "Did not become ready in time — check: docker logs ${NAME}" >&2
exit 1
