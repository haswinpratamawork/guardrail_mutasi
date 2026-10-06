# syntax=docker/dockerfile:1.6

# 3.12 matches the training venv that pickles the model artifact — see the
# note in pyproject.toml. Everything else follows the house service pattern.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml README.md ./
RUN uv sync --no-cache --all-extras
COPY . /app/

FROM base AS e2e-test
ENV PYTHONPATH=/app
# Unit tests gate the build. They skip the model-backed cases when no artifact
# is staged, so the build stays hermetic and needs no GCS access.
RUN .venv/bin/python -m pytest tests/unit -q && touch /tmp/tests.ok

FROM python:3.12-slim AS production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml README.md ./
# No --all-extras: production skips the test and dev extras.
RUN uv sync --no-cache

# Copying the marker forces BuildKit to run the test stage before this one.
# Without it, a build targeting production may skip e2e-test entirely.
COPY --from=e2e-test /tmp/tests.ok /tmp/tests.ok

# Create the runtime user BEFORE copying source, so COPY --chown can hand it
# ownership — source cloned on the host is often mode 700 and unreadable to a
# non-root uid, which breaks `import src` at startup.
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser . /app/
# src/models/artifacts is where the GCS artifact lands at boot; it must exist
# and be writable by appuser, and must stay empty in the image.
RUN mkdir -p /app/logs /app/src/models/artifacts \
    && chown -R appuser:appuser /app/logs /app/src/models/artifacts
USER appuser

# Make `src` importable in every process, including uvicorn --workers children:
# spawned subprocesses don't inherit the launch cwd on sys.path, and the
# editable install isn't reliable because uv sync runs before the source copy.
ENV PYTHONPATH=/app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8090/health/live || exit 1
CMD ["/bin/sh", "-c", "exec .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8090} --workers ${UVICORN_WORKERS:-2} --timeout-keep-alive 60"]
