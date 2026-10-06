"""Startup configuration checks — fail fast and loudly, before serving anything."""
from __future__ import annotations

import logging
import sys

from src.core.config import get_config

logger = logging.getLogger(__name__)


def validate_startup_configuration() -> None:
    cfg = get_config()
    problems: list[str] = []

    threshold = cfg.model["threshold"]
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        problems.append(f"model.threshold must be a number in [0, 1], got {threshold!r}")

    for section, key in (("request", "max_upload_mb"), ("request", "max_pages"), ("model", "embedding_batch_size")):
        value = cfg[section][key]
        if not isinstance(value, int) or value <= 0:
            problems.append(f"{section}.{key} must be a positive integer, got {value!r}")

    if not cfg.gcs["bucket_name"] or not cfg.gcs["object"]:
        problems.append("gcs.bucket_name and gcs.object are both required to fetch the model artifact")

    if problems:
        for problem in problems:
            logger.error("Startup configuration error: %s", problem)
        sys.exit(1)

    logger.info(
        "Configuration OK — threshold=%s, max_pages=%s, max_upload_mb=%s, artifact=gs://%s/%s",
        threshold, cfg.request["max_pages"], cfg.request["max_upload_mb"],
        cfg.gcs["bucket_name"], cfg.gcs["object"],
    )
