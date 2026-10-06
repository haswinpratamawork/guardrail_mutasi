"""Fetch the model artifact from GCS at startup.

Weights are never baked into the image (see .dockerignore) — the container
pulls them on boot, which is how the sibling services on this host work and
what lets a new model roll out without a rebuild.

Credentials come from Application Default Credentials, i.e. the workload's
attached service account. No keyfile.
"""
from __future__ import annotations

import fcntl
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_model(bucket_name: str, object_name: str, local_path: str | Path) -> Path:
    """Download gs://<bucket_name>/<object_name> to `local_path`, once.

    uvicorn runs several workers and they all boot at the same time, so the
    download is serialized on a lock file: the first worker fetches, the rest
    block and then find the artifact already there.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = local_path.with_suffix(local_path.suffix + ".lock")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("Model already present at %s (%.1f MB) — skipping download",
                        local_path, local_path.stat().st_size / 1e6)
            return local_path

        from google.cloud import storage

        logger.info("Downloading gs://%s/%s -> %s", bucket_name, object_name, local_path)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        if not blob.exists():
            raise FileNotFoundError(f"gs://{bucket_name}/{object_name} does not exist")

        # Download to a temp name and rename, so a killed pod can't leave a
        # truncated artifact that the next boot would load as if it were whole.
        tmp = local_path.with_suffix(local_path.suffix + ".partial")
        blob.download_to_filename(str(tmp))
        tmp.replace(local_path)
        logger.info("Downloaded %.1f MB", local_path.stat().st_size / 1e6)
        return local_path
