"""Logging with a request id woven through every line.

The id lives in a ContextVar set by RequestIdMiddleware, and a filter injects it
into the format string so `%(request_id)s` always resolves — including on log
records emitted by third-party libraries that know nothing about it.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_FORMAT = "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True


def setup_logging(level: str = "INFO", log_dir: str | Path = "./logs") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    formatter = logging.Formatter(_FORMAT)
    id_filter = RequestIdFilter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(id_filter)
    root.addHandler(stream)

    try:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(path / "service.log", maxBytes=10 * 1024 * 1024, backupCount=5)
        rotating.setFormatter(formatter)
        rotating.addFilter(id_filter)
        root.addHandler(rotating)
    except OSError as exc:
        # A read-only or missing log dir must not stop the service booting —
        # stdout is already attached and is what the container collects anyway.
        root.warning("File logging disabled (%s): %s", log_dir, exc)

    # pypdfium2 chatters at DEBUG about every page it renders.
    logging.getLogger("pypdfium2").setLevel(logging.WARNING)
