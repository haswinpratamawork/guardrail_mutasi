"""Singleton config: .env + config.yaml with ${VAR} / ${VAR:-default} interpolation.

Mirrors the loader the sibling services on this host use, so the deployment
story (env-file, nexus scripts, APP_ENVIRO switch) is the same one the team
already operates.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ${NAME} or ${NAME:-default}
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

_ROOT = Path(__file__).resolve().parents[2]


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str):
        return value

    def _sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or (default if default is not None else "")

    out = _VAR.sub(_sub, value)
    # YAML can't know that ${PORT:-8090} is an int, so re-coerce the obvious cases.
    if out.isdigit():
        return int(out)
    try:
        return float(out) if out.replace(".", "", 1).isdigit() and "." in out else out
    except ValueError:
        return out


class Config:
    def __init__(self, path: Path | None = None) -> None:
        load_dotenv(_ROOT / ".env", override=False)
        self.path = path or Path(os.environ.get("CONFIG_PATH", _ROOT / "config.yaml"))
        raw = yaml.safe_load(self.path.read_text())
        self._data: dict[str, Any] = _interpolate(raw)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def service(self) -> dict[str, Any]:
        return self._data["service"]

    @property
    def model(self) -> dict[str, Any]:
        return self._data["model"]

    @property
    def gcs(self) -> dict[str, Any]:
        return self._data["gcs"]

    @property
    def request(self) -> dict[str, Any]:
        return self._data["request"]

    @property
    def logging(self) -> dict[str, Any]:
        return self._data["logging"]

    def resolve(self, relative: str) -> Path:
        """Config paths are written relative to the repo root, not the cwd the
        process happens to start in."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else _ROOT / candidate


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Drop the cached singleton — tests patch the environment and reload."""
    global _config
    _config = None
