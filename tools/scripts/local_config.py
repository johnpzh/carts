"""Local, untracked CARTS workspace configuration."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


ENV_CARTS_HOME = "CARTS_HOME"
ENV_CARTS_CONFIG = "CARTS_CONFIG"
LOCAL_CONFIG_FILE = "carts.config"
INSTALL_DIR_NAME = ".install"
BUILD_DIR_NAME = "build"


def resolve_carts_home(carts_dir: Path) -> Path:
    """Resolve the generated-artifact root for this checkout.

    Precedence:
    1. CARTS_HOME environment variable
    2. CARTS_CONFIG local workspace file, defaulting to ./carts.config
    3. project checkout root
    """
    raw_home = os.environ.get(ENV_CARTS_HOME) or _read_config_home(carts_dir)
    if not raw_home:
        return carts_dir

    carts_home = Path(raw_home).expanduser()
    if not carts_home.is_absolute():
        carts_home = carts_dir / carts_home
    return carts_home.resolve()


def _read_config_home(carts_dir: Path) -> str | None:
    config_path = _resolve_config_path(carts_dir)
    if not config_path.is_file():
        return None

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    value = _lookup_home_value(data)
    return str(value) if value else None


def _lookup_home_value(data: dict[str, Any]) -> Any:
    if ENV_CARTS_HOME in data:
        return data[ENV_CARTS_HOME]
    if "home" in data:
        return data["home"]
    if "carts_home" in data:
        return data["carts_home"]

    carts_section = data.get("carts")
    if isinstance(carts_section, dict):
        return carts_section.get("home") or carts_section.get("carts_home")

    paths_section = data.get("paths")
    if isinstance(paths_section, dict):
        return paths_section.get("home") or paths_section.get("carts_home")

    return None


def _resolve_config_path(carts_dir: Path) -> Path:
    raw = os.environ.get(ENV_CARTS_CONFIG)
    if not raw:
        return carts_dir / LOCAL_CONFIG_FILE

    config_path = Path(raw).expanduser()
    if not config_path.is_absolute():
        config_path = carts_dir / config_path
    return config_path.resolve()
