"""Docker command adapter that loads implementation from docker/scripts."""

from __future__ import annotations

import sys
from pathlib import Path


_DOCKER_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "docker" / "scripts"
if str(_DOCKER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_DOCKER_SCRIPTS_DIR))

from docker_cli import (  # noqa: E402
    docker_build,
    docker_callback,
    docker_clean,
    docker_commit,
    docker_exec,
    docker_start,
    docker_status,
    docker_stop,
    docker_update,
)


__all__ = [
    "docker_callback",
    "docker_build",
    "docker_start",
    "docker_update",
    "docker_stop",
    "docker_clean",
    "docker_commit",
    "docker_status",
    "docker_exec",
]
