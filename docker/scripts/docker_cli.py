"""Docker subcommands for CARTS CLI."""

from __future__ import annotations

import typer

from docker_build_ops import docker_build
from docker_cleanup_ops import docker_clean, docker_commit
from docker_cluster_ops import docker_exec, docker_start, docker_status, docker_stop
from docker_update_ops import docker_update


def docker_callback(ctx: typer.Context):
    """Docker operations for multi-node execution."""
    if ctx.invoked_subcommand is None:
        from dekk import console

        console.print("Docker command requires a subcommand. Use -h/--help for options.")
        raise typer.Exit(1)


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
