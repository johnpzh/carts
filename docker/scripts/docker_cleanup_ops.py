"""Cleanup and image commit commands for Docker CLI."""

from __future__ import annotations

import tempfile
import time

import typer

from dekk import print_error, print_header, print_info, print_step, print_success, print_warning
from docker_shared import (
    BASE_IMAGE,
    BUILT_IMAGE,
    WORKSPACE_VOLUME,
    _container_exists,
    _docker,
    _docker_exec,
    _image_exists,
    _volume_exists,
)


def docker_clean():
    """Clean all Docker resources related to CARTS cluster."""
    print_header("CARTS Docker Clean")
    print_info("Removing CARTS Docker containers, images, and volumes")

    running_or_stopped = _docker("ps", "-a", "--format", "{{.Names}}", capture_output=True, check=False)
    if running_or_stopped.returncode == 0:
        names = [
            name.strip()
            for name in (running_or_stopped.stdout or "").splitlines()
            if name.strip().startswith("arts-node-")
        ]
        if names:
            print_step("Removing containers")
            _docker("rm", "-f", *names, check=False)

    print_step("Removing builder/update helper containers")
    _docker("rm", "-f", "arts-node-builder", "arts-node-update", "arts-node-commit-temp", check=False)

    print_step("Removing images")
    _docker("rmi", BUILT_IMAGE, check=False)
    _docker("rmi", BASE_IMAGE, check=False)

    print_step("Removing workspace volume")
    _docker("volume", "rm", WORKSPACE_VOLUME, check=False)

    print_step("Pruning Docker networks and system cache")
    _docker("network", "prune", "-f", check=False)
    _docker("system", "prune", "-af", check=False)

    print_success("Docker cleanup completed")


def docker_commit():
    """Commit current workspace from arts-node-1 into arts-node:built."""
    print_header("CARTS Docker Commit")

    if not _image_exists(BASE_IMAGE):
        print_error(f"Required image '{BASE_IMAGE}' not found")
        print_warning("Run 'carts docker build' first")
        raise typer.Exit(1)

    if not _container_exists("arts-node-1"):
        print_error("Container 'arts-node-1' not found")
        print_warning("Run 'carts docker start' first")
        raise typer.Exit(1)

    if not _volume_exists(WORKSPACE_VOLUME):
        print_error(f"Required volume '{WORKSPACE_VOLUME}' not found")
        print_warning("Run 'carts docker build' first")
        raise typer.Exit(1)

    _docker("rm", "-f", "arts-node-commit-temp", check=False)

    print_step("Starting temporary commit container")
    result = _docker("run", "-d", "--name", "arts-node-commit-temp", BASE_IMAGE, check=False)
    if result.returncode != 0:
        print_error("Failed to start temporary commit container")
        raise typer.Exit(result.returncode)

    try:
        time.sleep(2)

        print_step("Copying workspace into temporary container")
        with tempfile.TemporaryDirectory(prefix="carts-docker-commit-") as temp_copy_dir:
            cp_out = _docker("cp", "arts-node-1:/opt/carts/.", temp_copy_dir, check=False)
            if cp_out.returncode != 0:
                print_error("Failed to copy workspace from arts-node-1")
                raise typer.Exit(cp_out.returncode)

            _docker_exec(
                "arts-node-commit-temp",
                "rm -rf /opt/carts/* /opt/carts/.[!.]* /opt/carts/..?* 2>/dev/null || true; mkdir -p /opt/carts",
                check=False,
            )

            cp_in = _docker("cp", f"{temp_copy_dir}/.", "arts-node-commit-temp:/opt/carts/", check=False)
            if cp_in.returncode != 0:
                print_error("Failed to copy workspace to temporary container")
                raise typer.Exit(cp_in.returncode)

        verify = _docker_exec("arts-node-commit-temp", "test -d /opt/carts/.git", check=False)
        if verify.returncode != 0:
            print_error("Workspace copy verification failed (.git missing)")
            raise typer.Exit(1)

        if _image_exists(BUILT_IMAGE):
            print_step(f"Removing previous image: {BUILT_IMAGE}")
            _docker("rmi", BUILT_IMAGE, check=False)

        print_step(f"Committing container to image: {BUILT_IMAGE}")
        commit = _docker("commit", "arts-node-commit-temp", BUILT_IMAGE, capture_output=True, check=False)
        if commit.returncode != 0:
            print_error("Failed to create committed image")
            if commit.stderr:
                print_error(commit.stderr.strip())
            raise typer.Exit(commit.returncode)

        print_success(f"Created image: {BUILT_IMAGE}")

    finally:
        _docker("rm", "-f", "arts-node-commit-temp", check=False)
