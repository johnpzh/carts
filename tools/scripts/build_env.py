"""Shared build invocation helpers for CARTS CLI commands."""

from __future__ import annotations

import os
import shutil
from typing import Any

from scripts.local_config import ENV_CARTS_HOME


MAKE_COMMAND = "make"
ENV_CARTS_BUILD_JOBS = "CARTS_BUILD_JOBS"
ENV_CMAKE_BUILD_PARALLEL_LEVEL = "CMAKE_BUILD_PARALLEL_LEVEL"
MAKE_VAR_CARTS_LINKER_PATH = "CARTS_LINKER_PATH"
MAKE_VAR_LLVM_GCC_INSTALL_PREFIX = "LLVM_GCC_INSTALL_PREFIX"
MAKE_VAR_CMAKE = "CMAKE"
MAKE_VAR_INSTALL_DIR = "INSTALL_DIR"
MAKE_VAR_CARTS_BUILD_DIR = "CARTS_BUILD_DIR"
MAKE_VAR_ARTS_BUILD_DIR = "ARTS_BUILD_DIR"
MAKE_VAR_POLYGEIST_BUILD_DIR = "POLYGEIST_BUILD_DIR"
MAKE_VAR_LLVM_BUILD_DIR = "LLVM_BUILD_DIR"
MAKE_VAR_CARTS_INSTALL_DIR = "CARTS_INSTALL_DIR"
MAKE_VAR_ARTS_INSTALL_DIR = "ARTS_INSTALL_DIR"
MAKE_VAR_POLYGEIST_INSTALL_DIR = "POLYGEIST_INSTALL_DIR"
MAKE_VAR_LLVM_INSTALL_DIR = "LLVM_INSTALL_DIR"


def available_parallel_jobs() -> int:
    """Return the worker count to use for local build tools."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    detected = process_cpu_count() if process_cpu_count else os.cpu_count()
    return max(1, detected or 1)


def configured_make_vars(config: Any) -> list[str]:
    """Return Makefile variables required by the active workspace config."""
    make_vars: list[str] = [
        f"{ENV_CARTS_HOME}={config.carts_home}",
        f"{MAKE_VAR_INSTALL_DIR}={config.install_dir}",
        f"{MAKE_VAR_CARTS_BUILD_DIR}={config.carts_build_dir}",
        f"{MAKE_VAR_ARTS_BUILD_DIR}={config.arts_build_dir}",
        f"{MAKE_VAR_POLYGEIST_BUILD_DIR}={config.polygeist_build_dir}",
        f"{MAKE_VAR_LLVM_BUILD_DIR}={config.llvm_build_dir}",
        f"{MAKE_VAR_CARTS_INSTALL_DIR}={config.carts_install_dir}",
        f"{MAKE_VAR_ARTS_INSTALL_DIR}={config.arts_install_dir}",
        f"{MAKE_VAR_POLYGEIST_INSTALL_DIR}={config.polygeist_install_dir}",
        f"{MAKE_VAR_LLVM_INSTALL_DIR}={config.llvm_install_dir}",
    ]

    if getattr(config, "linker_path", None):
        make_vars.append(f"{MAKE_VAR_CARTS_LINKER_PATH}={config.linker_path}")
    if getattr(config, "gcc_install_prefix", None):
        make_vars.append(
            f"{MAKE_VAR_LLVM_GCC_INSTALL_PREFIX}={config.gcc_install_prefix}"
        )
    if cmake_path := shutil.which("cmake"):
        make_vars.append(f"{MAKE_VAR_CMAKE}={cmake_path}")
    return make_vars


def configured_make_command(config: Any, *args: str) -> list[str]:
    """Return a Makefile command that respects the resolved CARTS home."""
    return [MAKE_COMMAND, *configured_make_vars(config), *args]


def make_command_with_vars(make_vars: list[str], *args: str) -> list[str]:
    """Return a Makefile command from already-composed variables."""
    return [MAKE_COMMAND, *make_vars, *args]


def build_parallel_env() -> dict[str, str]:
    """Default builds to all visible CPUs while preserving user overrides."""
    jobs = (
        os.environ.get(ENV_CARTS_BUILD_JOBS)
        or os.environ.get(ENV_CMAKE_BUILD_PARALLEL_LEVEL)
        or str(available_parallel_jobs())
    )
    return {
        ENV_CARTS_BUILD_JOBS: jobs,
        ENV_CMAKE_BUILD_PARALLEL_LEVEL: jobs,
    }
