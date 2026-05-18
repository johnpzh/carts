#!/usr/bin/env python3
"""Project install driver for `dekk carts install`.

This script is intentionally small and imperative. Dekk owns the outer install
flow; this driver owns the CARTS-specific work that dekk cannot infer:

1. create configured build/install directories
2. initialize the required submodules
3. build LLVM, Polygeist, ARTS, and CARTS in order
"""

from __future__ import annotations
from pathlib import Path

from dekk import print_error, print_header, print_info, print_step, print_success

from scripts import (
    ARTS_NESTED_SUBMODULES,
    POLYGEIST_NESTED_SUBMODULES,
    SUBMODULE_ARTS,
    SUBMODULE_BENCHMARKS,
    SUBMODULE_POLYGEIST,
    run_subprocess,
)
from scripts.build_env import (
    available_parallel_jobs,
    build_parallel_env,
    configured_make_command,
)
from scripts.platform import get_config


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> None:
    result = run_subprocess(cmd, cwd=cwd, env=env, realtime=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed")


def _prepare_project_sources() -> None:
    config = get_config()
    project_root = config.carts_dir
    git_jobs = str(available_parallel_jobs())

    print_step("Preparing build directories...")
    for directory in [
        config.install_dir,
        config.install_dir / "bin",
        config.carts_home / "build",
        project_root / "external",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print_step("Synchronizing submodule metadata...")
    _run(["git", "submodule", "sync", "--recursive"], cwd=project_root, label="submodule sync")

    print_step("Initializing top-level submodules...")
    _run(
        [
            "git", "submodule", "update", "--init",
            "--depth", "1", "--single-branch", "--recommend-shallow",
            "--jobs", git_jobs,
            SUBMODULE_ARTS, SUBMODULE_POLYGEIST, SUBMODULE_BENCHMARKS,
        ],
        cwd=project_root,
        label="top-level submodule update",
    )

    arts_dir = project_root / SUBMODULE_ARTS
    print_step("Initializing ARTS nested submodules...")
    _run(["git", "submodule", "sync", "--recursive"], cwd=arts_dir, label="ARTS submodule sync")
    _run(
        [
            "git", "submodule", "update", "--init",
            "--depth", "1", "--single-branch", "--recommend-shallow",
            "--jobs", git_jobs,
            *ARTS_NESTED_SUBMODULES,
        ],
        cwd=arts_dir,
        label="ARTS nested submodule update",
    )

    polygeist_dir = project_root / SUBMODULE_POLYGEIST
    print_step("Initializing Polygeist nested submodules...")
    _run(
        ["git", "submodule", "sync", "--recursive"],
        cwd=polygeist_dir,
        label="Polygeist submodule sync",
    )
    _run(
        [
            "git", "submodule", "update", "--init",
            "--depth", "1", "--single-branch", "--recommend-shallow",
            "--jobs", git_jobs,
            *POLYGEIST_NESTED_SUBMODULES,
        ],
        cwd=polygeist_dir,
        label="Polygeist nested submodule update",
    )


def _build_project() -> None:
    config = get_config()
    project_root = config.carts_dir
    parallel_env = build_parallel_env()

    build_steps = [
        ("Building LLVM...", configured_make_command(config, "llvm")),
        ("Building Polygeist...", configured_make_command(config, "polygeist")),
        ("Building ARTS runtime...", configured_make_command(config, "arts")),
        ("Building CARTS...", configured_make_command(config, "build")),
    ]

    for label, cmd in build_steps:
        print_step(label)
        _run(cmd, cwd=project_root, label=label, env=parallel_env)


def main() -> int:
    print_header("CARTS Install")
    print_info("Running CARTS project install through dekk-managed environment setup.")
    print_info("Use `dekk carts install --wrap` if you also want a project-local `carts` wrapper.")

    try:
        _prepare_project_sources()
        _build_project()
    except RuntimeError as exc:
        print_error(str(exc))
        return 1

    print_header("Build Complete")
    print_success("CARTS build finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
