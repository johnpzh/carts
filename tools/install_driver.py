#!/usr/bin/env python3
"""Project install driver for `dekk carts install`.

This script is intentionally small and imperative. Dekk owns the outer install
flow; this driver owns the CARTS-specific work that dekk cannot infer:

1. create configured build/install directories
2. initialize the required submodules
3. generate agent-facing skills, hooks, MCP config, and editor adapters
4. build LLVM, Polygeist, ARTS, and CARTS in order
"""

from __future__ import annotations

import json
import sys
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

CARTS_MCP_SERVER = "carts"


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
        config.build_dir,
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


def _sync_agent_resources() -> None:
    config = get_config()
    project_root = config.carts_dir

    print_step("Generating agent skills and MCP resources...")
    _run(
        ["dekk", "carts", "skills", "generate"],
        cwd=project_root,
        label="agent resource generation",
    )
    _use_managed_python_for_mcp(project_root)


def _use_managed_python_for_mcp(project_root: Path) -> None:
    managed_python = str(Path(sys.executable).resolve())
    print_step(f"Configuring generated MCP launchers to use {managed_python}...")

    for relative in [
        "carts-plugin/.mcp.json",
        ".claude/settings.json",
        ".cursor/mcp.json",
        ".github/extensions/carts.json",
    ]:
        _rewrite_json_mcp_command(project_root / relative, managed_python)

    _rewrite_openai_agent_yaml(project_root / "agents/openai.yaml", managed_python)


def _rewrite_json_mcp_command(path: Path, managed_python: str) -> None:
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    servers = data.get("mcpServers")
    if isinstance(servers, dict):
        server = servers.get(CARTS_MCP_SERVER)
        if isinstance(server, dict) and server.get("command") != managed_python:
            server["command"] = managed_python
            changed = True

    mcp = data.get("mcp")
    if isinstance(mcp, dict) and mcp.get("command") != managed_python:
        mcp["command"] = managed_python
        changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _rewrite_openai_agent_yaml(path: Path, managed_python: str) -> None:
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    updated = text.replace("  command: python3\n", f"  command: {managed_python}\n")
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main() -> int:
    print_header("CARTS Install")
    print_info("Running CARTS project install through dekk-managed environment setup.")
    print_info("Use `dekk carts install --wrap` if you also want a project-local `carts` wrapper.")

    try:
        _prepare_project_sources()
        _sync_agent_resources()
        _build_project()
    except RuntimeError as exc:
        print_error(str(exc))
        return 1

    print_header("Build Complete")
    print_success("CARTS build finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
