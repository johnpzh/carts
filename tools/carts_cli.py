#!/usr/bin/env python3
"""CARTS CLI - Compiler for Asynchronous Runtime Systems."""

from pathlib import Path
import os
import sys

from dekk import Context, Exit, Option, Typer, print_error
from scripts.local_config import (
    prepend_runtime_library_env,
    resolve_install_dir,
)


def _bootstrap_runtime_library_paths() -> None:
    carts_dir = Path(__file__).resolve().parent.parent
    install_dir = resolve_install_dir(carts_dir)
    prepend_runtime_library_env(carts_dir, include_existing_env=False)
    llvm_symbolizer = install_dir / "llvm" / "bin" / "llvm-symbolizer"
    if llvm_symbolizer.is_file():
        os.environ["LLVM_SYMBOLIZER_PATH"] = str(llvm_symbolizer)


_bootstrap_runtime_library_paths()

from scripts.platform import get_config, set_verbose
from scripts import run_subprocess, SUBMODULE_BENCHMARKS
from scripts.build import build as build_cmd
from scripts.compile import (
    compile_cmd,
    pipeline as pipeline_cmd,
    cgeist,
    clang,
    mlir_translate as mlir_translate_cmd,
)
from scripts.docker import (
    docker_callback,
    docker_build,
    docker_start,
    docker_update,
    docker_stop,
    docker_clean as docker_clean_cmd,
    docker_commit,
    docker_status,
    docker_exec,
)
from scripts.test import test as test_cmd, lit as lit_cmd
from scripts.format import format_sources as format_cmd
from scripts.clean import run_local_clean, run_full_clean
from scripts.update import update as update_cmd
from scripts.examples import examples_app
from scripts.triage import triage_benchmark


# ============================================================================
# CLI Application
# ============================================================================

HELP_CTX = {"help_option_names": ["--help", "-h"]}
PASSTHROUGH_CTX = {
    **HELP_CTX,
    "allow_extra_args": True,
    "allow_interspersed_args": True,
    "ignore_unknown_options": True,
}
PASSTHROUGH_NO_INTERSPERSE_CTX = {
    **HELP_CTX,
    "allow_extra_args": True,
    "allow_interspersed_args": False,
    "ignore_unknown_options": True,
}

VERSION = "0.1.0"

app = Typer(
    name="carts",
    auto_activate=False,
    # Avoid eager dekk ExecutionContext capture for every command. The
    # compiler/benchmark commands do not consume that metadata, and the
    # hardware probe path can stall on systems where rocm-smi misbehaves.
    # Built-in commands that need the context still request it explicitly.
    auto_capture_env=False,
    fail_fast=False,
    add_doctor_command=True,
    add_version_command=True,
    project_version=VERSION,
    help="CARTS - Compiler for Asynchronous Runtime Systems",
    no_args_is_help=True,
    add_env_command=True,
    add_completion=False,
    context_settings=HELP_CTX,
)

# Docker subcommand group
docker_app = Typer(
    help="Docker operations for multi-node execution",
    context_settings=HELP_CTX,
)
app.add_typer(docker_app, name="docker")

# Examples subcommand group
app.add_typer(examples_app, name="examples", context_settings=HELP_CTX)


# ============================================================================
# Command Registration
# ============================================================================

# Build
app.command()(build_cmd)

# Compile pipeline
app.command(
    name="compile",
    context_settings=PASSTHROUGH_CTX,
)(compile_cmd)
app.command(name="pipeline")(pipeline_cmd)

app.command(
    name="cgeist",
    context_settings=PASSTHROUGH_NO_INTERSPERSE_CTX
)(cgeist)

app.command(
    name="clang",
    context_settings=PASSTHROUGH_NO_INTERSPERSE_CTX
)(clang)

app.command(name="mlir-translate", context_settings=PASSTHROUGH_NO_INTERSPERSE_CTX)(mlir_translate_cmd)
app.command(name="triage-benchmark")(triage_benchmark)

# Testing
app.command(name="test")(test_cmd)
app.command(
    name="lit",
    context_settings=PASSTHROUGH_NO_INTERSPERSE_CTX,
)(lit_cmd)

# Formatting
app.command(name="format")(format_cmd)

# Update submodules
app.command(name="update")(update_cmd)

# Docker subcommands
docker_app.callback(invoke_without_command=True)(docker_callback)
docker_app.command(name="build")(docker_build)
docker_app.command(name="start")(docker_start)
docker_app.command(name="update")(docker_update)
docker_app.command(name="stop")(docker_stop)
docker_app.command(name="clean")(docker_clean_cmd)
docker_app.command(name="commit")(docker_commit)
docker_app.command(name="status")(docker_status)
docker_app.command(
    name="exec",
    context_settings={**HELP_CTX, "allow_extra_args": True, "allow_interspersed_args": False},
)(docker_exec)


# ============================================================================
# Command Callbacks
# ============================================================================

@app.callback(invoke_without_command=True)
def main_callback(
    verbose: bool = Option(
        False, "--verbose", "-v", help="Enable verbose output"),
):
    """CARTS - Compiler for Asynchronous Runtime Systems"""
    set_verbose(verbose)
    if verbose:
        os.environ["CARTS_VERBOSE"] = "1"


# ============================================================================
# Benchmarks Command
# ============================================================================

@app.command(
    context_settings=PASSTHROUGH_NO_INTERSPERSE_CTX
)
def benchmarks(
    ctx: Context,
):
    """Build and manage CARTS benchmarks.

    Commands: list, run, build, clean

    Examples:
      carts benchmarks list
      carts benchmarks run polybench/2mm --size small
      carts benchmarks build --suite polybench
      carts benchmarks clean --all
    """
    config = get_config()
    benchmark_runner = (
        config.carts_dir / SUBMODULE_BENCHMARKS / "scripts" / "runner.py"
    )

    if not benchmark_runner.is_file():
        print_error(f"Benchmark runner not found at {benchmark_runner}")
        raise Exit(1)

    # Use the current interpreter from the activated dekk/conda environment.
    cmd = [sys.executable, str(benchmark_runner)]

    # Pass through all extra args (e.g., run, --verbose, --size, or `-- --help`)
    if ctx.args:
        cmd.extend(ctx.args)

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# Clean Command
# ============================================================================

@app.command()
def clean(
    all_builds: bool = Option(
        False, "--all", "-a", help="Clean all build directories (LLVM, Polygeist, ARTS, CARTS)"),
    docker_clean: bool = Option(
        False, "--docker", "-d", help="Clean Docker artifacts"),
):
    """Clean generated files in current directory."""
    if all_builds:
        run_full_clean()
    elif docker_clean:
        docker_clean_cmd()
    else:
        run_local_clean()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    app(prog_name="carts")


if __name__ == "__main__":
    main()
