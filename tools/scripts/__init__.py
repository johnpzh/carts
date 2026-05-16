"""CARTS CLI Scripts - Modular command implementations.

Environment setup (conda activation, PATH, env vars, wrapper installation) is
handled by dekk's project runner or the installed `carts` wrapper. Scripts only
need to resolve project paths and execute domain logic.
"""

import os
import subprocess
import sys
from pathlib import Path

from dekk import Colors, Exit, Symbols, console, print_debug, print_error

from scripts.platform import is_verbose

# ---------------------------------------------------------------------------
# Shared constants — single source of truth for paths, tools, and filenames
# ---------------------------------------------------------------------------

# Submodule paths (relative to carts_dir)
SUBMODULE_ARTS = "external/arts"
SUBMODULE_POLYGEIST = "external/Polygeist"
SUBMODULE_BENCHMARKS = "external/carts-benchmarks"

# Nested submodules required for each top-level submodule
ARTS_NESTED_SUBMODULES = ["third_party/hwloc", "third_party/jemalloc"]
POLYGEIST_NESTED_SUBMODULES = ["llvm-project"]

# Tool binary names
TOOL_CARTS_COMPILE = "carts-compile"
TOOL_CGEIST = "cgeist"
TOOL_CLANG = "clang"
TOOL_CLANG_FORMAT = "clang-format"
TOOL_LLVM_LIT = "llvm-lit"
TOOL_FILECHECK = "FileCheck"

# Config / manifest filenames
ARTS_CONFIG_FILENAME = "arts.cfg"
PIPELINE_MANIFEST_FILENAME = "pipeline-manifest.json"

# Makefile targets
MAKE_TARGET_BUILD = "build"
MAKE_TARGET_ARTS = "arts"
MAKE_TARGET_POLYGEIST = "polygeist"
MAKE_TARGET_LLVM = "llvm"

# ---------------------------------------------------------------------------
# Status formatting — maps status strings to Rich-styled symbols via dekk
# ---------------------------------------------------------------------------

_STATUS_STYLES: dict[str, tuple[str, str]] = {
    "PASS":    (Colors.SUCCESS, Symbols.PASS),
    "FAIL":    (Colors.ERROR,   Symbols.FAIL),
    "CRASH":   (Colors.ERROR,   Symbols.FAIL),
    "SKIP":    (Colors.WARNING, Symbols.SKIP),
    "TIMEOUT": (Colors.WARNING, Symbols.TIMEOUT),
}


def status_symbol(status_value: str) -> str:
    """Return a Rich-styled symbol for a test/example status string."""
    entry = _STATUS_STYLES.get(status_value.upper())
    if entry is None:
        return f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
    color, symbol = entry
    return f"[{color}]{symbol}[/{color}]"


def format_passed(count: int) -> str:
    return f"[{Colors.SUCCESS}]{Symbols.PASS} {count}[/{Colors.SUCCESS}] passed"


def format_failed(count: int) -> str:
    return f"[{Colors.ERROR}]{Symbols.FAIL} {count}[/{Colors.ERROR}] failed"


def format_skipped(count: int) -> str:
    return f"[{Colors.DEBUG}]{Symbols.SKIP} {count}[/{Colors.DEBUG}] skipped"


def format_summary_line(passed: int, failed: int, skipped: int) -> str:
    return f"{format_passed(passed)}  {format_failed(failed)}  {format_skipped(skipped)}"


def print_footer(title: str, style: str = Colors.SUCCESS) -> None:
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    console.print()
    console.print(Panel(Text.from_markup(title, style=style),
                        box=box.HEAVY, border_style=style, padding=(0, 1)))
    console.print()


def run_subprocess(
    cmd: list[str],
    capture_output: bool = False,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    realtime: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess with optional output capture.

    This is the shared subprocess helper used by all CARTS scripts.
    For long-running operations, prefer ``dekk.run_logged`` instead.
    """
    if is_verbose():
        print_debug(f"Running: {' '.join(str(c) for c in cmd)}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if realtime:
        process = subprocess.Popen(
            cmd, cwd=cwd, env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            sys.stdout.flush()
        process.wait()
        result = subprocess.CompletedProcess(
            cmd, process.returncode, stdout="", stderr="")
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result

    result = subprocess.run(
        cmd, capture_output=capture_output, check=check,
        cwd=cwd, env=merged_env,
        text=True if capture_output else None,
        timeout=timeout,
    )
    return result


def run_command_with_output(
    cmd: list[str],
    output_file: Path | None = None,
    cwd: Path | None = None,
) -> int:
    """Run a command, optionally redirecting stdout to a file."""
    if is_verbose():
        if output_file:
            print_debug(f"Running: {' '.join(str(c) for c in cmd)} > {output_file}")
        else:
            print_debug(f"Running: {' '.join(str(c) for c in cmd)}")

    try:
        if output_file:
            with open(output_file, 'w') as f:
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.PIPE, cwd=cwd, text=True)
                if result.returncode != 0 and result.stderr:
                    print_error(result.stderr)
                return result.returncode
        else:
            result = subprocess.run(cmd, cwd=cwd)
            return result.returncode
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return 1
    except Exception as e:
        print_error(f"Command failed: {e}")
        return 1


def carts_cli_path() -> Path:
    """Return the canonical CLI script path."""
    return Path(__file__).resolve().parent.parent / "carts_cli.py"


def carts_cli_command(*args: str) -> list[str]:
    """Build a command line that runs the CARTS CLI with the current Python."""
    return [sys.executable, str(carts_cli_path()), *args]
