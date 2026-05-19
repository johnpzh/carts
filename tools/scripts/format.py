"""Format command for CARTS CLI."""

from pathlib import Path
from typing import Iterable, List, Optional
import subprocess

from dekk import Argument, Exit, Option, console, print_error, print_info, print_success, print_warning
from scripts.platform import CartsConfig, get_config
from scripts import run_subprocess, TOOL_CLANG_FORMAT

# Source files handled by `carts format`. Only C/C++ are formatted; TableGen
# (.td) files use different syntax and must not be passed to clang-format.
FORMAT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc"
}
FORMAT_EXCLUDED_DIRS = {
    ".git", ".install", "build", "external", "third_party"
}


def _is_format_candidate(path: Path) -> bool:
    if path.suffix not in FORMAT_SUFFIXES:
        return False
    for part in path.parts:
        if part in FORMAT_EXCLUDED_DIRS:
            return False
    return True


def _collect_tracked_format_files(config: CartsConfig) -> List[Path]:
    """Collect tracked source files to format."""
    try:
        result = subprocess.run(
            ["git", "-C", str(config.carts_dir), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    files: List[Path] = []
    for line in result.stdout.splitlines():
        rel = Path(line.strip())
        if not rel:
            continue
        if _is_format_candidate(rel):
            abs_path = config.carts_dir / rel
            if abs_path.is_file():
                files.append(abs_path)
    return files


def _collect_format_files_from_paths(paths: Iterable[Path]) -> List[Path]:
    """Collect format targets from user-provided files/directories."""
    files: List[Path] = []
    for input_path in paths:
        path = input_path.resolve()
        if not path.exists():
            continue
        if path.is_file():
            if _is_format_candidate(path):
                files.append(path)
            continue
        for file_path in path.rglob("*"):
            if file_path.is_file() and _is_format_candidate(file_path):
                files.append(file_path)
    return files


def _resolve_clang_format(config: CartsConfig) -> Optional[Path]:
    """Find clang-format in the active CARTS install tree."""
    tool = config.get_llvm_tool(TOOL_CLANG_FORMAT)
    return tool if tool.is_file() else None


def format_sources(
    paths: Optional[List[Path]] = Argument(
        None,
        help=(
            "Files or directories to format. If omitted, formats tracked "
            "C/C++ sources in the repo."
        ),
    ),
    check_only: bool = Option(
        False, "--check", help="Check formatting without modifying files"),
    list_files: bool = Option(
        False, "--list", help="Print the files selected for formatting"),
):
    """Format source files with clang-format."""
    config = get_config()
    clang_format = _resolve_clang_format(config)
    if not clang_format:
        print_error(
            f"clang-format not found in {config.llvm_install_dir / 'bin'}."
        )
        print_info("Run `dekk carts build` and retry.")
        raise Exit(1)

    selected_paths = paths or []
    if selected_paths:
        files = _collect_format_files_from_paths(selected_paths)
    else:
        files = _collect_tracked_format_files(config)

    deduped = sorted({f.resolve() for f in files})
    if not deduped:
        print_warning("No eligible source files found for clang-format.")
        raise Exit(0)

    if list_files:
        for file_path in deduped:
            console.print(str(file_path))

    print_info(
        f"{'Checking' if check_only else 'Formatting'} {len(deduped)} files with {clang_format}"
    )

    failed = 0
    for file_path in deduped:
        cmd = [str(clang_format)]
        if check_only:
            cmd.extend(["--dry-run", "--Werror"])
        else:
            cmd.append("-i")
        cmd.append(str(file_path))
        result = run_subprocess(cmd, check=False)
        if result.returncode != 0:
            failed += 1

    if failed > 0:
        action = "check" if check_only else "format"
        print_error(f"clang-format {action} failed for {failed} file(s).")
        raise Exit(1)

    if check_only:
        print_success("Formatting check passed.")
    else:
        print_success("Formatting completed successfully.")
