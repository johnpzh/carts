"""Clean command logic for CARTS CLI."""

from pathlib import Path
import os
import shutil

from dekk import print_debug, print_info, print_success, print_error
from scripts.platform import get_config, is_verbose
from scripts import run_subprocess
from scripts.build_env import build_parallel_env, configured_make_command


# Files to clean
CLEAN_FILE_PATTERNS = [
    "*.mlir", "*.ll", "*.o", "*.so", "*.dylib", "*.a", "*.exe",
    "*.log", "*.tmp", ".artsPrintLock",
    ".carts-metadata.json", "*_arts_metadata.mlir", "core",
]

CLEAN_DIR_PATTERNS = [
    "build", "cmake-build-*", ".cache", "introspection", "counter", "counters",
]


def run_local_clean() -> None:
    """Clean local generated files."""
    print_info("Cleaning generated files in current directory...")

    removed_count = 0
    cwd = Path.cwd()

    # Remove files matching patterns
    for pattern in CLEAN_FILE_PATTERNS:
        for f in cwd.glob(pattern):
            if f.is_file():
                if is_verbose():
                    print_debug(f"Removing file: {f}")
                f.unlink()
                removed_count += 1

    # Remove directories matching patterns
    for pattern in CLEAN_DIR_PATTERNS:
        for d in cwd.glob(pattern):
            if d.is_dir():
                if is_verbose():
                    print_debug(f"Removing directory: {d}")
                shutil.rmtree(d)
                removed_count += 1

    # Remove executables that look like CARTS compilation outputs
    # Only remove files ending with _arts or _omp (our naming convention)
    for pattern in ["*_arts", "*_omp"]:
        for f in cwd.glob(pattern):
            if f.is_file() and os.access(f, os.X_OK):
                if is_verbose():
                    print_debug(f"Removing CARTS executable: {f}")
                f.unlink()
                removed_count += 1

    if removed_count == 0:
        print_info("No generated files found to clean")
    else:
        print_success(f"Cleaned {removed_count} files/directories")


def run_full_clean() -> None:
    """Clean all build directories (LLVM, Polygeist, ARTS, CARTS)."""
    config = get_config()
    carts_dir = config.carts_dir
    parallel_env = build_parallel_env()

    targets = ["llvm-clean", "polygeist-clean", "arts-clean", "clean"]

    for target in targets:
        print_info(f"Running make {target}...")
        result = run_subprocess(
            configured_make_command(config, target),
            cwd=carts_dir,
            env=parallel_env,
            check=False,
        )
        if result.returncode != 0:
            print_error(f"make {target} failed (exit {result.returncode})")

    print_success("All build directories cleaned")
