"""Local, untracked CARTS workspace configuration."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ENV_CARTS_HOME = "CARTS_HOME"
ENV_CARTS_BUILD_ROOT = "CARTS_BUILD_ROOT"
ENV_CARTS_CONFIG = "CARTS_CONFIG"
ENV_CARTS_INSTALL_ROOT = "CARTS_INSTALL_ROOT"
LOCAL_CONFIG_FILE = "carts.config"
INSTALL_DIR_NAME = ".install"
BUILD_DIR_NAME = "build"

_SHARED_LIBRARY_SUFFIXES = (".so", ".dylib", ".dll")
_LLVM_RUNTIME_LIBRARY_PREFIXES = (
    "libc++",
    "libc++abi",
    "libunwind",
    "libomp",
    "libompd",
    "libarcher",
)


@dataclass(frozen=True)
class RuntimePlatformInfo:
    is_linux: bool
    is_macos: bool
    is_windows: bool


def resolve_carts_home(carts_dir: Path) -> Path:
    """Resolve the generated-artifact root for this checkout.

    Precedence:
    1. CARTS_HOME environment variable
    2. CARTS_CONFIG local workspace file, defaulting to ./carts.config
    3. project checkout root
    """
    raw_home = os.environ.get(ENV_CARTS_HOME) or _read_config_home(carts_dir)
    if not raw_home:
        return carts_dir

    carts_home = Path(raw_home).expanduser()
    if not carts_home.is_absolute():
        carts_home = carts_dir / carts_home
    return carts_home.resolve()


def resolve_install_dir(carts_dir: Path) -> Path:
    """Resolve the active CARTS install root for this checkout."""
    raw_install = os.environ.get(ENV_CARTS_INSTALL_ROOT) or _read_config_path_value(
        carts_dir,
        "install",
        "install_dir",
        "install_root",
    )
    if raw_install:
        return _resolve_workspace_path(carts_dir, raw_install)
    return resolve_carts_home(carts_dir) / INSTALL_DIR_NAME


def resolve_build_dir(carts_dir: Path) -> Path:
    """Resolve the active CARTS build root for this checkout."""
    raw_build = os.environ.get(ENV_CARTS_BUILD_ROOT) or _read_config_path_value(
        carts_dir,
        "build",
        "build_dir",
        "build_root",
    )
    if raw_build:
        return _resolve_workspace_path(carts_dir, raw_build)
    return resolve_carts_home(carts_dir) / BUILD_DIR_NAME


def resolve_subproject_install_dir(carts_dir: Path, name: str) -> Path:
    """Resolve ``CARTS_HOME/.install/<name>`` for a managed subproject."""
    return resolve_install_dir(carts_dir) / name


def resolve_subproject_build_dir(carts_dir: Path, name: str) -> Path:
    """Resolve ``CARTS_HOME/build/<name>`` for a managed subproject."""
    return resolve_build_dir(carts_dir) / name


def discover_llvm_runtime_library_dirs(llvm_install_dir: Path) -> list[Path]:
    """Return LLVM runtime library directories without assuming a target triple.

    LLVM installs libc++/libomp runtime libraries either directly under
    ``lib``/``lib64`` or under a target-triple subdirectory such as
    ``x86_64-unknown-linux-gnu``.  Discover the target directories by contents
    instead of architecture-specific names so Linux and macOS installs use the
    same rule.
    """
    lib_dir = llvm_install_dir / "lib"
    lib64_dir = llvm_install_dir / "lib64"
    target_dirs: list[Path] = []
    if lib_dir.is_dir():
        for child in sorted(lib_dir.iterdir()):
            if child.is_dir() and _contains_llvm_runtime_library(child):
                target_dirs.append(child)

    return _existing_unique_dirs([*target_dirs, lib_dir, lib64_dir])


def managed_runtime_library_dirs(
    carts_dir: Path,
    *,
    include_existing_env: bool = True,
    platform_info: Any | None = None,
) -> list[Path]:
    """Return shared-library dirs managed by Dekk/CARTS for launched binaries.

    The list intentionally points at the active install tree instead of copying
    libraries into benchmark artifacts.  Managed dirs are ordered before
    inherited environment paths so the selected CARTS install wins while still
    preserving caller-provided dependencies.
    """
    info = platform_info or detect_platform_info()
    install_dir = resolve_install_dir(carts_dir)
    candidates: list[Path] = [
        install_dir / "carts" / "lib",
        install_dir / "carts" / "lib64",
        install_dir / "arts" / "lib",
        install_dir / "arts" / "lib64",
    ]
    candidates.extend(discover_llvm_runtime_library_dirs(install_dir / "llvm"))
    if is_windows_platform(info):
        candidates.extend(
            [
                install_dir / "carts" / "bin",
                install_dir / "arts" / "bin",
                install_dir / "llvm" / "bin",
            ]
        )

    candidates.append(Path(sys.prefix) / "lib")
    if is_windows_platform(info):
        candidates.extend(
            [
                Path(sys.prefix) / "DLLs",
                Path(sys.prefix) / "Library" / "bin",
                Path(sys.prefix) / "bin",
            ]
        )
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(conda_prefix) / "lib")
        if is_windows_platform(info):
            candidates.extend(
                [
                    Path(conda_prefix) / "DLLs",
                    Path(conda_prefix) / "Library" / "bin",
                    Path(conda_prefix) / "bin",
                ]
            )
    if include_existing_env:
        candidates.extend(_env_path_dirs(*runtime_library_env_vars(info)))
    return _existing_unique_dirs(candidates)


def managed_runtime_library_env(
    carts_dir: Path,
    *,
    include_existing_env: bool = True,
    platform_info: Any | None = None,
) -> str:
    return os.pathsep.join(
        str(path)
        for path in managed_runtime_library_dirs(
            carts_dir,
            include_existing_env=include_existing_env,
            platform_info=platform_info,
        )
    )


def prepend_runtime_library_env(
    carts_dir: Path,
    *,
    include_existing_env: bool = False,
    platform_info: Any | None = None,
) -> None:
    """Prepend managed runtime dirs to the platform dynamic-library env vars."""
    info = platform_info or detect_platform_info()
    paths = [
        str(path)
        for path in managed_runtime_library_dirs(
            carts_dir,
            include_existing_env=include_existing_env,
            platform_info=info,
        )
    ]
    if not paths:
        return

    for env_var in runtime_library_env_vars(info):
        existing = [
            entry for entry in os.environ.get(env_var, "").split(os.pathsep) if entry
        ]
        merged: list[str] = []
        seen: set[str] = set()
        for entry in [*paths, *existing]:
            if entry in seen:
                continue
            merged.append(entry)
            seen.add(entry)
        os.environ[env_var] = os.pathsep.join(merged)


def detect_platform_info() -> Any | None:
    """Return enough platform information for runtime library path setup.

    The richer Dekk ``PlatformInfo`` is passed in by ``scripts.platform`` for
    CLI commands.  This lightweight local detector keeps ``local_config``
    importable from lit and shims, where ``tools/scripts/platform.py`` may be
    importable as a top-level ``platform`` module and would otherwise shadow
    Python's standard library during Dekk imports.
    """
    return RuntimePlatformInfo(
        is_linux=sys.platform.startswith("linux"),
        is_macos=sys.platform == "darwin",
        is_windows=os.name == "nt",
    )


def is_macos_platform(platform_info: Any | None = None) -> bool:
    info = platform_info
    if info is not None and hasattr(info, "is_macos"):
        return bool(info.is_macos)
    return sys.platform == "darwin"


def is_linux_platform(platform_info: Any | None = None) -> bool:
    info = platform_info
    if info is not None and hasattr(info, "is_linux"):
        return bool(info.is_linux)
    return sys.platform.startswith("linux")


def runtime_library_env_vars(platform_info: Any | None = None) -> tuple[str, ...]:
    """Return dynamic-library search env vars for the current platform.

    On macOS we use ``DYLD_FALLBACK_LIBRARY_PATH`` rather than
    ``DYLD_LIBRARY_PATH``.  ``DYLD_LIBRARY_PATH`` is searched *before* a binary's
    install names/rpaths, so the managed ``.../lib`` dir shadows the system
    ``/usr/lib/libc++abi.dylib``.  The managed libc++abi (built from the in-tree
    libcxxabi) omits the C++26 type-aware allocation operators
    (``operator new/new[](size_t, std::__type_descriptor_t)``) that the
    bootstrap compiler emits weak references to, so shadowing the complete
    system libc++abi makes every managed tool abort with a dyld "Symbol not
    found: __ZnamSt19__type_descriptor_t".  The fallback variable is searched
    *after* install names/rpaths, so managed libc++/libomp/libarts still resolve
    via each binary's rpath while the system libc++abi provides the type-aware
    operators.  Do not switch this back to ``DYLD_LIBRARY_PATH`` without also
    fixing the managed libc++abi to export those operators.
    """
    info = platform_info or detect_platform_info()
    if is_windows_platform(info):
        return ("PATH",)
    if is_macos_platform(info):
        return ("DYLD_FALLBACK_LIBRARY_PATH",)
    if is_linux_platform(info):
        return ("LD_LIBRARY_PATH",)
    return ("DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH")


def is_windows_platform(platform_info: Any | None = None) -> bool:
    info = platform_info
    if info is not None and hasattr(info, "is_windows"):
        return bool(info.is_windows)
    return os.name == "nt"


def _read_config_home(carts_dir: Path) -> str | None:
    return _read_config_path_value(carts_dir, "home", "carts_home")


def _read_config_path_value(carts_dir: Path, *keys: str) -> str | None:
    config_path = _resolve_config_path(carts_dir)
    if not config_path.is_file():
        return None

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    value = _lookup_path_value(data, *keys)
    return str(value) if value else None


def _lookup_home_value(data: dict[str, Any]) -> Any:
    return _lookup_path_value(data, "home", "carts_home")


def _lookup_path_value(data: dict[str, Any], *keys: str) -> Any:
    env_aliases = {
        "home": ENV_CARTS_HOME,
        "carts_home": ENV_CARTS_HOME,
        "build": ENV_CARTS_BUILD_ROOT,
        "build_dir": ENV_CARTS_BUILD_ROOT,
        "build_root": ENV_CARTS_BUILD_ROOT,
        "install": ENV_CARTS_INSTALL_ROOT,
        "install_dir": ENV_CARTS_INSTALL_ROOT,
        "install_root": ENV_CARTS_INSTALL_ROOT,
    }

    for key in keys:
        env_key = env_aliases.get(key)
        if env_key and env_key in data:
            return data[env_key]
        if key in data:
            return data[key]

    carts_section = data.get("carts")
    if isinstance(carts_section, dict):
        for key in keys:
            value = carts_section.get(key)
            if value:
                return value

    paths_section = data.get("paths")
    if isinstance(paths_section, dict):
        for key in keys:
            value = paths_section.get(key)
            if value:
                return value

    return None


def _resolve_workspace_path(carts_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = carts_dir / path
    return path.resolve()


def _resolve_config_path(carts_dir: Path) -> Path:
    raw = os.environ.get(ENV_CARTS_CONFIG)
    if not raw:
        return carts_dir / LOCAL_CONFIG_FILE

    config_path = Path(raw).expanduser()
    if not config_path.is_absolute():
        config_path = carts_dir / config_path
    return config_path.resolve()


def _env_path_dirs(*var_names: str) -> list[Path]:
    paths: list[Path] = []
    for var_name in var_names:
        for entry in os.environ.get(var_name, "").split(os.pathsep):
            if entry:
                paths.append(Path(entry))
    return paths


def _existing_unique_dirs(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        result.append(resolved)
        seen.add(resolved)
    return result


def _contains_llvm_runtime_library(path: Path) -> bool:
    for child in path.iterdir():
        if not child.is_file() and not child.is_symlink():
            continue
        name = child.name
        if not name.startswith(_LLVM_RUNTIME_LIBRARY_PREFIXES):
            continue
        if _has_shared_library_suffix(name):
            return True
    return False


def _has_shared_library_suffix(name: str) -> bool:
    return any(suffix in name for suffix in _SHARED_LIBRARY_SUFFIXES)
