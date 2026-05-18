"""CARTS platform configuration and project path detection.

All environment setup (PATH, env vars, tool versions) is handled by dekk's
project runner, conda environment activation, and installed wrapper. This
module only provides:

* Project path detection (CARTS_DIR, install dirs)
* CARTS-specific compiler/linker flag construction
* Tool resolution helpers

Usage::

    from scripts.platform import CartsConfig, get_config

    config = get_config()
    config.get_llvm_tool("clang")
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dekk import PlatformDetector, PlatformInfo
from scripts.local_config import INSTALL_DIR_NAME, resolve_carts_home


@dataclass
class CartsConfig:
    """CARTS project paths and compiler flags.

    Auto-detects platform via ``dekk.PlatformDetector`` and configures:
    - Include paths for system headers and LLVM/ARTS libraries
    - Sysroot flags for cross-compilation support
    - Linker selection (ld64.lld on macOS, ld.lld on Linux)
    - Library paths and runtime dependencies
    """

    # Platform (from dekk)
    info: PlatformInfo

    # Project directories
    carts_dir: Path
    carts_home: Path
    install_dir: Path
    llvm_install_dir: Path
    polygeist_install_dir: Path
    arts_install_dir: Path
    carts_install_dir: Path

    # Include/library paths (auto-detected)
    llvm_include_path: Path = field(default_factory=Path)
    llvm_lib_path: Path = field(default_factory=Path)
    llvm_cxx_include_path: Path = field(default_factory=Path)
    macos_sdk_path: Path | None = None

    # Compiler flags
    include_flags: list[str] = field(default_factory=list)
    cgeist_sysroot_flags: list[str] = field(default_factory=list)
    clang_sysroot_flags: list[str] = field(default_factory=list)
    clang_library_flags: list[str] = field(default_factory=list)
    compile_library_flags: list[str] = field(default_factory=list)
    clang_libraries: list[str] = field(default_factory=list)
    compile_libraries: list[str] = field(default_factory=list)
    compile_flags: list[str] = field(default_factory=list)
    runtime_flags: list[str] = field(default_factory=list)
    linker_flags: list[str] = field(default_factory=list)
    linker_path: Path | None = None
    gcc_install_prefix: Path | None = None
    gcc_lib_path: Path | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls, script_dir: Path | None = None) -> CartsConfig:
        """Auto-detect platform and configure paths/flags."""
        if script_dir is None:
            script_dir = Path(__file__).parent.parent.resolve()

        info = PlatformDetector().detect()

        installed_bin_marker = f"{INSTALL_DIR_NAME}/carts/bin"
        if installed_bin_marker in str(script_dir):
            carts_dir = script_dir.parent.parent.parent
        else:
            carts_dir = script_dir.parent

        carts_home = resolve_carts_home(carts_dir)
        install_dir = carts_home / INSTALL_DIR_NAME

        config = cls(
            info=info,
            carts_dir=carts_dir,
            carts_home=carts_home,
            install_dir=install_dir,
            llvm_install_dir=install_dir / "llvm",
            polygeist_install_dir=install_dir / "polygeist",
            arts_install_dir=install_dir / "arts",
            carts_install_dir=install_dir / "carts",
        )

        config._setup_paths()
        config._setup_platform_flags()
        config._setup_common_flags()
        config._detect_linker()

        return config

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def _resolve_tool(self, install_dir: Path, name: str, *,
                      fallback_to_system: bool = False) -> Path:
        installed = install_dir / "bin" / name
        if installed.is_file() or not fallback_to_system:
            return installed
        system_path = shutil.which(name)
        return Path(system_path) if system_path else installed

    def get_llvm_tool(self, name: str, *, fallback_to_system: bool = False) -> Path:
        return self._resolve_tool(self.llvm_install_dir, name,
                                  fallback_to_system=fallback_to_system)

    def get_polygeist_tool(self, name: str) -> Path:
        return self._resolve_tool(self.polygeist_install_dir, name)

    def get_carts_tool(self, name: str) -> Path:
        return self._resolve_tool(self.carts_install_dir, name)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def clang_version_dir(self) -> Path:
        clang_lib = self.llvm_install_dir / "lib" / "clang"
        versions = sorted(clang_lib.iterdir()) if clang_lib.is_dir() else []
        return versions[-1] if versions else clang_lib / "18"

    # ------------------------------------------------------------------
    # Path and flag setup
    # ------------------------------------------------------------------

    def _setup_paths(self) -> None:
        self.llvm_include_path = self.clang_version_dir / "include"
        self.llvm_cxx_include_path = self.llvm_install_dir / "include" / "c++" / "v1"

        aarch64_path = self.llvm_install_dir / "lib" / "aarch64-unknown-linux-gnu"
        x86_64_path = self.llvm_install_dir / "lib" / "x86_64-unknown-linux-gnu"

        if aarch64_path.is_dir():
            self.llvm_lib_path = aarch64_path
        elif x86_64_path.is_dir():
            self.llvm_lib_path = x86_64_path
        else:
            self.llvm_lib_path = self.llvm_install_dir / "lib"

        self.macos_sdk_path = Path(
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

    def _setup_platform_flags(self) -> None:
        if self.info.is_macos:
            self._setup_macos_flags()
        elif self.info.is_linux:
            self._setup_linux_flags()
        elif self.info.is_windows:
            self._setup_windows_flags()

    def _setup_macos_flags(self) -> None:
        if self.macos_sdk_path and self.macos_sdk_path.exists():
            sys_include = self.macos_sdk_path / "usr" / "include"
            sys_cxx_include = sys_include / "c++" / "v1"

            self.include_flags.extend([
                f"-I{sys_include}",
                f"-I{sys_cxx_include}",
            ])

            self.cgeist_sysroot_flags = ["--sysroot", str(self.macos_sdk_path)]
            self.clang_sysroot_flags = ["-isysroot", str(self.macos_sdk_path)]

        if self.info.arch == "x86_64":
            self.compile_flags.extend(["-fno-pie", "-Wl,-no_pie"])

    def _setup_linux_flags(self) -> None:
        self._detect_gcc()

        self.include_flags.append("-I/usr/include")
        self.compile_flags.append(f"-I{self.llvm_cxx_include_path}")

        lib_flags = ["-L/usr/lib64", "-L/usr/lib"]
        if self.gcc_lib_path:
            lib_flags.append(f"-L{self.gcc_lib_path}")

        self.clang_library_flags.extend(lib_flags)
        self.compile_library_flags.extend(lib_flags)
        self.clang_libraries.extend(["-lpthread", "-lrt"])
        self.compile_libraries.extend(["-lpthread", "-lrt", "-ldl"])
        self.compile_flags.extend(["-fno-pie", "-no-pie"])
        self.runtime_flags.extend([
            "-stdlib=libc++", "-rtlib=compiler-rt", "--unwindlib=libunwind",
        ])

    def _detect_gcc(self) -> None:
        gcc = shutil.which("gcc")
        if not gcc:
            return

        try:
            result = subprocess.run(
                [gcc, "-print-search-dirs"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("install:"):
                        install_dir = line.split(":", 1)[1].strip().rstrip("/")
                        idx = install_dir.find("/lib/gcc/")
                        if idx > 0:
                            self.gcc_install_prefix = Path(install_dir[:idx])
                        break

            result = subprocess.run(
                [gcc, "-print-file-name=libstdc++.so"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "/" in result.stdout.strip():
                self.gcc_lib_path = Path(result.stdout.strip()).parent

        except subprocess.TimeoutExpired:
            pass

    def _setup_windows_flags(self) -> None:
        sdk_dir = os.environ.get(
            "WindowsSdkDir", os.environ.get("WINDOWSSDKDIR", ""))
        sdk_version = os.environ.get(
            "WindowsSDKVersion", os.environ.get("WINDOWSSDKVERSION", ""))
        vc_tools_dir = os.environ.get("VCToolsInstallDir", "")
        arch = os.environ.get("CARTS_WINDOWS_ARCH", "x64")

        if not sdk_dir or not sdk_version:
            raise RuntimeError(
                "Windows SDK environment not detected. Run from VS developer prompt.")
        if not vc_tools_dir:
            raise RuntimeError(
                "VCToolsInstallDir not set. Run from VS developer prompt.")

        sdk_dir = sdk_dir.rstrip("/\\")
        sdk_version = sdk_version.rstrip("/\\")
        vc_tools_dir = vc_tools_dir.rstrip("/\\")

        self.include_flags.extend([
            f"-I{sdk_dir}/Include/{sdk_version}/ucrt",
            f"-I{sdk_dir}/Include/{sdk_version}/shared",
            f"-I{sdk_dir}/Include/{sdk_version}/um",
            f"-I{vc_tools_dir}/include",
        ])

        self.clang_library_flags.extend([
            f"-L{sdk_dir}/Lib/{sdk_version}/ucrt/{arch}",
            f"-L{sdk_dir}/Lib/{sdk_version}/um/{arch}",
            f"-L{vc_tools_dir}/lib/{arch}",
        ])
        self.compile_library_flags.extend(self.clang_library_flags)

    def _setup_common_flags(self) -> None:
        self.include_flags.insert(0, f"-I{self.llvm_include_path}")
        self.include_flags.extend([
            f"-I{self.carts_install_dir}/include",
            f"-I{self.arts_install_dir}/include",
            f"-I{self.llvm_install_dir}/include",
        ])

        base_lib_flags = [
            f"-L{self.carts_install_dir}/lib",
            f"-L{self.carts_install_dir}/lib64",
            f"-L{self.arts_install_dir}/lib",
            f"-L{self.arts_install_dir}/lib64",
            f"-L{self.llvm_lib_path}",
            f"-L{self.llvm_install_dir}/lib",
            f"-L{self.llvm_install_dir}/lib64",
        ]
        self.clang_library_flags = base_lib_flags + self.clang_library_flags
        self.compile_library_flags = base_lib_flags + self.compile_library_flags

        base_libs = ["-lc++", "-lc++abi", "-lunwind"]
        self.clang_libraries = base_libs + self.clang_libraries
        self.compile_libraries = [
            "-lcartstest", "-lcartsbenchmarks", "-larts", "-lomp",
        ] + base_libs + [lib for lib in self.compile_libraries if lib not in base_libs]

        self.compile_flags.insert(0, "-O3")
        self.compile_flags.extend(self.runtime_flags)

    def _detect_linker(self) -> None:
        linker_candidates = []

        if self.info.is_macos:
            linker_candidates = [
                self.get_llvm_tool("ld64.lld"),
                self.get_llvm_tool("lld"),
            ]
        elif self.info.is_linux:
            linker_candidates = [
                self.get_llvm_tool("ld.lld"),
                self.get_llvm_tool("lld"),
            ]
        elif self.info.is_windows:
            linker_candidates = [
                self.get_llvm_tool("lld-link.exe"),
                self.get_llvm_tool("lld-link"),
            ]

        for candidate in linker_candidates:
            if candidate.is_file():
                self.linker_path = candidate
                break

        if self.linker_path:
            if self.info.is_windows:
                self.linker_flags.extend(
                    ["-fuse-ld=lld", f"--ld-path={self.linker_path}"])
            else:
                self.linker_flags.append(f"--ld-path={self.linker_path}")

        if self.info.is_linux or self.info.is_macos:
            if self.info.is_linux:
                # Use DT_RUNPATH so Slurm jobs can select the per-artifact ARTS
                # runtime snapshot through LD_LIBRARY_PATH without LD_PRELOAD.
                self.linker_flags.append("-Wl,--enable-new-dtags")

            rpaths = [
                self.carts_install_dir / "lib",
                self.arts_install_dir / "lib",
                self.llvm_install_dir / "lib",
                self.carts_install_dir / "lib64",
                self.arts_install_dir / "lib64",
                self.llvm_install_dir / "lib64",
                self.llvm_lib_path,
            ]

            existing: set[str] = set()
            for rp in rpaths:
                rpath_flag = f"-Wl,-rpath,{rp}"
                if rpath_flag not in existing:
                    self.linker_flags.append(rpath_flag)
                    existing.add(rpath_flag)


# ============================================================================
# Global State
# ============================================================================

_config: CartsConfig | None = None
_verbose: bool = False


def get_config() -> CartsConfig:
    """Get or create the platform configuration (singleton)."""
    global _config
    if _config is None:
        _config = CartsConfig.detect()
    return _config


def set_verbose(verbose: bool) -> None:
    global _verbose
    _verbose = verbose


def is_verbose() -> bool:
    return _verbose
