"""Build command for CARTS CLI."""

import shutil
from pathlib import Path
from typing import Optional

from dekk import (
    Colors,
    console,
    print_header,
    print_step,
    print_error,
    print_success,
    DependencyChecker,
    DependencySpec,
    Exit,
    Option,
)
from scripts.platform import get_config
from scripts import (
    run_subprocess,
    MAKE_TARGET_ARTS,
    MAKE_TARGET_BUILD,
    MAKE_TARGET_LLVM,
    MAKE_TARGET_POLYGEIST,
    SUBMODULE_BENCHMARKS,
)

_CLEAN_TARGETS = {
    MAKE_TARGET_BUILD: "clean",
    MAKE_TARGET_ARTS: "arts-clean",
    MAKE_TARGET_POLYGEIST: "polygeist-clean",
    MAKE_TARGET_LLVM: "llvm-clean",
}


# Counter profile mapping: level -> config file name
COUNTER_PROFILES = {
    0: "profile-none.cfg",       # All OFF - baseline performance
    1: "profile-timing.cfg",     # Timing only - minimal overhead (DEFAULT)
    2: "profile-workload.cfg",   # Workload characterization at CLUSTER level
    3: "profile-overhead.cfg",   # Full overhead analysis at CLUSTER level
}

def build(
    clean: bool = Option(False, "--clean", "-c", help="Run make clean before building"),
    arts: bool = Option(False, "--arts", "-a", help="Build only ARTS (mutually exclusive target flag)"),
    polygeist: bool = Option(
        False, "--polygeist", "-p", help="Build only Polygeist (mutually exclusive target flag)"),
    llvm: bool = Option(False, "--llvm", "-l", help="Build only LLVM (mutually exclusive target flag)"),
    debug_level: int = Option(
        1, "--debug", min=0, max=3,
        help="ARTS log level (--arts only): 0=error, 1=warn, 2=info, 3=debug"),
    counters_level: int = Option(
        0, "--counters",
        help="Counter profile (--arts only): 0=off, 1=timing, 2=workload, 3=overhead"),
    profile: Optional[Path] = Option(
        None, "--profile",
        help="Custom counter profile file path (overrides --counters)"),
    rdma: bool = Option(
        False, "--rdma",
        help="Build ARTS with RDMA RSockets transport (--arts only)"),
    cc: Optional[str] = Option(
        None, "--cc",
        help="C compiler for LLVM bootstrap (default: clang; use gcc on systems without clang)"),
    cxx: Optional[str] = Option(
        None, "--cxx",
        help="C++ compiler for LLVM bootstrap (default: clang++; use g++ with --cc gcc)"),
):
    """Build CARTS project using system clang."""
    config = get_config()

    print_header("CARTS Build")
    console.print(
        f"Platform: [{Colors.INFO}]{config.info.os}[/{Colors.INFO}] ({config.info.arch})")

    makefile = config.carts_dir / "Makefile"
    if not makefile.is_file():
        print_error("Makefile not found. Are you in the CARTS project root?")
        raise Exit(1)

    selected_targets = sum((1 if arts else 0, 1 if polygeist else 0, 1 if llvm else 0))
    if selected_targets > 1:
        print_error("Choose only one target flag among --arts, --polygeist, --llvm.")
        raise Exit(1)
    if rdma and not arts:
        print_error("--rdma is only valid with --arts.")
        raise Exit(1)

    # Determine build target
    if arts:
        target = MAKE_TARGET_ARTS
    elif polygeist:
        target = MAKE_TARGET_POLYGEIST
    elif llvm:
        target = MAKE_TARGET_LLVM
    else:
        target = MAKE_TARGET_BUILD

    console.print(f"Target: [{Colors.INFO}]{target}[/{Colors.INFO}]")

    # Build make variables
    make_vars = []

    if arts:
        # Expose the raw v2 ARTS runtime levels directly:
        #   0 -> ERROR only
        #   1 -> WARN
        #   2 -> INFO
        #   3 -> DEBUG (+ Debug build)
        make_vars.append(f"ARTS_LOG_LEVEL={debug_level}")
        if debug_level >= 3:
            make_vars.extend([
                "ARTS_BUILD_TYPE=Debug",
            ])
        if rdma:
            make_vars.append("ARTS_USE_RDMA=ON")

    # Counter levels: 0=off, 1=artsid, 2=deep
    # Levels 1+ require USE_COUNTERS and USE_METRICS
    if counters_level >= 1 or profile is not None:
        make_vars.extend(["ARTS_USE_COUNTERS=ON", "ARTS_USE_METRICS=ON"])

    # Always pass counter config path for ARTS builds
    if arts:
        if profile:
            # Use custom profile if provided
            if not profile.exists():
                print_error(f"Profile not found: {profile}")
                raise Exit(1)
            effective_counter_config = profile.resolve()
            console.print(f"Profile: [{Colors.INFO}]{profile}[/{Colors.INFO}]")
        else:
            # Use profile based on counters_level
            profile_name = COUNTER_PROFILES.get(
                counters_level, "profile-timing.cfg")
            effective_counter_config = (
                config.carts_dir
                / SUBMODULE_BENCHMARKS
                / "configs"
                / "profiles"
                / profile_name
            )
            console.print(f"Counter profile: [{Colors.INFO}]{profile_name}[/{Colors.INFO}]")
        make_vars.append(f"COUNTER_CONFIG_PATH={effective_counter_config}")

    # Pass platform-specific paths to make
    if config.linker_path:
        make_vars.append(f'CARTS_LINKER_PATH={config.linker_path}')
    if config.gcc_install_prefix:
        make_vars.append(f'LLVM_GCC_INSTALL_PREFIX={config.gcc_install_prefix}')

    # Pass bootstrap compiler overrides to make
    if cc:
        make_vars.append(f'LLVM_C_COMPILER={cc}')
    if cxx:
        make_vars.append(f'LLVM_CXX_COMPILER={cxx}')

    cmake_spec = DependencySpec(name="cmake", command="cmake", min_version="3.20")
    cmake_result = DependencyChecker().check(cmake_spec)
    cmake_path = shutil.which("cmake")
    if cmake_result.ok and cmake_path:
        make_vars.append(f"CMAKE={cmake_path}")

    if make_vars:
        console.print(f"Options: [{Colors.DEBUG}]{' '.join(make_vars)}[/{Colors.DEBUG}]")

    console.print()

    # Clean if requested
    if clean:
        print_step("Cleaning previous build...")
        clean_target = _CLEAN_TARGETS[target]
        run_subprocess(["make", clean_target], cwd=config.carts_dir, check=False)

    # Run build
    print_step(f"Building {target}...")
    cmd = ["make"] + make_vars + [target]

    result = run_subprocess(cmd, cwd=config.carts_dir, check=False)

    if result.returncode == 0:
        print_header("Build Complete")
        print_success("CARTS build completed successfully!")
    else:
        print_header("Build Failed")
        print_error("CARTS build failed!")
        raise Exit(1)
