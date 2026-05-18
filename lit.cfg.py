# -*- Python -*-
"""
Lit configuration for CARTS tests.

Tests are co-located with pass source code under lib/carts/dialect/*/test/,
with cross-cutting tests (CLI, verify) under tests/. This config discovers
tests across all these directories.

This file lives at the project root so lit can find it from any test directory
(lib/carts/dialect/*/test/, tests/cli/, tests/verify/).  When invoked via
CMake's lit.site.cfg.py, paths come from CMake variables.
"""

import os
import shutil
import sys
from pathlib import Path

import lit.formats
import lit.util

import lit.llvm
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config


config.name = "carts"
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = not lit.util.pythonize_bool(lit_shell_env)
config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)
config.suffixes = [".mlir"]

# --- Path setup ---
# Detect project root: lit.cfg.py lives at the project root.
project_root = getattr(config, "carts_source_dir", None) or os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(project_root, "tests")
tools_scripts_dir = os.path.join(project_root, "tools", "scripts")
if tools_scripts_dir not in sys.path:
    sys.path.insert(0, tools_scripts_dir)
from local_config import BUILD_DIR_NAME, INSTALL_DIR_NAME, resolve_carts_home

carts_home = str(resolve_carts_home(Path(project_root)))

# Source root is the project root so lit can discover test/ dirs under lib/.
config.test_source_root = project_root

# Out-of-source test execution: avoids polluting source tree with Output/ dirs.
build_dir = getattr(config, "carts_build_dir", None) or os.path.join(
    carts_home, BUILD_DIR_NAME, "carts"
)
config.test_exec_root = os.path.join(build_dir, "tests", "lit-output")

# Tell lit which subdirectories to scan for tests (IREE pattern).
config.test_subdirs = [
    os.path.join("lib", "carts", "dialect", "codir", "test"),
    os.path.join("lib", "carts", "dialect", "sde", "test"),
    os.path.join("lib", "carts", "dialect", "arts", "test"),
    os.path.join("lib", "carts", "dialect", "arts-rt", "test"),
    os.path.join("tests", "cli"),
    os.path.join("tests", "verify"),
]

# Keep llvm-lit resilient to interrupted writes or stray blank lines in the
# per-suite timing cache. TestTimes.py prefers the exec-root cache and falls
# back to the source-root cache.
for test_times_path in (
    os.path.join(config.test_exec_root, ".lit_test_times.txt"),
    os.path.join(config.test_source_root, ".lit_test_times.txt"),
):
    if os.path.exists(test_times_path):
        with open(test_times_path, "r") as time_file:
            cleaned_lines = []
            for line in time_file:
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    continue
                try:
                    float(fields[0])
                except ValueError:
                    continue
                cleaned_lines.append(line)
        with open(test_times_path, "w") as time_file:
            time_file.writelines(cleaned_lines)

# --- Tool paths ---
install_root = os.path.join(carts_home, INSTALL_DIR_NAME)
carts_bin_dir = os.path.join(install_root, "carts", "bin")
llvm_bin_dir = os.path.join(install_root, "llvm", "bin")
llvm_lib_dir = os.path.join(install_root, "llvm", "lib")
build_bin_dir = os.path.join(build_dir, "bin")

carts_compile_tool = getattr(config, "carts_compile_tool", None)
if not carts_compile_tool:
    carts_compile_tool = os.path.join(build_bin_dir, "carts-compile")
    if not os.path.exists(carts_compile_tool):
        carts_compile_tool = os.path.join(install_root, "carts", "bin", "carts-compile")

filecheck_tool = getattr(config, "filecheck_tool", None)
if not filecheck_tool:
    filecheck_tool = os.path.join(llvm_bin_dir, "FileCheck")

required_tools = [
    ("%carts-compile", carts_compile_tool),
    ("%FileCheck", filecheck_tool),
]

for subst, tool in required_tools:
    if not os.path.exists(tool):
        lit_config.fatal(
            f"Required tool '{tool}' was not found. "
            f"Run `dekk carts build` so the toolchain under {install_root} is up to date."
        )
    config.substitutions.append((subst, tool))

carts_tool = getattr(config, "carts_tool", None)
if not carts_tool:
    carts_wrapper = os.path.join(install_root, "bin", "carts")
    if os.path.exists(carts_wrapper):
        carts_tool = carts_wrapper
    else:
        dekk_tool = shutil.which("dekk")
        if dekk_tool:
            carts_tool = f"{dekk_tool} carts"

if carts_tool:
    config.substitutions.append(("%carts", carts_tool))

# --- Test data substitutions ---
inputs_dir = getattr(config, "carts_inputs_dir", None)
if not inputs_dir:
    inputs_dir = os.path.join(project_root, "tests", "inputs")
samples_dir = os.path.join(project_root, "samples")
arts_config_path = os.path.join(inputs_dir, "arts_8t.cfg")
arts_multinode_config_path = os.path.join(samples_dir, "arts_multinode.cfg")

config.substitutions.append(("%inputs_dir", inputs_dir))
config.substitutions.append(("%samples_dir", samples_dir))
config.substitutions.append(("%arts_config", arts_config_path))
config.substitutions.append(("%arts_multinode_config", arts_multinode_config_path))

# Export a few defaults so standard substitutions such as %S/%T are available.
config.llvm_tools_dir = llvm_bin_dir
config.llvm_lib_dir = llvm_lib_dir

# Add our installed bins to PATH for convenience.
config.environment["PATH"] = os.pathsep.join(
    [build_bin_dir, carts_bin_dir, llvm_bin_dir, config.environment.get("PATH", "")]
)

# macOS / Linux dyld search path. `carts-compile` is linked with
# @rpath/libzstd.1.dylib pointing at the dekk-managed conda env's lib/;
# propagate DYLD_LIBRARY_PATH / LD_LIBRARY_PATH from the outer shell (and
# fall back to the env's lib/) so lit subprocesses can resolve it.
_dekk_env_lib = os.path.join(project_root, ".dekk", "env", "lib")
for _lib_var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
    _inherited = os.environ.get(_lib_var, "")
    _entries = [_dekk_env_lib] + ([_inherited] if _inherited else [])
    config.environment[_lib_var] = os.pathsep.join(_entries)

# Directories that should not be treated as test directories.
config.excludes = ["inputs", "snapshots", "Output", "counters"]

llvm_config.use_default_substitutions()
