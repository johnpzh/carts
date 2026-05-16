"""Compile pipeline commands for CARTS CLI."""

from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from dekk import Argument, Colors, Context, Exit, Option, console, print_header, print_error, print_info, print_success, print_warning
from scripts.platform import CartsConfig, get_config, is_verbose
from scripts import (
    run_subprocess,
    run_command_with_output,
    ARTS_CONFIG_FILENAME,
    TOOL_CARTS_COMPILE,
    TOOL_CGEIST,
    TOOL_CLANG,
)

# Compiler flags used across multiple functions
FLAG_ARTS_CONFIG = "--arts-config"
FLAG_DIAGNOSE = "--diagnose"
FLAG_DIAGNOSE_OUTPUT = "--diagnose-output"
FLAG_ARTS_DEBUG = "--arts-debug"
FLAG_DEBUG_ONLY = "--debug-only"
FLAG_PIPELINE = "--pipeline"
FLAG_OUTPUT = "-o"


@dataclass(frozen=True)
class PipelineTokens:
    pipeline: List[str]
    start_from: List[str]
    pipeline_sequence: List[str]


@dataclass(frozen=True)
class PipelineManifestStep:
    name: str
    passes: List[str]
    depends_on: List[str]


@dataclass(frozen=True)
class PipelineManifest:
    tokens: PipelineTokens
    steps: List[PipelineManifestStep]
    epilogue_steps: List[PipelineManifestStep]
    dialect_groups: Dict[str, object]


def _parse_manifest_steps_field(
    data: Dict[str, object], key: str
) -> List[PipelineManifestStep]:
    """Parse a manifest step-list field."""
    steps_raw = data.get(key)
    if steps_raw is None:
        raise ValueError(f"field '{key}' must be a list")
    if not isinstance(steps_raw, list):
        raise ValueError(f"field '{key}' must be a list")

    steps: List[PipelineManifestStep] = []
    for step_raw in steps_raw:
        if not isinstance(step_raw, dict):
            raise ValueError(f"each '{key}' entry must be an object")
        name = step_raw.get("name")
        passes = step_raw.get("passes", [])
        depends_on = step_raw.get("dependsOn", [])
        if not isinstance(name, str):
            raise ValueError(f"manifest step field '{key}.name' must be a string")
        if not isinstance(passes, list) or not all(isinstance(p, str) for p in passes):
            raise ValueError(
                f"manifest step field '{key}.passes' must be a list of strings"
            )
        if (
            not isinstance(depends_on, list)
            or not all(isinstance(dep, str) for dep in depends_on)
        ):
            raise ValueError(
                f"manifest step field '{key}.dependsOn' must be a list of strings"
            )
        steps.append(PipelineManifestStep(
            name=name,
            passes=list(passes),
            depends_on=list(dict.fromkeys(depends_on)),
        ))
    return steps


_PIPELINE_MANIFEST_CACHE: Optional[PipelineManifest] = None
_DISALLOWED_PIPELINE_FLAGS = {
    FLAG_DEBUG_ONLY: FLAG_ARTS_DEBUG,
}



def _normalize_arts_debug(spec: str) -> str:
    """Normalize `--arts-debug` channels and reject empty specs."""
    channels = [channel.strip() for channel in spec.split(",") if channel.strip()]
    if not channels:
        raise ValueError(
            "Invalid --arts-debug value. Use comma-separated channels, "
            "for example --arts-debug=info,debug."
        )
    return ",".join(channels)


def _has_flag(args: List[str], flag: str) -> bool:
    """Return True when args contain a `--flag` token in either form."""
    for arg in args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _parse_pipeline_tokens_payload(payload: str) -> PipelineTokens:
    """Parse pipeline token arrays from the compiler manifest JSON."""
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("root JSON object must be a map")

    def require_string_list(key: str) -> List[str]:
        value = data.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"field '{key}' must be a list of strings")
        return list(dict.fromkeys(value))

    pipeline = require_string_list("pipeline")
    start_from = require_string_list("start_from")
    pipeline_sequence = require_string_list("pipeline_sequence")

    if not pipeline_sequence:
        raise ValueError("field 'pipeline_sequence' cannot be empty")
    if not start_from:
        raise ValueError("field 'start_from' cannot be empty")
    if not pipeline:
        raise ValueError("field 'pipeline' cannot be empty")

    return PipelineTokens(
        pipeline=pipeline,
        start_from=start_from,
        pipeline_sequence=pipeline_sequence,
    )


def _parse_pipeline_manifest_payload(payload: str) -> PipelineManifest:
    """Parse `carts-compile --print-pipeline-manifest-json` output."""
    tokens = _parse_pipeline_tokens_payload(payload)
    data = json.loads(payload)
    steps = _parse_manifest_steps_field(data, "pipeline_steps")
    epilogue_steps = _parse_manifest_steps_field(data, "epilogue_steps")
    dialect_groups_raw = data.get("dialect_groups", {})
    if not isinstance(dialect_groups_raw, dict):
        raise ValueError("field 'dialect_groups' must be an object when present")

    return PipelineManifest(
        tokens=tokens,
        steps=steps,
        epilogue_steps=epilogue_steps,
        dialect_groups=dialect_groups_raw,
    )


def _canonical_pipeline_token(
    manifest: PipelineManifest, token: Optional[str]
) -> Optional[str]:
    if token is None:
        return None
    if token == "complete":
        return token
    if token in manifest.tokens.pipeline_sequence:
        return token
    return None


def _accepted_pipeline_tokens(manifest: PipelineManifest) -> List[str]:
    return list(manifest.tokens.pipeline)


def _accepted_start_from_tokens(manifest: PipelineManifest) -> List[str]:
    return list(manifest.tokens.start_from)


def _pipeline_manifest_to_dict(manifest: PipelineManifest) -> Dict[str, object]:
    """Convert pipeline manifest dataclass into a JSON-serializable dict."""
    payload: Dict[str, object] = {
        "pipeline": manifest.tokens.pipeline,
        "start_from": manifest.tokens.start_from,
        "pipeline_sequence": manifest.tokens.pipeline_sequence,
        "pipeline_steps": [
            {
                "name": step.name,
                "passes": step.passes,
                "dependsOn": step.depends_on,
            }
            for step in manifest.steps
        ],
        "epilogue_steps": [
            {
                "name": step.name,
                "passes": step.passes,
                "dependsOn": step.depends_on,
            }
            for step in manifest.epilogue_steps
        ],
    }
    if manifest.dialect_groups:
        payload["dialect_groups"] = manifest.dialect_groups
    return payload


def _query_compiler_json(
    config: CartsConfig, flag: str, error_message: str
) -> str:
    """Run carts-compile JSON endpoint and return stdout payload."""
    carts_compile_bin = config.get_carts_tool(TOOL_CARTS_COMPILE)
    if not carts_compile_bin.is_file():
        print_error(f"carts-compile not found at {carts_compile_bin}")
        raise Exit(1)

    result = run_subprocess(
        [str(carts_compile_bin), flag],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print_error(error_message)
        raise Exit(result.returncode)
    return result.stdout or ""


def _get_pipeline_manifest(config: CartsConfig) -> PipelineManifest:
    """Load pipeline manifest from carts-compile once per process."""
    global _PIPELINE_MANIFEST_CACHE
    if _PIPELINE_MANIFEST_CACHE is not None:
        return _PIPELINE_MANIFEST_CACHE

    try:
        payload = _query_compiler_json(
            config,
            "--print-pipeline-manifest-json",
            "Failed to query pipeline manifest from carts-compile. "
            "Run `dekk carts build` and retry.",
        )
        _PIPELINE_MANIFEST_CACHE = _parse_pipeline_manifest_payload(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        print_error(f"Invalid pipeline manifest payload from carts-compile: {exc}")
        raise Exit(1)

    return _PIPELINE_MANIFEST_CACHE


@contextmanager
def _pushd(path: Path):
    """Temporarily change working directory."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _get_compile_workdir_override() -> Optional[Path]:
    """Return compile workdir override from environment, if set."""
    raw = os.environ.get("CARTS_COMPILE_WORKDIR")
    if not raw:
        return None
    workdir = Path(raw).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


_PATH_VALUE_FLAGS = {
    "-I",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-include",
    "-L",
    "-F",
    "-isysroot",
    "--sysroot",
}
_PATH_VALUE_PREFIXES = (
    "-I",
    "-L",
    "-F",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-include",
)
_PATH_EQUALS_PREFIXES = ("--sysroot=",)


def _resolve_relative_cli_path(token: str, base_dir: Path) -> str:
    """Resolve a path-valued CLI token relative to the original invocation cwd."""
    path = Path(token)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _normalize_path_flags(args: List[str], base_dir: Path) -> List[str]:
    """Stabilize path-valued compiler/linker flags before changing cwd."""
    normalized: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in _PATH_VALUE_FLAGS and i + 1 < len(args):
            normalized.append(arg)
            i += 1
            normalized.append(_resolve_relative_cli_path(args[i], base_dir))
        elif any(arg.startswith(prefix) for prefix in _PATH_EQUALS_PREFIXES):
            prefix, value = arg.split("=", 1)
            normalized.append(f"{prefix}={_resolve_relative_cli_path(value, base_dir)}")
        else:
            matched_prefix = None
            for prefix in _PATH_VALUE_PREFIXES:
                if arg.startswith(prefix) and len(arg) > len(prefix):
                    matched_prefix = prefix
                    break
            if matched_prefix is not None:
                normalized.append(
                    matched_prefix
                    + _resolve_relative_cli_path(arg[len(matched_prefix):], base_dir)
                )
            else:
                normalized.append(arg)
        i += 1
    return normalized


def _is_numeric(s: str) -> bool:
    """Check if a string is a numeric value (including negative)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _drop_flag_and_value(args: List[str], flag: str) -> List[str]:
    """Remove `--flag` tokens from passthrough args, including optional value."""
    filtered: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == flag:
            i += 1
            if i < len(args):
                next_arg = args[i]
                if not next_arg.startswith("-") or _is_numeric(next_arg):
                    i += 1
            continue
        if arg.startswith(f"{flag}="):
            i += 1
            continue
        filtered.append(arg)
        i += 1
    return filtered


def _reject_disallowed_pipeline_flags(args: List[str]) -> None:
    """Reject non-canonical flags that bypass ARTS compile conventions."""
    for disallowed, replacement in _DISALLOWED_PIPELINE_FLAGS.items():
        if _has_flag(args, disallowed):
            print_error(f"Unsupported option: {disallowed}. Use {replacement}.")
            raise Exit(1)


@dataclass
class CompileArgs:
    """Parsed passthrough arguments, classified by pipeline step.

    Classification is pattern-based so that new ``cl::opt`` flags added to
    ``carts-compile.cpp`` are automatically forwarded without any Python
    changes.  The rules are:

    1. A handful of *special* flags are consumed by the Python CLI itself
       (``-O3``, ``--diagnose``, ``--diagnose-output``).
    2. Linker flags are recognised by well-known prefixes/names
       (``-l``, ``-L``, ``-Wl,``, ``-Xlinker``, ``-framework``).
    3. Short single-dash flags (``-I``, ``-D``, ``-std``, ``-f``, …) go to
       cgeist.
    4. Everything else — i.e. any ``--`` long flag — is forwarded to the
       carts-compile pipeline binary.
    """

    cgeist: List[str] = field(default_factory=list)    # -I, -D, -std, etc.
    pipeline: List[str] = field(default_factory=list)   # --arts-config, --partition-*, etc.
    link: List[str] = field(default_factory=list)        # -l, -L, -Wl, -framework, etc.
    optimize: bool = False                               # -O3 passthrough
    diagnose: bool = False                               # --diagnose passthrough
    diagnose_output: Optional[Path] = None               # --diagnose-output passthrough

    # Linker flag prefixes (single-dash)
    _LINKER_PREFIXES = ("-l", "-L", "-Wl,")
    # Linker flags that consume the next argument
    _LINKER_VALUE_FLAGS = {"-Xlinker", "-framework"}

    @classmethod
    def parse(cls, args: List[str]) -> "CompileArgs":
        result = cls()
        i = 0
        while i < len(args):
            arg = args[i]
            # --- Special flags consumed by the Python CLI ---
            if arg in ("-O3", "--O3"):
                result.optimize = True
            elif arg == FLAG_DIAGNOSE:
                result.diagnose = True
            elif arg == FLAG_DIAGNOSE_OUTPUT:
                if i + 1 < len(args):
                    i += 1
                    result.diagnose_output = Path(args[i])
            # --- Linker flags ---
            elif arg.startswith(cls._LINKER_PREFIXES):
                result.link.append(arg)
            elif arg in cls._LINKER_VALUE_FLAGS:
                result.link.append(arg)
                if i + 1 < len(args):
                    i += 1
                    result.link.append(args[i])
            # --- Long flags (--*) -> pipeline (carts-compile binary) ---
            elif arg.startswith("--"):
                result.pipeline.append(arg)
                # If no "=", the next non-flag token is the value.
                if "=" not in arg and i + 1 < len(args):
                    nxt = args[i + 1]
                    if not nxt.startswith("-") or _is_numeric(nxt):
                        i += 1
                        result.pipeline.append(nxt)
            # --- Everything else -> cgeist (-I, -D, -std, -f, …) ---
            else:
                result.cgeist.append(arg)
            i += 1
        return result


# ============================================================================
# Cgeist Command
# ============================================================================

def cgeist(
    ctx: Context,
    input_file: Path = Argument(..., help="Input C/C++ file"),
):
    """Run cgeist with automatic include paths."""
    config = get_config()

    if not input_file.is_file():
        print_error(f"Input file '{input_file}' not found")
        raise Exit(1)

    cgeist_bin = config.get_polygeist_tool(TOOL_CGEIST)
    if not cgeist_bin.is_file():
        print_error(f"cgeist not found at {cgeist_bin}")
        raise Exit(1)

    # Build command
    cmd = [str(cgeist_bin)]
    cmd.extend(config.include_flags)
    cmd.extend(config.cgeist_sysroot_flags)
    cmd.append("--raise-scf-to-affine")

    # Filter out --arts-config (not supported by cgeist) and pass through rest
    if ctx.args:
        skip_next = False
        for arg in ctx.args:
            if skip_next:
                skip_next = False
                continue
            if arg == FLAG_ARTS_CONFIG:
                skip_next = True
                continue
            cmd.append(arg)

    cmd.append(str(input_file))

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# ============================================================================
# Clang Command
# ============================================================================

def clang(
    ctx: Context,
    input_file: Path = Argument(..., help="Input file"),
    output: Optional[Path] = Option(None, "-o", help="Output file"),
):
    """Compile C/C++ with LLVM clang and OpenMP."""
    config = get_config()

    if not input_file.is_file():
        print_error(f"Input file '{input_file}' not found")
        raise Exit(1)

    clang_bin = config.get_llvm_tool(TOOL_CLANG)
    if not clang_bin.is_file():
        print_error(f"LLVM clang not found at {clang_bin}")
        raise Exit(1)

    # Build command
    cmd = [str(clang_bin)]
    cmd.extend(config.runtime_flags)
    cmd.extend(config.include_flags)
    cmd.extend(config.clang_sysroot_flags)
    cmd.extend(config.clang_library_flags)
    cmd.extend(config.linker_flags)
    cmd.append("-fopenmp")
    cmd.append(str(input_file))
    cmd.extend(config.clang_libraries)

    if output:
        cmd.extend(["-o", str(output)])

    # Pass through extra args (e.g., -O3, -lm, etc.)
    if ctx.args:
        cmd.extend(ctx.args)

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# ============================================================================
# MLIR-Translate Command
# ============================================================================

def mlir_translate(
    ctx: Context,
    args: Optional[List[str]] = Argument(
        None, help="Arguments for mlir-translate"),
):
    """Run mlir-translate."""
    config = get_config()

    mlir_translate_bin = config.get_llvm_tool("mlir-translate")
    translated_args = list(args or [])
    if ctx.args:
        translated_args.extend(ctx.args)
    if not translated_args:
        print_error("Provide mlir-translate arguments, for example --help.")
        raise Exit(1)

    cmd = [str(mlir_translate_bin)] + translated_args

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# ============================================================================
# Pipeline Command
# ============================================================================

def pipeline(
    pipeline: Optional[str] = Option(
        None, FLAG_PIPELINE, help="Show passes for one pipeline step"),
    json_output: bool = Option(
        False, "--json", help="Print pipeline manifest JSON"),
):
    """Show pipeline steps and passes from carts-compile."""
    config = get_config()
    manifest = _get_pipeline_manifest(config)
    canonical_pipeline = _canonical_pipeline_token(manifest, pipeline)

    if pipeline and canonical_pipeline not in manifest.tokens.pipeline_sequence:
        print_error(f"Unknown pipeline step: '{pipeline}'")
        print_info("Available pipeline steps:")
        for token in manifest.tokens.pipeline_sequence:
            console.print(f"  - {token}")
        raise Exit(1)

    if json_output:
        print(json.dumps(_pipeline_manifest_to_dict(manifest), indent=2))
        return

    print_header("CARTS Pipeline")
    table = Table(title="Pipeline Order")
    table.add_column("#", style=Colors.INFO, justify="right")
    table.add_column("Pipeline", style=Colors.SUCCESS)
    table.add_column("Passes", style=Colors.STEP, justify="right")
    selected_pipelines = manifest.steps
    if pipeline:
        selected_pipelines = [
            entry for entry in manifest.steps if entry.name == canonical_pipeline
        ]
    for index, entry in enumerate(manifest.steps, start=1):
        if pipeline and entry.name != canonical_pipeline:
            continue
        table.add_row(str(index), entry.name, str(len(entry.passes)))
    console.print(table)

    for entry in selected_pipelines:
        console.print()
        console.print(f"[{Colors.INFO}]{entry.name}[/{Colors.INFO}]")
        for pass_name in entry.passes:
            console.print(f"  - {pass_name}")


# ============================================================================
# Compile Command (unified)
# ============================================================================

def compile_cmd(
    ctx: Context,
    input_file: Path = Argument(..., help="Input file (.c/.cpp/.mlir/.ll)"),
    output: Optional[Path] = Option(
        None, "-o",
        help="Output path (file for normal compile; directory with --all-pipelines)"),
    optimize: bool = Option(
        False, "-O3", help="Enable O3 optimizations"),
    debug: bool = Option(False, "-g", help="Generate debug symbols"),
    pipeline: Optional[str] = Option(
        None, FLAG_PIPELINE,
        help="Stop after pipeline stage token (see `dekk carts pipeline --json`)"),
    all_pipelines: bool = Option(
        False, "--all-pipelines",
        help="(.mlir input) write one output per pipeline step"),
    emit_llvm: bool = Option(
        False, "--emit-llvm", help="Emit LLVM IR output"),
    diagnose: bool = Option(
        False, FLAG_DIAGNOSE, help="Export diagnostic information"),
    diagnose_output: Optional[Path] = Option(
        None, FLAG_DIAGNOSE_OUTPUT, help="Output file for diagnostics"),
    arts_debug: Optional[str] = Option(
        None, FLAG_ARTS_DEBUG,
        help="Comma-separated ARTS debug channels (example: info,debug)"),
    start_from: Optional[str] = Option(
        None, "--start-from",
        help="Start pipeline at specified step (input must already match that step)"),
):
    """Unified compilation command.

    Routes based on input file type:

      .c/.cpp  Full pipeline: C -> MLIR -> LLVM IR -> executable

      .mlir    MLIR transformation pipeline via carts-compile binary

      .ll      Link LLVM IR with ARTS runtime -> executable

    Use --pipeline <step> to stop after a specific pipeline step.
    """
    passthrough_args = list(ctx.args or [])
    _reject_disallowed_pipeline_flags(passthrough_args)
    if arts_debug is not None:
        passthrough_args = _drop_flag_and_value(passthrough_args, FLAG_ARTS_DEBUG)

    normalized_arts_debug: Optional[str] = None
    if arts_debug:
        try:
            normalized_arts_debug = _normalize_arts_debug(arts_debug)
        except ValueError as exc:
            print_error(str(exc))
            raise Exit(1)

    ext = input_file.suffix.lower()
    if ext == ".ll":
        if pipeline:
            print_error("--pipeline is not supported for .ll input (link-only)")
            raise Exit(1)
        if start_from:
            print_error("--start-from is not supported for .ll input (link-only)")
            raise Exit(1)
        if all_pipelines:
            print_error("--all-pipelines is only supported for .mlir input")
            raise Exit(1)
        if emit_llvm or diagnose or diagnose_output:
            print_error(
                "--emit-llvm and diagnose options are not supported for .ll "
                "input (link-only)"
            )
            raise Exit(1)
        if normalized_arts_debug or _has_flag(passthrough_args, FLAG_ARTS_DEBUG):
            print_error("--arts-debug is only supported for carts-compile pipeline runs")
            raise Exit(1)
        _compile_from_ll(input_file, output, debug, passthrough_args)
        return

    if ext not in (".c", ".cpp", ".mlir"):
        print_error(
            f"Unsupported input '{input_file}'. Supported extensions: "
            ".c, .cpp, .mlir, .ll."
        )
        raise Exit(1)

    config = get_config()
    manifest = _get_pipeline_manifest(config)
    accepted_pipeline_tokens = _accepted_pipeline_tokens(manifest)
    accepted_start_from_tokens = _accepted_start_from_tokens(manifest)
    if pipeline and pipeline not in accepted_pipeline_tokens:
        print_error(f"Unknown pipeline step: '{pipeline}'")
        print_info("Available pipeline steps:")
        for pipeline_token in accepted_pipeline_tokens:
            console.print(f"  - {pipeline_token}")
        raise Exit(1)
    if start_from and start_from not in accepted_start_from_tokens:
        print_error(f"Unknown start-from pipeline step: '{start_from}'")
        print_info("Available start-from pipeline steps:")
        for pipeline_token in accepted_start_from_tokens:
            console.print(f"  - {pipeline_token}")
        raise Exit(1)

    if ext in (".c", ".cpp"):
        if emit_llvm:
            print_error("--emit-llvm is only supported for .mlir input")
            raise Exit(1)
        if start_from:
            print_error("--start-from is only supported for .mlir input")
            raise Exit(1)
        if all_pipelines:
            if pipeline:
                print_error("--all-pipelines cannot be combined with --pipeline")
                raise Exit(1)
            _compile_all_pipelines(get_config(), input_file, output,
                                   passthrough_args, optimize)
            return
        _compile_from_c(input_file, output, optimize, debug, pipeline,
                        diagnose, diagnose_output, normalized_arts_debug,
                        passthrough_args)
    else:
        if all_pipelines and (pipeline or start_from):
            print_error("--all-pipelines cannot be combined with --pipeline or --start-from")
            raise Exit(1)
        _compile_from_mlir(input_file, output, optimize, debug, pipeline,
                           all_pipelines, emit_llvm,
                           diagnose, diagnose_output, normalized_arts_debug,
                           start_from, passthrough_args)


# -----------------------------------------------------------------------------
# Compile: .c/.cpp input (full pipeline)
# -----------------------------------------------------------------------------

def _compile_from_c(
    input_file: Path,
    output: Optional[Path],
    optimize: bool,
    debug: bool,
    pipeline: Optional[str],
    diagnose: bool,
    diagnose_output: Optional[Path],
    arts_debug: Optional[str],
    passthrough_args: List[str],
) -> None:
    """Full pipeline for C/C++ input."""
    config = get_config()
    # Resolve early so changing cwd for artifact isolation does not break input lookup.
    input_file = input_file.resolve()

    if not input_file.is_file():
        print_error(f"Input file '{input_file}' not found")
        raise Exit(1)

    base_name = input_file.stem
    output_name = output if output else Path(f"{base_name}_arts")

    # Parse passthrough args
    parsed = CompileArgs.parse(passthrough_args)
    invocation_cwd = Path.cwd().resolve()

    extra_pipeline_args = parsed.pipeline[:]
    extra_link_args = _normalize_path_flags(parsed.link[:], invocation_cwd)
    cgeist_args = _normalize_path_flags(parsed.cgeist, invocation_cwd)

    if arts_debug:
        extra_pipeline_args.append(f"{FLAG_ARTS_DEBUG}={arts_debug}")

    enable_pipeline_o3 = optimize or parsed.optimize
    use_diagnose = diagnose or parsed.diagnose
    final_diagnose_output = diagnose_output or parsed.diagnose_output

    workdir_override = _get_compile_workdir_override()
    if workdir_override is None:
        _compile_c_pipeline(config, input_file, output_name, debug,
                            enable_pipeline_o3,
                            extra_pipeline_args, extra_link_args, cgeist_args,
                            use_diagnose, final_diagnose_output, pipeline)
        return

    with _pushd(workdir_override):
        _compile_c_pipeline(config, input_file, output_name, debug,
                            enable_pipeline_o3,
                            extra_pipeline_args, extra_link_args, cgeist_args,
                            use_diagnose, final_diagnose_output, pipeline)


# -----------------------------------------------------------------------------
# Compile: .mlir input (MLIR pipeline)
# -----------------------------------------------------------------------------

def _compile_from_mlir(
    input_file: Path,
    output: Optional[Path],
    optimize: bool,
    debug: bool,
    pipeline: Optional[str],
    all_pipelines: bool,
    emit_llvm: bool,
    diagnose: bool = False,
    diagnose_output: Optional[Path] = None,
    arts_debug: Optional[str] = None,
    start_from: Optional[str] = None,
    passthrough_args: Optional[List[str]] = None,
) -> None:
    """MLIR transformation pipeline."""
    config = get_config()
    carts_compile_bin = config.get_carts_tool(TOOL_CARTS_COMPILE)
    passthrough_args = passthrough_args or []

    # Pass --help through to carts-compile binary
    if "--help" in passthrough_args or "-h" in passthrough_args:
        result = run_subprocess([str(carts_compile_bin), "--help"], check=False)
        raise Exit(result.returncode)

    if all_pipelines:
        extra_args = list(passthrough_args)
        if arts_debug:
            extra_args.append(f"{FLAG_ARTS_DEBUG}={arts_debug}")
        _compile_all_pipelines(config, input_file, output, extra_args, optimize)
        return

    if not input_file.is_file():
        print_error(f"Input file '{input_file}' not found")
        raise Exit(1)

    cmd = [str(carts_compile_bin), str(input_file)]

    # Auto-discover arts.cfg if not explicitly provided
    arts_cfg = _find_arts_config(input_file, passthrough_args, config)
    if arts_cfg:
        cmd.extend([FLAG_ARTS_CONFIG, str(arts_cfg)])

    if optimize:
        cmd.append("--O3")
    if emit_llvm:
        cmd.append("--emit-llvm")
    if debug:
        cmd.append("-g")
    if pipeline:
        cmd.append(f"{FLAG_PIPELINE}={pipeline}")
    if diagnose:
        cmd.append(FLAG_DIAGNOSE)
    if diagnose_output:
        cmd.extend([FLAG_DIAGNOSE_OUTPUT, str(diagnose_output)])
    if arts_debug:
        cmd.append(f"{FLAG_ARTS_DEBUG}={arts_debug}")
    if start_from:
        cmd.append(f"{"--start-from"}={start_from}")
    if output:
        cmd.extend([FLAG_OUTPUT, str(output)])
    if passthrough_args:
        cmd.extend(passthrough_args)

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# -----------------------------------------------------------------------------
# Compile: .ll input (link only)
# -----------------------------------------------------------------------------

def _compile_from_ll(
    input_file: Path,
    output: Optional[Path],
    debug: bool,
    passthrough_args: List[str],
) -> None:
    """Link LLVM IR with ARTS runtime."""
    config = get_config()

    if not input_file.is_file():
        print_error(f"Input file '{input_file}' not found")
        raise Exit(1)

    if output is None:
        print_error("-o <output> is required for .ll input")
        raise Exit(1)

    cmd = _build_link_cmd(
        config,
        input_file,
        output,
        debug,
        passthrough_args,
    )

    result = run_subprocess(cmd, check=False)
    raise Exit(result.returncode)


# -----------------------------------------------------------------------------
# Compile Pipeline Helpers
# -----------------------------------------------------------------------------

def _find_arts_config(
    input_file: Path, pipeline_args: List[str], config: Optional["CartsConfig"] = None
) -> Optional[Path]:
    """Find arts.cfg for the compilation.

    Returns None if the user already supplied --arts-config.  Otherwise
    checks for arts.cfg next to the input file, then falls back to the
    default examples config.  When no config is found carts-compile
    will fail immediately — that is intentional.
    """
    for arg in pipeline_args:
        if arg == FLAG_ARTS_CONFIG or arg.startswith(f"{FLAG_ARTS_CONFIG}="):
            return None

    # Check next to the input file
    candidate = input_file.parent.resolve() / ARTS_CONFIG_FILENAME
    if candidate.is_file():
        return candidate

    # Fall back to the default examples config
    if config:
        default = config.carts_dir / "tests" / "examples" / ARTS_CONFIG_FILENAME
        if default.is_file():
            return default

    return None


def _build_cgeist_cmd(
    config: CartsConfig,
    input_file: Path,
    std_flag: str,
    passthrough_args: List[str],
    with_openmp: bool = False,
    with_debug_info: bool = False,
) -> List[str]:
    """Build cgeist (C-to-MLIR) command with standard flags."""
    cmd = [str(config.get_polygeist_tool(TOOL_CGEIST))]
    cmd.extend(config.include_flags)
    cmd.extend(config.cgeist_sysroot_flags)
    cmd.extend(["--raise-scf-to-affine", std_flag, "-O0", "-S",
                 "-D_POSIX_C_SOURCE=199309L"])  # Required for clock_gettime
    if with_openmp:
        cmd.append("-fopenmp")
    if with_debug_info:
        cmd.append("--print-debug-info")
    cmd.extend(passthrough_args)
    cmd.append(str(input_file))
    return cmd


def _build_link_cmd(
    config: CartsConfig,
    input_file: Path,
    output_file: Path,
    debug: bool,
    extra_args: List[str],
) -> List[str]:
    """Build clang link command for linking with ARTS runtime."""
    cmd = [str(config.get_llvm_tool(TOOL_CLANG))]
    cmd.extend(config.compile_flags)  # includes runtime_flags
    cmd.extend(config.clang_sysroot_flags)
    cmd.extend(config.compile_library_flags)
    cmd.extend(config.linker_flags)
    cmd.append(str(input_file))
    cmd.extend(["-o", str(output_file)])
    if debug:
        cmd.append("-g")
    cmd.extend(config.compile_libraries)
    cmd.extend(extra_args)
    return cmd


def _compile_c_pipeline(
    config: CartsConfig,
    input_file: Path,
    output_name: Path,
    debug: bool,
    enable_pipeline_o3: bool,
    pipeline_args: List[str],
    link_args: List[str],
    passthrough_args: Optional[List[str]] = None,
    diagnose: bool = False,
    diagnose_output: Optional[Path] = None,
    pipeline_stop: Optional[str] = None,
) -> None:
    """C/C++ compilation pipeline.

    Steps:
    1. Compilation (with OpenMP) - for ARTS transformation
    2. Apply ARTS transformations
    3. Link final executable (skipped if pipeline_stop is set)
    """
    base_name = input_file.stem
    extension = input_file.suffix
    std_flag = "-std=c++17" if extension == ".cpp" else "-std=c17"
    passthrough_args = passthrough_args or []
    skip_link = pipeline_stop is not None

    total_steps = 2 if skip_link else 3
    step_label = f"/{total_steps}]"

    print_header("CARTS Compile Pipeline")
    console.print(f"Input:  [{Colors.INFO}]{input_file}[/{Colors.INFO}]")
    if skip_link:
        console.print(f"Stop:   [{Colors.WARNING}]{pipeline_stop}[/{Colors.WARNING}]")
    else:
        console.print(f"Output: [{Colors.INFO}]{output_name}[/{Colors.INFO}]")
    console.print(
        f"Mode:   [{Colors.WARNING}]Compilation pipeline[/{Colors.WARNING}]")
    console.print()

    carts_compile_bin = config.get_carts_tool(TOOL_CARTS_COMPILE)

    # Output file paths
    par_mlir = Path(f"{base_name}.mlir")
    ll_file = Path(f"{base_name}-arts.ll")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Compilation (with OpenMP for ARTS transformation)
        task = progress.add_task(f"[1{step_label} Compilation...", total=None)
        cmd = _build_cgeist_cmd(config, input_file, std_flag, passthrough_args,
                                with_openmp=True, with_debug_info=True)
        if run_command_with_output(cmd, par_mlir) != 0:
            print_error("Failed compilation")
            raise Exit(1)
        progress.remove_task(task)
        print_success(f"[1{step_label} {par_mlir}")

        # Step 2: Apply ARTS transformations
        task = progress.add_task(f"[2{step_label} ARTS transformations...", total=None)
        cmd = [str(carts_compile_bin), str(par_mlir)]
        if enable_pipeline_o3:
            cmd.append("--O3")
        arts_cfg = _find_arts_config(input_file, pipeline_args, config)
        if arts_cfg:
            cmd.extend([FLAG_ARTS_CONFIG, str(arts_cfg)])
        if not skip_link:
            cmd.append("--emit-llvm")
        if debug:
            cmd.append("-g")
        if diagnose:
            cmd.append(FLAG_DIAGNOSE)
            effective_diagnose_output = diagnose_output or Path(
                f"{base_name}-diagnose.json")
            cmd.extend([FLAG_DIAGNOSE_OUTPUT, str(effective_diagnose_output)])
        if pipeline_stop:
            cmd.append(f"{FLAG_PIPELINE}={pipeline_stop}")
        cmd.extend(pipeline_args)

        out_file = par_mlir.with_suffix(f".{pipeline_stop}.mlir") if skip_link else ll_file
        if run_command_with_output(cmd, out_file) != 0:
            print_error("Failed ARTS transformations")
            raise Exit(1)
        progress.remove_task(task)
        print_success(f"[2{step_label} {out_file}")

        if skip_link:
            console.print()
            console.print(Panel(
                f"[{Colors.SUCCESS}]Pipeline output:[/{Colors.SUCCESS}] {out_file}\n"
                f"[{Colors.DIM}]Stopped at: {pipeline_stop}[/{Colors.DIM}]",
                title="Success",
                border_style=Colors.SUCCESS,
            ))
            return

        # Step 3: Link with ARTS runtime
        task = progress.add_task(f"[3{step_label} Final linking...", total=None)
        cmd = _build_link_cmd(
            config, ll_file, output_name, debug, link_args)
        if run_subprocess(cmd, check=False).returncode != 0:
            print_error("Failed final linking")
            raise Exit(1)
        progress.remove_task(task)
        print_success(f"[3{step_label} {output_name}")

    console.print()
    files_info = (
        f"[{Colors.SUCCESS}]Generated:[/{Colors.SUCCESS}] {output_name}\n"
        f"[{Colors.DIM}]MLIR: {par_mlir}[/{Colors.DIM}]\n"
        f"[{Colors.DIM}]LLVM IR: {ll_file}[/{Colors.DIM}]"
    )
    console.print(Panel(files_info, title="Success", border_style=Colors.SUCCESS))


# Matches MLIR's --mlir-print-ir-after-all boundary headers, e.g.
#   // -----// IR Dump After LowerAffinePass (lower-affine) //----- //
_PASS_DUMP_HEADER_RE = re.compile(
    r"^// -----// IR Dump After (\S+) \(([^)]+)\) //----- //\s*$",
    re.MULTILINE,
)

# Dialect-conversion boundaries in the ARTS pipeline. Each boundary is
# presented to users as a folder under <out>/boundaries/ containing the input
# state (before.mlir), the conversion pass dump(s), and the output state
# (after.mlir) — making the dialect transition directly inspectable. This
# table also drives phase bucketing for the per-stage and per-pass dumps via
# the helpers below.
#
# Each entry: (folder_name, [(stage_1based, pass_class_name), ...])
# Single source of truth for the dialect-architecture metadata used to bucket
# the dump tree. Each entry is:
#   (folder_name, target_phase_after, [(stage_token, pass_class), ...])
# where ``stage_token`` is the ``--pipeline=<token>`` name from the manifest
# (NOT a numeric index — the index is resolved at runtime so the layout
# survives stage reordering / insertions).
_DIALECT_BOUNDARIES: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    # OMP dialect → SDE dialect (inside the SDE planning stage).
    ("01_omp_to_sde", "sde",
     [("sde-planning", "ConvertOpenMPToSde")]),
    # SDE planning → CODIR codelet isolation.
    ("02_sde_to_codir", "codir",
     [("sde-to-codir", "ConvertSdeToCodir")]),
    # CODIR codelet boundary → ARTS core objects.
    ("03_codir_to_arts", "core",
     [("codir-to-arts", "ConvertCodirToArts")]),
    # ARTS core → arts_rt runtime dialect (inside the pre-lowering stage).
    ("04_arts_to_rt", "rt", [
        ("pre-lowering", "DbLowering"),
        ("pre-lowering", "EdtLowering"),
        ("pre-lowering", "EpochLowering"),
    ]),
]


def _phase_order() -> Tuple[str, ...]:
    """Phase progression. Initial phase plus each unique boundary target."""
    order: List[str] = []
    # Initial phase = the target of the first boundary (omp→sde puts us in sde).
    if _DIALECT_BOUNDARIES:
        order.append(_DIALECT_BOUNDARIES[0][1])
    for _name, target, _entries in _DIALECT_BOUNDARIES:
        if target not in order:
            order.append(target)
    return tuple(order)


def _initial_phase() -> str:
    return _phase_order()[0]


def _phase_dirname(phase: str) -> str:
    """e.g. 'sde' → '2_sde'. ``1_polygeist`` always sits ahead of phase 1."""
    return f"{_phase_order().index(phase) + 2}_{phase}"


def _llvm_dirname() -> str:
    """LLVM bucket sits after all phase buckets."""
    return f"{len(_phase_order()) + 2}_llvm"


def _phase_flip_passes() -> Dict[str, str]:
    """Pass-class name → phase the cursor advances to when this pass runs."""
    out: Dict[str, str] = {}
    for _name, target, entries in _DIALECT_BOUNDARIES:
        for _stage_token, cls in entries:
            out[cls] = target
    return out


def _stage_phase_map(pipeline_steps: List[str]) -> Dict[int, str]:
    """Stage idx (1-based) → phase the stage's OUTPUT IR belongs to.

    Walks ``pipeline_steps`` advancing a cursor whenever the current stage
    contains a boundary pass. If multiple boundaries fire in the same stage
    the last one wins (matches per-pass execution order).
    """
    # Per-stage final target (last boundary wins inside a stage).
    last_target: Dict[str, str] = {}
    for _name, target, entries in _DIALECT_BOUNDARIES:
        for stage_token, _cls in entries:
            last_target[stage_token] = target
    cursor = _initial_phase()
    out: Dict[int, str] = {}
    for idx, step in enumerate(pipeline_steps, start=1):
        if step in last_target:
            cursor = last_target[step]
        out[idx] = cursor
    return out


def _phase_dir(out_dir: Path, phase: str) -> Path:
    return out_dir / _phase_dirname(phase)


def _stage_mlir_path(
    out_dir: Path, idx: int, step: str, stage_phase: Dict[int, str],
) -> Path:
    """Per-stage MLIR snapshot path (in the phase that owns the output IR)."""
    return _phase_dir(out_dir, stage_phase[idx]) / "stages" / f"{idx:02d}_{step}.mlir"


def _find_stage_mlir(out_dir: Path, idx: int, step: str) -> Optional[Path]:
    """Find a stage's emitted MLIR across all phase buckets."""
    for phase in _phase_order():
        p = _phase_dir(out_dir, phase) / "stages" / f"{idx:02d}_{step}.mlir"
        if p.exists():
            return p
    return None


def _find_pass_dumps(
    out_dir: Path, idx: int, step: str, class_name: str,
) -> List[Path]:
    """Collect all per-pass dumps for (stage, class) across phase buckets."""
    matches: List[Path] = []
    for phase in _phase_order():
        sdir = _phase_dir(out_dir, phase) / "passes" / f"{idx:02d}_{step}"
        matches.extend(sorted(sdir.glob(f"*_{class_name}.mlir")))
    return matches


def _build_boundaries_view(
    out_dir: Path,
    boundaries_dir: Path,
    pipeline_steps: List[str],
) -> int:
    """Populate the boundaries/ view from existing stage + per-pass dumps.

    For each dialect boundary, materialize:
      - before.mlir: IR state just before the first conversion pass runs
      - <PassClass>.mlir: IR after each conversion pass (copied from per-pass
        dumps)
      - after.mlir: IR state after the last conversion pass runs

    Stage and per-pass MLIR are looked up across all phase buckets so this
    view stays valid under the dialect-keyed layout regardless of how many
    phases ``_DIALECT_BOUNDARIES`` defines.

    Returns the number of boundaries successfully populated.
    """
    boundaries_dir.mkdir(exist_ok=True)
    # Build a token → 1-based-index resolver from the live pipeline manifest.
    step_index = {step: i + 1 for i, step in enumerate(pipeline_steps)}
    populated = 0
    for bname, _target_phase, entries in _DIALECT_BOUNDARIES:
        # Resolve stage tokens → indices; skip silently if a token is unknown
        # (manifest may omit conditional stages or rename them).
        resolved = [(step_index[tok], tok, cls)
                    for tok, cls in entries if tok in step_index]
        if not resolved:
            continue
        bdir = boundaries_dir / bname
        bdir.mkdir(exist_ok=True)
        first_stage_idx = resolved[0][0]
        # "before" = output of the previous stage (or the raw input for
        # stage 1). Falls back to empty file if missing.
        before_src: Optional[Path] = None
        if first_stage_idx > 1:
            prev_stage_idx = first_stage_idx - 1
            prev_stage_token = pipeline_steps[prev_stage_idx - 1]
            before_src = _find_stage_mlir(out_dir, prev_stage_idx, prev_stage_token)
        if before_src and before_src.exists():
            (bdir / "before.mlir").write_bytes(before_src.read_bytes())

        copied = 0
        last_pass_file: Optional[Path] = None
        for stage_1based, stage_token, pass_class in resolved:
            matches = _find_pass_dumps(out_dir, stage_1based, stage_token, pass_class)
            if not matches:
                continue
            # For a class that ran multiple times (nested pass manager),
            # keep the first dump under its class name and suffix any
            # additional ones with _2, _3, ... so they all appear.
            for i, src in enumerate(matches):
                dst_name = (f"{pass_class}.mlir" if i == 0
                            else f"{pass_class}_{i + 1}.mlir")
                dst = bdir / dst_name
                dst.write_bytes(src.read_bytes())
                last_pass_file = dst
                copied += 1

        # "after" = last conversion pass's output, which by --mlir-print-ir-
        # after-all semantics IS the state right after the conversion ran.
        if last_pass_file is not None:
            (bdir / "after.mlir").write_bytes(last_pass_file.read_bytes())
            populated += 1

        # Drop a short README describing the boundary for reader orientation.
        passes_listed = ", ".join(f"{p}" for _, p in entries)
        (bdir / "README.md").write_text(
            f"# Dialect boundary: `{bname.split('_', 1)[1].replace('_', ' → ')}`\n\n"
            f"Conversion pass(es): {passes_listed}\n\n"
            f"- `before.mlir` — module state entering this boundary\n"
            f"- `<PassClass>.mlir` — module state after that specific pass ran\n"
            f"- `after.mlir` — module state leaving this boundary\n"
            f"\n_(file count this run: {copied} conversion pass dump(s))_\n",
            encoding="utf-8",
        )
    return populated


def _parse_per_pass_ir(stderr: str) -> List[Tuple[int, str, str, str]]:
    """Parse ``--mlir-print-ir-after-all`` stderr into per-pass dump entries.

    Each block starting with ``// -----// IR Dump After <Class> (<cli>) //-----//``
    becomes a tuple ``(pass_idx, class_name, header_line, body)`` in execution
    order. Caller is responsible for writing each entry to disk.
    """
    headers = list(_PASS_DUMP_HEADER_RE.finditer(stderr))
    out: List[Tuple[int, str, str, str]] = []
    for idx, match in enumerate(headers):
        class_name = match.group(1)
        header_line = match.group(0)
        body_start = match.end() + 1  # skip the trailing newline of the header
        body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(stderr)
        body = stderr[body_start:body_end].rstrip()
        out.append((idx + 1, class_name, header_line, body))
    return out
    return len(headers)


def _compile_all_pipelines(
    config: CartsConfig,
    source_file: Path,
    output_dir: Optional[Path],
    extra_args: List[str],
    optimize: bool,
) -> None:
    """Run the full compilation and dump artifacts for every phase.

    Layout (under ``output_dir`` or ``source_file.parent``) mirrors the 4-phase
    dialect architecture (lib/arts/dialect/{sde,core,rt}/ split):

    ::

        <out>/
          1_polygeist/                       # only emitted for .c / .cpp input
            <stem>.mlir                      # cgeist output (pre-ARTS MLIR)
          2_sde/                             # semantic dialect work
            stages/
              01_sde-input-normalization.mlir
              02_initial-cleanup.mlir
            passes/
              01_sde-input-normalization/
              02_initial-cleanup/
              03_sde-planning/               # SDE planning passes
          3_codir/                           # codelet isolation work
            stages/
              NN_sde-to-codir.mlir
            passes/
              NN_sde-to-codir/
          4_core/                            # ARTS core dialect work
            stages/
              NN_codir-to-arts.mlir          # output is core (post conversion)
              NN_edt-transforms.mlir … NN_epochs.mlir
            passes/
              NN_codir-to-arts/
              NN_edt-transforms/ … NN_epochs/
              NN_pre-lowering/               # core passes before rt-lowering
          5_rt/                              # arts_rt dialect + lowering
            stages/
              NN_pre-lowering.mlir
              NN_arts-rt-to-llvm.mlir
            passes/
              NN_pre-lowering/               # rt-lowering passes onward
              NN_arts-rt-to-llvm/
          6_llvm/
            <stem>.ll                        # final LLVM IR (--emit-llvm)
          boundaries/                        # dialect-conversion slices
            01_omp_to_sde/ 02_sde_to_codir/ 03_codir_to_arts/ 04_arts_to_rt/
          complete.mlir                      # final MLIR (after --O3 cleanup)

    Bucketing rules (all derived from ``_DIALECT_BOUNDARIES`` + the live
    pipeline manifest — no per-stage or per-phase literals are baked into
    this function, so adding/renaming/reordering pipeline stages or dialect
    boundaries only requires updating ``_DIALECT_BOUNDARIES``):
      - Per-stage MLIR is placed in the phase its output IR belongs to
        (see ``_stage_phase_map``).
      - Per-pass dumps are placed by an execution-order cursor that flips on
        boundary passes (see ``_phase_flip_passes``). The current codelet
        boundary uses separate stages for SDE planning, CODIR isolation, and
        ARTS materialization.

    For ``.mlir`` input the ``1_polygeist/`` phase is skipped (caller already
    has the pre-ARTS MLIR on disk).
    """
    if not source_file.is_file():
        print_error(f"Input file '{source_file}' not found")
        raise Exit(1)

    ext = source_file.suffix.lower()
    if ext not in (".c", ".cpp", ".mlir"):
        print_error(f"--all-pipelines: unsupported input extension '{ext}'. "
                    f"Expected .c, .cpp, or .mlir.")
        raise Exit(1)

    carts_compile_bin = config.get_carts_tool(TOOL_CARTS_COMPILE)

    out_dir = output_dir if output_dir else source_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _get_pipeline_manifest(config)
    pipeline_steps = manifest.tokens.pipeline_sequence

    stem = source_file.stem
    results: List[tuple] = []  # List of (pipeline_step, success_bool)

    print_header("CARTS Pipeline Dump")

    # Phase 1: Polygeist (C/C++ -> MLIR). Skipped for .mlir input.
    if ext in (".c", ".cpp"):
        phase1_dir = out_dir / "1_polygeist"
        phase1_dir.mkdir(exist_ok=True)
        mlir_input = phase1_dir / f"{stem}.mlir"
        print_info(f"Phase 1 (polygeist): {source_file.name} -> "
                   f"{mlir_input.relative_to(out_dir)}")
        std_flag = "-std=c++17" if ext == ".cpp" else "-std=c11"
        cgeist_cmd = _build_cgeist_cmd(
            config, source_file, std_flag, [], with_openmp=True,
            with_debug_info=True,
        )
        rc = run_command_with_output(cgeist_cmd, mlir_input)
        results.append(("1_polygeist", rc == 0))
        if rc != 0:
            print_error("Phase 1 (polygeist) failed; cannot continue.")
            _print_summary_table(results, out_dir)
            raise Exit(1)
    else:
        mlir_input = source_file

    # Derive layout from _DIALECT_BOUNDARIES + the live pipeline manifest so
    # the bucketing stays correct as stages are added/renamed/reordered.
    phase_order = _phase_order()
    stage_phase = _stage_phase_map(pipeline_steps)
    phase_flip = _phase_flip_passes()

    # Pre-create phase buckets (each gets stages/ + passes/) and the LLVM dir.
    for phase in phase_order:
        (_phase_dir(out_dir, phase) / "stages").mkdir(parents=True, exist_ok=True)
        (_phase_dir(out_dir, phase) / "passes").mkdir(parents=True, exist_ok=True)
    llvm_dir = out_dir / _llvm_dirname()
    llvm_dir.mkdir(exist_ok=True)

    base_cmd = [str(carts_compile_bin), str(mlir_input)]
    if optimize:
        base_cmd.append("--O3")
    arts_cfg = _find_arts_config(mlir_input, extra_args, config)
    if arts_cfg:
        base_cmd.extend([FLAG_ARTS_CONFIG, str(arts_cfg)])

    total_steps = 3 + len(pipeline_steps)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Phase 2: ARTS transforms", total=total_steps)

        # Phase 2a: one .mlir per core stage, bucketed by output dialect.
        for idx, pipeline_step in enumerate(pipeline_steps, start=1):
            progress.update(task, description=f"Phase 2 stage {idx:02d}: {pipeline_step}")
            out_file = _stage_mlir_path(out_dir, idx, pipeline_step, stage_phase)
            cmd = base_cmd + [f"{FLAG_PIPELINE}={pipeline_step}",
                              FLAG_OUTPUT, str(out_file)] + extra_args
            rc = run_subprocess(cmd, check=False).returncode
            results.append(
                (f"{_phase_dirname(stage_phase[idx])}/stages/"
                 f"{idx:02d}_{pipeline_step}",
                 rc == 0)
            )
            progress.advance(task)

        # Phase 2b: complete MLIR (all core + epilogue MLIR passes applied).
        progress.update(task, description="Phase 2: complete MLIR")
        complete_mlir = out_dir / "complete.mlir"
        cmd = base_cmd + [FLAG_OUTPUT, str(complete_mlir)] + extra_args
        rc = run_subprocess(cmd, check=False).returncode
        results.append(("complete.mlir", rc == 0))
        progress.advance(task)

        # Phase 3: LLVM IR emission.
        progress.update(task, description="Phase 3: LLVM IR")
        final_ll = llvm_dir / f"{stem}.ll"
        cmd = base_cmd + ["--emit-llvm", FLAG_OUTPUT, str(final_ll)] + extra_args
        rc = run_subprocess(cmd, check=False).returncode
        results.append((f"{_llvm_dirname()}/*.ll", rc == 0))
        progress.advance(task)

        # Phase 2c: per-pass IR dumps, routed to phase buckets by an
        # execution-order cursor that flips on boundary passes
        # (ConvertCodirToArts → core; rt-lowering passes → rt). Each per-pass
        # dump is written to <phase>/passes/<NN>_<stage>/<MMM>_<Class>.mlir.
        # Stages whose passes straddle a boundary (3 and 15) appear under
        # two phase buckets — that's intentional and informative.
        progress.update(task, description="Phase 2: per-pass IR dumps")
        total_dumps = 0
        stages_dumped = 0
        phase_cursor = _initial_phase()
        stage_cfg_args: List[str] = []
        if optimize:
            stage_cfg_args.append("--O3")
        if arts_cfg:
            stage_cfg_args.extend([FLAG_ARTS_CONFIG, str(arts_cfg)])

        for idx, stage in enumerate(pipeline_steps, start=1):
            progress.update(
                task,
                description=f"Per-pass ({idx}/{len(pipeline_steps)}: {stage})",
            )
            # Pre-stage input: the MLIR entry point for stage 1, previous
            # stage's emitted .mlir thereafter (looked up across phase dirs).
            if idx == 1:
                stage_input: Optional[Path] = mlir_input
            else:
                prev_stage = pipeline_steps[idx - 2]
                stage_input = _find_stage_mlir(out_dir, idx - 1, prev_stage)
            if stage_input is None or not stage_input.exists():
                continue  # previous stage failed, skip
            scratch_out = out_dir / f".discard_{idx:02d}.mlir"
            cmd = ([str(carts_compile_bin), str(stage_input),
                    f"--start-from={stage}",
                    f"{FLAG_PIPELINE}={stage}",
                    "--mlir-print-ir-after-all",
                    FLAG_OUTPUT, str(scratch_out)]
                   + stage_cfg_args + extra_args)
            proc = run_subprocess(cmd, capture_output=True, check=False)
            entries = _parse_per_pass_ir(proc.stderr or "")
            stage_had_dump = False
            for pass_idx, class_name, header_line, body in entries:
                # Boundary passes flip the cursor; the dump itself goes to the
                # phase the cursor points to AFTER the flip (its output IR is
                # already in the new dialect).
                if class_name in phase_flip:
                    phase_cursor = phase_flip[class_name]
                target_stage_dir = (
                    _phase_dir(out_dir, phase_cursor) / "passes"
                    / f"{idx:02d}_{stage}"
                )
                target_stage_dir.mkdir(parents=True, exist_ok=True)
                (target_stage_dir / f"{pass_idx:03d}_{class_name}.mlir").write_text(
                    f"{header_line}\n{body}\n", encoding="utf-8"
                )
                total_dumps += 1
                stage_had_dump = True
            if stage_had_dump:
                stages_dumped += 1
            try:
                scratch_out.unlink()
            except FileNotFoundError:
                pass

        label = (f"passes ({total_dumps} dumps, "
                 f"{stages_dumped}/{len(pipeline_steps)} stages)")
        results.append((label, total_dumps > 0))
        progress.advance(task)

        # Phase 2d: dialect-boundary view (OMP→SDE, SDE→CODIR,
        # CODIR→ARTS, ARTS→RT).
        # Materialized from the per-pass dumps we just wrote, by searching
        # across all phase buckets.
        progress.update(task, description="Phase 2: dialect boundaries")
        boundaries_dir = out_dir / "boundaries"
        populated = _build_boundaries_view(
            out_dir, boundaries_dir, pipeline_steps,
        )
        results.append(
            (f"boundaries ({populated}/{len(_DIALECT_BOUNDARIES)})",
             populated > 0)
        )

    _print_summary_table(results, out_dir)


def _print_summary_table(results: List[tuple], out_dir: Path) -> None:
    """Render the pipeline-dump summary + success line."""
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)

    table = Table(title="Pipeline Results")
    table.add_column("Pipeline", style=Colors.INFO)
    table.add_column("Status", justify="center")
    for pipeline_name, ok in results:
        status = (f"[{Colors.SUCCESS}]PASS[/{Colors.SUCCESS}]" if ok
                  else f"[{Colors.ERROR}]FAIL[/{Colors.ERROR}]")
        table.add_row(pipeline_name, status)
    console.print(table)

    console.print(
        f"\n[{Colors.SUCCESS}]{passed} passed[/{Colors.SUCCESS}], "
        f"[{Colors.ERROR}]{failed} failed[/{Colors.ERROR}] "
        f"out of {len(results)} pipeline outputs"
    )
    print_success(f"Pipeline dumps written to {out_dir}")
