"""ARTS configuration file parser, key definitions, and manipulation utilities.

Key names mirror the authoritative ``config_entries[]`` table in
``external/arts/libs/src/core/system/config.c:755-801``.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key names — single source of truth for Python code
# ---------------------------------------------------------------------------

# Threading
KEY_WORKER_THREADS = "worker_threads"
KEY_STACK_SIZE = "stack_size"

# Pinning
KEY_PIN = "pin"

# Scheduling
KEY_SCHEDULER = "scheduler"
KEY_DEQUE_TYPE = "deque_type"
KEY_WORKER_INIT_DEQUE_SIZE = "worker_init_deque_size"
KEY_ROUTE_TABLE_SIZE = "route_table_size"
KEY_AUTO_SHUTDOWN = "auto_shutdown"
KEY_MIN_ITERATIONS_PER_WORKER = "min_iterations_per_worker"

# GPU
KEY_GPU = "gpu"
KEY_GPU_LOCALITY = "gpu_locality"
KEY_GPU_FIT = "gpu_fit"
KEY_GPU_LC_SYNC = "gpu_lc_sync"
KEY_GPU_MAX_EDTS = "gpu_max_edts"
KEY_GPU_MAX_MEMORY = "gpu_max_memory"
KEY_GPU_P2P = "gpu_p2p"
KEY_GPU_ROUTE_TABLE_SIZE = "gpu_route_table_size"
KEY_FREE_DB_AFTER_GPU_RUN = "free_db_after_gpu_run"
KEY_RUN_GPU_GC_IDLE = "run_gpu_gc_idle"
KEY_RUN_GPU_GC_PRE_EDT = "run_gpu_gc_pre_edt"
KEY_DELETE_ZEROS_GPU_GC = "delete_zeros_gpu_gc"
KEY_GPU_BUFF_ON = "gpu_buff_on"

# Networking
KEY_SENDER_THREADS = "sender_threads"
KEY_RECEIVER_THREADS = "receiver_threads"
KEY_PORT_COUNT = "port_count"
KEY_MASTER_NODE = "master_node"
KEY_DEFAULT_PORTS = "default_ports"
KEY_NET_INTERFACE = "net_interface"

# Launcher / nodes
KEY_LAUNCHER = "launcher"
KEY_NODE_COUNT = "node_count"
KEY_NODES = "nodes"

# Debug
KEY_KILL_MODE = "kill_mode"
KEY_CORE_DUMP = "core_dump"

# Counters
KEY_COUNTER_FOLDER = "counter_folder"
KEY_COUNTER_CAPTURE_INTERVAL = "counter_capture_interval"

# Protocol
KEY_PROTOCOL = "protocol"

# Keys that are embedded into compiled binaries (used for validation)
EMBEDDED_KEYS: List[str] = [
    KEY_WORKER_THREADS,
    KEY_RECEIVER_THREADS,
    KEY_SENDER_THREADS,
    KEY_NODE_COUNT,
    KEY_LAUNCHER,
    KEY_PROTOCOL,
    KEY_DEFAULT_PORTS,
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_arts_cfg(path: Optional[Path]) -> Dict[str, str]:
    """Parse an ARTS config file (arts.cfg) into a key/value dict.

    This parser is intentionally simple: it ignores section headers and
    supports ``key=value`` lines, stripping whitespace and inline comments.
    """
    if not path or not path.exists():
        return {}

    values: Dict[str, str] = {}
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            # Strip common inline comments.
            value = value.split("#", 1)[0].split(";", 1)[0].strip()
            if key:
                values[key] = value
    except Exception:
        return {}
    return values


# ---------------------------------------------------------------------------
# Typed getters
# ---------------------------------------------------------------------------

def get_cfg_int(path: Optional[Path], key: str) -> Optional[int]:
    """Return an arts.cfg value as int, or None if missing/unparseable."""
    vals = parse_arts_cfg(path)
    if key not in vals:
        return None
    try:
        return int(vals[key])
    except Exception:
        logger.debug("Failed to convert arts.cfg key '%s' to int", key, exc_info=True)
        return None


def get_cfg_str(path: Optional[Path], key: str) -> Optional[str]:
    """Return an arts.cfg value as str, or None if missing/empty."""
    v = parse_arts_cfg(path).get(key)
    return v if v else None


def get_cfg_nodes(path: Optional[Path]) -> List[str]:
    """Parse the ``nodes=`` field into a list of hostnames.

    Returns ``["localhost"]`` when the key is absent.
    """
    cfg = parse_arts_cfg(path)
    nodes_str = cfg.get(KEY_NODES, "localhost")
    return [n.strip() for n in nodes_str.split(",") if n.strip()]


# ---------------------------------------------------------------------------
# Content manipulation (operates on the raw text of an arts.cfg file)
# ---------------------------------------------------------------------------

def upsert_cfg_value(content: str, key: str, value: object) -> str:
    """Replace or insert one top-level ARTS config assignment."""
    line = f"{key}={value}"
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, content, re.MULTILINE):
        return re.sub(pattern, line, content, flags=re.MULTILINE)
    return content.replace("[ARTS]", f"[ARTS]\n{line}", 1)


def comment_cfg_key(content: str, key: str, reason: str) -> str:
    """Comment out a key in arts.cfg content."""
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, content, re.MULTILINE):
        return re.sub(
            pattern, f"# {key}= ({reason})", content, flags=re.MULTILINE
        )
    return content


# ---------------------------------------------------------------------------
# Embedded config validation
# ---------------------------------------------------------------------------

def extract_embedded_cfg(artifacts_dir: Path) -> Dict[str, str]:
    """Extract embedded ARTS config values from LLVM IR or executable output."""
    candidates = sorted(artifacts_dir.glob("*-arts.ll"))
    if not candidates:
        candidates = sorted(artifacts_dir.glob("*_arts"))

    for candidate in candidates:
        try:
            text = candidate.read_text(errors="replace")
        except Exception:
            try:
                text = candidate.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                continue

        embedded: Dict[str, str] = {}
        for key in EMBEDDED_KEYS:
            match = re.search(
                rf"{re.escape(key)}=(.*?)(?:\\0A|\x00|\r|\n|\")",
                text,
            )
            if match:
                embedded[key] = match.group(1)
        if embedded:
            return embedded

    return {}


def validate_embedded_cfg(
    artifacts_dir: Path,
    expected_cfg: Optional[Path],
) -> Optional[str]:
    """Return an error string if the built artifact embeds the wrong config."""
    if expected_cfg is None or not expected_cfg.exists():
        return None

    expected = parse_arts_cfg(expected_cfg)
    if not expected:
        return f"Failed to parse expected arts.cfg: {expected_cfg}"

    embedded = extract_embedded_cfg(artifacts_dir)
    if not embedded:
        return (
            "Failed to inspect generated ARTS artifact for embedded config "
            f"in {artifacts_dir}"
        )

    mismatches: List[str] = []
    for key in EMBEDDED_KEYS:
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        embedded_value = embedded.get(key)
        if embedded_value != expected_value:
            mismatches.append(
                f"{key}: expected '{expected_value}', embedded '{embedded_value}'"
            )

    if not mismatches:
        return None

    return (
        "Generated ARTS artifact embeds a different config than the compile-time "
        f"arts.cfg. cfg={expected_cfg}, artifacts={artifacts_dir}. "
        + "; ".join(mismatches)
    )
