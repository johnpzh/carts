#!/usr/bin/env python3
"""Run a tool from the active CARTS-managed install root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_config import (
    prepend_runtime_library_env,
    resolve_install_dir,
)


LLVM_TOOLS = frozenset({
    "FileCheck",
    "clang-format",
    "llvm-lit",
})


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: managed_tool.py <tool> [args...]", file=sys.stderr)
        return 2

    tool_name = argv[1]
    if tool_name not in LLVM_TOOLS:
        print(f"unsupported CARTS-managed tool: {tool_name}", file=sys.stderr)
        return 2

    carts_dir = Path(__file__).resolve().parents[2]
    tool_path = resolve_install_dir(carts_dir) / "llvm" / "bin" / tool_name
    if not tool_path.is_file():
        print(f"CARTS-managed tool not found: {tool_path}", file=sys.stderr)
        return 127

    prepend_runtime_library_env(carts_dir, include_existing_env=False)
    os.execv(str(tool_path), [str(tool_path), *argv[2:]])
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
