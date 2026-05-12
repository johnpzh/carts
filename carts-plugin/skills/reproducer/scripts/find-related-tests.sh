#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <pattern>" >&2
  exit 2
fi

pattern=$1
cd "$(git rev-parse --show-toplevel)"

git grep -n "$pattern" -- lib/arts/dialect tests/e2e tests/e2e_multinode samples || true
