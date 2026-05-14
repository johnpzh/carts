#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <benchmark> <arts-config> [size=small] [nodes=2] [threads=4] [extra args...]" >&2
  exit 2
fi

benchmark=$1
arts_config=$2
size=${3:-small}
nodes=${4:-2}
threads=${5:-4}
shift $(( $# >= 5 ? 5 : $# ))

dekk carts benchmarks run "$benchmark" \
  --size "$size" \
  --arts-config "$arts_config" \
  --nodes "$nodes" \
  --threads "$threads" \
  --debug 2 \
  "$@"
