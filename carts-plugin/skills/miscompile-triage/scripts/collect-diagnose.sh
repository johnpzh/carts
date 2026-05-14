#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <input.{c,cpp,mlir}> <outdir> [extra dekk carts compile args...]" >&2
  exit 2
fi

input=$1
outdir=$2
shift 2

mkdir -p "$outdir"

dekk carts compile "$input" --diagnose --diagnose-output "${outdir}/diagnose.json" "$@" >/dev/null
dekk carts pipeline --json > "${outdir}/pipeline-manifest.json"

echo "${outdir}/diagnose.json"
