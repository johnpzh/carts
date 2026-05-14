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

stages=(
  openmp-to-arts
  post-db-refinement
  pre-lowering
)

for stage in "${stages[@]}"; do
  echo "[distributed-ir] dumping ${stage} -> ${outdir}/${stage}.mlir" >&2
  dekk carts compile "$input" --distributed-db --pipeline="$stage" "$@" \
    > "${outdir}/${stage}.mlir"
done
