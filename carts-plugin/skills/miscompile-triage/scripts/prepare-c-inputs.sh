#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <source.{c,cc,cpp}> <outdir> [extra cgeist args...]" >&2
  exit 2
fi

src=$1
outdir=$2
shift 2

mkdir -p "$outdir"
base=$(basename "$src")
stem=${base%.*}

dekk carts cgeist "$src" -O0 --print-debug-info -S --raise-scf-to-affine "$@" \
  -o "${outdir}/${stem}_seq.mlir"
dekk carts cgeist "$src" -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine "$@" \
  -o "${outdir}/${stem}_omp.mlir"

echo "${outdir}/${stem}_seq.mlir"
echo "${outdir}/${stem}_omp.mlir"
