#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <input.{c,cpp,mlir}> <outdir> [comma-separated stages] [extra dekk carts compile args...]" >&2
  exit 2
fi

input=$1
outdir=$2
stage_csv=${3:-sde-planning,sde-to-codir,codir-to-arts,edt-transforms,create-dbs,db-opt,post-db-refinement,pre-lowering}

if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

mkdir -p "$outdir"
IFS=',' read -r -a stages <<< "$stage_csv"

for stage in "${stages[@]}"; do
  echo "[miscompile] dumping ${stage}" >&2
  dekk carts compile "$input" --pipeline="$stage" "$@" > "${outdir}/${stage}.mlir"
done

dekk carts pipeline --json > "${outdir}/pipeline-manifest.json"
