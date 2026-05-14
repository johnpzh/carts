#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <input.{c,cpp,mlir}> <stage> <output.mlir> [extra dekk carts compile args...]" >&2
  exit 2
fi

input=$1
stage=$2
output=$3
shift 3

mkdir -p "$(dirname "$output")"
dekk carts compile "$input" --pipeline="$stage" "$@" > "$output"
