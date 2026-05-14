#!/usr/bin/env bash
set -euo pipefail

level=${1:-3}
shift $(( $# > 0 ? 1 : 0 ))

dekk carts build --arts --debug "$level" "$@"
