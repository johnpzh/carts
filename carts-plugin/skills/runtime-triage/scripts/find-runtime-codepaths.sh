#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git grep -n \
  -e 'CreateEpoch' \
  -e 'wait_on_epoch' \
  -e 'Continue' \
  -e 'ConvertArtsRtToLLVM' \
  -e 'EdtLowering' \
  -e 'artsActiveMessageWithDb' \
  -e 'distributed_db_init' \
  -- include lib docs external/arts .agents
