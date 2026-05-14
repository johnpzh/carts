#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git grep -n \
  -e 'DbDistributedOwnership' \
  -e 'DbDistributedEligibility' \
  -e 'DistributionPlanning' \
  -e 'ConvertSdeToArts' \
  -e 'distributed_db_init' \
  -e 'distributed_db_init_worker' \
  -e 'artsGetTotalNodes' \
  -e 'artsGuidGetRank' \
  -e '--distributed-db' \
  -e 'route = linearIndex' \
  -- docs include lib tools .agents
