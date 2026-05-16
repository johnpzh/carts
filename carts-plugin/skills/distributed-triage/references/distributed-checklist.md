# Distributed Debug Checklist

Use this checklist when a workload fails only with multiple nodes or `--distributed-db`.

## 1. Reproduction Inputs

- benchmark or input path
- `--arts-config` file
- node count
- thread count per node
- whether `--distributed-db` is enabled

## 2. IR Checks

Inspect these stage boundaries in order:

1. `sde-planning`
2. `codir-to-arts`
3. `post-db-refinement`
4. `pre-lowering`

Look for:

- `distribution_kind`, `distribution_pattern`, `distribution_version`
- writable task acquires that should preserve owner hints
- `DbAllocOp` instances marked `distributed`
- cases where SDE distribution planning should have produced distributed ARTS
  DB/EDT contracts but did not

## 3. Ownership Eligibility Questions

- Is the allocation host-level, not inside `arts.edt`?
- Does it have multiple DB blocks?
- Is the shape supported for distributed ownership?
- Are handle users restricted to allowed DB dependency flow?
- Is there at least one internode writer?
- Is the case rejected because it is read-only stencil-style internode use?

## 4. Runtime Checks

- Are logs retained with `--debug 2` or a runtime debug build?
- Do node-level JSON counters exist?
- Is remote work non-zero on nodes other than the primary node?
- Are there signs of thread starvation from too few worker threads per node?

## 5. Lowering/Runtime Split

Suspect lowering first if:

- routed work is absent in pre-lowering IR
- `distributed` markers are missing
- owner hints and partitioning disagree

Suspect runtime/config first if:

- IR looks correct but one node never executes work
- logs show progress starvation or configuration mismatch
- counters exist only for one node despite routed work
