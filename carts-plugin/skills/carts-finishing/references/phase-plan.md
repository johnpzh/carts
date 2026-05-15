# Phase Plan

The 11-phase ordering that brings every CARTS sample and benchmark green in single-node and multinode. Each phase has a clear stop condition; do not advance until it is met.

The corresponding tasks are #1–#11 in the project task list (`TaskList`). Tasks have `blockedBy` dependencies wired so the next-actionable item is always lowest-ID-pending-with-empty-blockedBy.

## Sequencing rationale

Phases are ordered to minimize attribution noise:

- **Decisions first** (phase 0): without charter answers, future placement decisions are arbitrary.
- **Baseline before any fix** (phase 1): without a snapshot, you cannot prove a regression-guard caught a regression vs. a pre-existing failure.
- **Diagnostic before structural** (phase 2 before 3): the recursion guard turns 8 crashes into legible diagnostics; the signal helps validate phase 3.
- **Single root-cause first** (phase 3): one fix unblocks 17 of 26 sample failures. Doing this before iterative fixes prevents whack-a-mole.
- **Single-node before multinode** (phases 4 and 7 before 6 and 8): multinode adds 7 unique failure modes on top of all single-node ones. Stabilize the base before adding axes.
- **Correctness before structural cleanup** (phase 9 last): pass-placement cleanup while correctness is broken makes attribution impossible.

**Do not reorder.** If you feel tempted, the rationale is wrong (update this doc) or the phase definition is wrong (update the task).

## Phase 0 — Resolve charter open questions (task #1)

**Type:** decision

**Stop condition:** `docs/architecture/charter-decisions.md` exists and answers the 5 open questions in `references/dialect-charter.md` "Open questions".

**Action:** use `AskUserQuestion` to surface the 5 questions one at a time. Each question has a recommendation in the charter doc; offer it as the first option.

**Why first:** without these answers, every subsequent placement decision (especially in phase 9) is arbitrary.

## Phase 1 — Establish CI baseline (task #2)

**Type:** baseline

**Stop condition:** `.results/baseline-YYYY-MM-DD.json` exists with: per-sample single-node status, per-sample multinode status (for the 9 currently-passing samples), per-benchmark small-size status. `tests/e2e/lit.local.cfg` has a multinode rule.

**Actions:**

1. `dekk carts test --suite e2e --json /tmp/single-node.json`
2. Add `tests/e2e/lit.local.cfg` rule for `arts_multinode.cfg`. Cover the 9 currently-passing samples (convolution, deps, jacobi/deps, matrix, matrixmul, parallel, parallel_for/{block,static}, task) under `--distributed-db`.
3. Run multinode lit subset; capture status.
4. `dekk carts benchmarks run --size small --json /tmp/bench.json`. Status snapshot.
5. Combine into `.results/baseline-YYYY-MM-DD.json`.

**Why now:** locks the green/red snapshot before any fix. Every later regression-guard run compares against this.

## Phase 2 — Recursion guard on copyArtsMetadataAttrs (task #3)

**Type:** targeted-fix

**Stop condition:** the 8 affected samples (concurrent, jacobi/for, mixed_access, mixed_orientation, parallel_for/{loops,reduction,single}, rows/chunks) transition from crash to legible diagnostic. Regression-guard passes.

**Action:** edit `lib/carts/dialect/arts/Analysis/heuristics/PartitioningHeuristics.cpp` near line 706. Add a depth counter + assertion when `copyArtsMetadataAttrs` recurses past a sane depth.

**Why now:** turns 8 crashes into diagnostics. Helps validate that phase 3 fixes the right thing.

**Effort:** ~1h.

## Phase 3 — Memref-of-memref heap-array flattening (task #4)

**Type:** targeted-fix (the highest-leverage single fix in the whole plan)

**Stop condition:** all 17 single-node-failing samples (11 compile-fail + 6 run-fail) compile and run. Regression-guard passes for all 26 samples.

**Actions:**

1. Edit `lib/carts/dialect/arts/Transforms/db/CreateDbs.cpp` `createDbAcquire()` to recognize heap-allocated arrays (`int *A = malloc(N*sizeof(int))` → `memref<?xmemref<?xT>>`) and produce N-element DBs of T instead of 1-element DBs holding a pointer.

2. Edit `lib/carts/dialect/arts/Transforms/RaiseMemRefDimensionality.cpp` to handle `memref<?xmemref<?>>` as a first-class heap-array pattern (covers the 3 samples that fail at stage 4 before reaching CreateDbs: dotproduct, parallel_for/stencil, stencil).

3. **Critical (anti-pattern 1):** the fix is in shape normalization. Do NOT touch `DbPartitioning::downgrade_to_coarse` or any partition-mode-selection heuristic. That layer is responding correctly to the wrong input shape; fixing it there would silently mask the issue.

**Effort:** ~7h primary + secondary.

## Phase 4 — Iterate samples to green, single-node (task #5)

**Type:** per-item iteration

**Stop condition:** all 26 samples pass single-node. `docs/compiler/sample-suite-triage.md` is refreshed.

**Workflow:** see `SKILL.md` "Per-item workflow." For each remaining failure:

1. Run, capture error + surface stage.
2. Walk the verification-barrier ladder.
3. Apply the triage decision tree.
4. Delegate to the matching specialist skill.
5. Apply the fix in the originating layer.
6. Run regression-guard.
7. Append fix-attribution log.
8. Move to next item.

**Order suggestion:** simplest first (parallel, task), then elementwise, then matmul, then reduction, then stencil, then wavefront. Each "easy" green builds confidence the rubric is working before tackling harder cases.

## Phase 5 — Core distributed ownership gates (task #6)

**Type:** targeted-fix

**Stop condition:** `DbDistributedOwnership` marks DBs with the `distributed` UnitAttr for elementwise, matmul, and reduction classifications when `--distributed-db` is set. The 9 originally-passing samples (now extended to all 26 if phase 4 is green) compile under `-O3 --distributed-db`.

**Action:** keep SDE distribution planning target-neutral and add the distributed eligibility gate in Core ownership/refinement. Core should consume SDE classification, window, and physical layout contracts, then use abstract-machine analysis to decide whether the realized DB/EDT shape is local or distributed.

**Effort:** ~2h.

**Why before phase 6:** without these gates, the 26 samples won't be marked for distribution and phase 6 has nothing to validate.

## Phase 6 — Iterate samples to green, multinode (task #7)

**Type:** per-item iteration

**Stop condition:** all 26 samples pass multinode (under `samples/arts_multinode.cfg`).

**Workflow:** same as phase 4, but consult `references/multinode-failures.md` BEFORE opening any source file. The 7 multinode-only failure modes are not in the single-node decision tree and you will misattribute without that reference.

**Order suggestion:** start simplest (parallel, task), then elementwise (matrix, convolution), then matmul (matrixmul), then reduction (deps, jacobi/deps), then stencil (jacobi/for, stencil), then wavefront (smith-waterman). Each fix must pass regression-guard against ALL prior-green samples in BOTH single-node and multinode.

## Phase 7 — Re-baseline benchmarks single-node (task #8)

**Type:** baseline

**Stop condition:** every benchmark in `external/carts-benchmarks/` has a current single-node status documented. Each regression vs the 2026-03-11 snapshot is either fixed or has a recorded reason.

**Action:** `dekk carts benchmarks run --size small`. Cross-reference with the 2026-03-11 snapshot. Many regressions likely already cleared by phase 3 (heap-array flattening); document which still remain and triage each.

**Why now and not earlier:** the 2026-03-11 benchmark snapshot is 5+ weeks old on this branch. Earlier baselining would record stale data; doing it after phase 3 picks up the fixes.

## Phase 8 — Iterate benchmarks to green, multinode (task #9)

**Type:** per-item iteration

**Stop condition:** parity between single-node and multinode benchmark green counts.

**Workflow:** same as phase 6 but on benchmark workloads.

**Order suggestion (smallest viable victory first):**

1. monte-carlo (already-verified distributed per the 2026-03-11 snapshot)
2. graph500 (already-verified distributed; OOM on large is a separate scaling issue)
3. PolybenchC (gemm, 2mm, 3mm — pure tensor kernels, benefit most directly from phase 3 fix)
4. kastors-jacobi (stencil class — exercises distributed halo)
5. seissol, sw4lite (grid-based, pre-allocated)
6. ml-kernels (perfect nests, narrow benefit)
7. lulesh, specfem3d (triple-indirected — known hard cases)

## Phase 9 — Pass-placement cleanup (task #10)

**Type:** structural

**Stop condition:** `docs/architecture/pass-placement.md` and `architecture-reaudit-2026-04-11.md` show no remaining TODOs. `docs/compiler/pipeline.md` reflects current state.

**Sub-steps (cheapest to most invasive):**

1. **2h** — Move `RaiseMemRefDimensionality` to `core/Conversion/PolygeistToArts/` (cross-dialect conversion currently in `core/Transforms/`).

2. **1h** — Verify `ScalarReplacement` lives in `rt/Transforms/` (the audit reports it as already moved; confirm and remove any stale references in `core/`).

3. **4h** — Wire or delete the SDE limbo passes:
   - `sde/Transforms/effect/distribution/BarrierElimination.cpp`
   - `sde/Transforms/dep/tensor/Interchange.cpp`
   - `sde/Transforms/state/codelet/{ConvertToCodelet,ScalarForwarding,TensorCleanup,TokenModeRefinement}.cpp`
   - `sde/Transforms/state/raising/LowerToMemref.cpp`

   For each: if it is phase 7/9/10 of `docs/plan.md`, wire it into `Compile.cpp`. Otherwise delete and update `docs/plan.md`.

4. **6h** — Add SDE-contract early-exit guards to the 4 ARTS PatternPipeline passes (StencilTilingND, MatmulReduction, LoopReordering, DepTransforms) so they defer when SDE has stamped a contract. Per `docs/plan.md` Phase 3A–3D.

5. **12h** — Decouple semantic detection from structural rewriting in DepTransforms / KernelTransforms. Move wavefront / Jacobi family detection into SDE (extend `PatternAnalysis` or add a later SDE wavefront-planning pass). Make core passes consumers, not detectors. Enforces Invariant 5.

Each sub-step must pass regression-guard against ALL samples and benchmarks in single-node and multinode.

**Why last:** restructuring while correctness is broken makes attribution impossible. Now that phases 1–8 are green, any regression introduced by structural cleanup is unambiguously the cleanup's fault.

## Phase 10 — Document + merge (task #11)

**Type:** docs

**Stop condition:** branch merged to `main` with passing CI for all samples + benchmarks single-node and multinode.

**Actions:**

1. Distill `docs/compiler/fix-attribution-log.md` into `docs/compiler/cps-failure-surfaces.md` and `ownership-proof-gaps.md` updates.
2. Update `docs/architecture/pass-placement.md` with final placement rules.
3. Update `docs/plan.md` status sections (replace 2026-04-12 snapshot with current).
4. Promote `docs/compiler/sample-suite-triage.md` from "snapshot" to "verified green baseline."
5. Merge `architecture/sde-restructuring` → `main`.
6. Mark all carts-finishing tasks `completed`.

## Delegation map

| Phase | Likely delegate(s) |
|---|---|
| 0 | none — direct user dialogue |
| 1 | none — direct CLI |
| 2 | `carts-pass-dev` if the depth-guard pattern is non-obvious |
| 3 | `carts-pass-dev` for the CreateDbs change; `carts-stage-diff` to verify the IR change at stage 5 |
| 4 | per item: `carts-miscompile-triage`, `carts-debug`, `carts-analysis-triage`, `carts-stage-diff` |
| 5 | `carts-pass-dev` for the gate additions |
| 6 | per item: `carts-distributed-triage` (always start here for multinode), then `carts-miscompile-triage` if the symptom reduces to wrong output |
| 7 | `carts-benchmark-triage` |
| 8 | `carts-distributed-triage` + `carts-benchmark-triage` |
| 9 | `carts-refactor-utils`, `carts-pass-dev`; consult `carts-review` before merging |
| 10 | none — direct user dialogue + git |
