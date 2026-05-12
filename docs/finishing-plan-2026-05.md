# CARTS Finishing Plan — 2026-05

End-to-end plan for closing out the architecture/sde-restructuring branch:
the single-node sample closure status, the multinode + benchmark work, the
architectural cleanups still pending, and the skills/competencies each task
needs.

Companion to `docs/plan.md` (SDE pipeline) and `docs/architecture/charter-decisions.md`.

## 1. Current state — 2026-05-10

### Suite

| Suite | Status |
|---|---|
| Pass tests | **93 pass + 1 XFAIL / 94 ✓** |
| E2E samples (single-node, `-j1`) | **19 pass + 7 XFAIL / 26 ✓** |
| Architecture | clean — `Polygeist → SDE → ARTS core → ARTS RT` |
| Cost model in core | **0** (charter D3) |
| Flakiness | eliminated (`-j 1` for e2e) |
| Original audit violations | **0 remaining** |

### Cleanups landed this branch

- **A** — Dead `Sde::*` attrs + `AffinityDb` + `placeEdtsByAffinity`
- **B** — Cost-model paths stripped from `DistributionHeuristics` + `EdtHeuristics`
- **C** — `DbPartitioning` pass deleted entirely (~5,000 lines)
- **D** — Use-after-free fix in `assembleAndApplyRewritePlan`
- **E** — `PersistentRegionCostModel` + `StructuredKernelPlanAnalysis` deleted
- **F** — Cost-model removed from `selectReductionStrategies`
- **G** — `core/Conversion/OmpToSde/` moved to `sde/Conversion/PolygeistToSde/`
- **Plus** — flakiness fix in lit harness; `parallel_for_block` shipped via
  pattern-matched globalize-inner-scf.for in CreateDbs.
- **Stage A** — the 5 tracked deterministic single-node sample failures are
  fixed (`task.cpp`, `smith-waterman.c`, `cholesky_static.c`, `stencil.c`,
  `matrixmul.c`), and stale `parallel_for_stencil.c` XFAIL was removed.

Total deletion: ~6,000+ lines of dead/cost-model code.

## 2. Single-node sample closure — tracked failures resolved

The 5 deterministic sample failures tracked by this plan are now fixed. Current
single-node e2e status, using the required serialized harness:

```bash
dekk carts lit -j1 --verbose tests/e2e
```

Result: **19 pass + 7 XFAIL / 26**.

Resolved samples:

| Sample | Status | Main fix area |
|---|---|---|
| `task.cpp` | PASS | `SdeCuCodeletOp` captures + scalar/codelet cloning |
| `smith-waterman.c` | PASS | stack-array threading + non-scalar single-region lowering |
| `cholesky_static.c` | PASS | stack-array threading + non-scalar single-region lowering |
| `stencil.c` | PASS | wrapper-mediated escape hoist + tensor safety guard |
| `matrixmul.c` | PASS | `local_accumulate` epoch serialization after `rec_dep` |
| `parallel_for_stencil.c` | PASS | stale XFAIL removed after it became green |

The notes below are retained as the pre-fix investigation record; they are no
longer open Stage A work items.

### Historical status update — 2026-05-10 evening

Active work landed in `RaiseToTensor.cpp` and `ConvertToCodelet.cpp`:

- **`isInSupportedParent`** extended to treat `scf.if` as transparent (with companion lowering support in `walkBlock`). Allocas whose load/store users are inside `scf.if` now reach the threading machinery.
- **`materializeReadOnlyMemref` + `traceTensorBoundaryMemref`** added in `ConvertToCodelet.cpp`. Outer read-only memrefs used in the codelet body get captured as tensors via `mu_token`, then re-materialized as local memrefs inside the codelet via `SdeMuTensorToMemrefOp`. This handles smith-waterman-style fixed-size stack arrays in a capture-respecting way.

**Effect on the alloca cluster:** the diagnostic shifted from `memref.alloca captured by cu_region` to the next layer down — `arith.addi using value defined outside the region` (cholesky_static), `polygeist.memref2pointer ...` (smith-waterman). The bug shape converged: all three (task, smith-waterman, cholesky_static) are now in **the same cluster** with the SAME fix path:

> Pure ops in the codelet body have operand chains that bottom out at outer-scope scalar SSA values (function arg, loop counter, computed value). The existing `clonePure` recursion silently produces invalid IR when the chain doesn't bottom out at mappable/cloneable values.

This was the pre-fix diagnosis: MEMREF capture was handled, but SCALAR SSA
capture still needed explicit support. Two viable paths were considered:

1. **Wrap scalars in 0-d tensors** — make `mu_data` + `mu_token` accept 0-d tensor wrappers; convert outer scalars to 0-d tensors at the cu_region boundary; extract back to scalars inside the codelet body via `tensor.extract`.
2. **Move scalar producers into the codelet body** — for outer pure ops whose ENTIRE operand chain is also "moveable or constant", move the chain into the codelet. The chain bottoms out at either constants (rematerializable) or impure task-private ops (movable per the agent design).

The landed implementation follows the capture/move path instead of relaxing
codelet isolation.

### 2.1 task.cpp — cu_codelet outer-SSA capture (resolved)

**Symptom:** `'arith.remsi' op using value defined outside the region`

**Root cause:** `SdeCuCodeletOp` has `IsolatedFromAbove` by design (intentional
contract — codelets are pure dataflow units with explicit inputs via mu_tokens).
`ConvertToCodelet.cpp::convertCuRegion` (lines 75–349, body cloning at 203–244)
fails when the codelet body references an impure outer op (`func.call @rand()`).
It can clone pure outer ops but rejects impure ones outright.

**Fix design (capture-based, NOT relaxation):**

In `ConvertToCodelet.cpp`, add a classification pass before body cloning:

```
classifyExternalCaptures():
  for each op in opsToMove:
    for each operand:
      if operand is mapped or internal → skip
      if defining op is pure → cloneable (existing path)
      if defining op is impure AND task-private → moveable
      if defining op is impure AND shared → reject

move impure task-private ops into codelet body in program order
clone body ops as before — moved-in ops resolve naturally
```

"Task-private" = all uses of the result are inside the cu_region body.

**Files:** `lib/arts/dialect/sde/Transforms/state/codelet/ConvertToCodelet.cpp`

**Effort:** ~55 lines net addition. Pattern already exists in
`RaiseMemrefToTensor.cpp:765–778` for cu_task — adapt to cu_region.

**Risks:** moving non-task-private impure ops would break semantics; strict
classification (every result user inside body) prevents this.

### 2.2 smith-waterman.c, cholesky_static.c — fixed-size stack arrays (resolved)

**Symptom:** `memref.alloca of type 'memref<33xi8>' captured by cu_region but
was not raised to tensor` (also `memref<33x33xi32>`, `memref<128x128xf64>`)

**Root cause:** `RaiseToTensor::isInSupportedParent` (line 91–109) rejects
load/store users inside `scf.if`. The samples have stack arrays used
inside argv-validation `if (argc >= N)` guards, so all access happens inside
scf.if, even though the alloca itself is in function entry.

Strategy A (move alloca inside cu_region) is rejected: both samples use the
arrays BEFORE the parallel region (initialization), INSIDE (compute), and
AFTER (verification). The lifetime extends beyond the cu_region.

**Fix design (proper threading, NOT verifier relaxation):**

Extend `isInSupportedParent` to recognize scf.if as supported when the access
is read-only within that scf.if. Read-only access means no merge is needed —
the tensor value flows through unchanged. The existing
`collectAccessedAllocas` machinery then adds the alloca as an iter_arg of the
enclosing cu_region/scf.for.

```
isAllocaReadOnlyInScfIf(loadOrStore, ifOp):
  collect all load/store users of the same alloca within ifOp
  return true iff all users are memref::LoadOp (no stores)

isInSupportedParent (extended):
  if parent is scf.if AND (op is read-only within it):
    walk through (transparent)
  else if parent is scf.if AND op is a write:
    return false (full threading still required)
```

This handles smith-waterman (all reads inside `if (argc >= 2)` guards) and
cholesky_static (all reads inside argument-check guards).

**Files:** `lib/arts/dialect/sde/Transforms/state/raising/RaiseToTensor.cpp`
(lines 85–140)

**Effort:** ~20 lines.

**Risks:** monotonic improvement — only relaxes a check, no previously-passing
sample can fail because of this. A future sample with WRITES inside scf.if
will still be rejected with a clear diagnostic. That case will need full
scf.if iter-arg threading (separate, larger fix).

### 2.3 stencil.c — wrapper-of-pointer SROA dominance (resolved)

**Symptom:** `memref.alloca of type 'memref<memref<?xi32>>' captured by
cu_region but was not raised to tensor`

**Root cause:** Polygeist generates a wrapper alloca holding a heap memref
pointer (`int *A = malloc(N)` → `memref<memref<?xi32>>`). MemrefNormalization's
`transformSimpleWrapper` SROA path correctly identifies the pattern, but the
dominance check at line 1848–1854 fails: the heap `memref.alloc` lives inside
inner scf.if, and some wrapper loads are in deeper-nested scf.if branches that
aren't strictly dominated.

The preprocessing hoist at line 302–371 only follows DIRECT users to detect
escapes. It misses transitive escape through wrappers (alloc → store-to-wrapper
→ load-from-wrapper-escapes-scf.if).

**Fix design:**

Extend the escape-detection in the preprocessing hoist to follow store-to-wrapper
chains:

```
For each escaping alloc:
  direct users → if any escapes parent scf.if, hoist (existing)
  wrapper-mediated users:
    for each direct memref.store storing this alloc into a wrapper:
      for each memref.load from that wrapper:
        if any load-user escapes parent scf.if, hoist
```

Once hoisted to dominate all transitively-reaching loads, the existing
`transformSimpleWrapper` SROA succeeds.

**Files:** `lib/arts/dialect/sde/Conversion/PolygeistToSde/MemrefNormalization.cpp`
(lines 302–371)

**Effort:** ~30 lines extension to the existing escape-detection loop.

**Risks:** hoisting too far — mitigated by only triggering on confirmed
transitive escape. Other samples whose allocs are intentionally inside scf.if
are unaffected (no transitive escape detected).

### 2.4 matrixmul.c — local_accumulate ordering (resolved)

**Symptom:** compiles cleanly, but N=32+ lost one or more `k` block
contributions for a `C` tile.

**Root cause:** each child EDT received the right loop bounds, but
`local_accumulate` launches were all placed into the same epoch without
preserving the sequential fold order for the coarse `C` DB update.

**Fix landed:**

- `EdtLowering.cpp` preserves `arts.reduction_strategy` on
  `arts_rt.edt_create`.
- `EpochLowering.cpp` detects `local_accumulate` creates in epochs that already
  need a blocking wait and inserts `arts_rt.wait_on_epoch` after the matching
  `arts_rt.rec_dep`, so dependencies are registered before the next
  local-accumulate launch is serialized.
- `tests/e2e/parallel_for_stencil.c` had a stale XFAIL and now runs as a normal
  passing e2e test.

**Verification:** `matrixmul.c` passes alone, in the focused 5-sample set, and
in the full serialized e2e run.

### 2.5 Remaining expected failures

The current non-green samples are explicit XFAILs and need separate triage
before claiming a fully passing 26/26 sample suite:

- `convolution.cpp`
- `deps.c`
- `dotproduct.c`
- `mixed_access.c`
- `mixed_orientation.c`
- `parallel_for_reduction.c`
- `parallel_for_single.c`

## 3. Architectural work still pending

### 3.1 Cleanup H — Move `arts.omp_dep` to SDE dialect

`arts.omp_dep` is currently a core dialect op (`core/IR/Ops.td:590`) but is
purely a pre-SDE Polygeist-bridge representation. It's:
- Created by `sde/Conversion/PolygeistToSde/HandleDeps.cpp`,
  `MemrefNormalization.cpp`
- Consumed by `sde/Conversion/OmpToSde/ConvertOpenMPToSde.cpp:161`
- Re-created post-SDE in `core/Conversion/SdeToArts/SdeToArtsPatterns.cpp:594`
  for stragglers

The user's architectural model is `Polygeist → SDE → ARTS core → ARTS RT`.
This op should live in SDE.

**Effort:** 5-touch refactor — move op def to `sde/IR/SdeOps.td`, update 5
producers/consumers, update tests.

**Defer until correctness is locked.**

### 3.2 Phase 6 — Multinode validation

Stop condition: all 26 samples pass under
`samples/arts_multinode.cfg` (2-process local TCP loopback).

Currently: zero multinode tests exist. `tests/e2e_multinode/lit.local.cfg`
infra stub is in place from Phase 1.

**Steps:**
1. After the remaining single-node XFAILs are fixed or explicitly descoped,
   add multinode-specific lit rules
2. For each sample, compile with `--distributed-db` and run with the multinode
   config
3. Triage multinode-only failures using
   `carts-finishing/references/multinode-failures.md`

**Effort:** unknown. Estimated 1-2 weeks of triage given the runtime
distributed-db code path is untested at scale.

### 3.3 Phase 7 + 8 — Benchmark suite

Re-baseline benchmarks at `external/carts-benchmarks/` against this branch
(2026-03-11 snapshot is 5+ weeks stale). Then green the suite single-node and
multinode.

**Effort:** depends on how many regressed. Phase 3's `parallel_for_block`
fix likely helped many parallel-loop benchmarks.

### 3.4 Phase 9 — Pass-placement final cleanup

After all correctness work is done, the final structural cleanup:
- Audit pass placement against the dialect charter
- Move any remaining misplaced passes
- Add SDE-contract guards to any core passes that re-derive what SDE stamps

The original audit listed 5 violations; with `DbPartitioning` deleted, they're
all resolved or moot. Phase 9 should re-audit and confirm.

### 3.5 Phase 10 — Documentation + branch merge

- Update `docs/architecture/pass-placement.md` to reflect current state
- Update `docs/compiler/sample-suite-triage.md` with the 19 pass + 7 XFAIL
  baseline
- Distill `docs/compiler/fix-attribution-log.md` into `cps-failure-surfaces.md`
  + `ownership-proof-gaps.md`
- Merge `architecture/sde-restructuring` → `main`

## 4. Skills + competencies needed

| Task | Skill | Required knowledge |
|---|---|---|
| 2.1 task.cpp codelet | MLIR pass dev | IRMapping, op cloning, cu_codelet semantics |
| 2.2 fixed-size arrays | MLIR pass dev | RaiseToTensor's threading model, alloca lifetime analysis |
| 2.3 stencil wrapper | MLIR pass dev | Dominance, escape analysis, transitive use chains |
| 2.4 matrixmul local_accumulate | RT lowering | reduction-strategy propagation, epoch waits, ARTS dependency registration order |
| 3.1 omp_dep move | TableGen + dialect surgery | MLIR op definitions, builder/parser/printer |
| 3.2 multinode | ARTS runtime | distributed-db, GUID coherence, halo bounds |
| 3.3 benchmarks | benchmark-triage skill | LULESH, PolybenchC, kastors-jacobi specifics |

The existing skills under `carts-plugin/skills/` cover much of this. Specifically:
- `carts-finishing` — the orchestrator skill (this plan is its target)
- `carts-miscompile-triage` — for matrixmul-style runtime correctness
- `carts-distributed-triage` — for Phase 6 multinode work
- `carts-stage-diff` — for IR diffing across stages
- `carts-analysis-triage` — for stale-analysis bugs
- `carts-pass-dev` — for new pass authoring (codelet capture, scf.if threading)
- `carts-simplify` and `carts-review` — for final complexity, utility, and
  regression-risk checks before calling a change done

A new skill may be useful: `carts-rt-lowering-triage` for matrixmul-class
epoch/dependency-registration issues. Filing as future work.

## 5. What's missing

Items not currently scoped but eventually needed:

1. **Runtime support for `default_ports=0` (OS-pick free port)** — would let
   e2e tests run in parallel without `-j 1`. Currently, e2e is serialized.
2. **Snapshot fixture for db_mode_tightening test** — currently XFAIL'd. Needs
   regenerated `tests/inputs/snapshots/activations_openmp_to_arts.mlir`.
3. **Full scf.if iter-arg threading in RaiseToTensor** — for samples with
   WRITES inside scf.if (none currently failing on this, but a real hardening).
4. **CPS forwarding correctness** — task #20 (deleted) noted CPS-8 carry
   re-analysis edge cases; keep this on the multinode/benchmark watch list.
5. **Benchmark performance baseline** — the 2026-03-11 numbers are stale.
   After Phase 7, generate fresh perf numbers per benchmark.
6. **Multi-node lit infrastructure** — currently a stub. Needs proper test
   harness with port-allocation strategy, log inspection, per-node counter
   diffing.

## 6. Execution order (recommended)

**Stage A — Tracked single-node closure: DONE**
1. Fixed 2.2 (fixed-size arrays) — unblocked 2 samples
2. Fixed 2.3 (stencil wrapper) — unblocked 1 sample
3. Fixed 2.1 (task codelet) — unblocked 1 sample
4. Fixed 2.4 (matrixmul local_accumulate) — unblocked 1 sample
5. Removed stale `parallel_for_stencil.c` XFAIL

After Stage A: e2e is at **19 pass + 7 XFAIL / 26** under `-j1`.

**Stage B — Multinode (1-2 weeks):**
5. Phase 6 multinode infrastructure + per-sample triage

**Stage C — Benchmarks (1 week):**
6. Phase 7 single-node baseline
7. Phase 8 multinode benchmarks

**Stage D — Cleanup + merge (3-5 days):**
8. Cleanup H (omp_dep dialect move)
9. Phase 9 (final pass placement audit)
10. Phase 10 (docs + merge)

Total: ~4-6 weeks of focused engineering.

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Each fix introduces regressions in passing samples | Pattern-match guards (proven in parallel_for_block fix); regression-guard checklist after every change |
| Multi-hour fixes don't fit /loop iterations | Acknowledged — schedule dedicated focused sessions; /loop is for diagnostics + small surgical fixes |
| Multinode triage longer than estimated | Stage B can be done incrementally; partial green is still progress |
| Benchmark perf regressions vs 2026-03-11 baseline | Charter D3 explicitly accepts perf changes — simpler architecture wins |
| local_accumulate serialization touches epoch lowering — broader impact | Test against all currently-passing samples and reduction-strategy cases |

## 8. Source-of-truth links

- This plan: `docs/finishing-plan-2026-05.md`
- Charter decisions: `docs/architecture/charter-decisions.md`
- SDE plan (older): `docs/plan.md`
- Sample triage: `docs/compiler/sample-suite-triage.md`
- Failure taxonomy: `docs/compiler/cps-failure-surfaces.md`,
  `docs/compiler/ownership-proof-gaps.md`
- Skill orchestrator: `carts-plugin/skills/carts-finishing/`
