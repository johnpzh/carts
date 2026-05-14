# Multinode-Only Failure Modes

These failure modes only appear in distributed runs. Single-node testing does not catch them. When a sample/benchmark passes single-node but fails under `--distributed-db` with `samples/arts_multinode.cfg` (or similar), this is your first reference.

## How to use

1. Confirm the failure is multinode-only: same compile flags except `--distributed-db`, single-node passes.
2. Match the runtime symptom to the surface table.
3. Inspect the named source location.
4. Apply the fix in the layer named under "fix usually belongs."
5. After fix, run the full regression guard including item 8 (multinode spot-check) on at least 3 different samples.

If the rubric here does not match, escalate to `carts-distributed-triage` with the captured logs.

## Failure modes

### 1. DB distribution attr missing or wrong

**Surface:** remote worker receives `GUID_NULL` or coarse-mode GUID when expecting a distributed handle.

**Symptom:** `(null reference)` at runtime. Worker hangs on `arts_db_acquire`. Or segfault.

**Root cause:** `DbDistributedOwnership` in Core DB refinement did not mark the DB, or the eligibility check is too conservative.

**Fix usually belongs in:**
- `lib/arts/dialect/core/Transforms/db/DbDistributedEligibility.cpp` (eligibility rules — too restrictive)
- `lib/arts/dialect/core/Transforms/db/DbDistributedOwnership.cpp` (attribute stamping — never reached)

### 2. Distributed window mismatch

**Surface:** Core materialization or DB refinement emits a stencil window that
is too narrow for a DB that Core will run distributed; remote workers cannot
reach halo elements.

**Symptom:** stencil values at boundary are garbage, or out-of-bounds access.

**Root cause:** the SDE halo/window contract is incomplete, or Core DB
refinement applied a local-only window while marking the DB as distributed.

**Fix usually belongs in:** SDE distribution planning only when the source
halo/window contract is wrong; otherwise Core DB refinement should validate the
`distributed` attr before narrowing.

### 3. Halo / remote-data ownership gap

**Surface:** distributed task needs data from a neighbor partition, but `min_offsets` / `max_offsets` are too narrow or missing.

**Symptom:** task hangs waiting for remote acquire to complete, or reads stale/wrong values from the halo region.

**Root cause:** Core DB refinement computed a halo for local-only operation and did not widen for distributed neighbors.

**Fix usually belongs in:** `lib/arts/dialect/core/Transforms/db/DbTransformsPass.cpp` or the DB refinement helper that stamps the window. When marking a DB distributed, validate that halo bounds cover all transitive neighbors. See `docs/compiler/ownership-proof-gaps.md` §2.3 for the full proof obligation.

### 4. GUID coherence (same data, different handles on nodes)

**Surface:** worker on node A acquires a DB with GUID_A; worker on node B acquires the same logical DB with GUID_B; they operate on incoherent copies.

**Symptom:** computation produces NaN or silently wrong values. Incoherent writes go undetected (no diagnostic).

**Root cause:** `DbAllocOp` + `DbAcquireOp` do not carry explicit node-locality or slice-coherence markers.

**Fix usually belongs in:** `ConvertArtsToLLVM` codegen — when lowering a distributed acquire, ensure the same logical DB across all nodes maps to coherent remote-slice handles.

### 5. Init-per-node ordering

**Surface:** epoch initializer runs on node 0 and tries to initialize all DBs for all nodes, but a distributed DB on node N is uninitialized.

**Symptom:** worker on node N hangs in init loop, or accesses uninitialized remote memory.

**Root cause:** `EpochLowering` does not gate DB initialization on local ownership, or SDE/Core materialization did not express per-node initialization as task work.

**Fix usually belongs in:** `lib/arts/dialect/rt/Conversion/ArtsToRt/EpochLowering.cpp` — check the `distributed` attr; if set, gate init to the owning node only.

### 6. Polygeist memref→pointer conversions across nodes

**Surface:** Polygeist generates `llvm.ptrtoint` or `memref.extract_aligned_pointer_as_index` to pass raw pointers across function boundaries; distributed worker on node B receives a pointer from node A and cannot dereference it.

**Symptom:** segfault or bus error on the remote worker; or opaque data corruption from pointer reinterpretation.

**Root cause:** `ConvertArtsToLLVM` / late lowering does not add serialization or marshalling for distributed handles.

**Fix usually belongs in:** codegen — wrap distributed DB pointers in serializable handles, or reconstruct the handle from `(GUID, partition metadata)` on the receiver.

### 7. Dep routing mismatch (distributed)

**Surface:** `CPSDepRouting` attribute says slot 0 is a distributed DB dep, but the actual dep at slot 0 is a local timing DB.

**Symptom:** worker hangs, or receives wrong dep signal from wrong partition.

**Root cause:** `EpochOpt` CPS-8 carry re-analysis, or `EpochLowering` propagation, did not account for mixed local/distributed deps.

**Fix usually belongs in:** `lib/arts/dialect/rt/Conversion/ArtsToRt/EpochLowering.cpp` — validate `CPSDepRouting` layout against actual carry arity; cross-check the `distributed` attr on each referenced DB.

## Diagnosis approach

If a sample passes single-node but fails multinode:

1. Generate dumps for both modes:

   ```bash
   dekk carts compile samples/<sample> -O3 -o /tmp/single
   dekk carts compile samples/<sample> --distributed-db -O3 -o /tmp/multi
   dekk carts compile samples/<sample> --distributed-db -O3 --all-pipelines -o /tmp/multi-stages/
   ```

2. Diff single-node vs distributed at each stage boundary:

   ```bash
   dekk carts compile samples/<sample> --pipeline=post-db-refinement > /tmp/single-post-db.mlir
   dekk carts compile samples/<sample> --distributed-db --pipeline=post-db-refinement > /tmp/multi-post-db.mlir
   diff /tmp/single-post-db.mlir /tmp/multi-post-db.mlir | head -200
   ```

3. Capture node-level logs: `arts.log`, `omp.log`, `cluster.json`, `n0.json`, `n1.json`. The distributed-triage skill has scripts for this.

4. If the bug reduces to wrong output rather than multinode-specific structure, hand off to `carts-miscompile-triage`.

## Reference docs

- `docs/architecture/arts-rt-dialect.md` — RT lowering contract
- `docs/compiler/ownership-proof-gaps.md` — formal ownership obligations
- `docs/compiler/cps-failure-surfaces.md` — CPS-related failure taxonomy
- `carts-plugin/skills/distributed-triage/SKILL.md` — sister triage skill
- `carts-plugin/skills/distributed-triage/references/distributed-checklist.md`
- `carts-plugin/skills/distributed-triage/references/distributed-codepaths.md`
