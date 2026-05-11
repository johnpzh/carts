# Charter Decisions

Resolutions for the 5 open questions identified by the architecture re-audit (`architecture-reaudit-2026-04-11.md`) and the pass-placement investigation (`pass-placement-investigation-2026-04-11.md`).

These decisions are the input to Phase 9 (pass-placement cleanup) of the carts-finishing plan. They are intentionally conservative — every "defer" answer keeps current behavior so phases 1–8 (correctness work) can proceed without re-architecting in parallel.

**The user may override any decision below at any time.** This file is the source of truth; updates here propagate to placement decisions and Phase 9 sub-tasks.

## D1. DestinationStyleOpInterface on SDE ops

**Decision:** Defer. Keep transient `linalg.generic` / `tensor` carriers as the SDE analysis window. Do not promote SDE ops to `DestinationStyleOpInterface` ops yet.

**Rationale:** the transient carrier model is the design that landed and works for current samples. Upgrading to DSI is an 8–10h refactor that competes for attention with correctness work. Once benchmarks are green (post Phase 8), reconsider — the upgrade unblocks linalg `tile-and-fuse`, but only matters once the structural plumbing it would compose with is correct.

**Implication:** the LoopReordering migration to SDE (D5) stays deferred too, since it depends on DSI.

## D2. Scope of `SdeStructuredSummaries`

**Decision:** Own all semantic planning. `SdeStructuredSummaries` (or a closely-coupled `SdeWavefrontAnalysis` in the same pass-group) becomes the single source for elementwise / stencil / matmul / reduction / wavefront / Jacobi family detection AND tile / halo geometry hints.

**Rationale:** the audit identified this as the largest violation of Invariant 5 (cost-model-driven decisions belong in the decision-owner). Splitting detection across SDE and core perpetuates the "core re-derives what SDE already computed" problem and blocks ARTS from being a pure consumer of contracts.

**Implication:** Phase 9 sub-step 5 (the 12h decoupling work) targets DepTransforms / KernelTransforms specifically — moving family detection into SDE and turning core passes into consumers.

## D3. ARTSCostModel everywhere

**Decision: REMOVE entirely.** No cost model. Distribution decisions are config-driven: if the runtime config has `node_count > 1`, do distributed; otherwise local. Block sizes split the static trip count evenly across `workerCfg.totalWorkers`. No thresholds, no pattern-aware heuristics, no amortization scoring.

**Rationale (revised 2026-05-10 by user):** the 42 voting heuristics in PartitioningHeuristics and the cost-model paths in DistributionHeuristics produce unverifiable tuning that masks correctness bugs. Simpler is better. SDE owns analysis (already stamps classification + halo + owner dims); core just realizes contracts; the runtime config decides where to run.

**Implication:**
- `DistributionHeuristics::computeLoopCoarseningDecision` is now a 15-line config-driven function.
- `chooseWavefront2DTilingPlan`, `evaluateStencilStripCostModel`, `computeCoarsenedBlockHint` deleted.
- `PartitioningHeuristics::evaluatePartitioningHeuristics` (42 H1.* heuristics) targeted for deletion in Cleanup C.
- `selectReductionStrategies` cost-model path in EdtTransformsPass targeted for deletion in Cleanup F.
- `AnalysisManager::getCostModel()` will eventually be removed once all callers are gone.

## D4. Backend-neutral SDE narrative

**Decision:** Reframe as ARTS-optimized SDE. Update docs to describe SDE as "the analysis-and-optimization layer for the ARTS runtime" rather than aspiring to be a portable scheduling IR for Legion / StarPU / GPU.

**Rationale:** the namespace, schedule kinds, conversion target, and contract attributes are all ARTS-tied. The portability claim misleads contributors and forces awkward abstractions. Reframing is honest, costs nothing, and does not preclude a future portable layer if one is justified later.

**Implication:** the SDE dialect docs (`docs/architecture/sde-dialect.md`, `sde-tensor-first-pipeline.md`) get a small intro update in Phase 10. No code change.

## D5. LoopReordering migration to SDE

**Decision:** Defer. LoopReordering stays in `core/Transforms/` until DSI (D1) lands. When DSI is reconsidered post-Phase 8, LoopReordering migrates to SDE as `SdeLoopInterchange` operating on `linalg.interchange`.

**Rationale:** LoopReordering operates structurally on `arts.for`. The cost-model-driven version belongs at the tensor / linalg level (SDE domain), but that requires DSI. Without DSI, moving the pass to SDE just relocates the same structural rewrite — no semantic improvement.

**Implication:** Phase 9 lists this as future work. Not in scope for the current plan.

## Summary

| ID | Question | Decision | Effort saved (deferred) | Phase 9 impact |
|---|---|---|---|---|
| D1 | DSI on SDE ops? | Defer | 8–10h | reduces |
| D2 | SdeStructuredSummaries owns all semantic planning? | Yes | (work moves into Phase 9) | step 5 in scope |
| D3 | ARTSCostModel everywhere? | Defer to post-Phase 8 | 12–16h | reduces |
| D4 | Backend-neutral SDE narrative? | Reframe as ARTS-optimized | 0 (docs only) | step 0 (Phase 10 docs) |
| D5 | LoopReordering migration to SDE? | Defer (blocked by D1) | depends on D1 | reduces |

## Revision log

| Date | Decision changed | New value | Rationale |
|---|---|---|---|
| 2026-05-10 | initial | all 5 set to defer/conservative | Phase 0 of carts-finishing plan; conservative defaults to focus phases 1–8 on correctness |
