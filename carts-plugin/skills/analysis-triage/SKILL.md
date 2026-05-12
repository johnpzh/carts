---
name: carts-analysis-triage
description: Use when behavior depends on pass order, graphs look stale, metadata is inconsistent, or a pass likely needs AnalysisDependencies or narrower invalidation.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<pass-name | file>]
parameters:
  - name: pipeline_root
    type: str
    gather: "Path to the pipeline definition or pass file to analyze"
---

# CARTS Analysis Triage

Goal: determine whether the bug is caused by stale cached analysis, incorrect dependency declarations, or an invalid phase-ordering assumption.

Use bundled helpers when they fit:
- `scripts/find-analysis-usage.sh` — grep AnalysisManager queries for a pass or token
- `scripts/find-invalidation-sites.sh` — grep invalidation APIs and direct invalidate calls
- `scripts/find-pass-dependencies.sh` — locate `AnalysisDependencies` arrays and declarations
- `scripts/scan-analysis-hotspots.sh` — broad scan of order-sensitive analysis sites

Read these before editing invalidation behavior:
- `references/analysis-hotspots.md`
- `../debug/references/failure-signatures.md`
- `../debug/references/codepath-map.md`

## Triage Order

1. Confirm the order-sensitive symptom.
   - Does rerunning a pass change the result?
   - Does `--start-from` differ from a full pipeline run?
   - Does inserting a rebuild/invalidation make the bug disappear?
2. Identify what the pass reads and mutates.
   - Look for `AM->getDbAnalysis()`, `AM->getEdtAnalysis()`, `AM->getLoopAnalysis()`, `AM->getMetadataManager()`
   - Check the pass-local `AnalysisDependencies` arrays
3. Check invalidation boundaries.
   - `AM->invalidate()`
   - `AM->invalidateAndRebuildGraphs(module)`
   - `AM->invalidateFunction(func)`
   - direct `dbAnalysis.invalidate()` / `edtAnalysis.invalidate()`
4. Prefer the narrowest correct fix.
   - add or correct `reads`
   - add or correct `invalidates`
   - move a query before mutation or after rebuild
   - replace blanket invalidation only when safety is preserved
5. Re-run the affected tests and any order-sensitive reproducer.

## Key Files

- `include/arts/dialect/core/Analysis/AnalysisDependencies.h`
- `include/arts/dialect/core/Analysis/AnalysisManager.h`
- `lib/arts/dialect/core/Analysis/AnalysisManager.cpp`
- `docs/compiler/phase-ordering-semantics.md`
- `docs/plans/phase-ordering-design.md`
- `docs/audits/2026-04-02-analysis-dependency-investigation.md`

## Guardrails

- Never bypass analysis APIs to “fix” a stale-analysis symptom
- Do not widen invalidation without checking downstream compile-time cost
- If a pass mutates DB/EDT structure, assume cached facts may be stale until proven otherwise

## Hand-off

- Wrong output caused by stale analysis -> `carts-miscompile-triage`
- Runtime crash after analysis fix -> `carts-runtime-triage`
- Need to verify op lifecycle across stages -> `/dialect-trace`
- Need pipeline stage comparison -> `/stage-diff`
- Need a regression test -> `/create-test`

## Validation

Close only after the minimal reproducer and the normal regression suite both agree.
