# Analysis Hotspots

Start here when the symptom depends on pass order, graph freshness, or stale metadata.

## ARTS APIs

- `include/carts/dialect/arts/Analysis/AnalysisDependencies.h`
- `include/carts/dialect/arts/Analysis/AnalysisManager.h`
- `lib/carts/dialect/arts/Analysis/AnalysisManager.cpp`

## High-Risk Docs

- `docs/compiler/phase-ordering-semantics.md`
- `tools/compile/Compile.cpp`
- `docs/compiler/pipeline.md`
- `docs/audits/2026-04-02-analysis-dependency-investigation.md`
- `docs/audits/2026-04-02-am-thread-safety-audit.md`

## High-Risk Patterns

- `AM->invalidate()`
- `invalidateAndRebuildGraphs`
- direct `dbAnalysis.invalidate()` or `edtAnalysis.invalidate()`
- missing or stale `k<Pass>_reads`
- queries after mutation without rebuild
- `--start-from` behavior that differs from a full pipeline run
