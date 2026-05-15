# ARTS-RT Optimization Notes

This directory is the compiler-facing home for `arts-rt` lowering and low-level
runtime-call optimization planning. The source tree still uses `rt/`, but the
conceptual layer is `arts-rt`. It is not a semantic planning layer. It receives
a DB/EDT/epoch shape chosen by SDE/CODIR/ARTS, then lowers and tightens the
runtime-call representation.

For the target dialect split, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md). `arts-rt`
should only lower runtime API shape after SDE has chosen logical work, CODIR
has isolated codelets, and ARTS has bound that work to EDT/DB/Epoch objects.
The target per-dialect docs live under
[`../dialects/arts-rt/`](../dialects/arts-rt/). ARTS-RT-owned analyses are
listed in [`../dialects/arts-rt/analysis.md`](../dialects/arts-rt/analysis.md),
and ARTS-RT-owned optimizations are listed in
[`../dialects/arts-rt/optimizations.md`](../dialects/arts-rt/optimizations.md).

## Boundary

ARTS-RT owns:

- `arts_rt` runtime-call-shaped IR.
- EDT launch, parameter packing, state packing, and dependency slot wiring.
- Epoch runtime descriptors and continuation plumbing.
- `dep_gep`, dependency DB acquire, DB pointer GEP, and byte/window-local
  runtime addressing.
- Low-level cleanup such as data pointer hoisting, scalar replacement, GUID
  range call optimization, runtime query hoisting, and LLVM-facing metadata.

ARTS-RT must not own:

- OpenMP semantics.
- Memref partition policy.
- Physical DB layout selection.
- EDT distribution policy.
- Epoch topology decisions that depend on SDE/CODIR/ARTS semantic proofs.

## Pipeline Spine

ARTS-RT-shaped work appears in the end of `pre-lowering` and in
`arts-to-llvm`:

```text
pre-lowering:
  DbLowering
  EdtLowering
  LICM
  DataPtrHoisting
  ScalarReplacement
  EpochLowering
  VerifyPreLowered

arts-to-llvm:
  ConvertArtsToLLVM
  LoweringContractCleanup
  GuidRangCallOpt
  RuntimeCallOpt
  DataPtrHoisting
  Mem2Reg
  ControlFlowSink
  VerifyLowered
```

`DbLowering` remains ARTS-owned, but it feeds the ARTS-RT-shaped EDT and epoch
lowering pipeline.

## Optimization Focus

Pursue ARTS-RT work only after SDE/CODIR/ARTS have produced the intended
DB/EDT/epoch shape and traces still show low-level overhead:

- launch and CPS/continuation overhead;
- dependency slot packing and window-local `dep_gep`/DB pointer access;
- `edt_param_pack`, `state_pack`, and relaunch schema churn;
- scalar memref traffic in outlined EDT loops;
- repeated GUID reserve or pure runtime topology queries;
- LLVM alias and vectorization metadata.

If the trace still shows coarse DBs, wrong task grain, missing wavefront shape,
or missing physical layout, the fix belongs in SDE, CODIR, or ARTS first.

## Verification

ARTS-RT changes need focused tests under `lib/arts/dialect/rt/test/` or the
relevant ARTS lowering tests, then benchmark traces that isolate launch,
dependency, CPS, or runtime-call overhead.
