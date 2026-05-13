# ARTS RT Optimization Notes

This directory is the compiler-facing home for RT lowering and low-level
runtime-call optimization planning. RT is not a semantic planning layer. It
receives a DB/EDT/epoch shape chosen by SDE and Core, then lowers and tightens
the runtime-call representation.

## Boundary

RT owns:

- `arts_rt` runtime-call-shaped IR.
- EDT launch, parameter packing, state packing, and dependency slot wiring.
- Epoch runtime descriptors and continuation plumbing.
- `dep_gep`, dependency DB acquire, DB pointer GEP, and byte/window-local
  runtime addressing.
- Low-level cleanup such as data pointer hoisting, scalar replacement, GUID
  range call optimization, runtime query hoisting, and LLVM-facing metadata.

RT must not own:

- OpenMP semantics.
- Tensor partition policy.
- Physical DB layout selection.
- EDT distribution policy.
- Epoch topology decisions that depend on SDE/Core semantic proofs.

## Pipeline Spine

RT-shaped work appears in the end of `pre-lowering` and in `arts-to-llvm`:

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

`DbLowering` remains Core-owned, but it feeds the RT-shaped EDT and epoch
lowering pipeline.

## Optimization Focus

Pursue RT work only after SDE/Core have produced the intended DB/EDT/epoch
shape and traces still show low-level overhead:

- launch and CPS/continuation overhead;
- dependency slot packing and window-local `dep_gep`/DB pointer access;
- `edt_param_pack`, `state_pack`, and relaunch schema churn;
- scalar memref traffic in outlined EDT loops;
- repeated GUID reserve or pure runtime topology queries;
- LLVM alias and vectorization metadata.

If the trace still shows coarse DBs, wrong task grain, missing wavefront shape,
or missing physical layout, the fix belongs in SDE or Core first.

## Verification

RT changes need focused tests under `lib/arts/dialect/rt/test/` or the relevant
Core lowering tests, then benchmark traces that isolate launch, dependency, CPS,
or runtime-call overhead.
