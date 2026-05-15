# Runtime Debug Checklist

Use this checklist after compilation succeeds but the produced program fails at runtime.

## Build Mode

- normal runtime: `dekk carts build --arts`
- verbose runtime logging: `dekk carts build --arts --debug 3`
- counters:
  - `--counters 1` timing
  - `--counters 2` workload characterization
  - `--counters 3` overhead analysis

## Symptom Split

- crash / assertion: inspect stderr first, then sanitizer/debug build
- hang / timeout: inspect epochs, continuation edges, and progress threads
- slow runtime: gather counters before changing compiler heuristics
- distributed-only issue: switch to `carts-distributed-triage`

## Artifacts to Keep

- executable stderr/stdout
- `arts.log`
- `omp.log`
- `cluster.json`
- `n*.json`

## Codepaths Worth Reading

- `lib/arts/dialect/core/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/arts/dialect/rt/Conversion/ArtsToRt/EdtLowering.cpp`
- `lib/arts/dialect/rt/Conversion/ArtsToRt/EpochLowering.cpp`
- `external/arts/` runtime code when compiler-generated structure looks correct
