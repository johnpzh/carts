---
name: carts-dialect-trace
description: Use when debugging lowering paths, understanding operation placement across SDE/CODIR/ARTS/ARTS-RT, or verifying dialect boundary invariants.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Agent
argument-hint: [<op-name> | boundary | verify]
parameters:
  - name: op_name
    type: str
    gather: "Operation to trace (e.g., 'arts.db_acquire', 'arts_rt.edt_create', 'sde.cu_task')"
---

# CARTS Dialect Operation Tracer

## Purpose

Trace an operation's lifecycle across the pipeline stages and dialect layers
(SDE, CODIR, ARTS, ARTS-RT). Understand where ops are created, transformed,
and destroyed.

## Dialects

| Dialect | Namespace | Stages | Purpose |
|---------|-----------|--------|---------|
| SDE | `sde::` | 3 (`sde-planning`) | Semantic contracts, pattern classification, MU/CU/SU planning |
| CODIR | `codir::` | 4-5 | Codelet isolation, explicit deps/params, token-local views |
| ARTS | `arts::` | 5-12 | DB/EDT/epoch orchestration and analysis |
| ARTS-RT | `arts_rt::` / `rt::` | 12-13 | Runtime ABI and LLVM-facing call mapping |

## Dialect Boundaries

```
C/OMP source
  → [Stage 3]    sde-planning: OMP → SDE (ConvertOpenMPToSde) →
                 SDE transforms and MU/CU/SU planning
  → [Stage 4]    sde-to-codir: SDE codelets → isolated CODIR codelets
  → [Stage 5]    codir-to-arts: CODIR deps/codelets → ARTS DB/acquire/EDT
  → [Stage 6-11] ARTS transforms (arts.edt, arts.db_*, arts.epoch, scf.for)
  → [Stage 12]   ARTS → ARTS-RT lowering in pre-lowering
                 (EdtLowering.cpp, EpochLowering.cpp, DbLowering.cpp)
  → [Stage 13]   ARTS-RT → LLVM lowering in arts-rt-to-llvm
                 (ConvertArtsRtToLLVM + rt-specific RuntimeCallOpt etc.)
```

## Verification Barriers

| Verifier | After Stage | Checks |
|----------|-------------|--------|
| VerifyCodir | 4 (sde-to-codir) | CODIR deps/params/yields are explicit and isolated |
| VerifySdeLowered | 5 (codir-to-arts) | No SDE ops survive |
| VerifyArtsObjectsOnly | 5 (codir-to-arts) | No transient semantic carrier survives into ARTS refinement |
| VerifyEdtCreated | 5 (codir-to-arts) | EDTs created from CODIR codelets |
| VerifyPreLowered | 12 (pre-lowering) | No arts.edt/epoch/db_* survive |
| VerifyLowered | 13 (arts-rt-to-llvm) | No ARTS/ARTS-RT ops survive |

## Op Lifecycle Quick Reference

### Core Ops
| Op | Created | Lowered | Stages Active |
|----|---------|---------|---------------|
| `arts.edt` | codir-to-arts (5) | pre-lowering (12) | 5-12 |
| `arts.db_alloc` | codir-to-arts (5) or create-dbs materialization (7) | pre-lowering (12) | 5/7-12 |
| `arts.db_acquire` | codir-to-arts (5) or create-dbs materialization (7) | pre-lowering (12) | 5/7-12 |
| `arts.db_ref` | create-dbs (7) | pre-lowering (12) | 7-12 |
| `arts.epoch` | codir-to-arts (5) or epochs (11) | pre-lowering (12) | 5-12 |
| `arts.barrier` | codir-to-arts (5) | epochs (11) | 5-11 |

### RT Ops
| Op | Created | Lowered | Stages Active |
|----|---------|---------|---------------|
| `arts_rt.edt_create` | pre-lowering (12) | arts-rt-to-llvm (13) | 12-13 |
| `arts_rt.create_epoch` | pre-lowering (12) | arts-rt-to-llvm (13) | 12-13 |
| `arts_rt.db_create` | pre-lowering (12) | arts-rt-to-llvm (13) | 12-13 |
| `arts_rt.rec_dep` | pre-lowering (12) | arts-rt-to-llvm (13) | 12-13 |

### SDE Ops
SDE planning ops are created in `sde-planning`; codelet ops are consumed by
`sde-to-codir`, and no SDE op may survive `codir-to-arts`:

| Op | Created | Lowered |
|----|---------|---------|
| `sde.cu_codelet` | ConvertOpenMPToSde / SDE planning (stage 3) | ConvertSdeToCodir (stage 4) |
| `sde.cu_task` | ConvertOpenMPToSde / SDE planning (stage 3) | ConvertSdeToCodir (stage 4); leftovers fail VerifySdeLowered |
| `sde.su_iterate` | ConvertOpenMPToSde / SDE planning (stage 3) | SDE-to-CODIR scheduling-unit materialization; leftovers fail VerifySdeLowered |
| `sde.mu_reduction_decl` | ConvertOpenMPToSde / SDE planning (stage 3) | SDE/CODIR reduction materialization; leftovers fail VerifySdeLowered |

## Tracing Commands

### Find where an op is created
```bash
grep -rn 'create<.*OpName>' lib/carts/ --include='*.cpp'
grep -rn 'builder\.create<.*OpName>' lib/carts/ --include='*.cpp'
```

### Find where an op is consumed/matched
```bash
grep -rn 'isa<.*OpName>\|dyn_cast<.*OpName>\|cast<.*OpName>' lib/carts/ --include='*.cpp'
```

### Find where an op is erased/replaced
```bash
grep -rn 'replaceOp\|eraseOp' lib/carts/ --include='*.cpp' | grep -i 'OpName'
```

### Find conversion patterns for an op
```bash
grep -rn 'OpName.*Pattern\|convert.*OpName' lib/carts/ --include='*.cpp'
```

## Key Source Locations

### Dialect Definitions
```
include/carts/dialect/arts/IR/Ops.td          # ARTS op definitions
include/carts/dialect/arts-rt/IR/RtOps.td          # ARTS-RT op definitions
include/carts/dialect/sde/IR/SdeOps.td        # SDE op definitions
include/carts/dialect/codir/IR/CodirOps.td   # CODIR op definitions
```

### Conversion Passes
```
lib/carts/dialect/codir/Transforms/          # SDE → CODIR → ARTS boundary
lib/carts/dialect/arts/Conversion/ArtsToRt/   # ARTS → ARTS-RT
lib/carts/dialect/arts-rt/Conversion/RtToLLVM/     # ARTS-RT → LLVM
lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ # ARTS-RT → LLVM cleanup/remaining
```

## Instructions

When asked to trace an op:

1. Identify the op and its dialect (SDE, core, or RT)
2. Search for creation sites (`builder.create<OpType>`)
3. Search for transformation sites (`isa<>`, pattern matches)
4. Search for lowering/erasure sites (`replaceOp`, conversion patterns)
5. Map findings to pipeline stages using the pass-placement reference
6. Report the complete lifecycle: created at stage X, transformed by Y, lowered at Z
7. Flag any unexpected cross-dialect references (ops used outside their expected stage range)
