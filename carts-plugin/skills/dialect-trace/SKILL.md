---
name: carts-dialect-trace
description: Use when debugging lowering paths, understanding operation placement across SDE/Core/RT, or verifying dialect boundary invariants.
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

Trace an operation's lifecycle across the 18 pipeline stages and 3 dialects
(SDE, ARTS core, ARTS RT). Understand where ops are created, transformed,
and destroyed.

## Three Dialects

| Dialect | Namespace | Stages | Purpose |
|---------|-----------|--------|---------|
| SDE | `sde::` | 3 (inside openmp-to-arts) | Semantic contracts, pattern classification |
| ARTS Core | `arts::` | 3-15 | Structural transforms, partitioning, distribution |
| ARTS RT | `arts_rt::` / `rt::` | 15-16 | 1:1 runtime call mapping |

## Dialect Boundaries

```
C/OMP source
  → [Stage 3]    inside openmp-to-arts: OMP → SDE (ConvertOpenMPToSde) →
                 SDE transforms → SDE → Core (ConvertSdeToArts)
  → [Stage 4-9]  Core transforms (arts.edt, arts.db_*, arts.epoch, scf.for)
  → [Stage 10]   Core → RT lowering in pre-lowering
                 (EdtLowering.cpp, EpochLowering.cpp, DbLowering.cpp)
  → [Stage 11]   RT → LLVM lowering in arts-to-llvm
                 (ConvertArtsToLLVM + rt-specific RuntimeCallOpt etc.)
```

## Verification Barriers

| Verifier | After Stage | Checks |
|----------|-------------|--------|
| VerifySdeLowered | 3 (openmp-to-arts) | No SDE ops survive |
| VerifyCoreObjectsOnly | 3 (openmp-to-arts) | No Core loop carrier or semantic parallel EDT survives |
| VerifyEdtCreated | 3 (openmp-to-arts) | EDTs created from OMP regions |
| VerifyPreLowered | 10 (pre-lowering) | No arts.edt/epoch/db_* survive |
| VerifyLowered | 11 (arts-to-llvm) | No ARTS ops survive |

## Op Lifecycle Quick Reference

### Core Ops
| Op | Created | Lowered | Stages Active |
|----|---------|---------|---------------|
| `arts.edt` | openmp-to-arts (3) | pre-lowering (10) | 3-10 |
| `arts.db_alloc` | create-dbs (5) | pre-lowering (10) | 5-10 |
| `arts.db_acquire` | create-dbs (5) | pre-lowering (10) | 5-10 |
| `arts.db_ref` | create-dbs (5) | pre-lowering (10) | 5-10 |
| `arts.epoch` | openmp-to-arts (3) or epochs (9) | pre-lowering (10) | 3-10 |
| `arts.barrier` | openmp-to-arts (3) | epochs (9) | 3-9 |

### RT Ops
| Op | Created | Lowered | Stages Active |
|----|---------|---------|---------------|
| `arts_rt.edt_create` | pre-lowering (15) | arts-to-llvm (16) | 15-16 |
| `arts_rt.create_epoch` | pre-lowering (15) | arts-to-llvm (16) | 15-16 |
| `arts_rt.db_create` | pre-lowering (15) | arts-to-llvm (16) | 15-16 |
| `arts_rt.rec_dep` | pre-lowering (15) | arts-to-llvm (16) | 15-16 |

### SDE Ops
SDE ops have a very short lifetime — they are created and lowered entirely
within stage 3 (`openmp-to-arts`):

| Op | Created | Lowered |
|----|---------|---------|
| `sde.cu_task` | ConvertOpenMPToSde (within stage 3) | ConvertSdeToArts (within stage 3) |
| `sde.su_access` | ConvertOpenMPToSde (within stage 3) | ConvertSdeToArts (within stage 3) |
| `sde.mu_reduction` | ConvertOpenMPToSde (within stage 3) | ConvertSdeToArts (within stage 3) |

## Tracing Commands

### Find where an op is created
```bash
grep -rn 'create<.*OpName>' lib/arts/ --include='*.cpp'
grep -rn 'builder\.create<.*OpName>' lib/arts/ --include='*.cpp'
```

### Find where an op is consumed/matched
```bash
grep -rn 'isa<.*OpName>\|dyn_cast<.*OpName>\|cast<.*OpName>' lib/arts/ --include='*.cpp'
```

### Find where an op is erased/replaced
```bash
grep -rn 'replaceOp\|eraseOp' lib/arts/ --include='*.cpp' | grep -i 'OpName'
```

### Find conversion patterns for an op
```bash
grep -rn 'OpName.*Pattern\|convert.*OpName' lib/arts/ --include='*.cpp'
```

## Key Source Locations

### Dialect Definitions
```
include/arts/dialect/core/IR/Ops.td          # Core op definitions
include/arts/dialect/rt/IR/RtOps.td          # RT op definitions
include/arts/dialect/sde/IR/SdeOps.td        # SDE op definitions
```

### Conversion Passes
```
lib/arts/dialect/sde/Conversion/SdeToArts/   # SDE → Core
lib/arts/dialect/core/Conversion/ArtsToRt/   # Core → RT
lib/arts/dialect/rt/Conversion/RtToLLVM/     # RT → LLVM
lib/arts/dialect/core/Conversion/ArtsToLLVM/ # Core → LLVM (remaining)
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
