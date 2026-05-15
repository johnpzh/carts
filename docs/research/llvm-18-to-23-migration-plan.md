# LLVM 18 → 23 Migration Plan for Polygeist/CARTS

**Date**: 2026-03-19
**Scope**: Full upgrade of Polygeist's bundled LLVM from 18.0.0git to 23.0.0
**Motivation**: Unblock OMP task baseline benchmarks (see `omp-task-investigation.md`)

---

## Executive Summary

Upgrading LLVM 18 → 23 across CARTS + Polygeist requires **~3,100 breaking call sites** + **~2,250 deprecation warnings** across 4 codebases. The migration is dominated by **mechanical transformations** (85% automatable), with the hardest parts being:
1. Polygeist's `ConvertPolygeistToLLVM.cpp` (typed pointers + pass infrastructure)
2. OpenMP dialect ODS redesign (`WsloopOp` now wraps `LoopNestOp`)
3. Makefile: `openmp` must move from `LLVM_ENABLE_PROJECTS` to `LLVM_ENABLE_RUNTIMES` (fatal error)

**Revised estimate**: 3-5 weeks for one developer (down from initial 6-10 weeks because Clang API analysis revealed NO breaking changes for cgeist).

---

## 1. Scope of Changes

### 1.1 Codebase Inventory

| Codebase | Files | Lines | Migration Effort |
|----------|-------|-------|-----------------|
| **CARTS passes** (`lib/carts/passes/`) | 50 .cpp | 25,098 | MEDIUM |
| **CARTS transforms/utils** (`lib/carts/dialect/arts/Transforms/`, `utils/`, etc.) | 79 .cpp | 32,109 | MEDIUM |
| **Polygeist passes** (`external/Polygeist/lib/`) | 24 .cpp + 11 others | 21,854 | HIGH |
| **cgeist frontend** (`external/Polygeist/tools/cgeist/`) | 9 .cc files | ~20,000 | LOW (no Clang API breaks) |
| **Total** | ~170 source files | ~99,000 | |

### 1.2 Change Categories (Consolidated)

| # | Category | Sites | Automatable | Breaks Build? | Difficulty |
|---|----------|-------|-------------|---------------|------------|
| 1 | Member cast removal (`.cast<T>()` → `cast<T>()`) | **757** | YES (script) | YES (removed) | Trivial |
| 2 | `OpBuilder::create<T>` → `OpType::create` | **2,254** | YES (script) | NO (deprecated warning) | Trivial |
| 3 | `GEN_PASS_CLASSES` → `GEN_PASS_DEF_*` | **73 files** | PARTIAL | YES (hard error) | Moderate |
| 4 | `applyPatternsAndFoldGreedily` → `applyPatternsGreedily` | **25** | YES (sed) | YES (removed) | Trivial |
| 5 | Typed LLVM pointer removal | **~111** | NO | YES | Hard |
| 6 | OpenMP dialect ODS redesign (op renames, LoopNestOp) | **~70** | NO | YES | Hard |
| 7 | LLVM dialect op signature changes | **~50** | NO | YES | Medium |
| 8 | `InsertPointTy` → `InsertPointOrErrorTy` (OMPIRBuilder) | ~15 | NO | YES (submodule) | Free |
| 9 | Delete `ConvertToOpaquePtr.cpp` pass | 1 file | YES | N/A | Trivial |
| 10 | Makefile: `openmp` → `LLVM_ENABLE_RUNTIMES` | 1 line | YES | YES (fatal) | Trivial |
| 11 | `notifyBlockCreated` → `notifyBlockInserted` (Polygeist) | 1 | NO | YES | Easy |
| 12 | CMake/TableGen updates | ~5 files | NO | MAYBE | Medium |

**Breaking total**: ~855 sites that will fail to compile + 73 files needing GEN_PASS_DEF
**Deprecation total**: ~2,254 sites that will produce warnings (can be deferred)

### Distribution by Codebase

| Codebase | Breaking | Deprecation Warnings |
|----------|----------|---------------------|
| CARTS (`lib/carts/` + `include/carts/`) | 185 casts + 4 greedy + GEN_PASS (50 files) | 565 create |
| Polygeist (`lib/polygeist/`) | 224 casts + 21 greedy + GEN_PASS (23 files) | 693 create |
| cgeist (`tools/cgeist/`) | 0 casts | 901 create |
| Polygeist other | 1 listener | 0 |

---

## 2. Detailed Change Breakdown

### 2.1 Member Cast Removal (Category 1) — ~375 sites

LLVM 19 removed `Value.cast<T>()`, `Value.dyn_cast<T>()`, `Value.isa<T>()` in favor of free functions.

**Distribution**:
| Location | `.cast<` | `.dyn_cast<` | `.isa<` | Total |
|----------|---------|-------------|--------|-------|
| CARTS `lib/carts/` | ~50 | ~70 | ~30 | ~151 |
| Polygeist `lib/polygeist/` | ~60 | ~100 | ~64 | ~224 |
| cgeist `tools/cgeist/` | 0 | 0 | 0 | 0 |
| **Total** | ~110 | ~170 | ~94 | ~375 |

**Automation**:
```bash
# Sed pattern for .cast<T>()/dyn_cast<T>()/isa<T>()
# These need care because x.cast<T>() → cast<T>(x) requires moving the argument
# Best done with clang-tidy or a custom script
sed -i 's/\.\(cast\|dyn_cast\|dyn_cast_or_null\|isa\)<\([^>]*\)>()/; # NOT a simple sed
```

**Recommended approach**: Write a Python script that parses `expr.cast<Type>()` → `cast<Type>(expr)` patterns. The regex is complex due to nested templates but well-defined.

### 2.2 OpBuilder::create → OpType::create (Category 2) — ~1,600 sites

LLVM 23 changed from `builder.create<OpType>(loc, args...)` to `OpType::create(builder, loc, args...)`.

**Distribution**:
| Location | Sites |
|----------|-------|
| CARTS `lib/carts/` | 499 |
| Polygeist `lib/polygeist/` | 693 |
| cgeist `tools/cgeist/` | 901 |
| CARTS `include/carts/` | ~2 |
| **Total** | ~2,095 |

**Note**: Not all `.create<` sites are `OpBuilder::create` — some are other template methods. The actual OpBuilder count after filtering is ~1,600.

**Automation**: `clang-tidy` modernize check or sed:
```bash
# Pattern: builder.create<OpType>(args...) → OpType::create(builder, args...)
# Complex to automate perfectly but ~95% can be caught with regex
```

### 2.3 GEN_PASS_CLASSES → GEN_PASS_DEF (Category 3) — 4 headers

LLVM 23 raises a **hard error** on `GEN_PASS_CLASSES`. Must use per-pass `GEN_PASS_DEF_PASSNAME` macros.

**Files requiring changes**:
| File | Status |
|------|--------|
| `include/carts/passes/PassDetails.h` | Uses `GEN_PASS_CLASSES` |
| `lib/carts/passes/PassDetails.h` | Uses `GEN_PASS_CLASSES` (duplicate) |
| `external/Polygeist/lib/polygeist/Passes/PassDetails.h` | Uses `GEN_PASS_CLASSES` |
| `external/Polygeist/tools/polymer/lib/Transforms/PassDetail.h` | Uses `GEN_PASS_CLASSES` |

**Migration pattern** (per LLVM 23 convention):
```cpp
// OLD (PassDetails.h shared by all passes):
#define GEN_PASS_CLASSES
#include "arts/passes/Passes.h.inc"

// NEW (each pass .cpp file defines its own macro):
// In FooPass.cpp:
#define GEN_PASS_DEF_FOOPASS
#include "arts/passes/Passes.h.inc"
```

Each of the **50 CARTS pass files** and **24 Polygeist pass files** will need their own `#define GEN_PASS_DEF_*` line. The shared `PassDetails.h` files can be deleted or replaced with a simpler header.

### 2.4 applyPatternsAndFoldGreedily Rename (Category 4) — 25 sites

Simple rename: `applyPatternsAndFoldGreedily` → `applyPatternsGreedily`

**Distribution**:
| Location | Sites |
|----------|-------|
| CARTS `lib/carts/` | 4 |
| Polygeist `lib/polygeist/` | 21 |
| **Total** | 25 |

**Automation**: Trivial sed.

### 2.5 Typed LLVM Pointer Removal (Category 5) — ~111 sites

LLVM 18→19 removed typed pointers (`LLVM::LLVMPointerType::get(elementType)` → `LLVM::LLVMPointerType::get(ctx)`). All pointer types are now opaque.

**Distribution**:
| Location | Sites | Notes |
|----------|-------|-------|
| CARTS `lib/carts/codegen/Codegen.cpp` | 1 | Single `LLVMPointerType` |
| CARTS passes | 11 | `LLVMPointerType` references |
| Polygeist `ConvertPolygeistToLLVM.cpp` | 28 | HIGHEST effort |
| Polygeist other passes | ~11 | |
| cgeist frontend | ~60 | `.isa<LLVMPointerType>()`, typed pointer construction |
| **Total** | ~111 | |

**Key files**:
- `ConvertPolygeistToLLVM.cpp` (28 sites): Most are `LLVMPointerType::get(elementType)` that need to become `LLVMPointerType::get(ctx)`, plus `BitcastOp` removals (16 instances — bitcast between pointer types is now a no-op)
- `ConvertToOpaquePtr.cpp` (285 lines): **DELETE ENTIRELY** — this pass converts typed→opaque, now unnecessary
- CARTS codegen passes: `AliasScopeGen.cpp`, `DataPtrHoisting.cpp` — use `LLVMPointerType` and `LLVMStructType`

**Semantic difficulty**: These changes aren't mechanical — need to understand what the typed pointer was expressing and whether the opaque replacement needs GEP index changes.

### 2.6 OpenMP Dialect ODS Redesign (Category 6) — ~70 sites

LLVM 19-23 massively restructured the OMP dialect:

**Op renames**:
| LLVM 18 | LLVM 23 |
|---------|---------|
| `TaskLoopOp` | `TaskloopOp` |
| `TaskGroupOp` | `TaskgroupOp` |
| `WsLoopOp` | `WsloopOp` |
| `ReductionDeclareOp` | `DeclareReductionOp` |

**CRITICAL semantic change — WsloopOp restructuring**:
In LLVM 23, `omp.wsloop` no longer carries `lowerBound`, `upperBound`, `step` as direct operands. Instead, it wraps an `omp.loop_nest` op that holds the loop bounds:
```
// LLVM 18:
omp.wsloop for (%iv) = (%lb) to (%ub) step (%step) { ... }

// LLVM 23:
omp.wsloop {
  omp.loop_nest (%iv) : index = (%lb) to (%ub) step (%step) { ... }
}
```
This means `ConvertOpenMPToArts.cpp` calls to `op.getLowerBound()`, `op.getUpperBound()`, `op.getStep()` (lines 117-119, 384-386, 707-709) must navigate from `WsloopOp` to the nested `LoopNestOp`. Similarly, Polygeist's `ParallelLower.cpp` and cgeist must emit the new structure.

**Accessor renames**:
| LLVM 18 | LLVM 23 |
|---------|---------|
| `getDepends()` | `getDependKinds()` |
| `getFinalExpr()` | `getFinal()` |
| `getInReductions()` | `getInReductionSyms()` |
| `getAllocatorsVars()` | `getAllocatorVars()` |
| `getGrainSize()` | `getGrainsize()` |
| `getUntiedAttr()` | `getUntied()` (returns bool) |
| `getMergeableAttr()` | `getMergeable()` (returns bool) |

**Files affected**:
| File | OMP refs | Notes |
|------|----------|-------|
| `lib/carts/passes/transforms/ConvertOpenMPToArts.cpp` | 42 | 11 patterns, all OMP op names |
| `lib/carts/passes/opt/general/RaiseMemRefDimensionality.cpp` | 12 | `omp::TaskOp`, dependencies |
| `lib/carts/passes/opt/general/HandleDeps.cpp` | 7 | OMP dep handling |
| `lib/carts/utils/Utils.cpp` | 4 | OMP utilities |
| `lib/carts/utils/LoopUtils.cpp` | 1 | |
| `lib/carts/dialect/arts/Analysis/metadata/MetadataRegistry.cpp` | 1 | |
| **Polygeist** `OpenMPOpt.cpp` | ~10 | WsLoopOp patterns |
| **Polygeist** `ConvertPolygeistToLLVM.cpp` | ~5 | omp::ParallelOp, WsLoopOp |

**Clause-based ODS**: LLVM 23 OMP ops use clause mixins (`OpenMP_Clause`). This changes how ops are defined in `.td` files but **doesn't affect C++ usage** beyond accessor renames — the C++ API is generated from the new ODS and uses the renamed accessors above.

### 2.7 LLVM Dialect Changes (Category 7) — ~50 sites

Key changes in the LLVM dialect itself:
- `LLVM::BitcastOp` removed for pointer-to-pointer casts (opaque pointers make these no-ops)
- `LLVM::GEPOp` constructor changed (typed pointers → element type as explicit arg)
- `LLVM::LoadOp`/`StoreOp` may have signature changes
- `LLVM::CallOp` vararg support

Primarily affects:
- `ConvertArtsToLLVM.cpp` (1829 lines, 7 `.create<` calls)
- `ConvertPolygeistToLLVM.cpp` (3388 lines, 156 `.create<` calls)
- `Codegen.cpp` (1 `LLVMPointerType`)

### 2.8 OMPIRBuilder Changes (Category 8) — ~15 sites

`OpenMPToLLVMIRTranslation.cpp` changes:
- `createTask` signature: 7 params → 11 params
- `createTaskloop`: doesn't exist in 18, new in 23
- Return type: `InsertPointTy` → `InsertPointOrErrorTy` (wrapped in `Expected<>`)
- `TaskContextStructManager` for privatization

These are **inside Polygeist's bundled LLVM** — when we update the LLVM submodule, these come for free. **No manual patching needed** — just point the submodule at a new commit.

### 2.9 Delete ConvertToOpaquePtr Pass (Category 9)

`external/Polygeist/lib/polygeist/Passes/ConvertToOpaquePtr.cpp` (285 lines) — this pass converts typed pointers to opaque pointers. With LLVM 23, there are no typed pointers to convert. **Delete entirely**.

Also remove:
- Its entry in `PassDetails.h`
- Its pass registration
- Its invocation in `driver.cc`

### 2.10 Makefile: OpenMP → LLVM_ENABLE_RUNTIMES (Category 10)

**CRITICAL**: LLVM 23 emits a `FATAL_ERROR` if `openmp` is in `LLVM_ENABLE_PROJECTS`.

**File**: `/home/raherrer/projects/carts/Makefile`, line 109

```makefile
# CURRENT (LLVM 18):
-DLLVM_ENABLE_PROJECTS='mlir;clang;lld;openmp'
-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind;compiler-rt'

# REQUIRED (LLVM 23):
-DLLVM_ENABLE_PROJECTS='mlir;clang;lld'
-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind;compiler-rt;openmp'
```

### 2.11 Polygeist Listener API (Category 11)

**File**: `external/Polygeist/lib/polygeist/Ops.cpp`, line ~4981

```cpp
// LLVM 18:
rewriter.getListener()->notifyBlockCreated(Tmp);

// LLVM 23:
rewriter.getListener()->notifyBlockInserted(Tmp, /*previous_parent=*/nullptr,
                                             /*previous=*/Region::iterator());
```

Only 1 site. Signature changed to `notifyBlockInserted(Block*, Region*, Region::iterator)`.

### 2.12 CMake / TableGen Updates (Category 12)

**TableGen**:
- `include/carts/passes/Passes.td`: May need `-name` argument update for `mlir_tablegen`
- OpenMP dialect `.td` files come with the LLVM submodule update
- CARTS dialect `.td` files (`include/carts/`) — need audit for any deprecated TableGen features

**CMake**:
- `include/carts/passes/CMakeLists.txt`: `mlir_tablegen(Passes.h.inc -gen-pass-decls -name Arts)` — should still work
- `lib/carts/passes/CMakeLists.txt`: `add_mlir_dialect_library(MLIRArtsTransforms ...)` — function may have changed signature
- MLIR CMake library names may have changed (e.g., `MLIROpenMPToLLVM` → check LLVM 23 equivalents)
- `external/Polygeist/CMakeLists.txt`: Submodule path for `llvm-project`

**Submodule strategy**:
- Current: `external/Polygeist/llvm-project/` → LLVM 18 (commit `26eb4285`)
- Target: Update submodule pointer to LLVM 23 tag/commit
- Polygeist's `.gitmodules` references `https://github.com/llvm/llvm-project.git`

---

## 3. Migration Strategy

### 3.1 Approach: Direct Jump (18 → 23)

Incremental version hops (18→19→20→21→22→23) are NOT worth the overhead. Each hop would require a full rebuild (~30 min), and the mechanical changes (casts, create, rename) are the same regardless. Better to do one big jump and fix everything at once.

### 3.2 Phase Plan

#### Phase 0: Preparation (1 day)
1. Create migration branch: `git checkout -b llvm-upgrade-18-to-23`
2. Ensure clean build on current LLVM 18
3. Run full test suite, save baseline results
4. Back up current `external/Polygeist/llvm-project` submodule state

#### Phase 1: Mechanical Transformations (2-3 days)
**These can be done BEFORE updating LLVM** (they use the new API style which LLVM 18 also accepts in many cases, or we apply them right after the submodule update).

1. **Cast removal** (375 sites): Python script to transform `.cast<T>()` → `cast<T>()`
2. **OpBuilder::create** (1,600 sites): Script to transform `builder.create<Op>(` → `Op::create(builder, `
3. **applyPatternsGreedily** rename (25 sites): sed
4. **Delete ConvertToOpaquePtr.cpp** and its references

#### Phase 2: LLVM Submodule Update (1 day)
1. Update `external/Polygeist/llvm-project` submodule to LLVM 23 tag
2. Update Polygeist's CMake to match new LLVM CMake conventions
3. Attempt first build — expect many failures
4. Fix `GEN_PASS_CLASSES` → `GEN_PASS_DEF_*` across all 74 pass files
5. Fix CMake library name changes

#### Phase 3: Typed Pointer Removal (3-5 days)
This is the hardest semantic work.

1. **ConvertPolygeistToLLVM.cpp** (28 sites + 16 BitcastOp): Remove typed pointer construction, fix GEP element types
2. **cgeist frontend** (~60 sites): Fix `.isa<LLVMPointerType>()` checks, pointer construction
3. **CARTS codegen passes** (12 sites): Fix `LLVMPointerType`, `LLVMStructType` usage
4. Remove all `opaquePointers` parameters/flags (now always-on)

#### Phase 4: OpenMP Dialect Adaptation (2-3 days)
1. **ConvertOpenMPToArts.cpp** (42 refs): Fix op names (`TaskLoopOp`→`TaskloopOp`), accessor renames
2. **RaiseMemRefDimensionality.cpp** (12 refs): Fix OMP op references
3. **HandleDeps.cpp** (7 refs): Fix dep-related accessor changes
4. **ConvertPolygeistToLLVM.cpp** (5 refs): Fix `omp::ParallelOp`, `omp::WsLoopOp` → `WsloopOp`
5. **Polygeist OpenMPOpt.cpp**: Fix WsLoop patterns
6. **cgeist CGStmt.cc**: Fix task/taskloop op construction (accessor/builder API changes)

#### Phase 5: Build + Test + Fix (3-5 days)
1. Iterative build-fix cycles until clean compile
2. Run CARTS test suite, fix any failures
3. Run benchmark suite (including the now-unblocked OMP task benchmarks!)
4. Fix any runtime failures

#### Phase 6: Validation (1-2 days)
1. Full test suite green
2. All benchmarks pass (ARTS + OMP baseline)
3. **OMP task benchmarks** compile and run correctly for both paths
4. Performance regression testing (no slowdowns from LLVM upgrade)

### 3.3 Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Polygeist internal API breaks | HIGH | MEDIUM | Polygeist lib is small (11 files), we maintain the fork |
| cgeist Clang API breaks | CRITICAL | **LOW** | Analysis confirmed NO breaking Clang API changes 18→23 |
| CARTS pass semantic bugs | HIGH | LOW | Well-tested pass suite, incremental approach |
| Build system breakage | MEDIUM | MEDIUM | CMake audit before starting |
| LLVM 23 regressions | LOW | LOW | LLVM 23 is mature (2.5 years of development) |
| Polygeist ops.td changes | MEDIUM | LOW | Polygeist dialect is stable, no LLVM-driven changes expected |

### 3.4 What We Get

After the upgrade:
1. **OMP task baseline works** — full `omp.task`/`omp.taskloop` lowering to LLVM IR
2. **9/11 task clauses** supported (untied, mergeable, priority, depend, if, final, private, firstprivate, detach)
3. **TaskContextStructManager** for proper firstprivate handling
4. **Full taskloop** with grainsize, num_tasks, nogroup, collapse, cancel
5. Access to 2.5 years of LLVM optimizations and bug fixes
6. Foundation for future MLIR features (e.g., transform dialect, mesh dialect)

---

## 4. Files Reference (By Migration Priority)

### Tier 1: Critical Path (must fix first to compile)

| File | Lines | Changes Needed |
|------|-------|----------------|
| `include/carts/passes/PassDetails.h` | 31 | GEN_PASS_DEF migration |
| `lib/carts/passes/PassDetails.h` | 30 | GEN_PASS_DEF migration |
| `external/Polygeist/lib/polygeist/Passes/PassDetails.h` | 39 | GEN_PASS_DEF migration |
| All 50 CARTS pass .cpp files | 25,098 | GEN_PASS_DEF per file |
| All 24 Polygeist pass .cpp files | 21,854 | GEN_PASS_DEF per file |
| `external/Polygeist/lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp` | 3,388 | 28 typed ptr + 156 create + 27 casts |
| `lib/carts/passes/transforms/ConvertArtsToLLVM.cpp` | 1,829 | 7 create + 3 casts + LLVM dialect changes |
| `lib/carts/passes/transforms/ConvertOpenMPToArts.cpp` | 824 | 42 OMP refs + 22 create + 1 cast |

### Tier 2: High-Impact (many changes but mostly mechanical)

| File | Lines | Changes Needed |
|------|-------|----------------|
| `external/Polygeist/tools/cgeist/Lib/clang-mlir.cc` | ~6,500 | 470 create + ~60 typed ptrs |
| `external/Polygeist/tools/cgeist/Lib/CGStmt.cc` | ~2,500 | 193 create + 5 typed ptrs |
| `external/Polygeist/tools/cgeist/Lib/CGCall.cc` | ~3,000 | 199 create |
| `external/Polygeist/lib/polygeist/Ops.cpp` | ~2,500 | 139 casts + 98 create |
| `lib/carts/passes/opt/db/DbPartitioning.cpp` | 2,495 | 18 create |
| `lib/carts/passes/opt/general/RaiseMemRefDimensionality.cpp` | 1,798 | 28 casts + 10 create + 12 OMP |

### Tier 3: Low-Impact (few changes each)

- 42 remaining CARTS pass/transform/util files with <10 changes each
- 13 remaining Polygeist pass files with <10 changes each

---

## 5. Automation Scripts

### 5.1 Cast Migration Script (Outline)

```python
#!/usr/bin/env python3
"""Migrate .cast<T>() / .dyn_cast<T>() / .isa<T>() to free functions."""
import re, sys

PATTERN = re.compile(
    r'(\b\w+(?:\([^)]*\))?)'  # expression before .cast
    r'\.(cast|dyn_cast|dyn_cast_or_null|isa)'
    r'<([^>]+)>'              # template arg
    r'\(\)'                   # empty parens
)

def replace(m):
    expr, func, type_arg = m.group(1), m.group(2), m.group(3)
    return f'{func}<{type_arg}>({expr})'

for path in sys.argv[1:]:
    text = open(path).read()
    new_text = PATTERN.sub(replace, text)
    if new_text != text:
        open(path, 'w').write(new_text)
        print(f'Updated: {path}')
```

### 5.2 OpBuilder::create Migration Script (Outline)

```python
#!/usr/bin/env python3
"""Migrate builder.create<OpType>(args) to OpType::create(builder, args)."""
# This is more complex due to:
# - Multi-line create calls
# - Different builder variable names (builder, rewriter, b, etc.)
# - Nested template args in OpType
# Best done with a proper AST tool or very careful regex
```

### 5.3 GEN_PASS_DEF Migration

For each pass `FooPass` in `Passes.td`:
1. Find the corresponding `.cpp` file
2. Replace `#include "PassDetails.h"` with:
   ```cpp
   #define GEN_PASS_DEF_FOOPASS
   #include "arts/passes/Passes.h.inc"
   ```
3. Update pass class to inherit from generated base: `struct FooPass : impl::FooPassBase<FooPass>`

---

## 6. Testing Strategy

### 6.1 Build Verification
```bash
carts build  # Full rebuild
```

### 6.2 Unit Tests
```bash
carts test   # Run full LIT test suite
```

### 6.3 OMP Task Baseline (The Whole Point)
```bash
cd external/carts-benchmarks/kastors-jacobi/jacobi-task-dep
make small-openmp CARTS=carts  # Should succeed after migration!
make small                      # Build both ARTS and OMP
make run-small                  # Run both and compare
```

### 6.4 Regression Testing
```bash
carts benchmarks run --suite=all --compare-baseline
```

---

## 7. Decision: Why Full Upgrade Over Patching

| Factor | Patch LLVM 18 (Option A) | Full Upgrade (Option C) |
|--------|--------------------------|------------------------|
| **Effort** | 3-5 days | 3-5 weeks |
| **OMP task support** | Partial (basic task only) | Full (9/11 clauses + taskloop) |
| **Taskloop** | Must implement from scratch | Comes free |
| **Future maintenance** | Carrying patches forever | Upstream-aligned |
| **LLVM improvements** | None | 2.5 years of optimizations |
| **Risk** | Low but limited | Medium but comprehensive |

The user chose Option C because:
1. Taskloop handler implementation from scratch (Option A) is substantial (~500 lines)
2. Carrying patches against LLVM creates ongoing maintenance burden
3. LLVM 23 provides production-quality task support with TaskContextStructManager
4. Access to 2.5 years of MLIR/LLVM improvements and bug fixes

---

## 8. Key Discoveries

### 8.1 Eliminated Risk: Clang API Compatibility
The initial estimate of 6-10 weeks was based on fear that cgeist's 13K lines of Clang internals would break. **Analysis confirmed NO breaking Clang API changes** between LLVM 18 and 23 for the APIs cgeist uses. This reduced the estimate by 40%.

### 8.2 Eliminated Risk: CARTS Pass Complexity
CARTS passes have **zero global/static mutable state** (confirmed by prior thread-safety audit). This means the migration is purely mechanical — no subtle shared-state bugs to worry about.

### 8.3 Identified Risk: ConvertPolygeistToLLVM.cpp
At 3,388 lines with 28 typed pointer sites, 156 `.create<` calls, and 27 cast removals, this is the single hardest file to migrate. It's also the ROOT CAUSE of the OMP baseline failure (missing TaskOp registration at line 3262).

### 8.4 New Discovery: WsloopOp Semantic Restructuring
LLVM 23 restructured `omp.wsloop` to wrap an `omp.loop_nest` op instead of carrying loop bounds directly. This is NOT a simple rename — it requires structural changes in:
- `ConvertOpenMPToArts.cpp`: Access bounds through nested `LoopNestOp`
- Polygeist `ParallelLower.cpp`: Emit `WsloopOp + LoopNestOp` structure
- cgeist: Generate the new MLIR structure from Clang AST

### 8.5 New Discovery: Makefile Fatal Error
LLVM 23 will refuse to build if `openmp` is in `LLVM_ENABLE_PROJECTS`. It must be moved to `LLVM_ENABLE_RUNTIMES`. This is a single-line fix in the Makefile but would be a confusing build failure without this knowledge.

### 8.6 Free Fixes with Submodule Update
The following come automatically when the LLVM submodule is updated:
- `OpenMPToLLVMIRTranslation.cpp`: Full task/taskloop lowering (2459→7681 lines)
- `OMPIRBuilder.cpp`: `createTask` with all clauses, `createTaskloop`
- OpenMP dialect `.td` files: Clause-based ODS
- All LLVM/MLIR bug fixes and optimizations since Sept 2023
