///==========================================================================///
/// File: DataPtrHoistingInternal.h
///
/// Local implementation contract for DataPtrHoisting. This header is
/// intentionally private to the data-ptr-hoisting implementation split and
/// should not be used as shared compiler infrastructure.
///==========================================================================///

#ifndef ARTS_DIALECT_RT_TRANSFORMS_DATAPTRHOISTINGINTERNAL_H
#define ARTS_DIALECT_RT_TRANSFORMS_DATAPTRHOISTINGINTERNAL_H

#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts/Utils/BlockedAccessUtils.h"
#include "carts/dialect/arts-rt/Utils/LoopInvarianceUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "polygeist/Ops.h"

#include <optional>

namespace mlir::arts::data_ptr_hoisting {

//===----------------------------------------------------------------------===//
// Struct / enum definitions
//===----------------------------------------------------------------------===//

struct NeighborCarryIndexPattern {
  Value baseIndex;
  Value belowCond;
  Value aboveCond;
  Value clampMax;
  Value forceZeroCond;
};

struct BlockedNeighborCarryPattern {
  Value baseGlobalIndex;
  Value blockSize;
  Value blockBaseOffset;
  Value clampMax;
  Value forceZeroCond;
  bool forceZeroWhenTrue = true;
};

enum class BoundaryLoopRegion { Left, Middle, Right };

struct BoundarySelectChain {
  Value lowCond;
  Value highCond;
  Value lowValue;
  Value midValue;
  Value highValue;
};

struct BoundaryIndexPattern {
  BoundarySelectChain chain;
  Value baseExpr;
  int64_t offset = 0;
};

struct MemoryAccessInfo {
  Value memref;
  SmallVector<Value> indices;
  SmallVector<Value> sizes;
  Value storedValue;
  Type resultType;
  bool isLoad = false;
  bool isStore = false;
};

struct LoopWindowAccessPattern {
  MemoryAccessInfo memInfo;
  polygeist::Pointer2MemrefOp ptr2memref;
  BoundarySelectChain ptrChain;
  bool hasBoundaryPtrChain = false;
  BoundaryIndexPattern indexPattern;
  unsigned partitionDim = 0;
};

//===----------------------------------------------------------------------===//
// Function declarations
//===----------------------------------------------------------------------===//

/// Check if an LLVM load is loading a data pointer from deps struct.
bool isDepsPtrLoad(LLVM::LoadOp loadOp);

/// Check if an LLVM load is loading a datablock pointer from a db_gep.
bool isDbPtrLoad(LLVM::LoadOp loadOp);

bool matchGreaterThanConstIcmp(Value cond, Value value, int64_t threshold);

bool hasDominatingBlockSizeGuard(scf::ForOp loop, Value blockSize,
                                 int64_t threshold);

bool getSingleHopOffsetThreshold(scf::ForOp loop, int64_t &threshold);

bool matchBlockedNeighborCarryIndex(Value value, scf::ForOp loop,
                                    BlockedNeighborCarryPattern &pattern);

bool matchCarryAdjust(Value value, Value &belowCond, Value &aboveCond);

bool matchBasePlusCarry(Value value, Value &baseIndex, Value &belowCond,
                        Value &aboveCond);

bool matchNeighborCarryIndex(Value value, scf::ForOp loop,
                             NeighborCarryIndexPattern &pattern);

Value buildLoopInvariantI1Not(OpBuilder &builder, Location loc, Value cond);

Value buildLoopInvariantI1And(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs);

Value buildDepPtrLoad(OpBuilder &builder, Location loc, rt::DepGepOp depGep,
                      ArrayRef<Value> indices);

Value buildGuardedDepPtrLoad(OpBuilder &builder, Location loc,
                             rt::DepGepOp depGep, ArrayRef<Value> indices,
                             Value guard, Value fallbackPtr);

Value buildNormalizedForceZeroCond(OpBuilder &builder, Location loc,
                                   const BlockedNeighborCarryPattern &pattern);

Value buildBlockedNeighborCenterIndex(
    OpBuilder &builder, Location loc,
    const BlockedNeighborCarryPattern &pattern, Value zero);

bool canHoistBlockedNeighborCacheAcross(
    scf::ForOp loop, int varyingIndex, ArrayRef<Value> indices,
    const BlockedNeighborCarryPattern &pattern);

scf::ForOp
findBlockedNeighborCacheHoistLoop(scf::ForOp loop, int varyingIndex,
                                  ArrayRef<Value> indices,
                                  const BlockedNeighborCarryPattern &pattern);

bool materializeBlockedNeighborPtrCache(LLVM::LoadOp loadOp, scf::ForOp loop);

/// Cache the current blocked dep-family pointer inside a monotone unit-step
/// loop and only reload it when the computed family index changes.
bool materializeMonotoneBlockedDepPtrCache(LLVM::LoadOp loadOp,
                                           scf::ForOp loop);

Value buildNeighborCandidateIndex(OpBuilder &builder, Location loc,
                                  const NeighborCarryIndexPattern &pattern,
                                  int64_t adjust);

std::optional<MemoryAccessInfo> getMemoryAccessInfo(Operation *op);

bool matchBlockedNeighborLocalIndex(Value value, scf::ForOp loop,
                                    Value baseGlobalIndex, Value blockSize);

bool rewriteBlockedNeighborLocalIndices(Operation *op, scf::ForOp loop,
                                        Value baseGlobalIndex, Value blockSize,
                                        Value replacementIndex);

bool matchSingleBlockBlockedDepIndex(Value value, scf::ForOp loop,
                                     Value &globalIndex,
                                     BlockedNeighborCarryPattern &pattern);

Value stripInvariantZeroFallback(Value value, scf::ForOp loop);

bool materializeSingleBlockBlockedDepView(LLVM::LoadOp loadOp, scf::ForOp loop);

bool versionSingleBlockUniformDepLoops(scf::ForOp loop);

/// Clone worker EDTs with verified single-block fast paths and rewrite launch
/// sites to dispatch to the fast clone when the launch-time param pack proves
/// the same condition as the worker-local guard.
int specializeVersionedSingleBlockEdtCreates(ModuleOp module);

bool getSelectOperands(Value value, Value &cond, Value &trueValue,
                       Value &falseValue);

bool matchLowerBoundaryCond(Value cond, Value candidate);

bool matchUpperBoundaryCond(Value cond, Value candidate);

bool tryBuildBoundaryIndexChain(Value firstCond, Value firstValue,
                                Value secondCond, Value secondValue,
                                Value middleValue, BoundarySelectChain &chain);

bool matchBoundaryIndexPattern(Value value, scf::ForOp loop,
                               BoundaryIndexPattern &pattern);

bool matchBoundaryPointerChain(Value value, Value lowCond, Value highCond,
                               BoundarySelectChain &chain);

std::optional<LoopWindowAccessPattern> matchLoopWindowAccess(Operation *op,
                                                             scf::ForOp loop);

Value chooseBoundaryRegionValue(BoundaryLoopRegion region, int64_t offset,
                                const BoundarySelectChain &chain);

bool rewriteLoopWindowAccess(Operation *op, scf::ForOp loop,
                             BoundaryLoopRegion region);

scf::ForOp cloneLoopWindowSegment(OpBuilder &builder, scf::ForOp loop,
                                  Value lowerBound, Value upperBound);

int rewriteLoopWindowSegment(scf::ForOp loop, BoundaryLoopRegion region);

bool versionLoopWindowAccesses(scf::ForOp loop, int &rewrittenAccesses);

bool materializeNeighborPtrCache(LLVM::LoadOp loadOp, scf::ForOp loop);

/// Find the highest loop that can legally hoist a pure, operand-only op.
scf::ForOp findInvariantOpHoistTarget(Operation *op, DominanceInfo &domInfo);

/// Hoist a load and its address computation (GEP) out of loops.
bool hoistLoadOutOfLoop(LLVM::LoadOp loadOp, scf::ForOp targetLoop);

} // namespace mlir::arts::data_ptr_hoisting

#endif // ARTS_DIALECT_RT_TRANSFORMS_DATAPTRHOISTINGINTERNAL_H
