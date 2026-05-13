///==========================================================================///
/// File: EpochOptInternal.h
///
/// Local implementation contract for EpochOpt. This header is intentionally
/// private to the epoch-opt implementation split and should not be used as
/// shared compiler infrastructure.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/edt/EpochAnalysis.h"
#include "arts/dialect/core/Analysis/heuristics/EpochHeuristics.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <climits>
#include <optional>
#include <string>

namespace mlir::arts::epoch_opt {

struct EpochNarrowingCounts {
  unsigned epochsNarrowed = 0;
  unsigned newEpochsCreated = 0;
};

llvm::StringRef asyncLoopStrategyToString(EpochAsyncLoopStrategy strategy);
ArtsPlanAsyncStrategy
asyncLoopStrategyToPlanAsyncStrategy(EpochAsyncLoopStrategy strategy);
bool loopContainsCpsChainExcludedDepPattern(scf::ForOp forOp);
bool loopContainsCpsDriverExcludedDepPattern(scf::ForOp forOp);
std::string makeAsyncLoopChainId(Operation *op);
std::string makeContinuationChainId(Operation *ownerOp, unsigned chainIdx);

using AttrNames::Operation::ContinuationForEpoch;
using AttrNames::Operation::ControlDep;
using AttrNames::Operation::CPSAdditiveParams;
using AttrNames::Operation::CPSAdvanceHasIvArg;
using AttrNames::Operation::CPSChainId;
using AttrNames::Operation::CPSDepRouting;
using AttrNames::Operation::CPSDirectRecreate;
using AttrNames::Operation::CPSForwardDeps;
using AttrNames::Operation::CPSInitIter;
using AttrNames::Operation::CPSIterCounterParamIdx;
using AttrNames::Operation::CPSLoopContinuation;
using AttrNames::Operation::CPSNumCarry;

bool isSlotOp(Operation *op, ArrayRef<EpochSlot> slots);
void ensureYieldTerminator(Block &block, Location loc);
void markClonedSlotEpochs(OpBuilder &attrBuilder, scf::IfOp ifOp);
Operation *cloneEpochSlot(OpBuilder &slotBuilder, EpochSlot slot,
                          IRMapping &mapping);
void cloneNonSlotArith(OpBuilder &builder, Block &body,
                       MutableArrayRef<EpochSlot> slots, IRMapping &mapping,
                       ArrayRef<Operation *> sequentialOps = {},
                       ArrayRef<Operation *> eagerSequentialOps = {},
                       bool restrictEagerSequentialOps = false);
Operation *getLastNonTerminatorOp(Block &block);
void emitAdvanceLogic(OpBuilder &builder, Location loc, Value iv,
                      Value tCurrent, Value ub, Value step, Block &body,
                      MutableArrayRef<EpochSlot> slots, StringRef targetChainId,
                      ArrayRef<Operation *> prefixSequentialOps = {},
                      ArrayRef<Operation *> allSequentialOps = {},
                      ArrayRef<Value> loopBackParams = {},
                      const IRMapping *seedMapping = nullptr);
void markAsContinuation(EdtOp edt, OpBuilder &builder, Operation *ownerOp,
                        unsigned chainIdx);
void normalizeAsyncLoopPlanAttrs(scf::ForOp forOp,
                                 EpochAsyncLoopStrategy strategy);
void copyNormalizedPlanAttrs(Operation *source, Operation *dest,
                             EpochAsyncLoopStrategy strategy);

EpochNarrowingCounts narrowEpochScopes(ModuleOp module);
unsigned processRegionForEpochFusion(Region &region,
                                     const EpochAnalysis &epochAnalysis,
                                     bool continuationEnabled);
unsigned fuseWorkerLoopsInEpoch(EpochOp epoch);
bool tryAmortizeRepeatedEpochLoop(EpochOp epochOp);
LogicalResult
transformToContinuation(EpochOp epochOp,
                        const EpochContinuationDecision &decision);
bool tryCPSLoopTransform(scf::ForOp forOp, const EpochAnalysis &epochAnalysis);
bool tryCPSChainTransform(scf::ForOp forOp, const EpochAnalysis &epochAnalysis);

} // namespace mlir::arts::epoch_opt

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H
