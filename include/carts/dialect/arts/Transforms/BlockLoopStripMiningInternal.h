///==========================================================================///
/// File: BlockLoopStripMiningInternal.h
///
/// Local implementation contract for BlockLoopStripMining. This header is
/// intentionally private to the block-loop-strip-mining implementation split
/// and should not be used as shared compiler infrastructure.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_BLOCKLOOPSTRIPMININGINTERNAL_H
#define ARTS_DIALECT_CORE_TRANSFORMS_BLOCKLOOPSTRIPMININGINTERNAL_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Utils/LoopStructureUtils.h"
#include "carts/utils/Debug.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::carts::arts::block_loop_strip_mining {

struct OffsetPatternGroup {
  int64_t offsetConst = 0;
  SmallVector<arith::DivUIOp> divOps;
  SmallVector<Value> remValues;
  SmallVector<Operation *> remPatternOps;
};

struct LoopNeighborhoodSegment {
  enum class BoundaryKind {
    constant,
    blockSizeMinusConstant,
    blockSize,
  };

  struct Boundary {
    BoundaryKind kind = BoundaryKind::constant;
    int64_t constant = 0;
  };

  Boundary localLower;
  Boundary localUpper;
  SmallVector<int64_t, 4> blockDeltas;
};

struct NeighborhoodPattern {
  int64_t offsetConst = 0;
  Value invariantBase;
  Value blockSizeVal;
  std::optional<int64_t> blockSizeConst;
  std::optional<arith::DivUIOp> divOp;
  Value remResult;
  SmallVector<Operation *, 4> patternOps;
};

struct NeighborhoodLoopInfo {
  Value invariantBase;
  Value blockSizeVal;
  std::optional<int64_t> blockSizeConst;
  std::optional<int64_t> dynamicOrderingGuardThreshold;
  unsigned matchedDbRefCount = 0;
  SmallVector<OffsetPatternGroup, 4> offsetGroups;
  SmallVector<LoopNeighborhoodSegment, 4> segments;
};

struct NeighborhoodExprInfo {
  Value invariantBase;
  int64_t offsetConst = 0;
};

struct SingleDimBlockLoopInfo {
  Value blockSizeVal;
  std::optional<int64_t> blockSizeConst;
  std::optional<int64_t> lowerBoundConst;
  Value lowerBoundDivHint;
  Value offsetVal;
  Value offsetDivHint;
  SmallVector<arith::DivUIOp> divOps;
  SmallVector<Value> remValues;
  SmallVector<Operation *> remPatternOps;
};

using NeighborhoodReplacementMap = llvm::DenseMap<Operation *, Value>;
using NeighborhoodEraseSet = llvm::DenseSet<Operation *>;

bool isGeneratedByStripMining(scf::ForOp loop);
void markGeneratedByStripMining(scf::ForOp loop);
void clearGeneratedStripMiningMarks(func::FuncOp func);

std::optional<Value> extractInvariantOffset(Value lhs, Value iv);
bool mergeInvariantBase(Value &currentBase, Value candidateBase);
std::optional<NeighborhoodExprInfo> extractNeighborhoodExprInfo(Value value,
                                                                Value iv);

OffsetPatternGroup &getOrCreateOffsetGroup(NeighborhoodLoopInfo &info,
                                           int64_t offsetConst);

std::optional<NeighborhoodPattern>
matchNeighborhoodPattern(Value lhs, Value rhs,
                         std::optional<arith::DivUIOp> divOp, Value remResult,
                         Value iv, ArrayRef<Operation *> patternOps);

NeighborhoodLoopInfo &getOrCreateNeighborhoodFamily(
    SmallVectorImpl<NeighborhoodLoopInfo> &families, Value blockSizeVal,
    std::optional<int64_t> blockSizeConst, Value invariantBase);

void recordNeighborhoodPattern(NeighborhoodLoopInfo &info,
                               const NeighborhoodPattern &pattern);

std::optional<NeighborhoodPattern>
matchNormalizedSignedRemainderNeighborhood(arith::SelectOp selectOp, Value iv);

bool recordRemPattern(Value lhs, Value rhs, Value remResult, Value iv,
                      SingleDimBlockLoopInfo &info,
                      ArrayRef<Operation *> patternOps);

bool matchNormalizedSignedRemainder(arith::SelectOp selectOp, Value iv,
                                    SingleDimBlockLoopInfo &info);

bool isAlignedOffset(Value offset, Value blockSize,
                     const std::optional<int64_t> &blockSizeConst,
                     Value *mulOther = nullptr);

bool isAlignedValue(Value value, Value blockSize,
                    const std::optional<int64_t> &blockSizeConst,
                    Value *divHint = nullptr);

void rewriteNestedNeighborhoodClone(
    Block &originalBlock, Block &clonedBlock,
    const NeighborhoodReplacementMap &replacements,
    const NeighborhoodEraseSet &eraseIfDead);

void rewriteNestedNeighborhoodClone(
    Operation *originalOp, Operation *clonedOp,
    const NeighborhoodReplacementMap &replacements,
    const NeighborhoodEraseSet &eraseIfDead);

LoopNeighborhoodSegment::Boundary makeConstantBoundary(int64_t constant);
LoopNeighborhoodSegment::Boundary makeBlockSizeMinusBoundary(int64_t constant);
LoopNeighborhoodSegment::Boundary makeBlockSizeBoundary();

SmallVector<LoopNeighborhoodSegment, 4>
buildConstantNeighborhoodSegments(const NeighborhoodLoopInfo &info);

std::optional<int64_t>
getDynamicNeighborhoodOrderingThreshold(const NeighborhoodLoopInfo &info);

SmallVector<LoopNeighborhoodSegment, 4>
buildDynamicNeighborhoodSegments(const NeighborhoodLoopInfo &info);

SmallVector<LoopNeighborhoodSegment, 4>
buildNeighborhoodSegments(NeighborhoodLoopInfo &info);

unsigned countTrackedNeighborhoodPatterns(const NeighborhoodLoopInfo &info);
bool hasNonZeroNeighborhoodOffset(const NeighborhoodLoopInfo &info);
unsigned countNeighborhoodDbRefUsers(scf::ForOp loop,
                                     const NeighborhoodLoopInfo &info);

bool finalizeNeighborhoodCandidate(scf::ForOp loop, NeighborhoodLoopInfo &info,
                                   DominanceInfo &domInfo);

bool isBetterNeighborhoodCandidate(const NeighborhoodLoopInfo &lhs,
                                   const NeighborhoodLoopInfo &rhs);

std::optional<NeighborhoodLoopInfo>
analyzeNeighborhoodLoop(scf::ForOp loop, DominanceInfo &domInfo);

std::optional<SingleDimBlockLoopInfo>
analyzeSingleDimBlockLoop(scf::ForOp loop, DominanceInfo &domInfo);

Value buildNeighborhoodBoundaryValue(
    OpBuilder &builder, Location loc,
    const LoopNeighborhoodSegment::Boundary &boundary, Value blockSize);

Value buildNeighborhoodRemValue(OpBuilder &builder, Location loc, Value local,
                                Value blockSize, int64_t offsetConst,
                                int64_t blockDelta);

bool stripMineNeighborhoodLoopImpl(
    scf::ForOp loop, const NeighborhoodLoopInfo &info,
    SmallVectorImpl<Value> *newResults = nullptr);

bool stripMineNeighborhoodLoop(scf::ForOp loop,
                               const NeighborhoodLoopInfo &info);

bool stripMineSingleDimBlockLoop(scf::ForOp loop,
                                 const SingleDimBlockLoopInfo &info);

} // namespace mlir::carts::arts::block_loop_strip_mining

#endif // ARTS_DIALECT_CORE_TRANSFORMS_BLOCKLOOPSTRIPMININGINTERNAL_H
