///==========================================================================///
/// File: BarrierElimination.cpp
///
/// Eliminate redundant SDE barriers between independent scheduling units.
/// When the write-set of the predecessor loop is provably disjoint from the
/// read-set of the successor loop, the barrier is marked for elimination so
/// ConvertSdeToArts skips generating arts.barrier.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_BARRIERELIMINATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/OperationAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(barrier_elimination);

using namespace mlir;
using namespace mlir::arts;

namespace {

/// Trace a carrier tensor operand back to its backing memref root through
/// mu_memref_to_tensor and tensor.cast/extract_slice chains.
static Value traceCarrierOperandToMemref(Value tensorOperand) {
  Value cur = tensorOperand;
  while (cur) {
    if (auto muOp = cur.getDefiningOp<sde::SdeMuMemrefToTensorOp>())
      return ValueAnalysis::stripMemrefViewOps(muOp.getMemref());
    if (auto castOp = cur.getDefiningOp<tensor::CastOp>()) {
      cur = castOp.getSource();
      continue;
    }
    if (auto sliceOp = cur.getDefiningOp<tensor::ExtractSliceOp>()) {
      cur = sliceOp.getSource();
      continue;
    }
    break;
  }
  return {};
}

/// Collect memref roots written by ops inside a region.
/// Handles both carrier-authoritative (linalg.generic DPS outputs) and
/// scalar (memref.store) bodies.
static void collectWriteSet(Region &region, llvm::DenseSet<Value> &writes) {
  region.walk([&](Operation *op) {
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      writes.insert(ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      for (Value output : generic.getDpsInits()) {
        Value root = traceCarrierOperandToMemref(output);
        if (root)
          writes.insert(root);
      }
    }
  });
}

/// Collect memref roots read by ops inside a region.
/// Handles both carrier-authoritative (linalg.generic DPS inputs) and
/// scalar (memref.load) bodies.
static void collectReadSet(Region &region, llvm::DenseSet<Value> &reads) {
  region.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      reads.insert(ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      for (Value input : generic.getDpsInputs()) {
        Value root = traceCarrierOperandToMemref(input);
        if (root)
          reads.insert(root);
      }
    }
  });
}

/// Check if two sets are disjoint.
static bool areSetsDisjoint(const llvm::DenseSet<Value> &a,
                            const llvm::DenseSet<Value> &b) {
  for (Value v : a)
    if (b.contains(v))
      return false;
  return true;
}

/// Find the su_iterate op that an operation represents. An su_iterate can
/// appear directly or nested inside an su_distribute wrapper.
static sde::SdeSuIterateOp findSuIterate(Operation *op) {
  if (auto suIt = dyn_cast<sde::SdeSuIterateOp>(op))
    return suIt;
  if (auto dist = dyn_cast<sde::SdeSuDistributeOp>(op)) {
    sde::SdeSuIterateOp result;
    dist.getBody().walk([&](sde::SdeSuIterateOp suIt) { result = suIt; });
    return result;
  }
  return {};
}

/// Return true if the operation is an SDE scheduling-unit container
/// (su_iterate or su_distribute wrapping one).
static bool isSuContainer(Operation *op) {
  return isa<sde::SdeSuIterateOp>(op) || isa<sde::SdeSuDistributeOp>(op);
}

struct BarrierEliminationPass
    : public arts::impl::BarrierEliminationBase<BarrierEliminationPass> {
  explicit BarrierEliminationPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    int eliminated = 0;

    getOperation().walk([&](sde::SdeSuBarrierOp barrier) {
      Block *block = barrier->getBlock();
      if (!block)
        return;

      Operation *predOp = nullptr;
      Operation *succOp = nullptr;

      // Walk backward to find predecessor su_iterate or su_distribute
      for (auto it = Block::reverse_iterator(barrier->getIterator());
           it != block->rend(); ++it) {
        if (isSuContainer(&*it)) {
          predOp = &*it;
          break;
        }
        if (!isMemoryEffectFree(&*it))
          break;
      }

      // Walk forward to find successor su_iterate or su_distribute
      for (auto it = std::next(barrier->getIterator()); it != block->end();
           ++it) {
        if (isSuContainer(&*it)) {
          succOp = &*it;
          break;
        }
        if (!isMemoryEffectFree(&*it))
          break;
      }

      if (!predOp || !succOp)
        return;

      auto predecessor = findSuIterate(predOp);
      auto successor = findSuIterate(succOp);
      if (!predecessor || !successor)
        return;

      // Both must have classification (analyzed)
      if (!predecessor.getStructuredClassificationAttr() ||
          !successor.getStructuredClassificationAttr())
        return;

      // Collect write set of predecessor, read set of successor.
      // Use the outer container's region to capture all memory ops.
      llvm::DenseSet<Value> predWrites;
      for (Region &region : predOp->getRegions())
        collectWriteSet(region, predWrites);

      llvm::DenseSet<Value> succReads;
      for (Region &region : succOp->getRegions())
        collectReadSet(region, succReads);

      if (predWrites.empty() || succReads.empty())
        return;

      if (areSetsDisjoint(predWrites, succReads)) {
        double syncCost = costModel ? costModel->getTaskSyncCost() : 0.0;
        barrier->setAttr(AttrNames::Operation::BarrierEliminated,
                         UnitAttr::get(barrier.getContext()));
        eliminated++;
        ARTS_DEBUG("Eliminated barrier (sync cost: " << syncCost << ")");
      }
    });

    ARTS_INFO("BarrierElimination: eliminated " << eliminated << " barrier(s)");

    // Phase 11: Nowait inference — conservative approach.
    // For reduction-only su_iterate ops (all memory operations touch only
    // the reduction accumulator memrefs), we can safely infer nowait because
    // the reduction strategy already handles synchronization.
    int nowaitInferred = 0;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      // Skip if already has nowait.
      if (op.getNowaitAttr())
        return;

      // Must be a reduction with a strategy selected.
      auto classification = op.getStructuredClassification();
      if (!classification ||
          *classification != sde::SdeStructuredClassification::reduction)
        return;
      if (!op.getReductionStrategyAttr())
        return;
      if (op.getReductionAccumulators().empty())
        return;

      // Collect the reduction accumulator memref bases.
      llvm::DenseSet<Value> accumulatorBases;
      for (Value acc : op.getReductionAccumulators())
        accumulatorBases.insert(ValueAnalysis::stripMemrefViewOps(acc));

      // Check all memory operations in the body — only accumulator
      // loads/stores are allowed for nowait inference. Handles both
      // carrier-authoritative (linalg.generic) and scalar (memref ops) bodies.
      bool allReductionLocal = true;
      op.getBody().walk([&](Operation *memOp) -> WalkResult {
        if (auto loadOp = dyn_cast<memref::LoadOp>(memOp)) {
          Value base = ValueAnalysis::stripMemrefViewOps(loadOp.getMemref());
          if (!accumulatorBases.contains(base)) {
            allReductionLocal = false;
            return WalkResult::interrupt();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(memOp)) {
          Value base = ValueAnalysis::stripMemrefViewOps(storeOp.getMemref());
          if (!accumulatorBases.contains(base)) {
            allReductionLocal = false;
            return WalkResult::interrupt();
          }
        } else if (auto generic = dyn_cast<linalg::GenericOp>(memOp)) {
          for (Value operand : generic.getDpsInputs()) {
            Value root = traceCarrierOperandToMemref(operand);
            if (root && !accumulatorBases.contains(root)) {
              allReductionLocal = false;
              return WalkResult::interrupt();
            }
          }
          for (Value operand : generic.getDpsInits()) {
            Value root = traceCarrierOperandToMemref(operand);
            if (root && !accumulatorBases.contains(root)) {
              allReductionLocal = false;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::skip(); // Don't walk into carrier body.
        }
        return WalkResult::advance();
      });

      if (allReductionLocal) {
        op.setNowaitAttr(UnitAttr::get(op.getContext()));
        nowaitInferred++;
        ARTS_DEBUG("Inferred nowait on reduction-only su_iterate");
      }
    });

    // Phase 11b: Elementwise nowait inference — elementwise loops have
    // disjoint per-iteration write sets by definition, so no implicit barrier
    // is needed if no subsequent su_iterate reads from the written memrefs.
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      if (op.getNowaitAttr())
        return;

      auto classification = op.getStructuredClassification();
      if (!classification)
        return;

      if (*classification != sde::SdeStructuredClassification::elementwise &&
          *classification !=
              sde::SdeStructuredClassification::elementwise_pipeline)
        return;

      // Collect write set — handles both carrier and scalar bodies.
      DenseSet<Value> writtenBases;
      bool hasSideEffects = false;

      op.getBody().walk([&](Operation *memOp) -> WalkResult {
        if (auto storeOp = dyn_cast<memref::StoreOp>(memOp)) {
          writtenBases.insert(
              ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
        } else if (isa<memref::LoadOp>(memOp)) {
          // Loads are fine.
        } else if (auto generic = dyn_cast<linalg::GenericOp>(memOp)) {
          for (Value output : generic.getDpsInits()) {
            Value root = traceCarrierOperandToMemref(output);
            if (root)
              writtenBases.insert(root);
          }
          return WalkResult::skip(); // Don't walk into carrier body.
        } else if (!isMemoryEffectFree(memOp) && !isa<scf::YieldOp>(memOp)) {
          hasSideEffects = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (hasSideEffects || writtenBases.empty())
        return;

      // Check if the immediate next su_iterate sibling reads from our
      // write set. If so, the barrier is needed.
      bool nextReadsOurWrites = false;
      Operation *nextOp = op->getNextNode();
      while (nextOp) {
        if (auto nextSu = dyn_cast<sde::SdeSuIterateOp>(nextOp)) {
          llvm::DenseSet<Value> nextReadBases;
          // Collect read set from successor — handles both paths.
          nextSu.getBody().walk([&](Operation *innerOp) -> WalkResult {
            if (auto loadOp = dyn_cast<memref::LoadOp>(innerOp)) {
              nextReadBases.insert(
                  ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
            } else if (auto generic = dyn_cast<linalg::GenericOp>(innerOp)) {
              for (Value input : generic.getDpsInputs()) {
                Value root = traceCarrierOperandToMemref(input);
                if (root)
                  nextReadBases.insert(root);
              }
              return WalkResult::skip();
            }
            return WalkResult::advance();
          });
          for (Value wb : writtenBases) {
            if (nextReadBases.contains(wb)) {
              nextReadsOurWrites = true;
              break;
            }
          }
          break; // only check immediate successor
        }
        // Skip barriers and memory-effect-free ops.
        if (!isMemoryEffectFree(nextOp) && !isa<sde::SdeSuBarrierOp>(nextOp))
          break;
        nextOp = nextOp->getNextNode();
      }

      if (!nextReadsOurWrites) {
        op.setNowaitAttr(UnitAttr::get(op.getContext()));
        nowaitInferred++;
        ARTS_DEBUG("Inferred nowait on elementwise su_iterate");
      }
    });

    ARTS_INFO("BarrierElimination: inferred nowait on " << nowaitInferred
                                                        << " op(s)");
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass>
createBarrierEliminationPass(sde::SDECostModel *costModel) {
  return std::make_unique<BarrierEliminationPass>(costModel);
}

} // namespace mlir::arts::sde
