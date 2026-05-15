///==========================================================================///
/// File: OwnershipProof.h
///
/// Proof-driven ownership analysis for lowering contracts.
///
/// Computes a 5-dimension proof vector that certifies various aspects of
/// ownership soundness for a distributed kernel:
///   - ownerDimReachability: owner dims are statically reachable
///   - partitionAccessMapping: partition-to-access mapping is sound
///   - haloLegality: halo widening is legal
///   - depSliceSoundness: dependency narrowing is sound
///   - relaunchStateSoundness: relaunch state is stable
///
/// When all 5 dimensions are proven, downstream passes can trust the
/// contract's spatial layout without re-deriving evidence from the graph.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_DB_OWNERSHIPPROOF_H
#define ARTS_DIALECT_CORE_ANALYSIS_DB_OWNERSHIPPROOF_H

#include "carts/Dialect.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace carts::arts {

class DbAnalysis;

struct OwnershipProof {
  bool ownerDimReachability = false;
  bool partitionAccessMapping = false;
  bool haloLegality = false;
  bool depSliceSoundness = false;
  bool relaunchStateSoundness = false;

  bool isFullyProven() const {
    return ownerDimReachability && partitionAccessMapping && haloLegality &&
           depSliceSoundness && relaunchStateSoundness;
  }

  unsigned provenCount() const {
    return static_cast<unsigned>(ownerDimReachability) +
           static_cast<unsigned>(partitionAccessMapping) +
           static_cast<unsigned>(haloLegality) +
           static_cast<unsigned>(depSliceSoundness) +
           static_cast<unsigned>(relaunchStateSoundness);
  }
};

/// Compute an ownership proof for a LoweringContractOp by inspecting its
/// attributes and the surrounding EDT/loop context.
OwnershipProof computeOwnershipProof(LoweringContractOp contractOp);

/// Stamp proof attributes onto a LoweringContractOp.
void stampOwnershipProof(LoweringContractOp contractOp,
                         const OwnershipProof &proof);

/// Read proof attributes from a LoweringContractOp (if present).
OwnershipProof readOwnershipProof(Operation *op);

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_DB_OWNERSHIPPROOF_H
