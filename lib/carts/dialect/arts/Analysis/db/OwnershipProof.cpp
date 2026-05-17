///==========================================================================///
/// File: OwnershipProof.cpp
///
/// Implementation of proof-driven ownership analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/db/OwnershipProof.h"
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

OwnershipProof
mlir::carts::arts::computeOwnershipProof(LoweringContractOp contractOp) {
  OwnershipProof proof;
  if (!contractOp)
    return proof;

  Value target = contractOp.getTarget();
  if (!target || !isa<MemRefType>(target.getType()))
    return proof;

  auto memrefType = cast<MemRefType>(target.getType());
  unsigned rank = memrefType.getRank();

  auto ownerDims = contractOp.getOwnerDims();
  auto depPattern = contractOp.getDepPattern();
  bool supportedBlockHalo = contractOp.getSupportedBlockHalo().value_or(false);
  bool narrowableDep = contractOp.getNarrowableDep().value_or(false);

  /// 1. ownerDimReachability: owner dims are present, non-empty, and within
  /// rank
  if (ownerDims && !ownerDims->empty()) {
    bool allValid = true;
    for (int64_t dim : *ownerDims) {
      if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        allValid = false;
        break;
      }
    }
    proof.ownerDimReachability = allValid;
  }

  /// 2. partitionAccessMapping: partition mode is block-compatible and
  ///    owner dims rank matches memref rank in the relevant dimensions
  if (proof.ownerDimReachability && depPattern) {
    auto partitionMode = contractOp->getAttrOfType<PartitionModeAttr>(
        ::mlir::carts::arts::AttrNames::Operation::PartitionMode);
    bool hasBlockCompatibleMode =
        !partitionMode || partitionMode.getValue() == PartitionMode::block ||
        partitionMode.getValue() == PartitionMode::stencil;
    bool ownerDimsWithinRank = true;
    if (ownerDims) {
      for (int64_t dim : *ownerDims) {
        if (dim >= static_cast<int64_t>(rank)) {
          ownerDimsWithinRank = false;
          break;
        }
      }
    }
    proof.partitionAccessMapping =
        hasBlockCompatibleMode && ownerDimsWithinRank;
  }

  /// 3. haloLegality: for stencil family, check offsets are present;
  ///    for uniform family, always true
  if (depPattern) {
    if (isStencilFamilyDepPattern(*depPattern)) {
      /// Stencil: need offsets to prove halo legality
      auto minOffsets = contractOp.getMinOffsets();
      auto maxOffsets = contractOp.getMaxOffsets();
      proof.haloLegality =
          (!minOffsets.empty() && !maxOffsets.empty()) || supportedBlockHalo;
    } else if (isUniformFamilyDepPattern(*depPattern)) {
      /// Uniform: no halo needed, always legal
      proof.haloLegality = true;
    }
    /// Unknown or other patterns: stays false
  }

  /// 4. depSliceSoundness: check narrowable_dep attr or uniform pattern
  if (depPattern) {
    if (narrowableDep) {
      proof.depSliceSoundness = true;
    } else if (isUniformFamilyDepPattern(*depPattern)) {
      proof.depSliceSoundness = true;
    }
  }

  proof.relaunchStateSoundness = true;

  return proof;
}

void mlir::carts::arts::stampOwnershipProof(LoweringContractOp contractOp,
                                     const OwnershipProof &proof) {
  if (!contractOp)
    return;
  auto *ctx = contractOp->getContext();
  contractOp->setAttr(::mlir::carts::arts::AttrNames::Operation::Proof::OwnerDimReachability,
                      BoolAttr::get(ctx, proof.ownerDimReachability));
  contractOp->setAttr(::mlir::carts::arts::AttrNames::Operation::Proof::PartitionAccessMapping,
                      BoolAttr::get(ctx, proof.partitionAccessMapping));
  contractOp->setAttr(::mlir::carts::arts::AttrNames::Operation::Proof::HaloLegality,
                      BoolAttr::get(ctx, proof.haloLegality));
  contractOp->setAttr(::mlir::carts::arts::AttrNames::Operation::Proof::DepSliceSoundness,
                      BoolAttr::get(ctx, proof.depSliceSoundness));
  contractOp->setAttr(::mlir::carts::arts::AttrNames::Operation::Proof::RelaunchStateSoundness,
                      BoolAttr::get(ctx, proof.relaunchStateSoundness));
}

OwnershipProof mlir::carts::arts::readOwnershipProof(Operation *op) {
  OwnershipProof proof;
  if (!op)
    return proof;
  if (auto attr = op->getAttrOfType<BoolAttr>(
          ::mlir::carts::arts::AttrNames::Operation::Proof::OwnerDimReachability))
    proof.ownerDimReachability = attr.getValue();
  if (auto attr = op->getAttrOfType<BoolAttr>(
          ::mlir::carts::arts::AttrNames::Operation::Proof::PartitionAccessMapping))
    proof.partitionAccessMapping = attr.getValue();
  if (auto attr = op->getAttrOfType<BoolAttr>(
          ::mlir::carts::arts::AttrNames::Operation::Proof::HaloLegality))
    proof.haloLegality = attr.getValue();
  if (auto attr = op->getAttrOfType<BoolAttr>(
          ::mlir::carts::arts::AttrNames::Operation::Proof::DepSliceSoundness))
    proof.depSliceSoundness = attr.getValue();
  if (auto attr = op->getAttrOfType<BoolAttr>(
          ::mlir::carts::arts::AttrNames::Operation::Proof::RelaunchStateSoundness))
    proof.relaunchStateSoundness = attr.getValue();
  return proof;
}
