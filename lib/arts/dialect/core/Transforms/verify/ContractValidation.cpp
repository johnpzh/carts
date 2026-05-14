///==========================================================================///
/// File: ContractValidation.cpp
///
/// Lightweight verification pass for LoweringContractOp integrity.
///
/// Walks all arts.lowering_contract operations and validates:
///   1. Target value exists and is a valid memref
///   2. If supported_block_halo is set, depPattern must be stencil family
///   3. min_offsets/max_offsets must have matching rank
///   4. block_shape rank > 0 if present
///   5. owner_dims values are non-negative
///
/// Additionally walks DbAcquireOps and warns if stencil attrs are present
/// but no lowering contract is attached to the source pointer.
///
/// Collects and logs statistics: total contracts, per-pattern counts, orphans.
///==========================================================================///

#include "arts/Dialect.h"
#define GEN_PASS_DEF_CONTRACTVALIDATION
#include "arts/dialect/core/Analysis/db/OwnershipProof.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

/// Debug
#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(contract_validation);

using namespace mlir;
using namespace mlir::arts;

namespace {

struct ContractValidationPass
    : public impl::ContractValidationBase<ContractValidationPass> {

  ContractValidationPass() = default;
  explicit ContractValidationPass(bool fail) { this->failOnError = fail; }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ARTS_INFO_HEADER(ContractValidationPass);

    unsigned totalContracts = 0;
    unsigned invalidContracts = 0;
    unsigned orphanAcquires = 0;
    DenseMap<ArtsDepPattern, unsigned> patternCounts;

    /// ---------------------------------------------------------------
    /// Phase 1: Validate every LoweringContractOp
    /// ---------------------------------------------------------------
    module.walk([&](LoweringContractOp contract) {
      ++totalContracts;

      /// Track per-pattern statistics
      if (auto dp = contract.getDepPattern())
        ++patternCounts[*dp];

      /// 1. Target value must exist and be a valid memref
      Value target = contract.getTarget();
      if (!target) {
        contract.emitWarning("lowering_contract has null target value");
        ++invalidContracts;
        return;
      }
      if (!isa<MemRefType>(target.getType())) {
        contract.emitWarning("lowering_contract target is not a memref type");
        ++invalidContracts;
        return;
      }

      /// Checks 2-5 (supported_block_halo/stencil, offset rank,
      /// block_shape, owner_dims non-negative) are enforced by
      /// LoweringContractOp::verify() and omitted here to avoid
      /// duplication.
    });

    /// ---------------------------------------------------------------
    /// Phase 2: Walk DbAcquireOps, warn if stencil attrs present but
    ///          no LoweringContractOp on the source pointer
    /// ---------------------------------------------------------------
    module.walk([&](DbAcquireOp acquire) {
      Operation *op = acquire.getOperation();
      bool hasStencilAttrs = hasSupportedBlockHalo(op) ||
                             getStencilMinOffsets(op).has_value() ||
                             getStencilMaxOffsets(op).has_value() ||
                             getStencilOwnerDims(op).has_value();
      if (!hasStencilAttrs)
        return;

      Value sourcePtr = acquire.getSourcePtr();
      LoweringContractOp contractOp = getLoweringContractOp(sourcePtr);
      if (!contractOp) {
        acquire.emitWarning(
            "db_acquire has stencil attributes but no lowering_contract on "
            "source pointer");
        ++orphanAcquires;
      }
    });

    /// ---------------------------------------------------------------
    /// Phase 3a: Compute and stamp ownership proofs
    /// ---------------------------------------------------------------
    unsigned proofsStamped = 0;
    unsigned fullyProven = 0;

    module.walk([&](LoweringContractOp contract) {
      Value target = contract.getTarget();
      if (!target || !isa<MemRefType>(target.getType()))
        return;

      OwnershipProof proof = computeOwnershipProof(contract);
      if (proof.provenCount() > 0) {
        stampOwnershipProof(contract, proof);
        ++proofsStamped;
        if (proof.isFullyProven())
          ++fullyProven;
      }
    });

    ARTS_DEBUG("  Ownership proofs: " << proofsStamped << " stamped, "
                                      << fullyProven << " fully proven");

    /// ---------------------------------------------------------------
    /// Phase 3b: Verify proof consistency
    /// ---------------------------------------------------------------
    unsigned proofInconsistencies = 0;

    module.walk([&](LoweringContractOp contract) {
      OwnershipProof proof = readOwnershipProof(contract.getOperation());
      if (proof.provenCount() == 0)
        return;

      /// If owner_dim_reachability is false but block partitioning is
      /// requested, emit warning
      auto partMode = contract->getAttrOfType<PartitionModeAttr>(
          AttrNames::Operation::PartitionMode);
      if (!proof.ownerDimReachability && partMode &&
          (partMode.getValue() == PartitionMode::block ||
           partMode.getValue() == PartitionMode::stencil)) {
        contract.emitWarning("proof inconsistency: owner_dim_reachability is "
                             "false but block partitioning is requested");
        ++proofInconsistencies;
      }

      /// If halo_legality is false but supported_block_halo is set
      if (!proof.haloLegality &&
          contract.getSupportedBlockHalo().value_or(false)) {
        contract.emitWarning("proof inconsistency: halo_legality is false but "
                             "supported_block_halo is set");
        ++proofInconsistencies;
      }
    });

    ARTS_DEBUG("  Proof inconsistencies: " << proofInconsistencies);

    /// ---------------------------------------------------------------
    /// Phase 4: Advisory proof-gap warnings
    /// ---------------------------------------------------------------
    unsigned proofGapWarnings = 0;

    if (warnOnProofGaps) {
      /// Ownership proof gap checks on LoweringContractOps
      module.walk([&](LoweringContractOp contract) {
        Value target = contract.getTarget();
        if (!target || !isa<MemRefType>(target.getType()))
          return;

        auto ownerDims = contract.getOwnerDims();
        auto blockShape = contract.getBlockShape();
        auto depPattern = contract.getDepPattern();

        /// owner_dims without block_shape
        if (ownerDims && !ownerDims->empty() && blockShape.empty()) {
          contract.emitWarning("proof gap: owner_dims without block_shape");
          ++proofGapWarnings;
        }

        /// stencil-family depPattern without supported_block_halo
        if (depPattern && isStencilFamilyDepPattern(*depPattern) &&
            !contract.getSupportedBlockHalo()) {
          contract.emitWarning(
              "proof gap: stencil pattern without supported_block_halo");
          ++proofGapWarnings;
        }

        /// narrowable_dep without owner_dims
        if (contract->hasAttr(AttrNames::Operation::Contract::NarrowableDep) &&
            (!ownerDims || ownerDims->empty())) {
          contract.emitWarning("proof gap: narrowable_dep without owner_dims");
          ++proofGapWarnings;
        }

        /// owner_dims value >= memref rank
        if (ownerDims) {
          unsigned rank = cast<MemRefType>(target.getType()).getRank();
          for (int64_t dim : *ownerDims) {
            if (dim >= static_cast<int64_t>(rank)) {
              contract.emitWarning(
                  "proof gap: owner_dims value >= memref rank");
              ++proofGapWarnings;
              break;
            }
          }
        }

        /// block_shape without owner_dims
        if (!blockShape.empty() && (!ownerDims || ownerDims->empty())) {
          contract.emitWarning("proof gap: block_shape without owner_dims");
          ++proofGapWarnings;
        }
      });
    }

    /// ---------------------------------------------------------------
    /// Phase 3: Log statistics
    /// ---------------------------------------------------------------
    ARTS_INFO("Contract validation summary: "
              << totalContracts << " contracts, " << invalidContracts
              << " invalid, " << orphanAcquires << " orphan acquires"
              << ", " << proofGapWarnings << " proof gap warnings"
              << ", " << proofsStamped << " proofs stamped (" << fullyProven
              << " fully proven, " << proofInconsistencies
              << " inconsistencies)");
    for (auto &entry : patternCounts) {
      ARTS_INFO("  Pattern " << stringifyArtsDepPattern(entry.first) << ": "
                             << entry.second);
    }

    /// Fail the pass if requested and there were problems
    if (failOnError && (invalidContracts > 0 || orphanAcquires > 0)) {
      emitError(module.getLoc())
          << "Contract validation failed: " << invalidContracts
          << " invalid contracts, " << orphanAcquires << " orphan acquires";
      signalPassFailure();
    }

    ARTS_INFO_FOOTER(ContractValidationPass);
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::arts::createContractValidationPass(bool failOnError) {
  return std::make_unique<ContractValidationPass>(failOnError);
}
