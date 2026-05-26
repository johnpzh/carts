#ifndef CARTS_DIALECT_SDE_UTILS_SDEATTRNAMES_H
#define CARTS_DIALECT_SDE_UTILS_SDEATTRNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::carts::sde::AttrNames {

/// SDE marker attached to host-side OpenMP regions that intentionally remain
/// unconverted by OmpToSde. Verifiers and downstream passes test for this
/// marker to allow the region to survive past the SDE objects-only check.
inline constexpr llvm::StringLiteral KeepHostOpenMP = "sde.keep_host_openmp";

/// CPS candidate annotations attached to SDE ops before CPS plan
/// materialization. Producers (BarrierElimination) and verifiers
/// (VerifySdeCpsPlan) share these names. The four attrs are set together as a
/// group; partial sets are an error.
inline constexpr llvm::StringLiteral CpsCandidateGroupId =
    "cps_candidate_group_id";
inline constexpr llvm::StringLiteral CpsCandidateStageIndex =
    "cps_candidate_stage_index";
inline constexpr llvm::StringLiteral CpsCandidateStageCount =
    "cps_candidate_stage_count";
inline constexpr llvm::StringLiteral CpsCandidateRequiresTokenizedDataflow =
    "cps_candidate_requires_tokenized_dataflow";

} // namespace mlir::carts::sde::AttrNames

#endif /// CARTS_DIALECT_SDE_UTILS_SDEATTRNAMES_H
