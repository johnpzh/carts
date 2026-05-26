#ifndef CARTS_DIALECT_ARTS_RT_UTILS_ARTSRTATTRNAMES_H
#define CARTS_DIALECT_ARTS_RT_UTILS_ARTSRTATTRNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::carts::arts_rt::AttrNames {

/// Operation-level marker attributes attached to non-arts_rt ops (the
/// outlined EDT function, runtime call ops, runtime epoch ops) that cannot
/// be inherent ODS attributes on an arts_rt op. The constants live here so
/// producers (ArtsToRt lowering) and consumers (ArtsRtToLLVM patterns)
/// share a single source of truth.
namespace Operation {

/// Stamped by EdtLowering on launch ops that are known to run on the local
/// node and therefore skip remote dispatch in the runtime-call lowering.
inline constexpr llvm::StringLiteral ReadyLocalLaunch =
    "arts.ready_local_launch";

/// Stamped on epoch lowerings whose caller does not contribute an active
/// participant count to the epoch (so the runtime call does not start a
/// caller-active epoch).
inline constexpr llvm::StringLiteral NoStartEpoch = "arts.no_start_epoch";

} // namespace Operation

/// Split launch state schema markers stamped on the host launch helper to
/// pin the state/dep payload layouts decided during ARTS lowering.
namespace LaunchState {

inline constexpr llvm::StringLiteral StateSchema = "arts.launch.state_schema";
inline constexpr llvm::StringLiteral DepSchema = "arts.launch.dep_schema";

} // namespace LaunchState

/// RT-facing loop execution hints copied onto outlined EDT functions. These
/// live on func/LLVM function ops, so they cannot be ODS accessors on
/// arts_rt ops; in-dialect producers use generated accessors and copy the
/// final values onto the outlined func through this shared name set.
namespace Rt {

inline constexpr llvm::StringLiteral VectorizeWidth =
    "arts.rt.vectorize_width";
inline constexpr llvm::StringLiteral UnrollFactor = "arts.rt.unroll_factor";
inline constexpr llvm::StringLiteral InterleaveCount =
    "arts.rt.interleave_count";

} // namespace Rt

} // namespace mlir::carts::arts_rt::AttrNames

#endif /// CARTS_DIALECT_ARTS_RT_UTILS_ARTSRTATTRNAMES_H
