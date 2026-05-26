#ifndef CARTS_DIALECT_ARTS_UTILS_ARTSATTRNAMES_H
#define CARTS_DIALECT_ARTS_UTILS_ARTSATTRNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::carts::arts::AttrNames {

/// Module-level marker attributes that decorate the top-level ModuleOp with
/// ARTS runtime configuration. These live on the builtin module (not on a
/// CARTS op), so they cannot be inherent ODS attributes; producers and
/// consumers across the pipeline share the names through this header.
namespace Module {

inline constexpr llvm::StringLiteral RuntimeConfigPath =
    "arts.runtime_config_path";
inline constexpr llvm::StringLiteral RuntimeConfigData =
    "arts.runtime_config_data";
inline constexpr llvm::StringLiteral RuntimeTotalWorkers =
    "arts.runtime_total_workers";
inline constexpr llvm::StringLiteral RuntimeTotalNodes =
    "arts.runtime_total_nodes";
inline constexpr llvm::StringLiteral RuntimeStaticWorkers =
    "arts.runtime_static_workers";

} // namespace Module

/// Operation-level marker attributes attached to non-ARTS ops (func.func,
/// LLVM funcs, scf.for, etc.) that cannot be inherent ODS attributes. The
/// constants live here so producers and consumers share a single source of
/// truth.
namespace Operation {

/// EDT identity. Stamped by partitioning analyses on the EDT's source op so
/// downstream passes can correlate IR sites with runtime EDT records.
inline constexpr llvm::StringLiteral ArtsId = "arts.id";

/// EDT creation-site identifier carried on calls/launches lowered into
/// runtime helper calls.
inline constexpr llvm::StringLiteral ArtsCreateId = "arts.create_id";

/// Symbol of the outlined EDT body function carried on the launch op.
/// Intentionally unprefixed for legacy compatibility with the host call
/// shim.
inline constexpr llvm::StringLiteral OutlinedFunc = "outlined_func";

/// PatternAnalysis revision counter used to invalidate downstream caches
/// when the semantic pattern family on a producer op changes.
inline constexpr llvm::StringLiteral PatternRevision = "arts.pattern_revision";

/// Marker stamped on loops produced by ARTS block-loop strip-mining so the
/// support pass can recognize and clear its own outputs.
inline constexpr llvm::StringLiteral StripMiningGenerated =
    "arts.block_loop_strip_mining.generated";

} // namespace Operation

} // namespace mlir::carts::arts::AttrNames

#endif /// CARTS_DIALECT_ARTS_UTILS_ARTSATTRNAMES_H
