///==========================================================================///
/// File: DeadIrCleanup.h
///
/// Shared dead-IR cleanup for dialect-neutral helper IR.
///==========================================================================///

#ifndef CARTS_UTILS_DEADIRCLEANUP_H
#define CARTS_UTILS_DEADIRCLEANUP_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace carts {

struct DeadIrCleanupResult {
  unsigned loads = 0;
  unsigned stores = 0;
  unsigned noOpStores = 0;
  unsigned allocas = 0;
  unsigned pureOps = 0;
  unsigned symbols = 0;

  unsigned total() const {
    return loads + stores + noOpStores + allocas + pureOps + symbols;
  }
};

/// Remove dead dialect-neutral helper IR:
/// - memref loads whose values are unused
/// - stores to stack slots that are never loaded through aliases
/// - no-op load/store self writes
/// - unused memref allocas
/// - unused memory-effect-free leaf ops
/// - private symbols with no symbol uses when requested
DeadIrCleanupResult runDeadIrCleanup(ModuleOp module,
                                     bool removeSymbols = true);

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_DEADIRCLEANUP_H
