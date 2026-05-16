///==========================================================================///
/// File: IdRegistry.h
///
/// Deterministic arts_id assignment for runtime-lowering operations.
///==========================================================================///
#ifndef CARTS_DIALECT_ARTS_RT_UTILS_IDREGISTRY_H
#define CARTS_DIALECT_ARTS_RT_UTILS_IDREGISTRY_H

#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>

namespace mlir {
namespace carts::arts {

/// Runtime-lowering ID registry for deterministic arts_id assignment.
/// This class tracks:
/// - Location -> ID mapping
/// - ID collision tracking
/// - Sequential ID assignment
class IdRegistry {
public:
  static constexpr auto AttrName = AttrNames::Operation::ArtsId;
  static constexpr int64_t DefaultStride = 1000;

  IdRegistry() = default;

  ///===------------------------------------------------------------------===///
  /// ID Operations
  ///===------------------------------------------------------------------===///

  /// Get existing arts.id attribute from operation, or 0 if none
  int64_t get(Operation *op) const;

  /// Get or create ID for operation. Returns existing ID if present,
  /// otherwise assigns based on location.
  int64_t getOrCreate(Operation *op);

private:
  int64_t assignFromLocation(Operation *op);
  void setIdAttribute(Operation *op, int64_t id);

private:
  llvm::DenseMap<Operation *, int64_t> operationCache;
  llvm::StringMap<int64_t> locationCache;
  llvm::DenseSet<int64_t> usedIds;
  int64_t nextId = 1;
};

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_RT_UTILS_IDREGISTRY_H
