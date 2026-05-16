///==========================================================================///
/// File: IdRegistry.h
///
/// Unified ID registry that manages all arts_id assignment and location
/// mapping.
///==========================================================================///
#ifndef ARTS_UTILS_METADATA_IDREGISTRY_H
#define ARTS_UTILS_METADATA_IDREGISTRY_H

#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>

namespace mlir {
namespace carts::arts {

/// Unified ID registry for deterministic arts_id assignment.
/// This class is the SINGLE source of truth for:
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

#endif // ARTS_UTILS_METADATA_IDREGISTRY_H
