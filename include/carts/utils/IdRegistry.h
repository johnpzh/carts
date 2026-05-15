///==========================================================================///
/// File: IdRegistry.h
///
/// Unified ID registry that manages all arts_id assignment and location
/// mapping.
///==========================================================================///
#ifndef ARTS_UTILS_METADATA_IDREGISTRY_H
#define ARTS_UTILS_METADATA_IDREGISTRY_H

#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace mlir {
namespace arts {

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
  /// Initialization
  ///===------------------------------------------------------------------===///

  /// Initialize registry by sorting all operations in module by location
  /// and assigning sequential IDs (1, 2, 3...)
  void initializeFromModule(ModuleOp module);

  /// Load ID mappings from existing .carts-metadata.json file
  void loadFromJson(llvm::StringRef jsonPath);

  /// Load from parsed JSON object
  void loadFromJsonObject(const llvm::json::Object &root);

  /// Reset all state
  void reset();

  ///===------------------------------------------------------------------===///
  /// ID Operations
  ///===------------------------------------------------------------------===///

  /// Get existing arts.id attribute from operation, or 0 if none
  int64_t get(Operation *op) const;

  /// Get or create ID for operation. Returns existing ID if present,
  /// otherwise assigns based on location.
  int64_t getOrCreate(Operation *op);

  /// Force-set arts.id attribute and record to avoid collisions
  void set(Operation *op, int64_t id);

  /// Compute deterministic ID from location without attaching attribute
  int64_t computeFromLocation(Operation *op);

  /// Allocate next sequential ID
  int64_t allocateSequential();

  ///===------------------------------------------------------------------===///
  /// Location Management
  ///===------------------------------------------------------------------===///

  /// Get ID for a location key (returns 0 if not found)
  int64_t getIdForLocation(llvm::StringRef locationKey) const;

  /// Record an ID as used
  void recordUsedId(int64_t id);

  /// Check if an ID is already used
  bool isIdUsed(int64_t id) const;

  ///===------------------------------------------------------------------===///
  /// State Queries
  ///===------------------------------------------------------------------===///

  bool isInitialized() const { return initialized; }
  int64_t getNextId() const { return nextId; }
  size_t getCacheSize() const { return locationCache.size(); }

private:
  int64_t assignFromLocation(Operation *op);
  void setIdAttribute(Operation *op, int64_t id);

private:
  llvm::DenseMap<Operation *, int64_t> operationCache;
  llvm::StringMap<int64_t> locationCache;
  llvm::DenseSet<int64_t> usedIds;
  int64_t nextId = 1;
  bool initialized = false;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_METADATA_IDREGISTRY_H
