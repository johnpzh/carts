///==========================================================================///
/// File: IdRegistry.cpp
///
/// Implementation of unified ID registry.
///==========================================================================///
#include "carts/dialect/arts-rt/Utils/IdRegistry.h"
#include "carts/utils/Debug.h"
#include "carts/utils/LocationMetadata.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

ARTS_DEBUG_SETUP(id_registry)

int64_t IdRegistry::get(Operation *op) const {
  if (!op)
    return 0;
  if (auto attr = op->getAttrOfType<IntegerAttr>(AttrName))
    return attr.getInt();
  return 0;
}

int64_t IdRegistry::getOrCreate(Operation *op) {
  if (!op)
    return 0;

  /// Check existing attribute
  if (auto existing = get(op))
    return existing;

  /// Check precomputed operation cache
  if (auto it = operationCache.find(op); it != operationCache.end()) {
    setIdAttribute(op, it->second);
    usedIds.insert(it->second);
    return it->second;
  }

  /// Assign from location
  int64_t id = assignFromLocation(op);
  if (id == 0)
    return 0;

  setIdAttribute(op, id);
  return id;
}

int64_t IdRegistry::assignFromLocation(Operation *op) {
  if (!op)
    return 0;

  if (auto it = operationCache.find(op); it != operationCache.end())
    return it->second;

  LocationMetadata loc = LocationMetadata::fromLocation(op->getLoc());
  if (loc.key.empty())
    return 0;

  /// Reuse first-seen location ID only if it has not been consumed by any
  /// operation. Otherwise allocate a new unique ID for this operation.
  if (auto it = locationCache.find(loc.key); it != locationCache.end()) {
    if (!usedIds.contains(it->second)) {
      operationCache[op] = it->second;
      ARTS_DEBUG("IdRegistry: Reusing ID=" << it->second << " for " << loc.key);
      return it->second;
    }
  }

  /// Assign new sequential.
  int64_t id = nextId++;
  operationCache[op] = id;
  locationCache.try_emplace(loc.key, id);
  usedIds.insert(id);
  ARTS_DEBUG("IdRegistry: Assigned new ID=" << id << " for " << loc.key);
  return id;
}

void IdRegistry::setIdAttribute(Operation *op, int64_t id) {
  OpBuilder builder(op->getContext());
  op->setAttr(AttrName, builder.getI64IntegerAttr(id));
}
