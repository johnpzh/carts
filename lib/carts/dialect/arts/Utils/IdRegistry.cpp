///==========================================================================///
/// File: IdRegistry.cpp
///
/// Implementation of unified ID registry.
///==========================================================================///
#include "carts/dialect/arts/Utils/IdRegistry.h"
#include "carts/utils/Debug.h"
#include "carts/dialect/arts/Utils/LocationMetadata.h"
#include "carts/dialect/arts/Utils/MetadataAttrNames.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

ARTS_DEBUG_SETUP(id_registry)

///===--------------------------------------------------------------------===///
/// Initialization
///===--------------------------------------------------------------------===///

void IdRegistry::initializeFromModule(ModuleOp module) {
  if (initialized) {
    ARTS_DEBUG("IdRegistry: Already initialized, skipping");
    return;
  }

  ARTS_DEBUG("IdRegistry: Initializing from module");

  /// Collect all operations with valid locations.
  /// Keep duplicate locations as separate entries so loops/memrefs generated
  /// from macro-expanded source lines still get distinct IDs.
  struct ParsedLoc {
    Operation *op = nullptr;
    LocationMetadata loc;
    uint64_t walkOrder = 0;
  };
  llvm::SmallVector<ParsedLoc> locations;
  uint64_t walkOrder = 0;

  module.walk([&](Operation *op) {
    LocationMetadata loc = LocationMetadata::fromLocation(op->getLoc());
    ++walkOrder;
    if (!loc.isValid() || loc.key.empty() || loc.key == "unknown:0:0")
      return;
    locations.push_back({op, loc, walkOrder});
  });

  /// Sort by (file, line, column, original walk order).
  llvm::sort(locations, [](const ParsedLoc &a, const ParsedLoc &b) {
    if (a.loc.file != b.loc.file)
      return a.loc.file < b.loc.file;
    if (a.loc.line != b.loc.line)
      return a.loc.line < b.loc.line;
    if (a.loc.column != b.loc.column)
      return a.loc.column < b.loc.column;
    return a.walkOrder < b.walkOrder;
  });

  /// Assign sequential IDs.
  for (const auto &entry : locations) {
    operationCache[entry.op] = nextId;
    /// Keep first-seen location -> id for deterministic fallback lookups.
    locationCache.try_emplace(entry.loc.key, nextId);
    usedIds.insert(nextId);
    ARTS_DEBUG("IdRegistry: ID=" << nextId << " -> " << entry.loc.key);
    nextId++;
  }

  initialized = true;
  ARTS_DEBUG("IdRegistry: Initialized "
             << operationCache.size() << " operation IDs, next ID: " << nextId);
}

void IdRegistry::loadFromJson(llvm::StringRef jsonPath) {
  if (initialized) {
    ARTS_DEBUG("IdRegistry: Already initialized, skipping JSON load");
    return;
  }

  ARTS_DEBUG("IdRegistry: Loading IDs from " << jsonPath);

  auto fileOrErr = llvm::MemoryBuffer::getFile(jsonPath);
  if (!fileOrErr) {
    ARTS_DEBUG("IdRegistry: No JSON file found at " << jsonPath);
    return;
  }

  auto json = llvm::json::parse((*fileOrErr)->getBuffer());
  if (!json) {
    ARTS_DEBUG("IdRegistry: Failed to parse JSON: "
               << llvm::toString(json.takeError()));
    return;
  }

  auto *root = json->getAsObject();
  if (!root) {
    ARTS_DEBUG("IdRegistry: JSON root is not an object");
    return;
  }

  loadFromJsonObject(*root);
}

void IdRegistry::loadFromJsonObject(const llvm::json::Object &root) {
  /// Helper to load a section (loops or memrefs)
  auto loadSection = [&](llvm::StringRef sectionName,
                         llvm::StringRef locationField) {
    auto *section = root.getObject(sectionName);
    if (!section)
      return;

    for (const auto &entry : *section) {
      const auto *obj = entry.second.getAsObject();
      if (!obj)
        continue;

      auto idVal = obj->getInteger("arts_id");
      if (!idVal)
        continue;

      /// Try to get location from nested object (new format)
      std::string locKey;
      if (auto fileStr = obj->getString("file")) {
        auto line = obj->getInteger("line").value_or(0);
        auto col = obj->getInteger("column").value_or(0);
        locKey = LocationMetadata::getBasename(*fileStr) + ":" +
                 std::to_string(line) + ":" + std::to_string(col);
      }

      /// Fall back to string key (old format)
      if (locKey.empty()) {
        if (auto key = obj->getString(locationField))
          locKey = key->str();
      }

      if (locKey.empty())
        continue;

      locationCache[locKey] = *idVal;
      usedIds.insert(*idVal);
      nextId = std::max(nextId, *idVal + 1);
      ARTS_DEBUG("IdRegistry: Loaded ID=" << *idVal << " for " << locKey);
    }
  };

  loadSection("loops", AttrNames::LoopMetadata::LocationKey);
  loadSection("memrefs", AttrNames::MemrefMetadata::AllocationId);

  initialized = true;
  ARTS_DEBUG("IdRegistry: Loaded " << locationCache.size()
                                   << " IDs from JSON, next ID: " << nextId);
}

void IdRegistry::reset() {
  operationCache.clear();
  locationCache.clear();
  usedIds.clear();
  nextId = 1;
  initialized = false;
  ARTS_DEBUG("IdRegistry: State reset");
}

///===----------------------------------------------------------------------===///
/// ID Operations
///===----------------------------------------------------------------------===///

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

void IdRegistry::set(Operation *op, int64_t id) {
  if (!op || id <= 0)
    return;

  setIdAttribute(op, id);
  operationCache[op] = id;
  usedIds.insert(id);

  LocationMetadata loc = LocationMetadata::fromLocation(op->getLoc());
  if (!loc.key.empty())
    locationCache.try_emplace(loc.key, id);
}

int64_t IdRegistry::computeFromLocation(Operation *op) {
  if (!op)
    return 0;

  /// Check existing first
  if (auto attr = op->getAttrOfType<IntegerAttr>(AttrName)) {
    int64_t existing = attr.getInt();
    usedIds.insert(existing);
    LocationMetadata loc = LocationMetadata::fromLocation(op->getLoc());
    if (!loc.key.empty())
      locationCache.try_emplace(loc.key, existing);
    return existing;
  }

  if (auto it = operationCache.find(op); it != operationCache.end())
    return it->second;

  return assignFromLocation(op);
}

int64_t IdRegistry::allocateSequential() {
  int64_t id = nextId++;
  usedIds.insert(id);
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

///===----------------------------------------------------------------------===///
/// Location Management
///===----------------------------------------------------------------------===///

int64_t IdRegistry::getIdForLocation(llvm::StringRef locationKey) const {
  auto it = locationCache.find(locationKey);
  return it != locationCache.end() ? it->second : 0;
}

void IdRegistry::recordUsedId(int64_t id) {
  if (id <= 0)
    return;
  usedIds.insert(id);
  nextId = std::max(nextId, id + 1);
}

bool IdRegistry::isIdUsed(int64_t id) const { return usedIds.contains(id); }
